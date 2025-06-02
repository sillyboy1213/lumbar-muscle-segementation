import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 unet_Encoder, uent_Decoder, BoundarySensitiveAllocationModule 已经定义在别处
from networks.u_net import unet_Encoder, uent_Decoder # 您的导入路径
from networks.custom_modules.RegionDetection import BoundarySensitiveAllocationModule # 您的导入路径

# --- 1. 定义 SpatialAttentionGate (SAG) 类 ---
class SpatialAttentionGate(nn.Module):
    def __init__(self, C_semantic, C_attention_key, C_alpha_intermediate):
        """
        Spatial Attention Gate (SAG) module based on the ESNet paper diagram.
        Args:
            C_semantic (int): Channels of the semantic features (X_dec_c, e.g., enc_final).
            C_attention_key (int): Channels of the aligned attention key (X_dec_k),
                                   derived from bbox features. Expected to be == C_semantic.
            C_alpha_intermediate (int): Intermediate channels for activation map alpha.
                                        (e.g., num_classes or C_semantic).
        """
        super(SpatialAttentionGate, self).__init__()
        if C_semantic != C_attention_key:
            # 根据图示 Psi_1 是 X_dec_c * X_dec_k, 通常要求通道数匹配或广播
            # 这里我们假设它们是匹配的，或者 X_dec_k 已经被处理成与 X_dec_c 通道数一致
            print(f"Warning: C_semantic ({C_semantic}) and C_attention_key ({C_attention_key}) for SAG Psi_1 differ. Element-wise product assumes compatibility.")

        self.C_semantic = C_semantic
        self.C_alpha_intermediate = C_alpha_intermediate

        # 1x1 Convolution for generating pre-softmax alpha (input from Psi_1 output)
        self.conv1x1_alpha = nn.Conv2d(C_semantic, C_alpha_intermediate, kernel_size=1, bias=False)

        # If C_alpha_intermediate (e.g., num_classes) is different from C_semantic,
        # another 1x1 conv is needed to expand/project alpha for Psi_2
        if C_alpha_intermediate != C_semantic:
            self.conv_expand_alpha = nn.Conv2d(C_alpha_intermediate, C_semantic, kernel_size=1, bias=False)
        else:
            self.conv_expand_alpha = None

    def forward(self, X_dec_c, X_dec_k):
        """
        Args:
            X_dec_c (Tensor): Semantic features (batch, C_semantic, H, W).
            X_dec_k (Tensor): Aligned attention key (batch, C_attention_key, H, W).
                              Should have C_attention_key == C_semantic.
        """
        # Psi_1: Element-wise product (X_dec_c * X_dec_k)
        attended_features_psi1 = X_dec_c * X_dec_k

        # 1x1 Convolution for alpha
        alpha_intermediate = self.conv1x1_alpha(attended_features_psi1)

        # Softmax function (over channel dimension for class-specific attention)
        activation_map_alpha = torch.softmax(alpha_intermediate, dim=1) # (B, C_alpha_intermediate, H, W)

        # Expand/project alpha if its channel dimension is different from X_dec_c for Psi_2
        alpha_for_psi2 = activation_map_alpha
        if self.conv_expand_alpha:
            alpha_for_psi2 = self.conv_expand_alpha(activation_map_alpha) # (B, C_semantic, H, W)

        # Psi_2: Element-wise product (X_dec_c * alpha_for_psi2)
        features_for_sigmoid = X_dec_c * alpha_for_psi2

        # Sigmoid
        gated_signal = torch.sigmoid(features_for_sigmoid)

        # Residual connection (Add)
        output_sag = X_dec_c + gated_signal

        return output_sag

# --- 2. 修改 MultiTaskUNetWithRegionDetection 类 ---
class MultiTaskUNetWithRegionDetection(nn.Module):
    def __init__(self, img_ch=1, num_classes=6, depth=2, enc_final_channels=512): # 添加 enc_final_channels
        super(MultiTaskUNetWithRegionDetection, self).__init__()
        self.num_classes = num_classes # 保存 num_classes 供 generate_one_hot_map 使用

        # U-Net 模块（分割任务）
        self.encoder = unet_Encoder(img_ch=img_ch, depth=depth) # 确保其输出通道与 enc_final_channels 匹配
        self.first_decoder = uent_Decoder(num_classes=num_classes)
        self.second_decoder = uent_Decoder(num_classes=num_classes)

        # 边界敏感分配模块（区域检测任务）
        # num_organs 通常是 num_classes - 1 (如果有一个背景类)
        self.region_detection = BoundarySensitiveAllocationModule(num_organs=num_classes - 1)

        # 用于处理 bbox_spatial_map (Q_a) 以生成 X_dec_k 的模块
        # 输入通道为 num_classes (来自 one_hot_map), 输出通道为 enc_final_channels
        # 这个模块是可训练的，对应图示中 X_dec_k 的生成过程
        self.bbox_map_to_Xdec_k_processor = nn.Sequential(
            nn.Conv2d(num_classes, enc_final_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 可能需要更多层或根据输入输出尺寸调整结构
        )

        # Spatial Attention Gate (SAG) 模块实例化
        self.sag_module = SpatialAttentionGate(
            C_semantic=enc_final_channels,
            C_attention_key=enc_final_channels, # X_dec_k 的通道数，与 enc_final 匹配
            C_alpha_intermediate=num_classes    # activation map alpha 的中间通道数，设为 num_classes
        )

    def generate_one_hot_map(self, bboxes, output_size_hw):
        """
        将边界框 (bboxes) 转换为独热码图像 (Q_a)
        :param bboxes: (batch, num_organs, 4)，边界框坐标, num_organs = num_classes - 1
        :param output_size_hw: 图像输出尺寸 (height, width)
        :return: 独热码图像 (batch, num_classes, height, width)
        """
        batch_size, num_organs, _ = bboxes.shape # num_organs 来自 bboxes
        
        # 初始化 one_hot_map，包含背景在内的所有类别
        # 假设类别0是背景
        one_hot_map = torch.zeros(batch_size, self.num_classes, output_size_hw[0], output_size_hw[1], device=bboxes.device)

        for i in range(batch_size):
            for j in range(num_organs): # 遍历检测到的 num_organs 个前景目标
                # bboxes[i, j] 对应器官 j, 映射到 one_hot_map 的通道 j+1 (因为通道0是背景)
                xmin, ymin, xmax, ymax = map(int, bboxes[i, j].tolist())

                # 确保坐标在图像边界内
                h, w = output_size_hw
                xmin_clamped = max(0, min(xmin, w - 1))
                ymin_clamped = max(0, min(ymin, h - 1))
                xmax_clamped = max(xmin_clamped, min(xmax, w)) # xmax 是slice的上限，可以等于w
                ymax_clamped = max(ymin_clamped, min(ymax, h)) # ymax 是slice的上限，可以等于h
                
                if xmin_clamped < xmax_clamped and ymin_clamped < ymax_clamped: # 确保是有效的框区域
                    one_hot_map[i, j + 1, ymin_clamped:ymax_clamped, xmin_clamped:xmax_clamped] = 1
        
        # 背景通道 (channel 0) 可以根据需要设定，例如：
        # one_hot_map[:, 0, :, :] = 1 - torch.max(one_hot_map[:, 1:, :, :], dim=1, keepdim=False)[0]
        # 当前实现中，背景通道默认为0，除非显式填充。

        return one_hot_map


    def forward(self, x):
        # 1. 经过 encoder 得到编码特征
        enc_features = self.encoder(x)
        enc_final = enc_features[-1]  # (batch, enc_final_channels, H_enc, W_enc), e.g., (B, 512, 32, 32) -> X_dec_c

        # 2. 经过第一个 decoder 得到初步分割结果
        first_decoder_logits = self.first_decoder(enc_final, enc_features[:-1])
        first_decoder_probs = torch.sigmoid(first_decoder_logits)  # (B, num_classes, H_orig, W_orig)

        # 3. 通过区域检测模块得到边界框
        # region_detection_output: (batch, num_organs, 4)
        region_detection_output = self.region_detection(first_decoder_probs)

        # 4. 生成独热码图像 Q_a (Explicit attention query)
        # one_hot_map: (batch, num_classes, H_orig, W_orig), e.g., (B, 6, 512, 512)
        one_hot_map = self.generate_one_hot_map(region_detection_output, x.shape[2:]) # x.shape[2:] is (H_orig, W_orig)

        # --- SAG 融合流程 ---
        # 5. 处理 Q_a (one_hot_map) 以生成 X_dec_k (Aligned attention key)
        # X_dec_k_processed: (batch, enc_final_channels, H_intermediate, W_intermediate)
        X_dec_k_processed = self.bbox_map_to_Xdec_k_processor(one_hot_map)
        # 插值使其空间维度与 enc_final (X_dec_c) 匹配
        # X_dec_k: (batch, enc_final_channels, H_enc, W_enc)
        X_dec_k = F.interpolate(X_dec_k_processed, size=enc_final.shape[2:], mode='bilinear', align_corners=True)
        # 注意：X_dec_k 在输入 SAG 之前不应使用 Sigmoid，这与您之前的代码不同，但符合图示。

        # 6. 将 X_dec_c (enc_final) 和 X_dec_k 送入 Spatial Attention Gate (SAG)
        decoder_input = self.sag_module(enc_final, X_dec_k) # SAG的输出作为下一个解码器的输入

        # 7. 经过第二个 decoder 得到最终结果
        segmentation_output_logits = self.second_decoder(decoder_input, enc_features[:-1])
        final_segmentation_output = torch.sigmoid(segmentation_output_logits) # (B, num_classes, H_orig, W_orig)

        # 返回最终分割结果，以及用于调试或计算辅助损失的中间图
        # 您之前的代码返回 (segmentation_output, one_hot_map, activation)
        # 'activation' 之前是 one_hot_map。现在可以考虑返回 X_dec_k 或 SAG 内部的 alpha。
        # 为了简单，我们先只返回主要输出和 one_hot_map。
        return final_segmentation_output, one_hot_map, X_dec_k # X_dec_k 作为 "activation" 示例


if __name__ == '__main__':
    # --- 主程序测试部分 ---
    num_classes = 6
    depth = 4 # unet_Encoder的深度
    enc_final_channels = 512 # 编码器最终输出通道数

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    net = MultiTaskUNetWithRegionDetection(
        img_ch=1,
        num_classes=num_classes,
        depth=depth,
        enc_final_channels=enc_final_channels
    ).to(device)

    total = sum([param.nelement() for param in net.parameters()])
    print(f"Number of parameters: {total / 1e6:.3f}M")

    batch_size = 2
    input_channels = 1
    H_orig, W_orig = 512, 512 # 原始输入图像尺寸
    input_tensor = torch.randn(batch_size, input_channels, H_orig, W_orig).to(device)

    # 运行模型
    try:
        output_seg, output_onehot, output_X_dec_k = net(input_tensor)

        print("Segmentation Output Shape:", output_seg.shape) # 期望 (B, num_classes, H_orig, W_orig)
        print("One-Hot Map (Q_a) Shape:", output_onehot.shape) # 期望 (B, num_classes, H_orig, W_orig)
        print("Aligned Attention Key (X_dec_k) Shape:", output_X_dec_k.shape) # 期望 (B, enc_final_channels, H_enc, W_enc)
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()