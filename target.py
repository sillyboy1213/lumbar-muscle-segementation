"""
本脚本定义了 MultiTaskUNetWithRegionDetection 模型，这是一个多任务学习模型，
旨在同时执行图像分割和区域检测（通过边界框）。

模型结构与核心组件:
1.  **U-Net 模块 (分割任务)**:
    *   `encoder`: 一个标准的U-Net编码器 (`unet_Encoder`)，用于从输入图像中提取多尺度特征。
    *   `first_decoder`: 第一个U-Net解码器 (`uent_Decoder`)，接收编码器的最终特征和跳跃连接特征，
        生成初步的分割结果。
    *   `second_decoder`: 第二个U-Net解码器 (`uent_Decoder`)，接收经过区域检测模块信息增强的特征
        以及编码器的跳跃连接特征，生成最终的精细化分割结果。

2.  **边界敏感分配模块 (Region Detection Task)**:
    *   `region_detection`: 一个 `BoundarySensitiveAllocationModule` 实例，用于根据初步分割结果
        预测目标区域的边界框。输入是第一个解码器的输出，输出是每个类别（不包括背景）的边界框坐标。

3.  **下采样模块 (`downsampler`)**:
    *   一个简单的卷积序列，用于将从边界框生成的 `activation_maps`（激活图）下采样，
        使其尺寸与U-Net编码器最深层特征图的尺寸相匹配，以便后续融合。

模型前向传播流程 (`forward` 方法):
1.  **编码**: 输入图像 `x` 通过 `encoder` 得到多层编码特征 `enc_features`，其中 `enc_final` 是最深层的特征。
2.  **初步分割**: `enc_final` 和 `enc_features[:-1]` (跳跃连接) 送入 `first_decoder`，得到初步分割图 `first_decoder_output`。
    该输出经过 Sigmoid 激活。
3.  **区域检测**: `first_decoder_output` 送入 `region_detection` 模块，得到预测的边界框 `region_detection_output`。
4.  **生成独热码图**: 调用 `generate_one_hot_map` 方法，将 `region_detection_output` (边界框) 转换为与输入图像尺寸相同的
    多通道独热编码图 `one_hot_map`。每个通道代表一个类别，目标区域为1，其余为0。背景通道也被添加。
5.  **生成激活图**: 当前实现中，`activation_maps` 直接使用了 `one_hot_map` (注释中提到这主要用于验证特征提取能力)。
    理想情况下，这里可以结合 `first_decoder_output` 和 `one_hot_map` 来生成更有信息的激活图。
    `activation` 变量存储了这一阶段的激活图，用于模型输出。
6.  **激活图处理**: `activation_maps` 经过 `downsampler` 进行下采样，然后通过双线性插值 (`F.interpolate`) 调整到
    与 `enc_final` 相同的空间维度，并应用 Sigmoid 激活。
7.  **特征融合 (输入到第二个解码器)**: 当前实现中，`decoder_input` 直接使用了处理后的 `activation_maps`
    (注释中提到这可能不正确，主要用于测试目标检测模块的特征提取能力)。
    理想情况下，这里应该将 `enc_final` 与处理后的 `activation_maps` 进行有效的融合（如逐元素相乘、相加或拼接）。
8.  **最终分割**: `decoder_input` (融合后的特征) 和 `enc_features[:-1]` (跳跃连接) 送入 `second_decoder`，
    得到最终的分割结果 `segmentation_output`。

模型输出:
*   `segmentation_output`: 最终的分割图 (batch, num_classes, height, width)。
*   `one_hot_map`: 根据检测到的边界框生成的独热编码图 (batch, num_classes, height, width)。
*   `activation`: 中间步骤生成的激活图 (batch, num_classes, height, width)。

辅助方法:
*   `generate_one_hot_map(bboxes, output_size)`:
    *   输入: 批量的边界框坐标 `bboxes` (batch, num_organs, 4) 和期望的输出图像尺寸 `output_size`。
    *   输出: 在指定尺寸上根据边界框生成的独热编码图 (batch, num_classes, height, width)，
             其中 `num_classes` 比 `num_organs` 多一个背景通道。

该模型试图通过结合区域检测信息来改善分割性能，使得分割网络能够更关注由检测模块定位出的目标区域。
"""
import time
import os
import torch
import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
import Dataset_gen
from torchvision import transforms

from utils.metrics import diceCoeffv2
import segmentation_models_pytorch as smp
from utils.loss import *
from utils import misc
from networks.custom_modules.RegionDetection import RegionDetectionModule,BoundarySensitiveAllocationModule
from networks.u_net import Baseline,unet_Encoder,uent_Decoder


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskUNetWithRegionDetection(nn.Module):
    def __init__(self, img_ch=1, num_classes=6, depth=2):
        super(MultiTaskUNetWithRegionDetection, self).__init__()

        # U-Net 模块（分割任务）
        self.encoder = unet_Encoder(img_ch=img_ch, depth=depth)
        self.first_decoder = uent_Decoder(num_classes=num_classes)
        self.second_decoder = uent_Decoder(num_classes=num_classes)

        # 边界敏感分配模块（区域检测任务）
        self.region_detection = BoundarySensitiveAllocationModule(num_organs=num_classes-1)

        # 下采样模块，使 decoder_input 匹配 encoder 输出的格式
        self.downsampler = nn.Sequential(
            nn.Conv2d(num_classes, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
        )

    def forward(self, x):
        # 1. 经过 encoder 得到编码特征
        enc_features = self.encoder(x)  # 
        enc_final = enc_features[-1]  # (batch, 512, 32, 32)
        # 2. 经过第一个 decoder 得到分割结果
        first_decoder_output = self.first_decoder(enc_final, enc_features[:-1])  # (batch, num_classes, 512, 512)
        first_decoder_output = torch.sigmoid(first_decoder_output)  # Sigmoid 激活
        

        # 3. 通过区域检测模块得到边界框
        region_detection_output = self.region_detection(first_decoder_output)  # (batch, num_classes-1, 4)

        # 4. 生成独热码图像 (batch, num_classes, 512, 512)
        one_hot_map = self.generate_one_hot_map(region_detection_output, x.shape[2:])
        # 5. 生成actiavtion map (batch, num_classes, 512, 512)
        # activation_maps  = first_decoder_output*one_hot_map  # (batch, num_classes, 512, 512)
        # activation_maps = activation_maps+one_hot_map  # (batch, num_classes, 512, 512)
        activation_maps = one_hot_map  #这里不对，只是验证一下能否提取到足够的特征
        activation = activation_maps
        #6.activation_maps做下采样和插值，变换为和enc_final一样的尺寸
        activation_maps = self.downsampler(activation_maps) 
        activation_maps = F.interpolate(activation_maps, size=enc_final.shape[2:], mode='bilinear', align_corners=True)
        activation_maps = torch.sigmoid(activation_maps)  # 修改为了先下采样和插值后再sigmoid激活
        #7. 和enc_final做逐元素相乘和相加
        # decoder_input = enc_final*activation_maps  # (batch, 512,32,32)
        # decoder_input = decoder_input+activation_maps  # (batch, 512,32,32)
        #测试，看看目标检测模块是否能提取到足够的特征
        decoder_input = activation_maps  # (batch, 512,32,32)  #不对啊
        # 8. 经过第二个 decoder 得到最终结果
        segmentation_output = self.second_decoder(decoder_input, enc_features[:-1])  # (batch, num_classes, 512, 512)
        return segmentation_output,one_hot_map,activation

    def generate_one_hot_map(self, bboxes, output_size):
        """
        将边界框 (bboxes) 转换为独热码图像
        :param bboxes: (batch, num_classes, 4)，边界框坐标
        :param output_size: 图像输出尺寸，如 (512, 512)
        :return: 独热码图像 (batch, num_classes, height, width)
        """
        batch_size, num_classes, _ = bboxes.shape
        one_hot_map = torch.zeros(batch_size, num_classes, *output_size, device=bboxes.device)

        for i in range(batch_size):
            for j in range(num_classes):
                # 获取边界框坐标，并转换为整数索引
                xmin, ymin, xmax, ymax = map(int, bboxes[i, j].tolist())

                # 限制坐标不超出图像边界
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(output_size[1], xmax), min(output_size[0], ymax)

                # 在独热图上标记目标区域
                one_hot_map[i, j, ymin:ymax, xmin:xmax] = 1
        # 添加背景通道（全 0），放在前面
        one_hot_map = torch.cat([torch.zeros(batch_size, 1, *output_size, device=bboxes.device), one_hot_map], dim=1)

        return one_hot_map


if __name__ == '__main__':
    num_classes = 6
    depth = 4

    net = MultiTaskUNetWithRegionDetection(num_classes=num_classes, depth=depth)
    net.to('cuda' if torch.cuda.is_available() else 'cpu')

        
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))

    batch_size = 2
    input_chi = 1
    input_tensor = torch.randn(batch_size, input_chi, 512, 512).to('cuda' if torch.cuda.is_available() else 'cpu')  # 示例输入
    output_seg, output_onehot, output_activation = net(input_tensor)

    print("Segmentation Output Shape:", output_seg.shape)
    print("One-Hot Map Shape:", output_onehot.shape)
    print("Activation Map Shape:", output_activation.shape)

    
