import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


"""
验证边界框检测模型能够在分割模型的结果上进行边界框检测。
这个脚本的主要功能是整合一个预训练的分割模型（U-Net）和一个预训练的边界框检测模型（ BoundarySensitiveAllocationModule ），并在单个图像上执行和可视化这个两阶段的流程 。

具体步骤如下：

1. 加载模型 ：
   - 加载一个使用 segmentation_models_pytorch 库预训练的 U-Net 模型，用于图像分割。
   - 加载一个在 `region.py` 中定义的预训练 BoundarySensitiveAllocationModule 模型，用于从分割结果中检测边界框。
2. 图像处理与分割 ：
   - 加载指定的输入图像（例如 Images\3.png ）。
   - 使用加载的 U-Net 模型对图像进行分割，生成预测的分割掩码（ pred_mask ）。
   - 对分割掩码应用 Sigmoid 函数得到概率图（ pred_mask_sigmoid ）。
3. 边界框检测 ：
   - 将 U-Net 输出的 Sigmoid 概率图作为输入，传递给 BoundarySensitiveAllocationModule 模型。
   - 模型预测出各个类别的边界框（ pred_bboxes ）。
4. 结果转换与组合 ：
   - 定义了一个函数 `bboxes_to_one_hot_map` 将预测的边界框转换回 one-hot 编码的图像表示（ one_hot_from_bbox ）。
   - 将 U-Net 的 Sigmoid 输出与从边界框生成的 one-hot 图进行逐元素乘法和加法，生成 activation_maps ，这可能是为了结合分割和检测结果。
5. 可视化 ：
   - 脚本包含大量的 matplotlib 可视化代码，用于展示：
     - U-Net 输出的原始通道和 Sigmoid 后的通道。
     - 从预测边界框生成的 one-hot 图。
     - 组合后的 activation_maps 。
     - 最终在原始输入图像上绘制预测的边界框。
总的来说，这个脚本是一个用于测试和调试的端到端流水线，它首先进行图像分割，然后利用分割结果进行边界框检测，并提供了详细的中间步骤和最终结果的可视化。
"""

def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=0)  # 取最大类别索引
    colour_codes = np.array(palette)
    x = colour_codes[x.astype(np.uint8)]
    return x

def mask_to_onehot(mask, palette):
        """
        Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
        hot encoding vector, C is usually 1 or 3, and K is the number of class.
        """
        semantic_map = []
        for colour in palette:
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
        return semantic_map

# ------------------------ 1. 定义数据集类 ------------------------
from datasetgen import SegmentationToBboxDataset

# ------------------------ 2. 定义模型 ------------------------
from region import BoundarySensitiveAllocationModule

# ------------------------ 3. 加载模型 ------------------------
def load_model(model_path, device):
    num_organs = 5  # 5 个类别
    model = BoundarySensitiveAllocationModule(num_organs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# 颜色映射（5 类目标 + 背景）
palette = [
    [0, 0, 0],  # 背景
    [170, 0, 255],  # 类别 1
    [0, 85, 255],  # 类别 2
    [170, 255, 0],  # 类别 3
    [85, 255, 0],  # 类别 4
    [255, 255, 127]  # 类别 5
]

# 5 种类别的边界框颜色
bbox_colors = ['r', 'g', 'b', 'y', 'm']

# ------------------------ 4. 测试并可视化 ------------------------
def test_and_visualize(model, dataset, index, device):
    image, gt_bboxes = dataset[index]  # 获取图像和真实边界框
    image = image.to(device).unsqueeze(0)  # 添加 batch 维度

    # 预测边界框
    with torch.no_grad():
        pred_bboxes = model(image).cpu().numpy().squeeze()  # 形状 (5, 4)
    
    print("Predicted Bounding Boxes:", pred_bboxes)
    print("Ground Truth Bounding Boxes:", gt_bboxes)

    # 转换 one-hot image 到 RGB
    image = image.squeeze(0).cpu().numpy()  # (6, 512, 512)
    image_rgb = onehot_to_mask(image, palette)  # 只对前 5 类处理

    # 读取 mask
    mask_path = os.path.join(dataset.masks_dir, dataset.mask_filenames[index])
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 绘制图像和边界框
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    #彩色图
    axes[0].imshow(mask)
    axes[0].set_title("Segmentation Mask")
    axes[0].axis("off")

    # 绘制预测结果
    axes[1].imshow(image_rgb)
    for i in range(5):  # 5 个类别
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bboxes[i]
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bboxes[i]

        # 画 **真实边界框**（蓝色）
        gt_rect = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                                    linewidth=2, edgecolor=bbox_colors[i], facecolor='none', label=f"GT Class {i+1}")
        axes[1].add_patch(gt_rect)

        # 画 **预测边界框**（红色）
        pred_rect = patches.Rectangle((pred_xmin, pred_ymin), pred_xmax - pred_xmin, pred_ymax - pred_ymin,
                                      linewidth=2, edgecolor=bbox_colors[i], linestyle='dashed', facecolor='none', label=f"Pred Class {i+1}")
        axes[1].add_patch(pred_rect)

    axes[1].set_title("Predicted vs Ground Truth Bounding Boxes")
    axes[1].axis("off")
    # axes[1].legend(loc="upper right")
    plt.show()

from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

def bboxes_to_one_hot_map(pred_bboxes, output_size=(512, 512), device='cpu'):
    """
    将 (5, 4) bbox 转为 (6, 512, 512) 的 one-hot 图（包含背景）
    """
    num_classes = pred_bboxes.shape[0]
    one_hot_map = torch.zeros(num_classes, *output_size, device=device)

    for j in range(num_classes):
        xmin, ymin, xmax, ymax = map(int, pred_bboxes[j])
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(output_size[1], xmax), min(output_size[0], ymax)

        if xmax > xmin and ymax > ymin:
            one_hot_map[j, ymin:ymax, xmin:xmax] = 1  # 前5类

    # 添加背景通道（全0）在最前面
    one_hot_map = torch.cat([torch.zeros(1, *output_size, device=device), one_hot_map], dim=0)  # (6, H, W)
    return one_hot_map




# ------------------------ 5. 执行测试 ------------------------
if __name__ == "__main__":
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件路径（请修改为你的数据集路径）
    images_dir = "label_png"  # 原始图片路径
    masks_dir = "label_png"  # 分割图片路径
    model_path = "region_detection_5_classes.pth"  # 训练好的模型路径
    transform = transforms.Compose([transforms.ToTensor()])
    # 加载数据集 & 模型
    dataset = SegmentationToBboxDataset(images_dir, masks_dir, transform=transform)
    model = load_model(model_path, device)

    net = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=6,                      # model output channels (number of classes in your dataset)
        )
    net.load_state_dict(torch.load(r"C:\Users\Admin\Desktop\Unet for Lumbar\checkpoint\unet-pretrained_depth=2_fold_1_dice_571999_numsclass_6.pth"))
    net.eval()
    net.to(device)
    img_path = r"C:\Users\Admin\Desktop\Unet for Lumbar\boundary\Images\3.png"
    img = Image.open(img_path)  # 打开图像
    img = img.convert('L')  # 转换为灰度图
    img = np.array(img).astype(np.float32)
    train_input_transform = transforms.Compose([transforms.ToTensor()])
    img = train_input_transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        pred_mask = net(img)
        pred_mask = pred_mask.cpu().numpy().squeeze()  # 形状 (6, 512, 512)
        print(pred_mask.shape)
    # pred_mask: (6, 512, 512)
    # 可视化每个通道 sigmoid 之前的预测图
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))  # 2行6列，第1行是原图，第2行是sigmoid后
    for i in range(6):
        # sigmoid 前
        axes[0, i].imshow(pred_mask[i], cmap='gray')
        axes[0, i].set_title(f'Raw Output - Class {i}')
        axes[0, i].axis('off')

    # 对每个通道做 sigmoid（注意是在 numpy 上要先转回 torch 再转回来）
    pred_mask_sigmoid = torch.sigmoid(torch.tensor(pred_mask)).numpy()
    print("pred_mask_sigmoid shape:", pred_mask_sigmoid.shape)
    for i in range(6):
        # sigmoid 后
        axes[1, i].imshow(pred_mask_sigmoid[i], cmap='gray')
        axes[1, i].set_title(f'Sigmoid Output - Class {i}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
    
    mask = torch.tensor(pred_mask_sigmoid)  # 转为 tensor
    print("mask shape1:", mask.shape)
    mask = mask.unsqueeze(0)  # 添加 batch 维度
    mask = mask.to(device)
    print("mask shape:", mask.shape)
    with torch.no_grad():
        pred_bboxes = model(mask)
        pred_bboxes = pred_bboxes.cpu().numpy().squeeze()  # 形状 (5, 4)
    print(pred_bboxes)
    # 将 pred_bboxes 转为 one-hot map，形状 (6, 512, 512)
    one_hot_from_bbox = bboxes_to_one_hot_map(pred_bboxes, output_size=(512, 512), device='cpu')

    # 可视化
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for i in range(6):
        axes[i].imshow(one_hot_from_bbox[i].cpu().numpy(), cmap='gray')
        axes[i].set_title(f'BBox Map - Class {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    one_hot_map = torch.tensor(one_hot_from_bbox)
    decoder_output = torch.tensor(pred_mask_sigmoid)
    print(one_hot_map.shape)
    print(decoder_output.shape)
    activation_maps  = decoder_output*one_hot_map  # (batch, num_classes, 512, 512)
    activation_maps = activation_maps+one_hot_map  # (batch, num_classes, 512, 512)
    activation_maps = F.sigmoid(activation_maps)  # (batch, num_classes, 512, 512)
    # 可视化
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for i in range(6):
        axes[i].imshow(activation_maps[i].cpu().numpy(), cmap='gray')
        axes[i].set_title(f'BBox Map - Class {i}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()  
    mask_path = r'C:\Users\Admin\Desktop\Unet for Lumbar\boundary\label_png\3.png'
    # 读取原图并恢复为 PIL 图像用于画图
    orig_img = Image.open(mask_path).convert('RGB')  # 转为 RGB 方便显示彩色 bbox
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(orig_img)

    # 画出每个边界框（共5类，不包含背景）
    for i, bbox in enumerate(pred_bboxes):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f'Class {i+1}', color='red', fontsize=10, weight='bold')

    ax.set_title("Predicted Bounding Boxes on Image")
    ax.axis('off')
    plt.show()


