import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
from torchvision import transforms

# ------------------------ 1. Mask 映射函数 ------------------------
def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=0)  # 取最大类别索引
    colour_codes = np.array(palette)
    x = colour_codes[x.astype(np.uint8)]
    return x

# ------------------------ 2. 导入数据集与模型 ------------------------
from datasetgen import SegmentationToBboxDataset
from region import BoundarySensitiveAllocationModule

# ------------------------ 3. 加载模型 ------------------------
def load_model(model_path, device):
    num_organs = 5
    model = BoundarySensitiveAllocationModule(num_organs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ------------------------ 4. 可视化测试函数 ------------------------
def test_and_visualize(model, dataset, index, device):
    image, gt_bboxes = dataset[index]  # 获取图像和真实边界框
    image = image.to(device)

    # 保存原始图像以备可视化
    original_image = image.clone()

    # --- 小幅随机旋转图像 ---
    angle = random.uniform(-10, 10)
    image = TF.rotate(image, angle)
    print(f"\nIndex {index}: Applied rotation = {angle:.2f}°")

    # 添加 batch 维度
    image_batch = image.unsqueeze(0)

    # 模型预测
    with torch.no_grad():
        pred_bboxes = model(image_batch).cpu().numpy().squeeze()  # (5, 4)

    print("Predicted Bounding Boxes:\n", pred_bboxes)
    print("Ground Truth Bounding Boxes:\n", gt_bboxes)

    # 转换 one-hot 图像为 RGB
    image_np = original_image.cpu().numpy()  # (6, H, W)
    image_rgb = onehot_to_mask(image_np, palette)

    # 读取 mask（用于可视化）
    mask_path = os.path.join(dataset.masks_dir, dataset.mask_filenames[index])
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # 左图：分割掩膜
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title("Segmentation Mask")
    axes[0].axis("off")

    # 右图：RGB 图 + 边界框
    axes[1].imshow(image_rgb)
    for i in range(5):  # 类别 1~5
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bboxes[i]
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bboxes[i]

        # 真实框（实线）
        gt_rect = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                                    linewidth=2, edgecolor=bbox_colors[i], facecolor='none', label=f"GT Class {i+1}")
        axes[1].add_patch(gt_rect)

        # 预测框（虚线）
        pred_rect = patches.Rectangle((pred_xmin, pred_ymin), pred_xmax - pred_xmin, pred_ymax - pred_ymin,
                                      linewidth=2, edgecolor=bbox_colors[i], linestyle='dashed', facecolor='none', label=f"Pred Class {i+1}")
        axes[1].add_patch(pred_rect)

    axes[1].set_title("Predicted vs Ground Truth BBoxes")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# ------------------------ 5. 调色板 & 框颜色 ------------------------
palette = [
    [0, 0, 0],           # 背景
    [170, 0, 255],       # 类别 1
    [0, 85, 255],        # 类别 2
    [170, 255, 0],       # 类别 3
    [85, 255, 0],        # 类别 4
    [255, 255, 127]      # 类别 5
]

bbox_colors = ['r', 'g', 'b', 'y', 'm']

# ------------------------ 6. 主函数执行 ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置（请根据实际情况修改）
    images_dir = "./data/label_png"
    masks_dir = "./data/label_png"
    model_path = "region_detection_5_classes.pth"

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SegmentationToBboxDataset(images_dir, masks_dir, transform=transform)
    model = load_model(model_path, device)

    for i in range(len(dataset)):
        test_and_visualize(model, dataset, i, device)
