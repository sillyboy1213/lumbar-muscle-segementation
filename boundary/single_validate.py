import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader

# ------------------------ 1. 定义数据集类 ------------------------
class SegmentationToBboxDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # 读取原始图像
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # 归一化
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 计算边界框
        bbox = self.mask_to_bbox(mask)
        
        # 处理形状
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) → (C, H, W)
        bbox = torch.tensor(bbox, dtype=torch.float32)

        return image, bbox

    def mask_to_bbox(self, mask):
        """计算分割掩码的外接边界框"""
        ys, xs = np.where(mask > 0)  # 找到所有前景点
        if len(xs) == 0 or len(ys) == 0:
            return np.array([0, 0, 1, 1])  # 防止无目标

        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)

        return np.array([xmin, ymin, xmax, ymax])

# ------------------------ 2. 定义模型 ------------------------
from single_region import BoundarySensitiveAllocationModule

# ------------------------ 3. 加载模型 ------------------------
def load_model(model_path, device):
    num_organs = 1  # 只有一个类别
    model = BoundarySensitiveAllocationModule(num_organs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ------------------------ 4. 测试并可视化 ------------------------
def test_and_visualize(model, dataset, index, device):
    image, gt_bbox = dataset[index]  # 获取图像和真实边界框
    image = image.to(device).unsqueeze(0)  # 添加 batch 维度

    # 预测边界框
    with torch.no_grad():
        pred_bbox = model(image).cpu().numpy().squeeze()  # 确保形状为 (4,)
    
    print("Predicted Bounding Box:", pred_bbox)
    print("Ground Truth Bounding Box:", gt_bbox)

    # 读取原始图像 & mask
    img_path = os.path.join(dataset.images_dir, dataset.image_filenames[index])
    mask_path = os.path.join(dataset.masks_dir, dataset.mask_filenames[index])

    orig_image = cv2.imread(img_path)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 绘制原始 mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(mask, cmap='gray')
    axes[0].set_title("Segmentation Mask")
    axes[0].axis("off")

    # 绘制原图
    axes[1].imshow(orig_image)
    
    # 画 **真实边界框**（蓝色）
    gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox
    gt_rect = patches.Rectangle((gt_xmin, gt_ymin), gt_xmax - gt_xmin, gt_ymax - gt_ymin,
                                linewidth=2, edgecolor='b', facecolor='none', label="Ground Truth")
    axes[1].add_patch(gt_rect)

    # 画 **预测边界框**（红色）
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bbox
    pred_rect = patches.Rectangle((pred_xmin, pred_ymin), pred_xmax - pred_xmin, pred_ymax - pred_ymin,
                                  linewidth=2, edgecolor='r', facecolor='none', label="Predicted")
    axes[1].add_patch(pred_rect)

    # 添加标题和图例
    axes[1].set_title("Predicted vs Ground Truth Bounding Box")
    axes[1].axis("off")
    axes[1].legend(loc="upper right")

    plt.show()

# ------------------------ 5. 执行测试 ------------------------
if __name__ == "__main__":
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 文件路径（请修改为你的数据集路径）
    images_dir = "label_png"  # 原始图片路径
    masks_dir = "R3"  # 分割图片路径
    model_path = "region_detection.pth"  # 训练好的模型路径

    # 加载数据集 & 模型
    dataset = SegmentationToBboxDataset(images_dir, masks_dir)
    model = load_model(model_path, device)

    for i in range(len(dataset)):   
        test_and_visualize(model, dataset, i, device)
