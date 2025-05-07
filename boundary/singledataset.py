import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationToBboxDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        :param images_dir: 原始图片文件夹路径
        :param masks_dir: 分割图片文件夹路径
        :param transform: 适用于原始图片的变换
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filenames)
    
    def mask_to_bbox(self, mask):
        """
        从单通道 mask 提取边界框
        :param mask: (H, W) 二值分割图，前景为 1，背景为 0
        :return: (4,) 形式的边界框 [xmin, ymin, xmax, ymax]
        """
        y_indices, x_indices = np.where(mask > 0)  # 获取前景像素坐标
        if len(y_indices) > 0 and len(x_indices) > 0:
            xmin, xmax = x_indices.min(), x_indices.max()
            ymin, ymax = y_indices.min(), y_indices.max()
            return torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        else:
            return torch.tensor([0, 0, 0, 0], dtype=torch.float32)  # 没有前景时返回空 bbox

    def __getitem__(self, idx):
        # 加载原始图片
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV 默认 BGR 需转换为 RGB
        
        # 加载分割 mask
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为单通道灰度图
        
        # 计算边界框
        bbox = self.mask_to_bbox(mask)

        # 处理原始图片
        if self.transform:
            image = self.transform(image)

        return image, bbox


from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches


if __name__ == '__main__':
    # 测试数据集
    # 目录路径
    images_dir = "label_png"
    masks_dir = "R3"
    # 数据变换（可选）
    transform = transforms.Compose([
        transforms.ToTensor(),])
        # 创建数据集
    dataset = SegmentationToBboxDataset(images_dir, masks_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 测试加载一个样本
    image, bbox = dataset[0]
    print(f"Image shape: {image.shape}, Bbox: {bbox}")
    def visualize_bbox(image, mask, bbox):
        """
        显示原始 mask 和绘制边界框的原始图片
        :param image: 原始图像 (H, W, 3) 格式
        :param mask: 分割 mask (H, W) 格式
        :param bbox: 计算出的边界框 [xmin, ymin, xmax, ymax]
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # 显示分割 mask
        axes[0].imshow(mask, cmap='gray')
        axes[0].set_title("Segmentation Mask")
        axes[0].axis("off")

        # 显示原始图片并绘制边界框
        axes[1].imshow(image)
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        axes[1].add_patch(rect)
        axes[1].set_title("Bounding Box on Image")
        axes[1].axis("off")

        plt.show()
    for i in range(len(dataset)):
        # 取数据集中的一个样本进行可视化
        image, bbox = dataset[i]

        # 由于 image 经过 ToTensor 归一化，需要转换回 numpy 格式并反归一化
        image_np = image.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
        image_np = (image_np * 0.5 + 0.5).clip(0, 1)  # 反归一化到 [0, 1] 范围

        # 读取对应的 mask
        mask_path = os.path.join(masks_dir, dataset.mask_filenames[i])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 显示结果
        visualize_bbox(image_np, mask, bbox)