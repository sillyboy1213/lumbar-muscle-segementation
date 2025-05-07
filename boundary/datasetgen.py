"""
此脚本定义了一个 PyTorch 数据集类 `SegmentationToBboxDataset`，
用于处理多类别分割任务。注意，由于区域检测模块是输入图像分割的结果，所以数据集

主要功能包括：
1. 从指定的图像目录和掩码目录加载彩色分割掩码图像。
2. 使用预定义的调色板 (`palette`) 将彩色掩码转换为 one-hot 编码张量。
3. 从彩色掩码中为每个类别（背景除外）提取边界框坐标。
4. 提供 `__getitem__` 方法，返回 one-hot 编码的图像张量和对应的边界框张量。
5. 包含辅助函数 `mask_to_onehot` 和 `onehot_to_mask` 用于掩码格式转换。
6. 在 `if __name__ == '__main__':` 块中包含测试和可视化代码。
"""
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

palette = [[0, 0, 0], [170, 0, 255], [0, 85, 255], [170, 255, 0], [85, 255, 0], [255, 255, 127]]
num_classes = len(palette) - 1  # 5 个类别

class SegmentationToBboxDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        # self.image_filenames = sorted(os.listdir(images_dir))
        # self.mask_filenames = sorted(os.listdir(masks_dir))
        self.image_filenames = os.listdir(images_dir)
        self.mask_filenames = os.listdir(masks_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filenames)
    
    def mask_to_bboxes(self, mask):
        bboxes = torch.zeros((num_classes, 4), dtype=torch.float32)
        for class_idx, color in enumerate(palette[1:]):
            class_mask = np.all(mask == np.array(color), axis=-1)
            y_indices, x_indices = np.where(class_mask)
            if len(y_indices) > 0 and len(x_indices) > 0:
                xmin, xmax = x_indices.min(), x_indices.max()
                ymin, ymax = y_indices.min(), y_indices.max()
                bboxes[class_idx] = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
            else:
                raise ValueError("Empty mask")
        return bboxes
    
    def mask_to_onehot(self, mask, palette):
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


    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_filenames[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[idx])
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        bboxes = self.mask_to_bboxes(mask)
        
        # print(image.shape)  # (H, W, C)
        image = self.mask_to_onehot(image, palette)
        if self.transform:
            image = self.transform(image)
        # image = self.mask_to_onehot(image.permute(1, 2, 0), palette)
        return image, bboxes


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)

    colour_codes = np.array(palette)
    x = colour_codes[x.astype(np.uint8)]
    return x

if __name__ == '__main__':
    images_dir = "label_png"
    masks_dir = "label_png"
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SegmentationToBboxDataset(images_dir, masks_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    def visualize_bbox(image_to_show, mask, bboxes):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(mask)
        axes[0].set_title("Segmentation Mask")
        axes[0].axis("off")
        
        # 修改这里，显示传入的 image_to_show
        axes[1].imshow(image_to_show) 
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            if xmin == xmax and ymin == ymax:
                continue
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            axes[1].add_patch(rect)
        axes[1].set_title("Bounding Boxes on Image")
        axes[1].axis("off")
        plt.show()
    
    for i in range(len(dataset)):
        image, bboxes = dataset[i] # image 是 one-hot (6, 512, 512)
        print(image.shape)
        print(bboxes.shape)
        # image_np = image.permute(1, 2, 0).numpy() # 这个是 (512, 512, 6)，不能直接显示
        
        # 加载原始彩色掩码用于显示
        mask_path = os.path.join(masks_dir, dataset.mask_filenames[i])
        mask_rgb = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB) # 转为 RGB
        
        # 调用可视化函数，传入 mask_rgb 用于显示
        visualize_bbox(mask_rgb, mask_rgb, bboxes)
