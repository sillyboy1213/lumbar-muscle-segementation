import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import albumentations as A
import torch

# 您的调色板定义保持不变
palette = [[0, 0, 0], [170, 0, 255], [0, 85, 255], [170, 255, 0], [85, 255, 0], [255, 255, 127]]
num_classes = 6

def make_dataset(root, mode, fold):
    # ... 此函数保持不变 ...
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'val{}.txt'.format(fold))).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))
            items.append(item)
    return items

# ==================== 新增的辅助函数 ====================
def convert_rgb_to_id_mask(rgb_mask, palette):
    """
    将RGB掩码根据调色板转换为单通道的类别ID掩码。
    
    Args:
        rgb_mask (np.array): 形状为 (H, W, 3) 的RGB掩码图像。
        palette (list): 颜色列表，列表的索引对应类别ID。

    Returns:
        np.array: 形状为 (H, W) 的单通道ID掩码。
    """
    # 创建一个空白的单通道掩码用于存放类别ID
    id_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    
    # 遍历调色板中的每一种颜色（从类别1开始，类别0是背景，默认为0）
    for class_id, color in enumerate(palette):
        if class_id == 0: # 跳过背景色
            continue
        
        # 找到所有等于当前颜色的像素位置
        # np.all(..., axis=2) 用于确保R, G, B三个通道都匹配
        matching_pixels = np.all(rgb_mask == color, axis=2)
        
        # 在这些位置上，将类别ID写入id_mask
        id_mask[matching_pixels] = class_id
        
    return id_mask
# =========================================================

class Dataset(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)
        self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.transform = transform
        self.augmentation = A.Compose([
            A.Rotate(limit=15, p=0.8, border_mode=0),
        ])

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        
        # img = Image.open(img_path).convert('L') # 输入图像转为灰度
        img = Image.open(img_path).convert('RGB') # 修改后的代码
        rgb_mask_img = Image.open(mask_path).convert('RGB') # 确保掩码是RGB格式

        # 将图像和掩码转换为NumPy数组
        img_np = np.array(img).astype(np.float32)
        rgb_mask_np = np.array(rgb_mask_img).astype(np.uint8)

        # ==================== 核心改动点 ====================
        # 调用新函数，将RGB掩码精确转换为单通道ID掩码
        id_mask = convert_rgb_to_id_mask(rgb_mask_np, self.palette)
        # ====================================================

        # 应用数据增强 (对图像和ID掩码)
        if self.mode == 'train':
            augmented = self.augmentation(image=img_np, mask=id_mask)
            img_np = augmented['image']
            id_mask = augmented['mask']

        # --------------- Mask R-CNN 数据准备 ------------------
        # 从单通道ID掩码中提取实例信息
        obj_ids = np.unique(id_mask)
        obj_ids = obj_ids[obj_ids != 0]

        if len(obj_ids) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "masks": torch.zeros((0, id_mask.shape[0], id_mask.shape[1]), dtype=torch.uint8),
                "image_id": torch.tensor([index])
            }
        else:
            masks = (id_mask == obj_ids[:, None, None])
            
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                # 增加一个检查，防止实例过小或不存在
                if pos[0].size > 0 and pos[1].size > 0:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                else: # 如果增强后某个小实例消失了，可以跳过
                    continue

            if not boxes: # 如果所有实例都消失了
                 return self.__getitem__((index + 1) % len(self.imgs)) # 加载下一张图，避免空目标

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(obj_ids, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            target['masks'] = masks
            target['image_id'] = torch.tensor([index])

        # 标准化和转换为 Tensor (对图像进行)
        # img_tensor = torch.from_numpy(img_np / 255.0).float().unsqueeze(0) # 添加通道维度
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float()
        #  # 使用 .repeat() 将单通道张量复制3次
        # img_tensor = img_tensor.repeat(3, 1, 1)
        # # 现在 img_tensor 的形状是 (3, H, W)
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

# 导入DataLoader来加载数据
from torch.utils.data import DataLoader

if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)

    def demo():
        train_path = r'./Dataset'
        val_path = r'./Dataset'
        test_path = r'./Dataset'

        train_set = Dataset(train_path, 'train', 1)
        # The collate_fn is crucial for Mask R-CNN when batching images
        # with a variable number of objects.
        train_loader = DataLoader(train_set, batch_size=2, shuffle=True,
                                collate_fn=lambda x: tuple(zip(*x)))

        for images, targets in train_loader:
            print("Image shape:", images[0].shape)
            # print("Targets:", targets) # This can be very long, so it's commented out
            if targets and targets[0]:
                print("Boxes shape:", targets[0]['boxes'].shape)
                print("Labels shape:", targets[0]['labels'].shape)
                print("Masks shape:", targets[0]['masks'].shape)

                # --- Visualization of the first image in the batch ---
                # Permute from (C, H, W) to (H, W, C) for plotting
                image = images[0].permute(1, 2, 0).numpy()
                
                # Squeeze out the channel dimension if it's 1
                if image.shape[2] == 1:
                    image = image.squeeze(axis=2)

                # Convert tensors to NumPy arrays for visualization
                boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)
                masks = targets[0]['masks'].cpu().numpy() # This is the main NumPy conversion
                labels = targets[0]['labels'].cpu().numpy()

                # Prepare a color image for drawing boxes and masks
                # Convert grayscale float (0-1) to BGR uint8 (0-255)
                vis_image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

                fig, ax = plt.subplots(1, figsize=(10, 10))
                
                # Create a composite mask for visualization
                composite_mask = np.zeros_like(vis_image, dtype=np.uint8)

                for i, box in enumerate(boxes):
                    # --- THIS IS THE FIX ---
                    # 'masks' is already a NumPy array, so we just index it.
                    # No need for .squeeze() if shape is (H,W), or .numpy() again.
                    mask = masks[i]
                    # -----------------------

                    # Assign a unique color for each class for the visualization mask
                    color = plt.cm.get_cmap('tab10')(labels[i] % 10)[:3] # Use a colormap
                    color = tuple(c * 255 for c in color) # Convert to 0-255 scale
                    
                    # Apply the colored mask
                    composite_mask[mask > 0] = color

                    # Draw the bounding box and label text
                    cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(vis_image, f"Class: {labels[i]}", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Blend the original image with the masks
                final_image = cv2.addWeighted(vis_image, 0.7, composite_mask, 0.3, 0)
                
                ax.imshow(final_image)
                ax.set_axis_off()
                plt.show()
                # We only visualize the first image of the batch, so break the loop
                break

    demo()