import os
import cv2
import numpy as np
from PIL import Image
from torch.utils import data
import utils.helpers as helpers
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

palette = [[0, 0, 0],  [170, 0, 255], [0, 85, 255], [170, 255, 0],[85, 255, 0],   
           [255, 255, 127]]  # one-hot的颜色表
num_classes = 6  # 包括背景

def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test']
    items = []
    # ... (make_dataset 函数保持不变) ...
    if mode == 'train':
        img_path = os.path.join(root, 'Images')
        mask_path = os.path.join(root, 'Labels')
        if 'Augdata' in root:
            data_list = os.listdir(os.path.join(root, 'Labels'))
        else:
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
    else:
        img_path = os.path.join(root, 'Images')
        try:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'test.txt')).readlines()]
        except FileNotFoundError:
            raise FileNotFoundError(f"文件test.txt不存在!")
        for it in data_list:
            item = (os.path.join(img_path,it))
            items.append(item)
    return items

class Dataset(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)
        self.palette = palette
        self.mode = mode
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.joint_transform = joint_transform
        self.center_crop = center_crop
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = A.Compose([
            A.Rotate(limit=30, p=0.8, border_mode=0),
            # 其他增强可以根据需要添加
        ])

    def __getitem__(self, index):
        img_path, mask_path = self.imgs [index] if self.mode != 'test' else (self.imgs [index], None)
        file_name = os.path.basename(mask_path) if mask_path else os.path.basename(img_path)

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path) if mask_path else None

        if self.joint_transform is not None and mask is not None:
            img, mask = self.joint_transform(img, mask)
        if self.center_crop is not None:
            img = self.center_crop(img)
            if mask is not None:
                mask = self.center_crop(mask)

        img = np.array(img).astype(np.float32)
        if mask is not None:
            mask = np.array(mask).astype(np.uint8) # 确保掩码是 uint8 类型

        if self.mode == 'train' and mask is not None:
            augmented = self.augmentation(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # --------------- Mask R-CNN 数据准备 ------------------
        if mask is not None:
            # 获取所有唯一的类别 ID (不包括背景 0)
            obj_ids = np.unique(mask)
            obj_ids = obj_ids [obj_ids != 0]

            # 将掩码分割成每个实例的二值掩码
            masks = mask == obj_ids[:, None, None]

            # 获取每个实例的边界框
            num_objs = len(obj_ids)
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks [i])
                xmin = np.min(pos [1])
                xmax = np.max(pos [1])
                ymin = np.min(pos [0])
                ymax = np.max(pos [0])
                boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            # 类别标签
            labels = torch.as_tensor(obj_ids, dtype=torch.int64) # 类别 ID 1-5

            # 转换为 torch.Tensor
            masks = torch.as_tensor(masks, dtype=torch.uint8)

            target = {}
            target['boxes'] = boxes
            target['labels'] = labels
            target['masks'] = masks
            target['image_id'] = torch.tensor([index]) # 可以添加图像ID
        else:
            target = None

        # 标准化和转换为 Tensor (对图像进行)
        img = img / 255.0
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0) # 灰度图添加通道维度
        img = torch.as_tensor(img, dtype=torch.float32)
        if self.transform is not None:
            img = self.transform(img)

        if target is not None and self.target_transform is not None:
            # 注意：通常不对 target 做额外的 tensor transform，因为 boxes 和 labels 是数值
            pass

        if self.mode == 'test':
            return img, file_name
        else:
            return img, target

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

        center_crop = None
        train_input_transform = transforms.Compose([]) # Mask R-CNN 通常在内部处理标准化
        target_transform = transforms.Compose([])

        train_set = Dataset(train_path, 'train', 1,
                              joint_transform=None, center_crop=center_crop,
                              transform=train_input_transform, target_transform=target_transform)
        train_loader = DataLoader(train_set, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x))) # 需要自定义 collate_fn

        for images, targets in train_loader:
            print("Image shape:", images [0].shape)
            print("Targets:", targets)
            if targets is not None and len(targets) > 0 and targets [0] is not None:
                print("Boxes shape:", targets [0]['boxes'].shape)
                print("Labels shape:", targets [0]['labels'].shape)
                print("Masks shape:", targets [0]['masks'].shape)

                # 可视化第一个图像及其标注
                image = images [0].permute(1, 2, 0).numpy()
                if image.shape[-1] == 1:
                    image = image.squeeze()
                boxes = targets [0]['boxes'].cpu().numpy().astype(np.int32)
                masks = targets [0]['masks'].cpu().numpy()
                labels = targets [0]['labels'].cpu().numpy()

                fig, ax = plt.subplots(1, figsize=(10, 10))
                ax.imshow(image, cmap='gray')

                for i, box in enumerate(boxes):
                    cv2.rectangle(image.astype(np.uint8).copy(), (box [0], box [1]), (box [2], box [3]), (0, 255, 0), 2)
                    mask = masks [i].squeeze().numpy()
                    # 叠加掩码 (为了可视化)
                    masked_image = image.copy()
                    masked_image [mask > 0.5] = [1, 0, 0] # Red color for mask
                    ax.imshow(masked_image, alpha=0.5)
                    ax.text(box [0], box [1] - 10, f"Class: {labels [i]}", fontsize=10, color='yellow')

                plt.show()

    demo()