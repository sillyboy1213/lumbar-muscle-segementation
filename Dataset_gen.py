"""
本脚本定义了用于加载和预处理椎旁肌分割任务所需的数据集。
主要功能包括：

1.  **常量定义**:
    *   `palette`: 定义了用于将分割掩码转换为one-hot编码的颜色调色板。每个颜色对应一个类别。
    *   `num_classes`: 指定了数据集中分割的总类别数量（包括背景）。

2.  **`make_dataset(root, mode, fold)` 函数**:
    *   根据指定的根目录 (`root`)、模式 (`mode`：'train', 'val', 'test') 和交叉验证的折数 (`fold`)，
        收集图像及其对应掩码的文件路径。
    *   对于训练集和验证集，它会从预定义的文本文件（如 `train1.txt`, `val1.txt`）或直接从目录中读取文件列表。
    *   对于测试集，它会从 `test.txt` 文件中读取图像文件列表（测试集通常只有图像，没有标签）。
    *   返回一个包含元组的列表，每个元组包含 (图像路径, 掩码路径) 或仅 图像路径（对于测试集）。

3.  **`Dataset(data.Dataset)` 类**:
    *   继承自 `torch.utils.data.Dataset`，是PyTorch中用于自定义数据集的标准方式。
    *   **`__init__(self, root, mode, fold, joint_transform=None, center_crop=None, transform=None, target_transform=None)`**:
        *   初始化数据集对象。
        *   调用 `make_dataset` 来获取图像和掩码的路径列表。
        *   存储颜色调色板、模式、以及各种可选的图像变换（联合变换、中心裁剪、输入图像变换、目标掩码变换）。
        *   定义了基于 `albumentations` 库的数据增强流程（例如旋转），仅在训练模式下应用。
    *   **`__getitem__(self, index)`**:
        *   根据索引 `index` 获取单个数据样本。
        *   加载图像和掩码文件（使用Pillow库）。
        *   将图像转换为灰度图。
        *   按顺序应用指定的变换：联合变换、中心裁剪。
        *   将图像和掩码转换为NumPy数组。
        *   如果处于训练模式，则应用数据增强。
        *   使用 `helpers.mask_to_onehot` 函数将掩码转换为one-hot编码格式。
        *   应用单独的输入图像变换和目标掩码变换（通常是转换为PyTorch张量）。
        *   返回一个元组 `((img, mask), file_name)`，其中 `img` 和 `mask` 是处理后的张量，`file_name` 是原始掩码的文件名。
    *   **`__len__(self)`**:
        *   返回数据集中样本的总数。

4.  **`if __name__ == '__main__':` 块**:
    *   包含一个 `demo()` 函数，用于测试数据集加载器的功能。
    *   它会创建一个 `Dataset` 实例和一个 `DataLoader`，然后迭代加载数据，
        并使用 `matplotlib.pyplot` 和 `utils.helpers` 中的函数来可视化加载的图像和处理后的掩码。

该脚本是训练深度学习分割模型前数据准备的关键部分，确保了数据能够以正确的格式和增强方式送入模型。
"""
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

# 图像标签的颜色和名称定义
'''
label:
- color: '#000000'
  name: __background__
- color: '#aa0000'
  name: L1
- color: '#005500'
  name: R1
- color: '#aa007f'
  name: L2
- color: '#00557f'
  name: R2
- color: '#aa00ff'
  name: L3
- color: '#0055ff'
  name: R3
- color: '#55ff00'
  name: L
- color: '#ffff7f'
  name: R
- color: '#aaff00'
  name: S
'''
# palette = [[0, 0, 0], [170, 0, 0], [0, 85, 0], [170, 0, 127], 
#            [0, 85, 127], [170, 0, 255], [0, 85, 255], [170, 255, 0]]  # one-hot的颜色表
palette = [[0, 0, 0],  [170, 0, 255], [0, 85, 255], [170, 255, 0],[85, 255, 0],   
           [255, 255, 127]]  # one-hot的颜色表
# palette = [[0, 0, 0], [170, 0, 0],] 
num_classes = 6  # 分类数,不包含L和R

# 用于根据给定的数据集路径创建数据项
def make_dataset(root, mode, fold):
    assert mode in ['train', 'val', 'test']  # 检查模式是否为 'train'、'val' 或 'test'
    items = []
    if mode == 'train':  # 如果是训练集
        img_path = os.path.join(root, 'Images')  # 图像路径
        mask_path = os.path.join(root, 'Labels')  # 标签路径

        if 'Augdata' in root:  # 当使用增广后的训练集
            data_list = os.listdir(os.path.join(root, 'Labels'))  # 获取标签文件列表
        else:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'train{}.txt'.format(fold))).readlines()]  # 从文件读取训练数据列表
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))  # 将图像和标签的路径作为一项添加
            items.append(item)
    elif mode == 'val':  # 如果是验证集
        img_path = os.path.join(root, 'Images')  # 图像路径
        mask_path = os.path.join(root, 'Labels')  # 标签路径
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'val{}.txt'.format(fold))).readlines()]  # 从文件读取验证数据列表
        for it in data_list:
            item = (os.path.join(img_path, it), os.path.join(mask_path, it))  # 将图像和标签的路径作为一项添加
            items.append(item)
    else:  # 如果是测试集
        img_path = os.path.join(root, 'Images')  # 图像路径
        try:
            data_list = [l.strip('\n') for l in open(os.path.join(root, 'test.txt')).readlines()]  # 从文件读取测试数据列表
        except:
            raise FileNotFoundError(f"文件test.txt不存在!")  # 如果找不到文件则报错
        for it in data_list:
            item = (os.path.join(img_path,it))  # 只有图像路径
            items.append(item)
    return items

# 引入torchvision的transforms库进行图像处理
from torchvision import transforms

# 定义数据集类
class Dataset(data.Dataset):
    def __init__(self, root, mode, fold, joint_transform=None, center_crop=None, transform=None, target_transform=None):
        self.imgs = make_dataset(root, mode, fold)  # 加载数据集
        self.palette = palette  # 定义颜色表
        self.mode = mode  # 数据集模式（训练、验证、测试）
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')  # 如果没有图片数据，则抛出异常
        self.joint_transform = joint_transform  # 图像联合变换
        self.center_crop = center_crop  # 中心裁剪
        self.transform = transform  # 输入图像的转换
        self.target_transform = target_transform  # 目标图像的转换
         # 设定数据增强
        self.augmentation = A.Compose([
            # A.HorizontalFlip(p=0.5),  # 50% 概率水平翻转,水平翻转大概率有问题
            A.Rotate(limit=30, p=0.8, border_mode=0),  # 80% 概率 ±30° 旋转
        ])
    def __getitem__(self, index):
        # 获取图像和标签路径
        img_path, mask_path = self.imgs[index]
        file_name = mask_path.split('\\')[-1]  # 获取文件名

        img = Image.open(img_path)  # 打开图像
        mask = Image.open(mask_path)  # 打开标签
        img = img.convert('L')  # 转换为灰度图像
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)  # 如果有联合变换，进行处理
        if self.center_crop is not None:
            img, mask = self.center_crop(img, mask)  # 如果有中心裁剪，进行处理
        # img = np.array(img)  # 转换为NumPy数组
        # mask = np.array(mask)  # 转换为NumPy数组
        img = np.array(img).astype(np.float32)  # 转换为 float32
        mask = np.array(mask).astype(np.float32)
                # 进行数据增强
        if self.mode == 'train':
          augmented = self.augmentation(image=img, mask=mask)
          img = augmented['image']
          mask = augmented['mask']
        mask = helpers.mask_to_onehot(mask, self.palette)  # 将标签转为one-hot编码
        # print(img.shape, mask.shape) (512, 512) (512, 512, 8)
        # shape from (H, W, C) to (C, H, W)
        # img = img.transpose([2, 0, 1])  # 如果是彩色图像，调整维度
        # mask = mask.transpose([2, 0, 1])  # 如果是彩色标签，调整维度
        # print(img.shape, mask.shape)
        if self.transform is not None:
            img = self.transform(img)  # 进行输入图像的转换
        if self.target_transform is not None:
            mask = self.target_transform(mask)  # 进行目标图像的转换
        # print(img.shape, mask.shape)    #torch.Size([1, 512, 512]) torch.Size([8, 512, 512])
        return (img, mask), file_name  # 返回图像、标签和文件名

    def __len__(self):
        return len(self.imgs)  # 返回数据集的长度

# 导入DataLoader来加载数据
from torch.utils.data import DataLoader

if __name__ == '__main__':
  np.set_printoptions(threshold=9999999)  # 设置NumPy数组打印选项

  # 测试加载数据类
  def demo():
      train_path = r'.\Dataset'  # 训练集路径
      val_path = r'.\Dataset'  # 验证集路径
      test_path = r'.\Dataset'  # 测试集路径

      # center_crop = joint_transforms.CenterCrop(256)
      center_crop = None  # 没有使用中心裁剪
      test_center_crop = transforms.CenterCrop(256)  # 使用torchvision的CenterCrop进行裁剪
      train_input_transform = transforms.Compose([transforms.ToTensor()])  # 输入图像转换为Tensor
      target_transform = transforms.Compose([transforms.ToTensor()])  # 标签转换为Tensor

      # 创建数据集实例
      train_set = Dataset(train_path, 'train', 1,
                            joint_transform=None, center_crop=center_crop,
                            transform=train_input_transform, target_transform=target_transform)
      train_loader = DataLoader(train_set, batch_size=1, shuffle=False)  # 使用DataLoader加载数据

      for (input, mask), file_name in train_loader:
          # 打印图像和标签的形状
          # print(input.shape)
          # print(mask.shape)
#         torch.Size([1, 1, 512, 512])
# torch.Size([1, 8, 512, 512])
          img = np.array(input.squeeze())  # 将输入图像转换为NumPy数组
          # print(img.shape)
          plt.imshow(img)  # 显示图像
          plt.show()  # 展示图像
          img = helpers.array_to_img(np.expand_dims(input.squeeze(), 2))  # 将Tensor转为PIL图像并展示
          plt.imshow(img)
          plt.show()
          # 将gt反one-hot回去以便进行可视化
          # palette = [[0, 0, 0], [246, 16, 16], [16, 136, 246]]
          palette = [[0, 0, 0],  [170, 0, 255], [0, 85, 255], [170, 255, 0],[85, 255, 0],   
           [255, 255, 127]]  # one-hot的颜色表
          gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose(1, 2, 0), palette)  # 将one-hot标签转为图像
          plt.imshow(gt)  # 展示标签图像
          plt.show()  # 展示标签图像
      
  demo()
