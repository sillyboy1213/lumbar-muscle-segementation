"""
此脚本定义了用于边界框检测的 PyTorch 模块。

包含两个主要的类：

1. RegionDetectionModule:
   - 一个卷积神经网络 (CNN) 模型，用于从单个输入通道 (例如，一个器官的掩码) 预测边界框。
   - 架构：
     - 输入: (batch, 1, 512, 512) 的张量。
     - MaxPool2d: 核大小为 2，步幅为 2。
     - Conv2d (conv1): 1 输入通道, 3 输出通道, 核大小 3, 步幅 3, ReLU 激活。
     - Conv2d (conv2): 3 输入通道, 9 输出通道, 核大小 3, 步幅 3, ReLU 激活。
     - Flatten: 将特征图展平。
     - Linear (fc1): 输入特征数 (9*28*28), 输出 128, ReLU 激活。
     - Linear (fc2): 输入 128, 输出 4 (预测的边界框坐标 [x_min, y_min, x_max, y_max])。
   - 输出: (batch, 4) 的张量。

2. BoundarySensitiveAllocationModule:
   - 一个包装模块，为每个需要检测的器官（类别）实例化一个独立的 `RegionDetectionModule`。
   - 它接收一个多通道的 one-hot 编码掩码作为输入，其中每个通道代表一个器官（通常第一个通道是背景，会被跳过）。
   - 对每个非背景通道，它使用对应的 `RegionDetectionModule` 来预测该器官的边界框。
   - 输入: (batch, num_organs + 1, 512, 512) 的 one-hot 编码掩码张量 (包含背景通道)。
   - 处理: 遍历每个器官通道 (跳过索引 0 的背景通道)，将其传递给相应的 `RegionDetectionModule`。
   - 输出: (batch, num_organs, 4) 的张量，包含每个器官预测的边界框。
"""
import torch
import torch.nn as nn

class RegionDetectionModule(nn.Module):
    def __init__(self):
        super(RegionDetectionModule, self).__init__()

        # 最大池化层
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层 conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=3)
       
        # 卷积层 conv2
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=9, kernel_size=3, stride=3)
        
        # 全连接层 fc1
        self.infeature = 9 * 28 * 28
        self.fc1 = nn.Linear(in_features=self.infeature, out_features=128)
        
        # 全连接层 fc2
        self.fc2 = nn.Linear(in_features=128, out_features=4)

    def forward(self, x):
        # 输入形状: (batch, 1, 512, 512)
        batch_size, _, height, width = x.shape

        # 最大池化
        pooled = self.max_pool(x)
        # 卷积层 conv1
        conv1_out = self.conv1(pooled)
        # conv1_out = self.bn1(conv1_out)  # 应用 BatchNorm
        conv1_out = torch.relu(conv1_out)
        # 卷积层 conv2
        conv2_out = self.conv2(conv1_out)
        # conv2_out = self.bn2(conv2_out)  # 应用 BatchNorm
        conv2_out = torch.relu(conv2_out)
        # 展平特征图
        flattened = conv2_out.view(batch_size, -1)
        # 全连接层 fc1
        fc1_out = self.fc1(flattened)
        # fc1_out = self.bn3(fc1_out)  # 应用 BatchNorm
        fc1_out = torch.relu(fc1_out)
        # 全连接层 fc2
        fc2_out = self.fc2(fc1_out)

        # 输出形状: (batch, 4)
        return fc2_out



class BoundarySensitiveAllocationModule(nn.Module):
    def __init__(self, num_organs):
        super(BoundarySensitiveAllocationModule, self).__init__()
        self.num_organs = num_organs
        self.detectors = nn.ModuleList([RegionDetectionModule() for _ in range(num_organs)])

    def forward(self, masked):
        #masked形状为(batch,num_organs,512,512)
        bboxes = []
        for i, detector in enumerate(self.detectors):
            bbox = detector(masked[:, i+1:i+2, :, :])  # 形状为 (batch, 1, 4)   #跳过背景
            bboxes.append(bbox)
        return torch.stack(bboxes, dim=1)  # 形状为 (batch, num_organs, 4)
    
