import torch
import torch.nn as nn

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



# class BoundarySensitiveAllocationModule(nn.Module):
#     def __init__(self, num_organs):
#         super(BoundarySensitiveAllocationModule, self).__init__()
#         self.num_organs = num_organs
#         self.detectors = nn.ModuleList([RegionDetectionModule() for _ in range(num_organs)])

#     def forward(self, masked):
#         #masked形状为(batch,num_organs,512,512)
#         bboxes = []
#         for i, detector in enumerate(self.detectors):
#             bbox = detector(masked[:, i:i+1, :, :])  # 形状为 (batch, 1, 4)   #tiao
#             bboxes.append(bbox)
#         return torch.stack(bboxes, dim=1)  # 形状为 (batch, num_organs, 4)

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






# 示例使用
if __name__ == "__main__":
    # 假设输入是 onehot 形式，形状为 (batch, num_classes, 512, 512)
    batch_size = 2
    num_classes = 5
    input_tensor = torch.randn(batch_size, num_classes, 512, 512)  # 示例输入
    model = BoundarySensitiveAllocationModule(num_classes)
    output = model(input_tensor)
    print(output.shape)  # 输出形状: (batch, num_classes, 4)
    
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))