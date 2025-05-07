import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from single_region import BoundarySensitiveAllocationModule
from singledataset import SegmentationToBboxDataset
from torchvision import transforms
batch_size = 4
num_epochs = 400
learning_rate = 1e-4
images_dir = "label_png"
masks_dir = "R3"
    # 数据变换（可选）
transform = transforms.Compose([transforms.ToTensor(),])
# 1. 加载数据
dataset = SegmentationToBboxDataset(images_dir, masks_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_organs = 1  # 只有一个类别
model = BoundarySensitiveAllocationModule(num_organs).to(device)

# 3. 定义损失函数和优化器
criterion = nn.SmoothL1Loss()  # 适用于边界框回归
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 训练模型
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for images, gt_bboxes in dataloader:
        images, gt_bboxes = images.to(device), gt_bboxes.to(device)

        optimizer.zero_grad()
        pred_bboxes = model(images)  # 预测边界框 (batch, num_organs, 4)

        loss = criterion(pred_bboxes.squeeze(1), gt_bboxes)  # 计算损失
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

# 5. 保存模型
torch.save(model.state_dict(), "region_detection.pth")
print("训练完成，模型已保存！")

