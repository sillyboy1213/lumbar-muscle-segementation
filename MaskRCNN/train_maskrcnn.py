import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
from tqdm import tqdm

# 从您的数据加载脚本中导入Dataset类
from datagen import Dataset 

# --- 1. 超参数与设置 ---
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# 类别数 = 5种肌肉 + 1背景
NUM_CLASSES = 6 
BATCH_SIZE = 4 # 根据您的显存大小调整
NUM_EPOCHS = 50 # 训练轮次
LEARNING_RATE = 0.001 # 学习率
WEIGHT_DECAY = 0.0005 # 权重衰减

# 请根据您的实际路径进行修改
DATA_ROOT = './Dataset'
CHECKPOINT_PATH = '/Users/xiaochen/Desktop/Lumbar for github/MaskRCNN/checkpoints_maskrcnn'
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


# --- 2. 模型定义 ---
# train_maskrcnn.py

def get_model_instance_segmentation(num_classes):
    """
    加载并修改预训练的Mask R-CNN模型。
    """
    # 加载在COCO上预训练的Mask R-CNN模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')

    # ==================== 不再修改输入通道 ====================
    # 以下修改 conv1 的代码块需要被删除或注释掉
    # original_conv1 = model.backbone.body.conv1
    # original_weights = original_conv1.weight.data
    # new_conv1 = torch.nn.Conv2d(1, original_conv1.out_channels, 
    #                             kernel_size=original_conv1.kernel_size,
    #                             stride=original_conv1.stride, 
    #                             padding=original_conv1.padding,
    #                             bias=(original_conv1.bias is not None))
    # new_conv1.weight.data[:, 0, :, :] = original_weights.mean(dim=1)
    # model.backbone.body.conv1 = new_conv1
    # =========================================================

    # --- 修改模型头部以匹配新的类别数 (这部分代码保持不变) ---
    # 1. 替换box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # 2. 替换mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def collate_fn(batch):
    """
    自定义的collate_fn，用于处理批次中包含不同数量目标的情况。
    """
    return tuple(zip(*batch))


# --- 3. 训练与验证主逻辑 ---
def main():
    # 初始化模型并移动到设备
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.to(DEVICE)

    # 准备数据加载器
    # 假设交叉验证使用第1折
    dataset_train = Dataset(root=DATA_ROOT, mode='train', fold=1)
    dataset_val = Dataset(root=DATA_ROOT, mode='val', fold=1)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    # 定义学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    best_val_loss = float('inf')

    # 开始训练循环
    for epoch in range(NUM_EPOCHS):
        # --- 训练部分 ---
        model.train()
        train_loss_total = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for images, targets in train_pbar:
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # 模型直接返回损失字典
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss_total += losses.item()
            train_pbar.set_postfix(loss=losses.item())

        avg_train_loss = train_loss_total / len(train_loader)
        
        # --- 验证部分 ---
        model.eval()
        val_loss_total = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
        
        with torch.no_grad():
            for images, targets in val_pbar:
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                # 在验证时，模型也可以计算损失
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                val_loss_total += losses.item()
                val_pbar.set_postfix(loss=losses.item())

        avg_val_loss = val_loss_total / len(val_loader)

        print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}\n")

        # 更新学习率
        lr_scheduler.step()

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CHECKPOINT_PATH, 'maskrcnn_best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved to {save_path} with validation loss: {best_val_loss:.4f}")

    print("Training finished.")


if __name__ == "__main__":
    # 设置 num_workers > 0 在 Windows 上需要这个
    # 在 macOS 或 Linux 上通常不是必需的
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()