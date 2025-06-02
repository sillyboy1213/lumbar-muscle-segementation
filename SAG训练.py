import time
import os
import torch
import random
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter # 移除 TensorBoardX
import wandb # 导入 wandb
from torch.optim import lr_scheduler
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
import Dataset_gen
from torchvision import transforms

from utils.metrics import diceCoeffv2
import segmentation_models_pytorch as smp
from utils.loss import *
from utils import misc

# 假设 MultiTaskUNetWithRegionDetection 已经集成了 SAG
# from networks.u_net import Baseline # 如果 model_type == "unet"
# from target import MultiTaskUNetWithRegionDetection # 如果 model_type == "multitask"


# 超参设置
crop_size = 256
batch_size = 2
n_epoch = 100
early_stop__eps = 1e-3
early_stop_patience = 15
initial_lr = 5e-4
threshold_lr = 1e-6
weight_decay = 1e-6
optimizer_type = 'adam'
scheduler_type = 'ReduceLR'
label_smoothing = 0.01
aux_loss = False
gamma = 0.5
alpha = 0.85
model_number = random.randint(1, int(1e6))

num_classes = 6
enc_final_channels = 512 # 为 MultiTaskUNetWithRegionDetection 指定编码器最终通道数

model_type = "multitask"

# 根据模型类型导入相应的网络结构
if model_type == "unet-pretrained":
    import segmentation_models_pytorch as smp
elif model_type == "unet":
    from networks.u_net import Baseline
elif model_type == "multitask":
    from target import MultiTaskUNetWithRegionDetection # 确保 target.py 中的模型已集成SAG

root_path = './'
fold = 1
depth = 2 # U-Net编码器的深度 (对于自定义U-Net或MultiTask模型中的U-Net部分)
loss_name = 'mix'
reduction = ''
model_name = '{}_depth={}_fold_{}_{}_{}{}_numsclass_{}'.format(model_type, depth, fold, loss_name, reduction, model_number, num_classes)

# WandB 项目名称 - 请替换为您自己的项目名称
WANDB_PROJECT_NAME = "SAG train" # <<<--- 请修改为您的WandB项目名称

# 数据集路径配置
train_path = os.path.join(root_path, 'Dataset')
val_path = os.path.join(root_path, 'Dataset')


def main():
    """
    主函数，执行模型训练的整个流程。
    """
    # --- WandB 初始化 ---
    config_wandb = {
        "crop_size": crop_size,
        "batch_size": batch_size,
        "n_epoch": n_epoch,
        "initial_lr": initial_lr,
        "threshold_lr": threshold_lr,
        "weight_decay": weight_decay,
        "optimizer_type": optimizer_type,
        "scheduler_type": scheduler_type,
        "label_smoothing": label_smoothing,
        "aux_loss": aux_loss,
        "gamma": gamma,
        "alpha": alpha,
        "num_classes": num_classes,
        "model_type": model_type,
        "fold": fold,
        "depth": depth,
        "loss_name": loss_name,
        "reduction": reduction,
        "enc_final_channels": enc_final_channels,
        "model_script_name": model_name # 用于区分不同的运行实例
    }
    wandb.init(
        project=WANDB_PROJECT_NAME,
        name=model_name, # 使用脚本生成的model_name作为wandb的run name
        config=config_wandb
    )

    # 定义网络模型
    if model_type == "unet-pretrained":
        net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=num_classes,
        )
    elif model_type == "unet":
        net = Baseline(num_classes=num_classes, depth=depth)
    elif model_type == "multitask":
        # 初始化多任务U-Net，包含区域检测模块和SAG
        net = MultiTaskUNetWithRegionDetection(
            img_ch=1, # 假设输入是单通道
            num_classes=num_classes,
            depth=depth,
            enc_final_channels=enc_final_channels # 传递给模型
        )

        # === 加载预训练好的 region_detection 模块权重 ===
        checkpoint_path = './region_detection_5_classes.pth'
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
            )
            # 确保 checkpoint 是 state_dict
            if not isinstance(checkpoint, dict): # 有些checkpoint可能直接保存模型而不是state_dict
                 # 如果是直接保存的模型对象，则取其 state_dict
                if hasattr(checkpoint, 'state_dict'):
                    checkpoint = checkpoint.state_dict()
                else: # 如果无法处理，则抛出错误
                    raise TypeError(f"Checkpoint at {checkpoint_path} is not a state_dict or a model object with state_dict.")

            net.region_detection.load_state_dict(checkpoint)
            print(f"Loaded pre-trained weights for region_detection from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading pre-trained weights for region_detection: {e}")
            print("Proceeding with randomly initialized region_detection module.")


        # === 冻结 region_detection 模块的参数 ===
        for param in net.region_detection.parameters():
            param.requires_grad = False
        print("Froze parameters for region_detection module.")

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    print(f"Model moved to {device}.")
    wandb.watch(net, log_freq=100) # 可选: 使用wandb.watch()记录模型梯度和参数

    center_crop = None
    input_transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.ToTensor()])

    train_set = Dataset_gen.Dataset(train_path, 'train', fold,
                                    joint_transform=None, center_crop=center_crop,
                                    transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    val_set = Dataset_gen.Dataset(val_path, 'val', fold,
                                  joint_transform=None, transform=input_transform, center_crop=center_crop,
                                  target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    if loss_name == 'dice':
        criterion = SoftDiceLoss(num_classes).to(device)
    elif loss_name == 'mix':
        criterion = WBCE_Dice_Loss(num_classes,size=512,weight=(1.0,1.0,1.0,1.0,1.0,1.0)).to(device)
    else: # 添加一个默认或错误处理
        raise ValueError(f"Unsupported loss function: {loss_name}")


    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=initial_lr, weight_decay=weight_decay)
    print(f"Optimizer: AdamW, Initial LR: {initial_lr}, Weight Decay: {weight_decay}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

    if scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)
    elif scheduler_type == 'ReduceLR':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    else:
        scheduler = None
    
    if scheduler:
        print(f"Learning rate scheduler: {scheduler_type}")
    else:
        print("No learning rate scheduler.")

    train(train_loader, val_loader, net, criterion, optimizer, scheduler, None, n_epoch, 0)

    wandb.finish() # 结束WandB运行


def train(train_loader, val_loader, net, criterion, optimizer, scheduler, warm_scheduler, num_epoches,
          iters):
    """
    执行模型训练和验证的循环。
    """
    class_names = ["L3", "R3", "S", "L", "R"] # 假设的类别名称 (num_classes - 1 个)
    if len(class_names) != num_classes -1 :
        # 如果类别名数量与前景类不符，使用通用名称
        class_names = [f"Class_{i+1}" for i in range(num_classes -1)]


    for epoch in range(1, num_epoches + 1):
        st_epoch = time.time() # 记录整个epoch的开始时间
        train_class_dices_epoch_sum = np.array([0] * (num_classes - 1), dtype=np.float64)
        val_class_dices_epoch_sum = np.array([0] * (num_classes - 1), dtype=np.float64)
        
        train_losses_epoch = []
        val_losses_epoch = []

        # 训练模型阶段
        net.train()
        st_batch_log = time.time() # 用于间歇性日志记录的时间戳
        for batch_idx, ((input_data, mask), file_name) in enumerate(train_loader, 1):
            X = input_data.to('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
            y = mask.to('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
            
            optimizer.zero_grad()
            
            output, one_hot_mask, activation_or_Xdec_k = net(X) # 假设第三个输出现在是 X_dec_k
            # output_seg = torch.sigmoid(output_logits) # Sigmoid通常在损失函数内部或之后处理
                                                    # 如果损失函数如BCEWithLogitsLoss则不需要提前sigmoid

            # 确保 output 在计算损失前是 logits (如果损失函数期望 logits)
            # 如果损失函数期望概率，则需要 sigmoid
            # 您的 WBCE_Dice_Loss 可能内部处理了 sigmoid，或者期望 logits 或 probs
            # 为简单起见，我们假设您的损失函数期望概率值(0,1)，因此应用sigmoid
            output_probs = torch.sigmoid(output)


            loss = criterion(output_probs, y) # 或者 criterion(output, y) 如果损失函数处理sigmoid
            loss.backward()
            optimizer.step()
            
            iters += 1
            train_losses_epoch.append(loss.item())

            current_batch_class_dice = []
            for i in range(1, num_classes):
                cur_dice = diceCoeffv2(output_probs[:, i:i + 1, :], y[:, i:i + 1, :]).cpu().item()
                current_batch_class_dice.append(cur_dice)
            
            current_batch_mean_dice = sum(current_batch_class_dice) / len(current_batch_class_dice) if len(current_batch_class_dice) > 0 else 0.0
            train_class_dices_epoch_sum += np.array(current_batch_class_dice)

            # 打印训练信息 (每隔N个batch或者一定时间间隔)
            # 为了减少日志冗余，这里可以调整打印频率
            if batch_idx % (len(train_loader) // 2) == 0 or batch_idx == len(train_loader) : # 大约打印2-3次
                batch_log_time = time.time() - st_batch_log
                string_print = (f'Epoch: {epoch}/{num_epoches} - Batch: {batch_idx}/{len(train_loader)} - Iter: {iters} - Loss: {loss.item():.4f} - MeanDice: {current_batch_mean_dice:.4f} - Time: {batch_log_time:.2f}s')
                # for i, name in enumerate(class_names):
                #     string_print += f' - {name}: {current_batch_class_dice[i]:.4f}'
                misc.log(string_print)
                st_batch_log = time.time() # 重置间歇性日志的时间戳
        
        # --- Epoch Training Summary ---
        avg_train_loss = np.average(train_losses_epoch) if train_losses_epoch else 0
        avg_train_class_dices = train_class_dices_epoch_sum / len(train_loader) if len(train_loader) > 0 else np.zeros_like(train_class_dices_epoch_sum)
        avg_train_mean_dice = avg_train_class_dices.sum() / avg_train_class_dices.size if avg_train_class_dices.size > 0 else 0

        log_dict_train = {
            "epoch": epoch,
            "train/epoch_loss": avg_train_loss,
            "train/epoch_mean_dice": avg_train_mean_dice,
            "train/learning_rate": optimizer.param_groups[0]['lr']
        }
        for i, name in enumerate(class_names):
            log_dict_train[f"train/dice_{name}"] = avg_train_class_dices[i]
        wandb.log(log_dict_train)

        print(f'Epoch {epoch}/{num_epoches} Summary (Train): Loss: {avg_train_loss:.4f}, MeanDice: {avg_train_mean_dice:.4f}')
        dice_str_train = " ".join([f"{name}: {avg_train_class_dices[i]:.4f}" for i, name in enumerate(class_names)])
        print(f"Train Dices: {dice_str_train}")


        # --- Validation Phase ---
        net.eval()
        with torch.no_grad(): # 关闭梯度计算
            for batch_idx_val, ((input_val, mask_val), file_name_val) in tqdm(enumerate(val_loader, 1), total=len(val_loader), desc=f"Epoch {epoch} Val"):
                val_X = input_val.to('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
                val_y = mask_val.to('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

                pred_logits_val, _, _ = net(val_X)
                pred_probs_val = torch.sigmoid(pred_logits_val)
                
                val_loss_item = criterion(pred_probs_val, val_y)
                val_losses_epoch.append(val_loss_item.item())
                
                current_val_class_dice = []
                for i in range(1, num_classes):
                    current_val_class_dice.append(diceCoeffv2(pred_probs_val[:, i:i + 1, :], val_y[:, i:i + 1, :]).cpu().item())
                val_class_dices_epoch_sum += np.array(current_val_class_dice)

        # --- Epoch Validation Summary ---
        avg_val_loss = np.average(val_losses_epoch) if val_losses_epoch else 0
        avg_val_class_dices = val_class_dices_epoch_sum / len(val_loader) if len(val_loader) > 0 else np.zeros_like(val_class_dices_epoch_sum)
        avg_val_mean_dice = avg_val_class_dices.sum() / avg_val_class_dices.size if avg_val_class_dices.size > 0 else 0

        log_dict_val = {
            "epoch": epoch, # Ensure epoch is logged with val metrics for proper x-axis alignment
            "val/epoch_loss": avg_val_loss,
            "val/epoch_mean_dice": avg_val_mean_dice
        }
        for i, name in enumerate(class_names):
            log_dict_val[f"val/dice_{name}"] = avg_val_class_dices[i]
        wandb.log(log_dict_val)
        
        print(f'Epoch {epoch}/{num_epoches} Summary (Val): Loss: {avg_val_loss:.4f}, MeanDice: {avg_val_mean_dice:.4f}')
        dice_str_val = " ".join([f"{name}: {avg_val_class_dices[i]:.4f}" for i, name in enumerate(class_names)])
        print(f"Val Dices: {dice_str_val}")
        
        epoch_duration = time.time() - st_epoch
        print(f"Epoch {epoch} completed in {epoch_duration:.2f}s. LR: {optimizer.param_groups[0]['lr']:.2e}")

        if scheduler:
            if scheduler_type == 'ReduceLR':
                scheduler.step(avg_val_loss) # ReduceLROnPlateau needs a metric to monitor
            elif scheduler_type == 'StepLR':
                scheduler.step()
            # For PolyLR or other schedulers, ensure correct step call if they are uncommented

    print('----------------------------------------------------------')
    save_dir = os.path.join(root_path, 'checkpoint')
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, f'{model_name}_epoch{epoch}.pth') # Include epoch in filename
    torch.save(net.state_dict(), model_save_path)
    
    print(f'Model saved to {model_save_path}')
    print(f'Stopped at epoch {epoch}')
    print('----------------------------------------------------------')
    # WandB finish is called at the end of main()

if __name__ == '__main__':
    # 确保 `target.py` 中定义了 `MultiTaskUNetWithRegionDetection`
    # 并且它接受 `img_ch`, `num_classes`, `depth`, `enc_final_channels` 参数
    # 并且其内部已经集成了 SAG 模块。
    # `unet_Encoder`, `uent_Decoder`, `BoundarySensitiveAllocationModule` 也需要被正确定义。
    try:
        from SAG import MultiTaskUNetWithRegionDetection # 尝试导入，确保存在
        from networks.u_net import Baseline, unet_Encoder, uent_Decoder
        from networks.custom_modules.RegionDetection import BoundarySensitiveAllocationModule
    except ImportError as e:
        print(f"Could not import necessary model class definitions: {e}")
        print("Please ensure 'target.py' contains 'MultiTaskUNetWithRegionDetection' with SAG integration,")
        print("and other U-Net components ('unet_Encoder', 'uent_Decoder', 'BoundarySensitiveAllocationModule') are correctly defined and importable.")
        print("Exiting.")
        sys.exit(1)
    main()