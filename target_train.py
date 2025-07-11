# train_deeplabv3plus.py

import time
import os
import torch
import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
import Dataset_gen
from torchvision import transforms

from utils.metrics import diceCoeffv2
import segmentation_models_pytorch as smp # smp库已包含DeepLabV3+
from utils.loss import *
from utils import misc

# 超参设置
crop_size = 256
batch_size = 2
n_epoch = 50
early_stop__eps = 1e-3
early_stop_patience = 15
initial_lr = 1e-4
threshold_lr = 1e-6
weight_decay = 1e-5
optimizer_type = 'adam'
scheduler_type = 'ReduceLR'
label_smoothing = 0.01
aux_loss = False
gamma = 0.5
alpha = 0.85
model_number = random.randint(1, int(1e6))

num_classes = 6

# ========================= 关键改动点 1 =========================
# 将模型类型更改为 'deeplabv3plus' 来激活新的模型
model_type = "deeplabv3plus"  # <--- MODIFIED (was "multitask")
# ================================================================

if model_type == "unet-pretrained":
    from networks.u_net import Baseline # 假设原来这里是 smp.Unet
elif model_type == "unet":
    from networks.u_net import Baseline
elif model_type == "multitask":
    from target import MultiTaskUNetWithRegionDetection
# <--- NEW: DeepLabV3+ 无需额外导入，smp 已包含

root_path = './'
fold = 1
depth = 2 # 对于DeepLabV3+，此参数无用，但为保持变量名统一而保留
loss_name = 'mix'
reduction = ''
# ========================= 关键改动点 2 =========================
# 更新模型名称以反映变化
model_name = '{}_fold_{}_{}_{}{}_numsclass_{}'.format(model_type, fold, loss_name, reduction, model_number, num_classes) # <--- MODIFIED
# ================================================================

writer = SummaryWriter(os.path.join(root_path, 'log/train', model_name + '_{}fold'.format(fold) + str(int(time.time()))))
val_writer = SummaryWriter(os.path.join(os.path.join(root_path, 'log/val', model_name) + '_{}fold'.format(fold) + str(int(time.time()))))

train_path = os.path.join(root_path, 'Dataset')
val_path = os.path.join(root_path, 'Dataset')


def main():
    # 定义网络
    if model_type == "unet-pretrained":
        net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=num_classes,
        )
    # ========================= 关键改动点 3 =========================
    # 添加新的模型分支来实例化 DeepLabV3+
    elif model_type == "deeplabv3plus": # <--- NEW
        net = smp.DeepLabV3Plus(
            encoder_name="resnet34",        # 您可以选择不同的encoder, e.g., "efficientnet-b7"
            encoder_weights="imagenet",     # 使用 'imagenet' 预训练权重
            in_channels=1,                  # 您的MRI是单通道图像
            classes=num_classes,            # 输出通道数
        )
    # ================================================================
    elif model_type == "unet":
        net = Baseline(num_classes=num_classes, depth=depth)
    elif model_type == "multitask":
        net = MultiTaskUNetWithRegionDetection(num_classes=num_classes, depth=depth)
        
    net.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    center_crop = None
    input_transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.ToTensor()])

    train_set = Dataset_gen.Dataset(train_path, 'train', fold, joint_transform=None, center_crop=center_crop,
                                    transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    val_set = Dataset_gen.Dataset(val_path, 'val', fold,
                                  joint_transform=None, transform=input_transform, center_crop=center_crop,
                                  target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    if loss_name == 'dice':
        criterion = SoftDiceLoss(num_classes).to('cuda' if torch.cuda.is_available() else 'cpu')
    elif loss_name == 'mix':
        criterion = WBCE_Dice_Loss(num_classes,size=512,weight=(1.0,1.0,1.0,1.0,1.0,1.0)).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    if scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    elif scheduler_type == 'ReduceLR':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    else:
        scheduler = None

    train(train_loader, val_loader, net, criterion, optimizer, scheduler, None,  n_epoch, 0)


def train(train_loader, val_loader, net, criterion, optimizer, scheduler, warm_scheduler, num_epoches,
          iters):
    for epoch in range(1, num_epoches + 1):
        st = time.time()
        train_class_dices = np.array([0] * (num_classes - 1), dtype=np.float64)
        val_class_dices = np.array([0] * (num_classes - 1), dtype=np.float64)
        val_dice_arr = []
        train_losses = []
        val_losses = []

        net.train()
        for batch, ((input, mask), file_name) in enumerate(train_loader, 1):
            X = input.to('cuda' if torch.cuda.is_available() else 'cpu')
            y = mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            
            # ========================= 关键改动点 4 =========================
            # smp.DeepLabV3Plus 只返回一个分割图，不再有多任务的其它输出
            output = net(X) # <--- MODIFIED
            # output,one_hot_mask,activation = net(X) # <--- 原来的多任务输出
            # ================================================================

            output = torch.sigmoid(output)
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            iters += 1
            train_losses.append(loss.item())

            class_dice = []
            for i in range(1, num_classes):
                cur_dice = diceCoeffv2(output[:, i:i + 1, :], y[:, i:i + 1, :]).cpu().item()
                class_dice.append(cur_dice)

            mean_dice = sum(class_dice) / len(class_dice)
            train_class_dices += np.array(class_dice)
            
            string_print = ('epoch: {} - iters: {} - loss: {:.4f} - mean: {:.4f} - '
                            'L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f} - time: {:.2f}s').format(
                            epoch, iters, loss.data.cpu(), mean_dice, 
                            class_dice[0], class_dice[1], class_dice[2], class_dice[3], class_dice[4], 
                            time.time() - st)

            misc.log(string_print)
            st = time.time()

        train_loss = np.average(train_losses)
        train_class_dices = train_class_dices / batch
        train_mean_dice = train_class_dices.sum() / train_class_dices.size

        writer.add_scalar('main_loss', train_loss, epoch)
        writer.add_scalar('main_dice', train_mean_dice, epoch)

        print('epoch {}/{} - train_loss: {:.4f} - train_mean: {:.4f} - '
            'L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f}'.format(
            epoch, num_epoches, train_loss, train_mean_dice,
            train_class_dices[0], train_class_dices[1], train_class_dices[2], train_class_dices[3], train_class_dices[4]
        ))

        net.eval()
        with torch.no_grad(): # 在验证时使用 no_grad() 以节省显存和计算
            for val_batch, ((input, mask), file_name) in tqdm(enumerate(val_loader, 1)):
                val_X = input.to('cuda' if torch.cuda.is_available() else 'cpu')
                val_y = mask.to('cuda'if torch.cuda.is_available() else 'cpu')
                
                # ========================= 关键改动点 5 =========================
                # 同样，验证时也只接收一个输出
                pred = net(val_X) # <--- MODIFIED
                # pred,one_hot_mask,activation = net(val_X) # <--- 原来的多任务输出
                # ================================================================

                pred = torch.sigmoid(pred)
                val_loss = criterion(pred, val_y)

                val_losses.append(val_loss.item())
                pred = pred.cpu().detach()
                val_class_dice = []
                for i in range(1, num_classes):
                    val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

                val_dice_arr.append(val_class_dice)
                val_class_dices += np.array(val_class_dice)

        val_loss = np.average(val_losses)
        val_dice_arr = np.array(val_dice_arr)
        val_class_dices = val_class_dices / val_batch
        val_mean_dice = val_class_dices.sum() / val_class_dices.size

        val_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        val_writer.add_scalar('main_loss', val_loss, epoch)
        val_writer.add_scalar('main_dice', val_mean_dice, epoch)

        print('val_loss: {:.4f} - val_mean: {:.4f} - '
            'L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f}'
            .format(val_loss, val_mean_dice,
                    val_class_dices[0], val_class_dices[1], val_class_dices[2], val_class_dices[3], val_class_dices[4]
        ))

        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

    print('----------------------------------------------------------')
    save_dir = os.path.join(root_path, 'checkpoint')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(net.state_dict(), os.path.join(save_dir, '{}.pth'.format(model_name)))
    
    print('model saved')
    print('stoped epoch {}'.format(epoch))
    print('----------------------------------------------------------')
    writer.close()
    val_writer.close()

if __name__ == '__main__':
    main()