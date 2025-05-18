"""
最基础的train代码。用于训练Unet或者Unet-pretrained
前者是没有预训练权重的unet，后者是有预训练权重的unet

"""


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
import segmentation_models_pytorch as smp
from utils.loss import *
from utils import misc
# from datasets import bladder 
# import Bones
# import utils.image_transforms as joint_transforms
# import utils.transforms as extended_transforms
# from utils.loss import *
# from utils.metrics import diceCoeffv2

# from utils.pytorchtools import EarlyStopping
# from utils.LRScheduler import PolyLR

# 超参设置
crop_size = 256  # 输入裁剪大小   不要裁剪
batch_size = 2  # batch size
n_epoch = 150  # 训练的最大epoch
early_stop__eps = 1e-3  # 早停的指标阈值
early_stop_patience = 15  # 早停的epoch阈值
initial_lr = 1e-4  # 初始学习率
threshold_lr = 1e-6  # 早停的学习率阈值
weight_decay = 1e-6  # 学习率衰减率，学长说weight_decay应该小于学习率至少两个档位
optimizer_type = 'adam'  # adam, sgd
scheduler_type = 'ReduceLR'  # ReduceLR, StepLR, poly
label_smoothing = 0.01
aux_loss = False
gamma = 0.5
alpha = 0.85
model_number = random.randint(1, int(1e6))

num_classes = 6

model_type = "unet-pretrained"

if model_type == "unet-pretrained":
    import segmentation_models_pytorch as smp
elif model_type == "unet":
    from networks.u_net import Baseline

root_path = './'
fold = 1  # 训练集k-fold, 可设置1, 2, 3, 4, 5
depth = 2  # unet编码器的卷积层数
loss_name = 'mix'  # dice, bce, wbce, dual, wdual
reduction = ''  # aug
model_name = '{}_depth={}_fold_{}_{}_{}{}_numsclass_{}'.format(model_type, depth, fold, loss_name, reduction, model_number, num_classes)

# 训练日志
writer = SummaryWriter(os.path.join(root_path, 'log/train', model_name + '_{}fold'.format(fold) + str(int(time.time()))))
val_writer = SummaryWriter(os.path.join(os.path.join(root_path, 'log/val', model_name) + '_{}fold'.format(fold) + str(int(time.time()))))

# 训练集路径
# train_path = os.path.join(root_path, 'media/Datasets/bladder/Augdata_5folds', 'train{}'.format(fold), 'npy')
train_path = os.path.join(root_path, 'Dataset')
val_path = os.path.join(root_path, 'Dataset')


def main():
    # 定义网络
    # net = Baseline(num_classes, depth=depth).to('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == "unet-pretrained":
        net = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,                      # model output channels (number of classes in your dataset)
        )
    elif model_type == "unet":
        net = Baseline(num_classes=num_classes, depth=depth)
    net.to('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据预处理
    # center_crop = joint_transforms.CenterCrop(crop_size)
    center_crop = None
    input_transform = transforms.Compose([transforms.ToTensor()])
    target_transform = transforms.Compose([transforms.ToTensor()])

    # 训练集加载
    train_set = Dataset_gen.Dataset(train_path, 'train', fold, joint_transform=None, center_crop=center_crop,
                                    transform=input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    # 验证集加载
    val_set = Dataset_gen.Dataset(val_path, 'val', fold,
                                  joint_transform=None, transform=input_transform, center_crop=center_crop,
                                  target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # 定义损失函数
    if loss_name == 'dice':
        criterion = SoftDiceLoss(num_classes).to('cuda' if torch.cuda.is_available() else 'cpu')
    elif loss_name == 'mix':
        criterion = WBCE_Dice_Loss(num_classes,size=512,weight=(1.0,1.0,1.0,1.0,1.0,1.0)).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # 定义早停机制
    # early_stopping = EarlyStopping(early_stop_patience, verbose=True, delta=early_stop__eps,
    #                                path=os.path.join(root_path, 'checkpoint', '{}.pth'.format(model_name)))

    # 定义优化器
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    # 定义学习率衰减策略
    if scheduler_type == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    elif scheduler_type == 'ReduceLR':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # elif scheduler_type == 'poly':
    #     scheduler = PolyLR(optimizer, max_iter=n_epoch, power=0.9)
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

        # 训练模型
        net.train()
        for batch, ((input, mask), file_name) in enumerate(train_loader, 1):
            X = input.to('cuda' if torch.cuda.is_available() else 'cpu')
            y = mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            output = net(X)
            output = torch.sigmoid(output)
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            iters += 1
            train_losses.append(loss.item())
            # print(output.detach().cpu().numpy().shape)
            # gt = helpers.onehot_to_mask(output[0].detach().cpu().numpy().transpose([1, 2, 0]), Bones.palette)
            # gt = helpers.array_to_img(gt)
            # plt.imshow(gt)
            # plt.show(block=False)
            # plt.pause(0.2)
            # plt.close()
            class_dice = []
            for i in range(1, num_classes):  # 跳过背景类
                cur_dice = diceCoeffv2(output[:, i:i + 1, :], y[:, i:i + 1, :]).cpu().item()
                class_dice.append(cur_dice)

            mean_dice = sum(class_dice) / len(class_dice)
            train_class_dices += np.array(class_dice)

            # 重新匹配类别名称
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

        # print('epoch {}/{} - train_loss: {:.4} - train_mean_dice: {:.4} - dice_bladder: {:.4} - dice_tumor: {:.4}'.format(
        #         epoch, num_epoches, train_loss, train_mean_dice, train_class_dices[0], train_class_dices[1]))
        # 更新 print 输出格式，保持一致
        print('epoch {}/{} - train_loss: {:.4f} - train_mean: {:.4f} - '
            'L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f}'.format(
            epoch, num_epoches, train_loss, train_mean_dice,
            train_class_dices[0],  # L3
            train_class_dices[1],  # R3
            train_class_dices[2],  # S
            train_class_dices[3],  # L
            train_class_dices[4]   # R
        ))


        # 验证模型
        net.eval()
        for val_batch, ((input, mask), file_name) in tqdm(enumerate(val_loader, 1)):
            val_X = input.to('cuda' if torch.cuda.is_available() else 'cpu')
            val_y = mask.to('cuda' if torch.cuda.is_available() else 'cpu')

            pred = net(val_X)
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

        # print('val_loss: {:.4} - val_mean_dice: {:.4} - bladder: {:.4}- tumor: {:.4}'
        #     .format(val_loss, val_mean_dice, val_class_dices[0], val_class_dices[1]))
        print('val_loss: {:.4f} - val_mean: {:.4f} - '
            'L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f}'
            .format(val_loss, val_mean_dice,
                    val_class_dices[0],  # L3
                    val_class_dices[1],  # R3
                    val_class_dices[2],  # S
                    val_class_dices[3],  # L
                    val_class_dices[4]   # R
        ))


        print('lr: {}'.format(optimizer.param_groups[0]['lr']))

        # early_stopping(val_mean_dice, net, epoch)
        # if early_stopping.early_stop or optimizer.param_groups[0]['lr'] < threshold_lr:
        #     print("Early stopping")
        #     # 结束模型训练
        #     break

    print('----------------------------------------------------------')
    #存储模型
    save_dir = os.path.join(root_path, 'checkpoint')
    os.makedirs(save_dir, exist_ok=True)  # 如果不存在，则创建
    torch.save(net.state_dict(), os.path.join(save_dir, '{}.pth'.format(model_name)))
    
    # print('save epoch {}'.format(early_stopping.save_epoch))
    print('model saved')
    print('stoped epoch {}'.format(epoch))
    print('----------------------------------------------------------')
    # 训练结束后，关闭 SummaryWriter
    writer.close()
    val_writer.close()

if __name__ == '__main__':
    main()
