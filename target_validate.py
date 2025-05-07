"""
本代码用于验证多任务模型，基于validate.py
"""

import os
import cv2
import torch
import shutil
# import utils.image_transforms as joint_transforms
from torch.utils.data import DataLoader
# import utils.transforms as extended_transforms
# import Bones
from utils.loss import *
from networks.u_net import Baseline
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
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
from networks.vit_seg_modeling import VisionTransformer as ViT_seg  # 导入Vision Transformer模型类
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg  # 导入Vision Transformer模型的配置
crop_size = 256
val_path = r'./Dataset'
# center_crop = joint_transforms.CenterCrop(crop_size)
center_crop  = None
val_input_transform = transforms.Compose([
    transforms.ToTensor(),])
target_transform = transforms.Compose([
    transforms.ToTensor(),])


val_set = Dataset_gen.Dataset(val_path, 'val', 1,
                              joint_transform=None, transform=val_input_transform, center_crop=center_crop,
                              target_transform=target_transform)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

# palette = [[0, 0, 0], [170, 0, 0], [0, 85, 0], [170, 0, 127], 
#            [0, 85, 127], [170, 0, 255], [0, 85, 255], [170, 255, 0]]  # one-hot的颜色表
palette = [[0, 0, 0],  [170, 0, 255], [0, 85, 255], [170, 255, 0],[85, 255, 0],   
           [255, 255, 127]]  # one-hot的颜色表
num_classes = 6  # 分类数,不包含L和R
model_type = "multitask"
if model_type == "unet-pretrained":
        net = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=num_classes,                      # model output channels (number of classes in your dataset)
        )

elif model_type == "unet":
        net = Baseline(num_classes=num_classes, depth=2)
        
elif model_type == "transunet":
    # 获取ViT的配置并设置参数
    vit_name = 'R50-ViT-B_16'
    n_skip = 3  # 设置跳跃连接数
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = num_classes  # 设置类别数
    config_vit.n_skip = n_skip  # 设置跳跃连接数
    img_size = 512
    vit_patches_size = 16
    # 如果ViT模型名包含 'R50'，则设置patch的网格大小
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size /vit_patches_size))

    # 初始化Vision Transformer模型并加载预训练权重
    net = ViT_seg(config_vit, img_size=img_size, num_classes=num_classes)
    net.load_from(weights=np.load(config_vit.pretrained_path))  # 从指定路径加载预训练权重
elif model_type =="multitask":
    from target import MultiTaskUNetWithRegionDetection
    net = MultiTaskUNetWithRegionDetection(num_classes=num_classes, depth=2)
net.to('cuda' if torch.cuda.is_available() else 'cpu')
# net.load_state_dict(torch.load("./checkpoint/transunet_depth=2_fold_1_dice_865804.pth"))  #transunet
# net.load_state_dict(torch.load("./checkpoint/unet_depth=2_fold_1_dice_218155.pth"))  #unet
net.load_state_dict(torch.load("./checkpoint/multitask_depth=2_fold_1_mix_993320_numsclass_6.pth"))  #unet-pretrained
net.eval()


def auto_val(net):
    # Dice 系数累积
    dices = 0
    class_dices = np.array([0] * (num_classes - 1), dtype=np.float64)

    save_path = './results'
    img_path = os.path.join(save_path, 'images')
    pred_path = os.path.join(save_path, f'pred_{model_type}')
    gt_path = os.path.join(save_path, 'gt')

    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    val_dice_arr = []
    for (input, mask), file_name in tqdm(val_loader):
        file_name = file_name[0].split('.')[0]

        X = input.cuda()
        pred,one_hot_mask,activation = net(X)
        pred = torch.sigmoid(pred)
        pred = pred.cpu().detach()

        # 原图
        m1 = np.array(input.squeeze())
        m1 = helpers.array_to_img(np.expand_dims(m1, 2))

        # gt
        gt = helpers.onehot_to_mask(np.array(mask.squeeze()).transpose([1, 2, 0]), palette)
        gt = helpers.array_to_img(gt)

        # pred
        save_pred = helpers.onehot_to_mask(np.array(pred.squeeze()).transpose([1, 2, 0]), palette)
        save_pred_png = helpers.array_to_img(save_pred)

        #展示one_hot_mask,activation
        one_hot_mask = one_hot_mask.cpu().detach()
        
        # 去掉 batch 维度，变成 (6, 512, 512)
        one_hot_mask = one_hot_mask.squeeze(0)  # shape: (6, 512, 512)

        # 可视化每一个通道
        fig, axes = plt.subplots(1, 6, figsize=(18, 3))
        for i in range(6):
            axes[i].imshow(one_hot_mask[i], cmap='gray')
            axes[i].set_title(f'Class {i}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
        activation = activation.cpu().detach()
        #输出one_hot_mask的最大值
        # print(one_hot_mask.max())
        # #输出activation的最大值
        # print(activation.max())
        # one_hot_mask = helpers.onehot_to_mask(np.array(one_hot_mask.squeeze()).transpose([1, 2, 0]), palette)
        # one_hot_mask = helpers.array_to_img(one_hot_mask)
        # activation = helpers.onehot_to_mask(np.array(activation.squeeze()).transpose([1, 2, 0]), palette)
        # activation = helpers.array_to_img(activation)
        
        # plt.imshow(one_hot_mask)
        # plt.show()
        # plt.imshow(activation)
        # plt.show()
        # plt.imshow(save_pred_png)
        # plt.show()
        # 保存 PNG
        m1.save(os.path.join(img_path, file_name + '.png'))
        gt.save(os.path.join(gt_path, file_name + '.png'))  
        save_pred_png.save(os.path.join(pred_path, file_name + '.png'))

        # 计算每个类别的 dice 系数
        class_dice = []
        for i in range(1, num_classes):
            class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :]))

        mean_dice = sum(class_dice) / len(class_dice)
        val_dice_arr.append(class_dice)
        dices += mean_dice
        class_dices += np.array(class_dice)

        # **更新打印信息**
        print('Val mean: {:.4f} - L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f}'
            .format(mean_dice, class_dice[0], class_dice[1], class_dice[2], class_dice[3], class_dice[4]))

    val_mean_dice = dices / len(val_loader)
    val_class_dice = class_dices / len(val_loader)

    # **更新最终统计信息**
    print('Final Val mean: {:.4f} - L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f}'
        .format(val_mean_dice, val_class_dice[0], val_class_dice[1], val_class_dice[2], val_class_dice[3], val_class_dice[4]))



if __name__ == '__main__':
    np.set_printoptions(threshold=9999999)
    auto_val(net)


      # print(output.detach().cpu().numpy().shape)
            # gt = helpers.onehot_to_mask(output[0].detach().cpu().numpy().transpose([1, 2, 0]), Bones.palette)
            # gt = helpers.array_to_img(gt)
            # plt.imshow(gt)
            # plt.show(block=False)
            # plt.pause(0.2)
            # plt.close()