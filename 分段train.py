"""
本脚本用于尝试分段训练多任务模型。在这里目标检测模块是加载预训练好的权重，并在训练过程中冻结其权重。
目标检测模块的权重在训练过程中不会被更新。
在训练过程中，仅更新分割任务的权重，而不更新区域检测任务的权重。
模型的输出包括分割结果和区域检测结果。
主要流程包括：
1. 设置超参数和训练配置。
2. 根据模型类型选择并初始化网络结构。
3. 加载并预处理训练集和验证集数据。
4. 配置损失函数、优化器和学习率调度器。
5. 进行训练和验证循环，记录损失和指标。
6. 保存训练好的模型参数。
适用于医学图像分割任务，支持多类别分割和区域检测。
"""
import time
import os
import torch
import random
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter  # 用于可视化训练过程
from torch.optim import lr_scheduler  # 学习率调度器
from tqdm import tqdm  # 进度条库
import sys
import matplotlib.pyplot as plt
import numpy as np
import Dataset_gen # 自定义的数据集加载模块
from torchvision import transforms # Pytorch中常用的图像变换

from utils.metrics import diceCoeffv2 # 自定义的Dice系数计算函数
import segmentation_models_pytorch as smp # 预训练的分割模型库
from utils.loss import * # 自定义的损失函数
from utils import misc # 其他辅助函数，例如日志记录
# from datasets import bladder
# import Bones
# import utils.image_transforms as joint_transforms
# import utils.transforms as extended_transforms
# from utils.loss import *
# from utils.metrics import diceCoeffv2

# from utils.pytorchtools import EarlyStopping
# from utils.LRScheduler import PolyLR

# 超参设置
crop_size = 256  # 输入裁剪大小   注意：当前设置为不裁剪
batch_size = 2  # 每个批次的样本数量
n_epoch = 100  # 训练的总轮数
early_stop__eps = 1e-3  # 早停机制的指标阈值 (当前未使用)
early_stop_patience = 15  # 早停机制的耐心轮数 (当前未使用)
initial_lr = 5e-4  # 初始学习率
threshold_lr = 1e-6  # 早停的学习率阈值 (当前未使用)
weight_decay = 1e-6  # 优化器的权重衰减系数，学长说weight_decay应该小于学习率至少两个档位
optimizer_type = 'adam'  # 优化器类型: 'adam' 或 'sgd'
scheduler_type = 'ReduceLR'  # 学习率调度器类型: 'ReduceLR', 'StepLR', 'poly'
label_smoothing = 0.01 # 标签平滑系数 (当前未使用)
aux_loss = False # 是否使用辅助损失 (当前未使用)
gamma = 0.5 # 学习率衰减的gamma值 (用于StepLR)
alpha = 0.85 # 损失函数中的alpha值 (具体用途取决于所选损失函数)
model_number = random.randint(1, int(1e6)) # 为模型名称生成一个随机数，用于区分不同的训练

num_classes = 6 # 分割的类别数量 (包括背景)

model_type = "multitask" # 模型类型: "unet-pretrained", "unet", "multitask"

# 根据模型类型导入相应的网络结构
if model_type == "unet-pretrained":
    import segmentation_models_pytorch as smp # 使用预训练的U-Net
elif model_type == "unet":
    from networks.u_net import Baseline # 使用自定义的U-Net基线模型
elif model_type == "multitask":
    from target import MultiTaskUNetWithRegionDetection # 使用自定义的多任务U-Net模型

root_path = './' # 项目根目录
fold = 1  # 训练集k-fold交叉验证的折数, 可设置1, 2, 3, 4, 5
depth = 2  # U-Net编码器的卷积层数 (对于自定义U-Net)
loss_name = 'mix'  # 损失函数名称: 'dice', 'bce', 'wbce', 'dual', 'wdual', 'mix'
reduction = ''  # 数据增强相关的标识 (当前为空)
# 构建模型名称，用于保存日志和模型文件
model_name = '{}_depth={}_fold_{}_{}_{}{}_numsclass_{}'.format(model_type, depth, fold, loss_name, reduction, model_number, num_classes)

# 设置TensorBoard日志记录器
# 训练日志保存路径
writer = SummaryWriter(os.path.join(root_path, 'log/train', model_name + '_{}fold'.format(fold) + str(int(time.time()))))
# 验证日志保存路径
val_writer = SummaryWriter(os.path.join(os.path.join(root_path, 'log/val', model_name) + '_{}fold'.format(fold) + str(int(time.time()))))

# 数据集路径配置
# train_path = os.path.join(root_path, 'media/Datasets/bladder/Augdata_5folds', 'train{}'.format(fold), 'npy') # 示例：膀胱数据集路径
train_path = os.path.join(root_path, 'Dataset') # 训练集实际路径
val_path = os.path.join(root_path, 'Dataset') # 验证集实际路径


def main():
    """
    主函数，执行模型训练的整个流程。
    """
    # 定义网络模型
    # net = Baseline(num_classes, depth=depth).to('cuda' if torch.cuda.is_available() else 'cpu') # 备选：直接使用自定义U-Net
    if model_type == "unet-pretrained":
        net = smp.Unet(
        encoder_name="resnet34",        # 选择编码器, 例如 mobilenet_v2 或 efficientnet-b7
        encoder_weights="imagenet",     # 使用 "imagenet" 预训练权重进行编码器初始化
        in_channels=1,                  # 模型输入通道数 (灰度图像为1, RGB图像为3等)
        classes=num_classes,            # 模型输出通道数 (数据集中类别数量)
        )
    elif model_type == "unet":
        net = Baseline(num_classes=num_classes, depth=depth) # 初始化自定义U-Net
    elif model_type == "multitask":
        # 初始化多任务U-Net，包含区域检测模块
        net = MultiTaskUNetWithRegionDetection(num_classes=num_classes, depth=depth)

        # === 加载预训练好的 region_detection 模块权重 ===
        # 指定预训练权重文件路径
        checkpoint_path = './region_detection_5_classes.pth'
        # 加载权重，map_location确保在没有GPU时也能加载
        checkpoint = torch.load(checkpoint_path, map_location= 'cuda' if torch.cuda.is_available() else 'cpu')
        # 将加载的权重载入到模型的区域检测部分
        net.region_detection.load_state_dict(checkpoint)
        print(f"Loaded pre-trained weights for region_detection from {checkpoint_path}")

        # === 冻结 region_detection 模块的参数 ===
        # 遍历区域检测模块的所有参数
        for param in net.region_detection.parameters():
            param.requires_grad = False # 设置参数不需要梯度更新，即在训练中保持不变
        print("Froze parameters for region_detection module.")

    # 将模型移动到GPU（如果可用），否则使用CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    print(f"Model moved to {device}.")

    
    # 数据预处理设置
    # center_crop = joint_transforms.CenterCrop(crop_size) # 中心裁剪 (当前未使用)
    center_crop = None # 不进行中心裁剪
    # 定义输入图像的转换：转换为Tensor
    input_transform = transforms.Compose([transforms.ToTensor()])
    # 定义目标掩码的转换：转换为Tensor
    target_transform = transforms.Compose([transforms.ToTensor()])

    # 训练集加载
    # 使用自定义的Dataset_gen.Dataset类加载训练数据
    train_set = Dataset_gen.Dataset(train_path, 'train', fold, # 数据路径，模式（训练），交叉验证折数
                                    joint_transform=None, center_crop=center_crop, # 联合变换和中心裁剪（当前为None）
                                    transform=input_transform, target_transform=target_transform) # 输入和目标变换
    # 使用DataLoader创建训练数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0表示在主进程中加载数据

    # 验证集加载
    # 使用自定义的Dataset_gen.Dataset类加载验证数据
    val_set = Dataset_gen.Dataset(val_path, 'val', fold, # 数据路径，模式（验证），交叉验证折数
                                  joint_transform=None, transform=input_transform, center_crop=center_crop, # 联合变换和中心裁剪（当前为None）
                                  target_transform=target_transform) # 输入和目标变换
    # 使用DataLoader创建验证数据加载器
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0) # batch_size=1用于逐个评估验证样本

    # 定义损失函数
    if loss_name == 'dice':
        # 使用SoftDiceLoss，适用于多类别分割
        criterion = SoftDiceLoss(num_classes).to(device)
    elif loss_name == 'mix':
        # 使用加权的BCE损失和Dice损失的组合
        # weight参数可以为不同类别设置不同的权重
        criterion = WBCE_Dice_Loss(num_classes,size=512,weight=(1.0,1.0,1.0,1.0,1.0,1.0)).to(device)
    
    # # 定义早停机制 (当前已注释掉)
    # early_stopping = EarlyStopping(early_stop_patience, verbose=True, delta=early_stop__eps,
    #                                path=os.path.join(root_path, 'checkpoint', '{}.pth'.format(model_name)))

    # 定义优化器
    # === 只优化 requires_grad=True 的参数 ===
    # 使用AdamW优化器，并通过filter筛选出需要梯度更新的参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=initial_lr, weight_decay=weight_decay)
    print(f"Optimizer: AdamW, Initial LR: {initial_lr}, Weight Decay: {weight_decay}")
    print(f"Number of trainable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")


    # 定义学习率衰减策略
    if scheduler_type == 'StepLR':
        # 每隔step_size个epoch，学习率乘以gamma
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=gamma)
    elif scheduler_type == 'ReduceLR':
        # 当验证集损失在patience个epoch内没有改善时，学习率乘以factor
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    # elif scheduler_type == 'poly': # PolyLR (当前已注释掉)
    #     scheduler = PolyLR(optimizer, max_iter=n_epoch, power=0.9)
    else:
        scheduler = None # 不使用学习率调度器
    
    if scheduler:
        print(f"Learning rate scheduler: {scheduler_type}")
    else:
        print("No learning rate scheduler.")

    # 调用训练函数开始训练
    # 参数：训练加载器，验证加载器，网络模型，损失函数，优化器，学习率调度器，预热调度器(None)，总epoch数，初始迭代次数(0)
    train(train_loader, val_loader, net, criterion, optimizer, scheduler, None,  n_epoch, 0)


def train(train_loader, val_loader, net, criterion, optimizer, scheduler, warm_scheduler, num_epoches,
          iters):
    """
    执行模型训练和验证的循环。

    Args:
        train_loader: 训练数据加载器。
        val_loader: 验证数据加载器。
        net: 神经网络模型。
        criterion: 损失函数。
        optimizer: 优化器。
        scheduler: 学习率调度器。
        warm_scheduler: 预热调度器 (当前未使用)。
        num_epoches: 总训练轮数。
        iters: 当前迭代次数。
    """
    for epoch in range(1, num_epoches + 1): # 迭代每个epoch
        st = time.time() # 记录epoch开始时间
        # 初始化每个类别的Dice系数累加器 (不包括背景类)
        train_class_dices = np.array([0] * (num_classes - 1), dtype=np.float64)
        val_class_dices = np.array([0] * (num_classes - 1), dtype=np.float64)
        val_dice_arr = [] # 存储验证集每个样本的各类别Dice
        train_losses = [] # 存储训练过程中的每个batch的损失
        val_losses = [] # 存储验证过程中的每个batch的损失

        # 训练模型阶段
        net.train() # 设置模型为训练模式
        # 遍历训练数据加载器中的每个batch
        for batch, ((input, mask), file_name) in enumerate(train_loader, 1):
            # 将输入数据和掩码移动到指定设备 (GPU或CPU)
            X = input.to('cuda' if torch.cuda.is_available() else 'cpu')
            y = mask.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            optimizer.zero_grad() # 清空之前的梯度
            # 前向传播，获取模型输出
            # output: 分割图, one_hot_mask: 区域检测的one-hot掩码, activation: 区域检测的激活图
            output, one_hot_mask, activation = net(X)
            output = torch.sigmoid(output) # 对分割图应用sigmoid激活函数，将输出值映射到(0,1)范围
            
            loss = criterion(output, y) # 计算损失
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新模型参数
            
            iters += 1 # 迭代次数加1
            train_losses.append(loss.item()) # 记录当前batch的损失值

            # 可视化部分 (当前已注释掉)
            # print(output.detach().cpu().numpy().shape)
            # gt = helpers.onehot_to_mask(output[0].detach().cpu().numpy().transpose([1, 2, 0]), Bones.palette)
            # gt = helpers.array_to_img(gt)
            # plt.imshow(gt)
            # plt.show(block=False)
            # plt.pause(0.2)
            # plt.close()

            # 计算每个类别的Dice系数 (跳过背景类，索引从1开始)
            class_dice = []
            for i in range(1, num_classes):  # num_classes是总类别数，通常第一个类别是背景
                # output[:, i:i + 1, :] 选择第i个类别的预测结果
                # y[:, i:i + 1, :] 选择第i个类别的真实标签
                cur_dice = diceCoeffv2(output[:, i:i + 1, :], y[:, i:i + 1, :]).cpu().item()
                class_dice.append(cur_dice)

            mean_dice = sum(class_dice) / len(class_dice) # 计算当前batch的平均Dice
            train_class_dices += np.array(class_dice) # 累加每个类别的Dice

            # 打印训练信息
            # 重新匹配类别名称，这里假设类别顺序为 L3, R3, S, L, R (对应class_dice的索引0到4)
            string_print = ('epoch: {} - iters: {} - loss: {:.4f} - mean: {:.4f} - '
                            'L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f} - time: {:.2f}s').format(
                            epoch, iters, loss.data.cpu(), mean_dice, 
                            class_dice[0], class_dice[1], class_dice[2], class_dice[3], class_dice[4], 
                            time.time() - st) # 计算并打印处理当前batch所用时间

            misc.log(string_print) # 使用自定义的log函数记录信息
            st = time.time() # 重置开始时间，用于计算下一个batch的处理时间
        
        # 计算整个epoch的平均训练损失和各类别平均Dice
        train_loss = np.average(train_losses)
        train_class_dices = train_class_dices / batch # batch是训练集中的总batch数
        train_mean_dice = train_class_dices.sum() / train_class_dices.size # 计算所有非背景类的平均Dice

        # 使用TensorBoard记录训练损失和Dice
        writer.add_scalar('main_loss', train_loss, epoch)
        writer.add_scalar('main_dice', train_mean_dice, epoch)

        # 打印epoch的训练总结信息
        # 更新 print 输出格式，保持与 batch log 一致
        print('epoch {}/{} - train_loss: {:.4f} - train_mean: {:.4f} - '
            'L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f}'.format(
            epoch, num_epoches, train_loss, train_mean_dice,
            train_class_dices[0],  # L3
            train_class_dices[1],  # R3
            train_class_dices[2],  # S
            train_class_dices[3],  # L
            train_class_dices[4]   # R
        ))


        # 验证模型阶段
        net.eval() # 设置模型为评估模式 (不计算梯度，关闭dropout等)
        # 使用tqdm显示验证进度条
        for val_batch, ((input, mask), file_name) in tqdm(enumerate(val_loader, 1)):
            # 将验证数据移动到指定设备
            val_X = input.to('cuda' if torch.cuda.is_available() else 'cpu')
            val_y = mask.to('cuda' if torch.cuda.is_available() else 'cpu')

            # 前向传播获取预测结果
            pred, one_hot_mask, activation = net(val_X)
            pred = torch.sigmoid(pred) # 应用sigmoid激活
            val_loss = criterion(pred, val_y) # 计算验证损失

            val_losses.append(val_loss.item()) # 记录验证损失
            pred = pred.cpu().detach() # 将预测结果移到CPU并分离计算图，以便后续Numpy操作
            
            # 计算验证集每个类别的Dice系数
            val_class_dice = []
            for i in range(1, num_classes): # 跳过背景类
                val_class_dice.append(diceCoeffv2(pred[:, i:i + 1, :], mask[:, i:i + 1, :].cpu())) # mask也需要移到CPU

            val_dice_arr.append(val_class_dice) # 存储当前样本的各类别Dice
            val_class_dices += np.array(val_class_dice) # 累加各类别Dice

        # 计算整个epoch的平均验证损失和各类别平均Dice
        val_loss = np.average(val_losses)
        val_dice_arr = np.array(val_dice_arr) # 转换为Numpy数组
        val_class_dices = val_class_dices / val_batch # val_batch是验证集中的总batch数
        val_mean_dice = val_class_dices.sum() / val_class_dices.size # 计算所有非背景类的平均Dice

        # 使用TensorBoard记录验证信息
        val_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch) # 记录当前学习率
        val_writer.add_scalar('main_loss', val_loss, epoch) # 记录验证损失
        val_writer.add_scalar('main_dice', val_mean_dice, epoch) # 记录验证平均Dice

        # 打印epoch的验证总结信息
        print('val_loss: {:.4f} - val_mean: {:.4f} - '
            'L3: {:.4f} - R3: {:.4f} - S: {:.4f} - L: {:.4f} - R: {:.4f}'
            .format(val_loss, val_mean_dice,
                    val_class_dices[0],  # L3
                    val_class_dices[1],  # R3
                    val_class_dices[2],  # S
                    val_class_dices[3],  # L
                    val_class_dices[4]   # R
        ))
        
        # 根据学习率调度器类型更新学习率
        if scheduler_type == 'StepLR':
            scheduler.step()
        elif scheduler_type == 'ReduceLR':
            scheduler.step(val_loss) # ReduceLROnPlateau需要验证损失作为参数

        print('lr: {}'.format(optimizer.param_groups[0]['lr'])) # 打印更新后的学习率

        # 早停机制 (当前已注释掉)
        # early_stopping(val_mean_dice, net, epoch)
        # if early_stopping.early_stop or optimizer.param_groups[0]['lr'] < threshold_lr:
        #     print("Early stopping")
        #     # 结束模型训练
        #     break

    print('----------------------------------------------------------')
    # 存储模型权重
    save_dir = os.path.join(root_path, 'checkpoint') # 模型保存目录
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，则创建
    # 保存模型的状态字典
    torch.save(net.state_dict(), os.path.join(save_dir, '{}.pth'.format(model_name)))
    
    # print('save epoch {}'.format(early_stopping.save_epoch)) # 打印早停保存的epoch (如果使用早停)
    print('model saved to {}'.format(os.path.join(save_dir, '{}.pth'.format(model_name))))
    print('stopped epoch {}'.format(epoch)) # 打印训练结束时的epoch
    print('----------------------------------------------------------')
    # 训练结束后，关闭 SummaryWriter
    writer.close()
    val_writer.close()

if __name__ == '__main__':
    # 当脚本作为主程序运行时，调用main函数
    main()
