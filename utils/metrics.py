import torch
import torch.nn as nn
import numpy as np
import math
import scipy.spatial
import scipy.ndimage.morphology

"""
True Positive （真正， TP）预测为正的正样本
True Negative（真负 , TN）预测为负的负样本 
False Positive （假正， FP）预测为正的负样本
False Negative（假负 , FN）预测为负的正样本
"""

# 计算批量像素准确率
# 输入：predict - 4D张量，预测结果，形状为(batch_size, num_classes, height, width)
# 输入：target - 3D张量，真实标签，形状为(batch_size, height, width)
# 输出：pixel_correct - 正确预测的像素数量
# 输出：pixel_labeled - 标记的像素总数，这里背景为0，+1后背景为1，所以背景被算在标记中。
def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(predict, 1)  ## 预测结果中，最大值的索引，形状为(batch_size, height, width)
    predict = predict.cpu().numpy() + 1  ## 将预测结果转换为numpy数组，并加1，使背景为1
    target = target.cpu().numpy() + 1  ## 将真实标签转换为numpy数组，并加1，使背景为1
    pixel_labeled = np.sum(target > 0)  ## 标记的像素总数，这里背景为0，+1后背景为1，所以背景被算在标记中。
    pixel_correct = np.sum((predict == target) * (target > 0))  ## 正确预测的像素数量
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


# 计算批量的交并比(IoU)
# 输入：predict - 4D张量，预测结果
# 输入：target - 3D张量，真实标签
# 输入：nclass - 类别数量
# 输出：area_inter - 每个类别的交集面积
# 输出：area_union - 每个类别的并集面积
def batch_intersection_union(predict, target, nclass):
    """计算批量的交并比
    参数:
        predict: 输入的4D预测张量
        target: 3D标签张量
        nclass: 类别数量(整数)
    """
    _, predict = torch.max(predict, 1)  # 获取每个像素位置上预测概率最大的类别索引
    mini = 1  # 设置直方图的最小值为1
    maxi = nclass  # 设置直方图的最大值为类别数
    nbins = nclass  # 设置直方图的bin数为类别数
    predict = predict.cpu().numpy() + 1  # 将预测结果转为numpy数组并加1(使类别从1开始)
    target = target.cpu().numpy() + 1  # 将目标标签转为numpy数组并加1
    predict = predict * (target > 0).astype(predict.dtype)  # 将预测结果中未标注区域(target=0)的预测置为0
    intersection = predict * (predict == target)  # 计算预测正确的区域(intersection)
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))  # 计算每个类别的交集面积
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))  # 计算每个类别的预测面积
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))  # 计算每个类别的真实标签面积
    area_union = area_pred + area_lab - area_inter  # 计算并集面积(area_pred + area_lab - area_inter)
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"  # 确保交集面积小于等于并集面积
    return area_inter, area_union  # 返回交集和并集面积


def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)  # 转换预测为numpy数组
    im_lab = np.asarray(im_lab)  # 转换标签为numpy数组

    pixel_labeled = np.sum(im_lab > 0)  # 计算标记的像素总数（不包括背景）
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))  # 计算正确预测的像素数量
    return pixel_correct, pixel_labeled


# 计算单张图像的交并比(IoU)
# 输入：im_pred - 预测图像，numpy数组
# 输入：im_lab - 真实标签图像，numpy数组
# 输入：num_class - 类别数量
# 输出：area_inter - 每个类别的交集面积
# 输出：area_union - 每个类别的并集面积
def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


# 计算Dice系数
# 计算公式：dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：smooth - 平滑项，防止分母为0
# 输出：score - Dice系数，范围[0,1]
def diceCoeff(pred, gt, smooth=1e-5, ):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    score = (2 * intersection + smooth) / (unionset + smooth)

    return score.sum() / N


# 计算单个样本的Dice系数
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：smooth - 平滑项
# 输出：score - Dice系数
def diceFlat(pred, gt, smooth=1e-5):
    intersection = ((pred * gt).sum()).item()

    unionset = (pred.sum() + gt.sum()).item()
    score = (2 * intersection + smooth) / (unionset + smooth)
    return score


# 计算Dice系数的另一种实现
# 计算公式：dice = (2 * tp) / (2 * tp + fp + fn)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - Dice系数
def diceCoeffv2(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() / N


# 计算Dice系数的第三种实现
# 计算公式：dice = (2 * tp) / (2 * tp + fp + fn)
# 使用布尔运算计算tp、fp、fn
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - Dice系数
def diceCoeffv3(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
    # 转为float，以防long类型之间相除结果为0
    score = (2 * tp + eps).float() / (2 * tp + fp + fn + eps).float()

    return score.sum() / N


# 计算Jaccard系数（IoU）
# 计算公式：TP / (TP + FP + FN)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - Jaccard系数
def jaccard(pred, gt, eps=1e-5):
    """TP / (TP + FP + FN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp.float() + eps) / ((tp + fp + fn).float() + eps)
    return score.sum() / N


# 计算单个样本的Jaccard系数
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - Jaccard系数
def jaccardFlat(pred, gt, eps=1e-5):
    pred_flat = pred.squeeze()
    gt_flat = gt.squeeze()
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))
    score = (tp.float() + eps) / ((tp + fp + fn).float() + eps)
    return score


# 计算Jaccard系数的另一种实现
# 计算公式：TP / (TP + FP + FN)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - Jaccard系数
def jaccardv2(pred, gt, eps=1e-5):
    """TP / (TP + FP + FN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp

    score = (tp + eps).float() / (tp + fp + fn + eps).float()
    return score.sum() / N


# 计算Tversky系数
# 计算公式：TP / (TP + (1-alpha) * FP + alpha * FN)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输入：alpha - 假阴性惩罚系数
# 输出：score - Tversky系数
def tversky(pred, gt, eps=1e-5, alpha=0.7):
    """TP / (TP + (1-alpha) * FP + alpha * FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (tp + eps) / (tp + (1 - alpha) * fp + alpha * fn + eps)
    return score.sum() / N


# 计算准确率
# 计算公式：(TP + TN) / (TP + FP + FN + TN)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - 准确率
def accuracy(pred, gt, eps=1e-5):
    """(TP + TN) / (TP + FP + FN + TN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = ((tp + tn).float() + eps) / ((tp + fp + tn + fn).float() + eps)

    return score.sum() / N


# 计算精确率
# 计算公式：TP / (TP + FP)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - 精确率
def precision(pred, gt, eps=1e-5):
    """TP / (TP + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))

    score = (tp.float() + eps) / ((tp + fp).float() + eps)

    return score.sum() / N


# 计算敏感度（召回率）
# 计算公式：TP / (TP + FN)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - 敏感度
def sensitivity(pred, gt, eps=1e-5):
    """TP / (TP + FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp.float() + eps) / ((tp + fn).float() + eps)

    return score.sum() / N


# 计算特异度
# 计算公式：TN / (TN + FP)
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - 特异度
def specificity(pred, gt, eps=1e-5):
    """TN / (TN + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))

    score = (tn.float() + eps) / ((fp + tn).float() + eps)

    return score.sum() / N


# 计算召回率（与敏感度相同）
# 输入：pred - 预测张量
# 输入：gt - 真实标签张量
# 输入：eps - 平滑项
# 输出：score - 召回率
def recall(pred, gt, eps=1e-5):
    return sensitivity(pred, gt)


# 计算3D表面距离的类
# 用于计算两个3D对象表面之间的距离度量
class Surface(object):
    # The edge points of the mask object.
    __mask_edge_points = None
    # The edge points of the reference object.
    __reference_edge_points = None
    # The nearest neighbours distances between mask and reference edge points.
    __mask_reference_nn = None
    # The nearest neighbours distances between reference and mask edge points.
    __reference_mask_nn = None
    # Distances of the two objects surface points.
    __distance_matrix = None

    # 初始化Surface对象
    # 输入：mask - 掩码图像
    # 输入：reference - 参考图像
    # 输入：physical_voxel_spacing - 物理体素间距
    # 输入：mask_offset - 掩码偏移量
    # 输入：reference_offset - 参考偏移量
    # 输入：connectivity - 连通性参数
    def __init__(self, mask, reference, physical_voxel_spacing=[1, 1, 1], mask_offset=[0, 0, 0],
                 reference_offset=[0, 0, 0], connectivity=1):
        self.connectivity = connectivity
        # compute edge images
        mask_edge_image = self.compute_contour(mask)
        reference_edge_image = self.compute_contour(reference)
        mask_pts = mask_edge_image.nonzero()
        mask_edge_points = list(zip(mask_pts[0], mask_pts[1], mask_pts[2]))
        reference_pts = reference_edge_image.nonzero()
        reference_edge_points = list(zip(reference_pts[0], reference_pts[1], reference_pts[2]))
        # check if there is actually an object present
        if 0 >= len(mask_edge_points):
            raise Exception('The mask image does not seem to contain an object.')
        if 0 >= len(reference_edge_points):
            raise Exception('The reference image does not seem to contain an object.')
        # add offsets to the voxels positions and multiply with physical voxel spacing
        # to get the real positions in millimeters
        physical_voxel_spacing = scipy.array(physical_voxel_spacing)
        mask_edge_points = scipy.array(mask_edge_points, dtype='float64')
        mask_edge_points += scipy.array(mask_offset)
        mask_edge_points *= physical_voxel_spacing
        reference_edge_points = scipy.array(reference_edge_points, dtype='float64')
        reference_edge_points += scipy.array(reference_offset)
        reference_edge_points *= physical_voxel_spacing
        # set member vars
        self.__mask_edge_points = mask_edge_points
        self.__reference_edge_points = reference_edge_points

    # 计算最大对称表面距离
    # 输出：最大对称表面距离
    def get_maximum_symmetric_surface_distance(self):
        # Get the maximum of the nearest neighbour distances
        A_B_distance = self.get_mask_reference_nn().max()
        B_A_distance = self.get_reference_mask_nn().max()
        # compute result and return
        return min(A_B_distance, B_A_distance)

    # 计算均方根对称表面距离
    # 输出：均方根对称表面距离
    def get_root_mean_square_symmetric_surface_distance(self):
        # get object sizes
        mask_surface_size = len(self.get_mask_edge_points())
        reference_surface_sice = len(self.get_reference_edge_points())
        # get minimal nearest neighbours distances
        A_B_distances = self.get_mask_reference_nn()
        B_A_distances = self.get_reference_mask_nn()
        # square the distances
        A_B_distances_sqrt = A_B_distances * A_B_distances
        B_A_distances_sqrt = B_A_distances * B_A_distances
        # sum the minimal distances
        A_B_distances_sum = A_B_distances_sqrt.sum()
        B_A_distances_sum = B_A_distances_sqrt.sum()
        # compute result and return
        return math.sqrt(1. / (mask_surface_size + reference_surface_sice)) * math.sqrt(
            A_B_distances_sum + B_A_distances_sum)

    # 计算平均对称表面距离
    # 输出：平均对称表面距离
    def get_average_symmetric_surface_distance(self):
        # get object sizes
        mask_surface_size = len(self.get_mask_edge_points())
        reference_surface_sice = len(self.get_reference_edge_points())
        # get minimal nearest neighbours distances
        A_B_distances = self.get_mask_reference_nn()
        B_A_distances = self.get_reference_mask_nn()
        # sum the minimal distances
        A_B_distances = A_B_distances.sum()
        B_A_distances = B_A_distances.sum()
        # compute result and return
        return 1. / (mask_surface_size + reference_surface_sice) * (A_B_distances + B_A_distances)

    # 获取掩码到参考的最近邻距离
    # 输出：最近邻距离数组
    def get_mask_reference_nn(self):
        # Note: see note for @see get_reference_mask_nn
        if None == self.__mask_reference_nn:
            tree = scipy.spatial.cKDTree(self.get_mask_edge_points())
            self.__mask_reference_nn, _ = tree.query(self.get_reference_edge_points())
        return self.__mask_reference_nn

    # 获取参考到掩码的最近邻距离
    # 输出：最近邻距离数组
    def get_reference_mask_nn(self):
        if self.__reference_mask_nn is None:
            tree = scipy.spatial.cKDTree(self.get_reference_edge_points())
            self.__reference_mask_nn, _ = tree.query(self.get_mask_edge_points())
        return self.__reference_mask_nn

    # 获取掩码边缘点
    # 输出：边缘点坐标数组
    def get_mask_edge_points(self):
        return self.__mask_edge_points

    # 获取参考边缘点
    # 输出：边缘点坐标数组
    def get_reference_edge_points(self):
        return self.__reference_edge_points

    # 计算轮廓
    # 输入：array - 输入数组
    # 输出：轮廓数组
    def compute_contour(self, array):
        footprint = scipy.ndimage.morphology.generate_binary_structure(array.ndim, self.connectivity)
        # create an erode version of the array
        erode_array = scipy.ndimage.morphology.binary_erosion(array, footprint)
        array = array.astype(bool)
        # xor the erode_array with the original and return
        return array ^ erode_array


import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion

def compute_hausdorff_distance(pred, gt, num_classes):
    """
    计算每个类别的 Hausdorff Distance (HD)。
    
    Args:
        pred (torch.Tensor): 模型预测的概率图，shape = [B, C, H, W]
        gt (torch.Tensor): 真实标签图，shape = [B, H, W]
        num_classes (int): 类别数，包括背景类

    Returns:
        hd_dict (dict): 每个类别的 Hausdorff Distance，字典形式 {类别编号: HD}
    """
    # 将预测转换为类别标签 (softmax 输出转为类别标签)
    pred_label = torch.argmax(pred, dim=1)  # shape: [B, H, W]
    
    hd_dict = {}  # 用于保存每个类别的 Hausdorff Distance
    
    for b in range(pred.shape[0]):  # 每张图像
        for c in range(1, num_classes):  # 跳过背景类（通常为 0）
            # 生成二值掩码
            pred_mask = (pred_label[b] == c).cpu().numpy().astype(np.uint8)
            gt_mask = (gt[b] == c).cpu().numpy().astype(np.uint8)
            
            # 如果该类没有出现，则跳过
            if np.sum(pred_mask) == 0 or np.sum(gt_mask) == 0:
                continue
            
            # 计算边缘点
            pred_edge = pred_mask ^ binary_erosion(pred_mask)
            gt_edge = gt_mask ^ binary_erosion(gt_mask)
            
            # 提取边缘点的坐标
            pred_points = np.argwhere(pred_edge)
            gt_points = np.argwhere(gt_edge)

            # 计算 Hausdorff 距离（正向和反向）
            hd = max(
                directed_hausdorff(pred_points, gt_points)[0],
                directed_hausdorff(gt_points, pred_points)[0]
            )
            
            if c not in hd_dict:
                hd_dict[c] = []
            hd_dict[c].append(hd)
    
    # 计算每个类别的平均 Hausdorff Distance
    for c in hd_dict:
        hd_dict[c] = np.mean(hd_dict[c])  # 平均每张图像的 HD
    
    return hd_dict





if __name__ == '__main__':
    # shape = torch.Size([2, 3, 4, 4])
    # 模拟batch_size = 2
    '''
    1 0 0= bladder
    0 1 0 = tumor
    0 0 1= background 
    '''
    pred = torch.Tensor([[
        [[0, 1, 0, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 1, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]],
        [
            [[0, 1, 0, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 1, 1],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]]
    ])

    gt = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]],
        [
            [[0, 1, 1, 0],
             [1, 0, 0, 1],
             [1, 0, 0, 1],
             [0, 1, 1, 0]],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 1, 1, 0],
             [0, 0, 0, 0]],
            [[1, 0, 0, 1],
             [0, 1, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 1]]]
    ])

    dice1 = diceCoeff(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    dice2 = jaccard(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    dice3 = diceCoeffv3(pred[:, 0:1, :], gt[:, 0:1, :], activation=None)
    print(dice1, dice2, dice3)
