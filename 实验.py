import torch
import os
import numpy as np
from PIL import Image
import logging
from typing import List, Dict, Tuple, Any

# 尝试从 utils.helpers 导入 mask_to_onehot
# 如果此模块或函数在您的环境中不可用，您需要确保它存在或提供正确的实现。
try:
    from utils.helpers import mask_to_onehot
except ImportError:
    logger = logging.getLogger(__name__) # 获取一个临时的logger实例用于早期错误报告
    logger.error("错误: 无法从 'utils.helpers' 导入 'mask_to_onehot'。请确保该模块和函数可用。")
    # 定义一个占位符函数以允许脚本的其余部分至少可以被解析，但运行时会出问题
    def mask_to_onehot(mask: np.ndarray, palette: List[List[int]]) -> np.ndarray:
        raise NotImplementedError("mask_to_onehot 未正确导入或实现。")
    # 或者直接退出
    # import sys
    # sys.exit("依赖项 'utils.helpers.mask_to_onehot' 未找到。")


# =============== 获取logger实例 ===============
# 日志的详细配置（handlers, formatter）将在主程序块中进行
logger = logging.getLogger(__name__)


# =============== 指标计算函数 ===============
def compute_metrics(pred_tensor: torch.Tensor, gt_tensor: torch.Tensor, num_classes: int, eps: float = 1e-5, alpha: float = 0.7) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """
    计算各种分割指标。

    参数:
        pred_tensor (torch.Tensor): 预测的one-hot张量，形状为 (N, K, H, W)，N为批大小, K为类别数。
        gt_tensor (torch.Tensor): 真实的one-hot张量，形状为 (N, K, H, W)。
        num_classes (int): 类别总数 (包括背景)。
        eps (float): 防止除以零的小常数。
        alpha (float): Tversky指数的超参数，控制FP和FN的权重。

    返回:
        Tuple[Dict[str, float], Dict[str, List[float]]]:
            - mean_metrics: 一个字典，包含所有类别平均后的各项指标。
            - class_metrics: 一个字典，包含每个类别（不含背景）的各项指标列表。
    """
    N, K, H, W = pred_tensor.shape
    class_metrics = {
        "dice": [], "jaccard": [], "tversky": [], "accuracy": [],
        "precision": [], "sensitivity": [], "specificity": []
    }

    for i in range(1, num_classes):  # 跳过背景类别 (索引0)
        pred = pred_tensor[:, i, :, :].contiguous().view(-1)
        gt = gt_tensor[:, i, :, :].contiguous().view(-1)

        tp = torch.sum((pred == 1) & (gt == 1)).float()
        fp = torch.sum((pred == 1) & (gt == 0)).float()
        fn = torch.sum((pred == 0) & (gt == 1)).float()
        tn = torch.sum((pred == 0) & (gt == 0)).float()

        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        jaccard = (tp + eps) / (tp + fp + fn + eps)
        tversky = (tp + eps) / (tp + (1 - alpha) * fp + alpha * fn + eps)
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        sensitivity = (tp + eps) / (tp + fn + eps)
        specificity = (tn + eps) / (tn + fp + eps)

        class_metrics["dice"].append(dice.item())
        class_metrics["jaccard"].append(jaccard.item())
        class_metrics["tversky"].append(tversky.item())
        class_metrics["accuracy"].append(accuracy.item())
        class_metrics["precision"].append(precision.item())
        class_metrics["sensitivity"].append(sensitivity.item())
        class_metrics["specificity"].append(specificity.item())

    mean_metrics = {k: np.mean(v) if v else 0.0 for k, v in class_metrics.items()}
    return mean_metrics, class_metrics


# =============== 主评估函数 ===============
def evaluate_segmentation_performance(
    val_path: str,
    gt_folder_name: str = 'gt',
    pred_folder_name: str = 'pred_multitask',
    num_classes: int = 6,
    palette: List[List[int]] = None,
    top_n_images: int = 5,
    alpha_tversky: float = 0.7) -> Dict[str, Any]:
    """
    对分割结果进行评估，计算各项指标，并记录日志。
    日志记录器 (logger) 应该在此函数被调用前配置好。

    参数:
        val_path (str): 包含 'gt' 和 'pred' 子文件夹的验证结果的根路径。
        gt_folder_name (str): 真实标签图像所在文件夹的名称。
        pred_folder_name (str): 预测结果图像所在文件夹的名称。
        num_classes (int): 类别总数 (包括背景)。
        palette (List[List[int]]): one-hot编码时使用的颜色调色板。
        top_n_images (int): 要报告的Dice得分最高的图片数量。
        alpha_tversky (float): Tversky指数的alpha参数。

    返回:
        Dict[str, Any]: 包含平均指标和得分最高图片列表的字典。
    """

    if palette is None:
        if num_classes == 6: # 与原始脚本的默认行为一致
             palette = [[0, 0, 0], [170, 0, 255], [0, 85, 255],
                        [170, 255, 0], [85, 255, 0], [255, 255, 127]]
        else:
            palette = [[i, i, i] for i in range(num_classes)] # 简易灰度调色板
            logger.warning(f"未提供调色板，已为 {num_classes} 类生成一个简单的灰度调色板: {palette}")

    if len(palette) != num_classes:
        logger.error(f"调色板中的颜色数量 ({len(palette)}) 与 num_classes ({num_classes}) 不匹配。")
        raise ValueError(f"调色板长度 ({len(palette)}) 必须等于 num_classes ({num_classes}).")

    gt_path = os.path.join(val_path, gt_folder_name)
    pred_path = os.path.join(val_path, pred_folder_name)

    logger.info(f"开始评估...")
    logger.info(f"真实标签路径 (GT Path): {gt_path}")
    logger.info(f"预测结果路径 (Pred Path): {pred_path}")
    logger.info(f"类别数量 (Num Classes): {num_classes} (包括背景)")
    # logger.info(f"调色板 (Palette): {palette}") # 调色板可能很长，选择性记录
    logger.info(f"Tversky Alpha: {alpha_tversky}")

    try:
        gt_files = sorted([f for f in os.listdir(gt_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        pred_files = sorted([f for f in os.listdir(pred_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError as e:
        logger.error(f"错误：找不到GT或Pred路径: {e}")
        return {"error": f"找不到路径: {e}", "mean_metrics": {}, "mean_class_metrics": {}, "top_dice_images": []}

    if not gt_files:
        logger.warning(f"在路径 '{gt_path}' 中没有找到真实标签文件。")
        return {"error": "GT文件夹为空。", "mean_metrics": {}, "mean_class_metrics": {}, "top_dice_images": []}
    if not pred_files:
        logger.warning(f"在路径 '{pred_path}' 中没有找到预测结果文件。")
        return {"error": "Pred文件夹为空。", "mean_metrics": {}, "mean_class_metrics": {}, "top_dice_images": []}

    if len(gt_files) != len(pred_files):
        logger.warning(f"真实标签文件数量 ({len(gt_files)}) 与预测结果文件数量 ({len(pred_files)}) 不一致！将尝试匹配文件名。")
        common_files = sorted(list(set(gt_files) & set(pred_files)))
        if not common_files:
            logger.error("GT和Pred文件夹中没有共同的文件名。无法进行评估。")
            return {"error": "GT和Pred文件夹中无共同文件。", "mean_metrics": {}, "mean_class_metrics": {}, "top_dice_images": []}
        gt_files = [f for f in common_files] # 确保使用对齐后的列表
        pred_files = [f for f in common_files] # 确保使用对齐后的列表 (顺序应由sorted保证)
        logger.info(f"将处理 {len(common_files)} 个共同文件。")


    all_images_metrics_data = []
    aggregated_mean_metrics = {
        "dice": [], "jaccard": [], "tversky": [], "accuracy": [],
        "precision": [], "sensitivity": [], "specificity": []
    }
    aggregated_class_metrics = {
        "dice": [], "jaccard": [], "tversky": [], "accuracy": [],
        "precision": [], "sensitivity": [], "specificity": []
    }

    for gt_file, pred_file in zip(gt_files, pred_files):
        # 在对齐后，gt_file 和 pred_file 应该相同，如果原始文件名不完全对应，
        # common_files 步骤已经处理了这个问题。
        # 如果仍需严格检查，可以保留 if gt_file != pred_file: continue
        
        try:
            gt_img_pil = Image.open(os.path.join(gt_path, gt_file))
            pred_img_pil = Image.open(os.path.join(pred_path, pred_file))

            gt_img_np = np.array(gt_img_pil).astype(np.float32)
            pred_img_np = np.array(pred_img_pil).astype(np.float32)
            
            # 使用导入的 mask_to_onehot 函数
            gt_onehot = mask_to_onehot(gt_img_np, palette)
            pred_onehot = mask_to_onehot(pred_img_np, palette)

            if gt_onehot.shape[-1] != num_classes or pred_onehot.shape[-1] != num_classes:
                logger.error(f"文件 {gt_file}: one-hot编码后的类别数与配置的 num_classes ({num_classes}) 不符。 "
                             f"GT shape: {gt_onehot.shape}, Pred shape: {pred_onehot.shape}. 跳过此文件。")
                continue

            gt_tensor = torch.from_numpy(gt_onehot).permute(2, 0, 1).unsqueeze(0).float()
            pred_tensor = torch.from_numpy(pred_onehot).permute(2, 0, 1).unsqueeze(0).float()

            mean_img_metrics, class_img_metrics = compute_metrics(pred_tensor, gt_tensor, num_classes=num_classes, alpha=alpha_tversky)
            
            all_images_metrics_data.append({
                "filename": gt_file,
                "mean_dice": mean_img_metrics["dice"],
                "mean_jaccard": mean_img_metrics["jaccard"],
                "mean_tversky": mean_img_metrics["tversky"],
                "mean_accuracy": mean_img_metrics["accuracy"],
                "mean_precision": mean_img_metrics["precision"],
                "mean_sensitivity": mean_img_metrics["sensitivity"],
                "mean_specificity": mean_img_metrics["specificity"],
                "class_dice": class_img_metrics["dice"],
            })

            for metric_name in aggregated_mean_metrics.keys():
                aggregated_mean_metrics[metric_name].append(mean_img_metrics[metric_name])
            for metric_name in aggregated_class_metrics.keys():
                aggregated_class_metrics[metric_name].append(class_img_metrics[metric_name])

            log_msg = f"文件: {gt_file} | "
            for k, v in mean_img_metrics.items(): log_msg += f"平均{k.capitalize()}: {v:.4f} | "
            logger.info(log_msg.strip(" | "))
            for metric_name, values in class_img_metrics.items():
                if values: logger.info(f"  各类别{metric_name.capitalize()}: {np.round(values, 4)}")
            logger.info("-" * 80)

        except Exception as e:
            logger.error(f"处理文件 {gt_file} 时发生错误: {e}", exc_info=True)
            continue

    if not all_images_metrics_data:
        logger.error("没有成功处理任何图像文件。无法计算平均指标或筛选最佳图像。")
        return {"error": "未能成功处理任何图像。", "mean_metrics": {}, "mean_class_metrics": {}, "top_dice_images": []}

    final_mean_metrics = {}
    logger.info("=============== 总体平均指标 ===============")
    for metric_name, values_list in aggregated_mean_metrics.items():
        if values_list:
            mean_val = np.mean(values_list)
            final_mean_metrics[metric_name] = mean_val
            logger.info(f"所有样本的平均 {metric_name.capitalize()}: {mean_val:.4f}")
        else:
            final_mean_metrics[metric_name] = 0.0
            logger.info(f"所有样本的平均 {metric_name.capitalize()}: N/A (无数据)")

    final_mean_class_metrics = {}
    for metric_name, list_of_class_metrics_for_images in aggregated_class_metrics.items():
        if list_of_class_metrics_for_images and list_of_class_metrics_for_images[0]:
            stacked_class_metrics = np.array(list_of_class_metrics_for_images)
            mean_over_images_per_class = np.mean(stacked_class_metrics, axis=0)
            final_mean_class_metrics[metric_name] = mean_over_images_per_class.tolist()
            logger.info(f"--- 各类别平均 {metric_name.capitalize()} (跳过背景类0) ---")
            for i, class_metric_val in enumerate(mean_over_images_per_class):
                logger.info(f"  类别 {i+1} 平均 {metric_name.capitalize()}: {class_metric_val:.4f}")
        else:
            final_mean_class_metrics[metric_name] = []
            logger.info(f"--- 各类别平均 {metric_name.capitalize()}: N/A (无数据) ---")
    logger.info("=" * 60)

    sorted_images_by_dice = sorted(all_images_metrics_data, key=lambda x: x['mean_dice'], reverse=True)
    num_to_report = min(top_n_images, len(sorted_images_by_dice))
    top_images_report = []

    if num_to_report > 0:
        logger.info(f"\n=============== Dice得分最高的 {num_to_report} 张图片 ===============")
        for i in range(num_to_report):
            image_data = sorted_images_by_dice[i]
            logger.info(f"Top {i+1}: 文件名: {image_data['filename']}, "
                        f"平均Dice: {image_data['mean_dice']:.4f}, "
                        f"平均Jaccard: {image_data['mean_jaccard']:.4f}")
            top_images_report.append({
                "filename": image_data['filename'],
                "mean_dice": image_data['mean_dice'],
                "mean_jaccard": image_data['mean_jaccard'],
                "class_dice": image_data['class_dice']
            })
    else:
        logger.info("没有图像数据可用于排名。")
        
    logger.info("评估完成。")
    return {
        "mean_metrics": final_mean_metrics,
        "mean_class_metrics": final_mean_class_metrics,
        "top_dice_images": top_images_report
    }

# =============== 主程序入口 ===============
if __name__ == '__main__':
    # --- 日志配置 ---
    # 获取根logger或特定名称的logger
    # logger 实例已在模块级别通过 logging.getLogger(__name__) 获取
    logger.setLevel(logging.INFO) # 为logger设置最低日志级别

    # 清除已存在的handlers，防止在重复运行脚本（例如在IPython中）时重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建日志格式器
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. 控制台 Handler
    console_h = logging.StreamHandler()
    console_h.setLevel(logging.INFO) # 控制台handler的级别
    console_h.setFormatter(log_formatter)
    logger.addHandler(console_h)

    # 2. 文件 Handler
    log_output_directory = "./evaluation_logs"  # <<< 指定日志文件存放的文件夹路径
    log_filename = "evaluation_results.log"    # <<< 指定日志文件名

    # 创建日志输出文件夹（如果不存在）
    try:
        os.makedirs(log_output_directory, exist_ok=True)
        log_file_path = os.path.join(log_output_directory, log_filename)

        file_h = logging.FileHandler(log_file_path, mode='a') # 'a' 表示追加模式, 'w' 表示覆盖模式
        file_h.setLevel(logging.INFO) # 文件handler的级别
        file_h.setFormatter(log_formatter)
        logger.addHandler(file_h)
        
        logger.info(f"日志记录已配置。同时输出到控制台和文件: {log_file_path}")
    except Exception as e:
        logger.error(f"配置日志文件处理器时发生错误: {e}", exc_info=True)
        logger.info("将仅使用控制台日志记录。")
    # --- 日志配置结束 ---


    # --- 配置评估参数 ---
    # !!! 用户需要修改以下路径和参数以匹配其实际数据和需求 !!!
    validation_data_path = r'./results'  # 包含 'gt' 和 'pred' 子文件夹的根路径
                                          # 例如: './my_experiment/validation_output'
    gt_subfolder = 'gt'                   # 真实标签图像所在文件夹的名称
    pred_subfolder = 'pred_multitask'     # 预测结果图像所在文件夹的名称
    
    # 示例调色板 (请根据您的数据集调整)
    # 6个类别 (包括背景)
    example_palette = [
        [0, 0, 0],      # 背景
        [170, 0, 255],  # 类别 1
        [0, 85, 255],   # 类别 2
        [170, 255, 0],  # 类别 3
        [85, 255, 0],   # 类别 4
        [255, 255, 127] # 类别 5
    ]
    num_actual_classes = len(example_palette) # 类别总数，包括背景

    logger.info("开始执行评估脚本...")
    logger.info(f"请确保路径 '{os.path.join(validation_data_path, gt_subfolder)}' 和 "
                f"'{os.path.join(validation_data_path, pred_subfolder)}' 中包含有效的图像数据。")

    # 调用评估函数
    results = evaluate_segmentation_performance(
        val_path=validation_data_path,
        gt_folder_name=gt_subfolder,
        pred_folder_name=pred_subfolder,
        num_classes=num_actual_classes,
        palette=example_palette,
        top_n_images=5,         # 报告Dice得分最高的5张图片
        alpha_tversky=0.7       # Tversky指数的alpha值
    )

    # 处理返回结果 (可选)
    if results:
        if "error" in results and results["error"]:
            logger.error(f"评估函数返回错误: {results['error']}")
        else:
            logger.info("\n=============== evaluate_segmentation_performance 函数返回的评估结果摘要 ===============")
            if results.get("mean_metrics"):
                 logger.info(f"总体平均指标: {results['mean_metrics']}")
            # logger.info(f"各类别平均指标: {results.get('mean_class_metrics')}") # 可能较长
            if results.get("top_dice_images"):
                logger.info(f"Top Dice 图片: {results['top_dice_images']}")
    else:
        logger.warning("评估函数没有返回任何结果。")

    logger.info("评估脚本执行完毕。")