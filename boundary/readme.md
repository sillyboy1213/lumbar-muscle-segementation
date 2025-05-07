# 边界框检测模块 (Boundary Detection Module)

此目录包含用于从分割掩码预测边界框的 PyTorch 代码，即对应沈学长的多任务模块中的区域检测模块。

## 概述

该模块的核心功能是训练一个轻量化的目标检测模型，该模型接收分割掩码（通常是 one-hot 编码格式）作为输入，并为每个分割区域（类别）输出一个边界框坐标。每一类别的边界框预测由一个独立的 `RegionDetectionModule` 实例处理。

主要包含两种模式：

1.  **单目标检测**: 针对只有一个前景类别的情况进行训练和预测。相关代码通常包含 `single` 字样或在多目标代码中被注释掉。这部分主要是用于验证单目标检测模型的性能。
2.  **多目标检测**: 针对有多个前景类别（例如，不同的器官）的情况进行训练和预测。这是当前代码库中的主要模式。

## 核心模型 (`region.py`)

*   `RegionDetectionModule`: 一个基础的卷积神经网络 (CNN)，接收单通道掩码（代表一个类别）并预测其边界框 (4个坐标)。
*   `BoundarySensitiveAllocationModule`: 一个包装器模块，它为每个目标类别（不包括背景）创建一个 `RegionDetectionModule` 实例。它接收多通道的 one-hot 编码掩码，并将每个通道传递给相应的检测器以获得所有类别的边界框。

## 文件说明

*   `region.py`: 定义上述核心的 PyTorch 模型 (`RegionDetectionModule`, `BoundarySensitiveAllocationModule`)。
*   `datasetgen.py`: 定义 `SegmentationToBboxDataset` 类，用于加载分割掩码图像，将其转换为 one-hot 编码张量，并计算相应的真实边界框。这是**多目标**版本的数据加载器。 (文件中注释掉的代码块是**单目标**版本的数据加载逻辑，与 `singledataset.py` 类似)。
*   `singledataset.py`: (在之前的对话中提到，可能存在或已被整合/注释) 定义用于**单目标**检测的数据集加载器。
*   `train.py`: 包含模型训练脚本。当前活动代码用于训练**多目标**模型 (`BoundarySensitiveAllocationModule` with `num_classes > 1`)。文件中注释掉的代码块是用于训练**单目标**模型的配置 (与 `singletrain.py` 可能的功能类似)。
*   `singletrain.py`: (在之前的对话中提到，可能存在或已被整合/注释) 用于训练**单目标**检测模型的脚本。
*   `validate.py`: 加载训练好的**多目标**模型 (`region_detection_5_classes.pth`)，对数据集中的样本进行预测，并使用 `matplotlib` 可视化预测边界框与真实边界框的对比。
*   `merge.py`: 整合预训练的分割模型 (如 U-Net) 和此处的边界框检测模型。它对单个图像执行分割，然后使用分割结果（转换后的 one-hot 掩码）进行边界框预测，并可视化整个流程。


## 文件夹说明

*   `data/`: 包含训练、验证或测试所需的数据集文件，包括标签和图像。
*   `label_png/`: 存放 PNG 格式的标签（分割掩码）图像。被多目标版本的 `datasetgen.py` 和 `train.py` 使用。注意这里由于沈学长的代码标注存在重叠，所以类别数从8变成了6（包含背景）。
*   `R3/`: 存放掩码或其他数据。被**单目标**版本的训练代码引用,用于测试单目标检测模型的性能。
*   `Images/`: 包含原始图像。
*   `__pycache__/`: Python 自动生成的字节码缓存目录。

## 使用

1.  **准备数据**: 确保 `label_png` (用于多目标) 或 `R3` (用于单目标) 目录包含正确的分割掩码图像。
2.  **训练**: 运行 `train.py` 来训练模型。根据需要调整脚本中的参数（如 `batch_size`, `num_epochs`, `learning_rate`, `num_classes`, 数据目录）。训练好的多目标模型默认保存为 `region_detection_5_classes.pth`。
3.  **验证/可视化**: 运行 `validate.py` 来加载训练好的模型并查看其在数据集上的预测效果。
4.  **集成测试**: 运行 `merge.py` (可能需要配置分割模型路径) 来查看分割和边界框检测的联合效果。