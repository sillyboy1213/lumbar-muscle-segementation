import os, shutil
from sklearn.model_selection import KFold
import random

# 按K折交叉验证划分数据集
#输入为原始数据集路径，保存路径，以及K折数
#输出为K个训练集和验证集，每个训练集和验证集分别有train.txt和val.txt文件
def dataset_kfold(dataset_dir, save_path,fold=5):
    data_list = os.listdir(dataset_dir)

    # kf = KFold(5, False, 12345)  # 使用5折交叉验证
    kf = KFold(n_splits=fold, shuffle=False)

    for i, (tr, val) in enumerate(kf.split(data_list), 1):
        print(len(tr), len(val))
        if os.path.exists(os.path.join(save_path, 'train{}.txt'.format(i))):
            # 若该目录已存在，则先删除，用来清空数据
            print('清空原始数据中...')
            os.remove(os.path.join(save_path, 'train{}.txt'.format(i)))
            os.remove(os.path.join(save_path, 'val{}.txt'.format(i)))
            print('原始数据已清空。')

        for item in tr:
            file_name = data_list[item]
            with open(os.path.join(save_path, 'train{}.txt'.format(i)), 'a') as f:
                f.write(file_name)
                f.write('\n')

        for item in val:
            file_name = data_list[item]
            with open(os.path.join(save_path, 'val{}.txt'.format(i)), 'a') as f:
                f.write(file_name)
                f.write('\n')

#从Images里随机复制8张图像作为测试集
def get_test_set(dataset_dir, save_path, num=8):
    img_dir = os.path.join(dataset_dir, 'Images')
    # 确保 Images 目录存在
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"错误：{img_dir} 目录不存在！")
    # 获取所有图片文件名
    data_list = os.listdir(img_dir)
    # 随机选择 num 张图片（避免超出范围）
    test_set = random.sample(data_list, min(num, len(data_list)))

    # 确保 save_path 存在
    os.makedirs(save_path, exist_ok=True)

    # 定义 test.txt 文件路径
    test_file = os.path.join(save_path, 'test.txt')

    # 删除旧的 test.txt 文件（如果存在）
    if os.path.exists(test_file):
        os.remove(test_file)

    # 写入新的测试集文件
    with open(test_file, 'w') as f:
        f.writelines(f"{item}\n" for item in test_set)

    print(f"已保存 {len(test_set)} 张测试图片到 {test_file}")
    
        

if __name__ == '__main__':
    dataset_kfold('./data/Labels',
                  './Dataset')
    get_test_set('./data',
                 './Dataset')
    