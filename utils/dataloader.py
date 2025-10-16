"""创建训练、验证和测试数据加载器（从config读取比例，自定义数据集实现）"""
import torch
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset


class WaveletDataset(Dataset):
    """替代ABIDEWaveletDataset的自定义数据集类
    功能：加载h5格式的小波特征和对应的标签
    """
    def __init__(self, h5_path, label_path):
        # 加载特征数据（h5文件）
        self.h5_data = h5py.File(h5_path, 'r')
        self.sample_ids = list(self.h5_data.keys())  # 样本ID列表（与标签文件对应）

        # 加载标签数据（csv文件）
        label_df = pd.read_csv(label_path)
        # 假设标签文件包含'SUB_ID'（样本ID）和'DX_GROUP'（标签：1/0）
        self.label_dict = {
            f"{int(row['subject_id']):07d}": row['label'] - 1  # 统一ID格式为7位，标签转为0/1
            for _, row in label_df.iterrows()
            if not pd.isna(row['label'])
        }

        # 过滤无效样本（无标签或特征的样本）
        self.valid_ids = [
            sid for sid in self.sample_ids
            if sid in self.label_dict and self.h5_data[sid].shape[0] > 0
        ]
        print(f"加载数据集：有效样本数 {len(self.valid_ids)}/{len(self.sample_ids)}")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        sid = self.valid_ids[idx]
        # 读取特征（假设h5中每个样本存储为[num_rois, num_scales, feat_dim]）
        features = self.h5_data[sid][()]
        # 读取标签（转为0/1）
        label = self.label_dict[sid]
        return torch.FloatTensor(features), torch.LongTensor([label])[0]

    def get_class_weights(self):
        """计算类别权重（用于不平衡数据集）"""
        labels = [self.label_dict[sid] for sid in self.valid_ids]
        class_counts = np.bincount(labels)
        total = len(labels)
        # 权重公式：total / (num_classes * class_counts)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)


def create_data_loaders(config, shuffle=True, num_workers=0):
    """创建训练、验证和测试数据加载器（从config读取比例）

    Args:
        config: 配置字典（包含数据路径和比例）
        shuffle: 是否打乱训练集
        num_workers: 数据加载线程数
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    # 从config获取路径和比例
    h5_path = config['data']['h5_path']
    label_path = config['data']['label_path']
    batch_size = config['training']['batch_size']

    # 从config读取比例（默认train:0.8, val:0.1, test:0.1）
    train_ratio = config['training'].get('train_ratio', 0.7)
    val_ratio = config['training'].get('val_ratio', 0.15)
    test_ratio = config['training'].get('test_ratio', 0.15)

    # 验证比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"数据集比例总和必须为1，当前为{total_ratio:.2f}")

    # 创建完整数据集
    full_dataset = WaveletDataset(h5_path, label_path)
    dataset_size = len(full_dataset)

    # 计算各数据集大小
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size  # 确保总和正确

    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"数据集分割：")
    print(f"训练集: {len(train_dataset)} 样本 ({train_ratio:.0%})")
    print(f"验证集: {len(val_dataset)} 样本 ({val_ratio:.0%})")
    print(f"测试集: {len(test_dataset)} 样本 ({test_ratio:.0%})")

    return train_loader, val_loader, test_loader, full_dataset.get_class_weights()