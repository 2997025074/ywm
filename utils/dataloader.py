import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class ABIDEWaveletDataset(Dataset):
    """ABIDE小波特征数据集加载器"""

    def __init__(self, h5_file_path, label_csv_path, transform=None):
        """
        Args:
            h5_file_path: HDF5特征文件路径
            label_csv_path: 标签CSV文件路径
            transform: 数据变换
        """
        self.h5_file_path = h5_file_path
        self.transform = transform

        # 加载标签
        self.labels_df = pd.read_csv(label_csv_path)
        self.subject_ids = self.labels_df['subject_id'].tolist()

        # 验证HDF5文件中的受试者
        with h5py.File(h5_file_path, 'r') as f:
            h5_subjects = list(f.keys())

        # 取交集
        self.valid_subjects = [sid for sid in self.subject_ids if sid in h5_subjects]
        print(f"有效受试者数量: {len(self.valid_subjects)}")

    def __len__(self):
        return len(self.valid_subjects)

    def __getitem__(self, idx):
        subject_id = self.valid_subjects[idx]

        # 从HDF5加载特征
        with h5py.File(self.h5_file_path, 'r') as f:
            features = f[subject_id]['features'][:]  # [116, 64, 72]

        # 从DataFrame获取标签
        label_row = self.labels_df[self.labels_df['subject_id'] == subject_id].iloc[0]
        label = label_row['label']

        # 转换为张量
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            features = self.transform(features)

        return features, label

    def get_class_weights(self):
        """计算类别权重（用于处理不平衡数据）"""
        labels = self.labels_df[self.labels_df['subject_id'].isin(self.valid_subjects)]['label'].values
        class_counts = np.bincount(labels)
        total = len(labels)
        class_weights = total / (len(class_counts) * class_counts)
        return torch.tensor(class_weights, dtype=torch.float32)


def create_data_loaders(h5_path, label_path, batch_size=8, train_ratio=0.8,
                        shuffle=True, num_workers=0):
    """创建训练和验证数据加载器"""

    # 完整数据集
    full_dataset = ABIDEWaveletDataset(h5_path, label_path)

    # 分割训练集和验证集
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
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

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    return train_loader, val_loader, full_dataset.get_class_weights()


# 测试代码
if __name__ == "__main__":
    # 测试数据加载器
    h5_path = "C:/Users/29970/Desktop/BrainNetworkTransformer-main/data/useful_wavelet_features.h5"
    label_path = "C:/Users/29970/Desktop/BrainNetworkTransformer-main/data/labels.csv"

    if os.path.exists(h5_path) and os.path.exists(label_path):
        train_loader, val_loader, class_weights = create_data_loaders(
            h5_path, label_path, batch_size=4
        )

        # 测试一个批次
        for features, labels in train_loader:
            print(f"特征形状: {features.shape}")  # [4, 116, 64, 72]
            print(f"标签形状: {labels.shape}")  # [4]
            print(f"类别权重: {class_weights}")
            break
    else:
        print("测试文件不存在，请先运行cwt-processor.py生成特征文件")