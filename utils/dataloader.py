"""创建训练、验证和测试数据加载器（从config读取比例，自定义数据集实现）"""
import torch
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset


class WaveletDataset(Dataset):
    """替代ABIDEWaveletDataset的自定义数据集类"""

    def __init__(self, h5_path, label_path):
        # 加载特征数据（h5文件）
        self.h5_data = h5py.File(h5_path, 'r')
        self.sample_ids = list(self.h5_data.keys())

        # 加载标签数据（csv文件）
        label_df = pd.read_csv(label_path)
        self.label_dict = {}
        for _, row in label_df.iterrows():
            try:
                sub_id = f"{int(row['subject_id']):07d}"
                label = int(row['label'])
                self.label_dict[sub_id] = label
            except (ValueError, KeyError):
                continue

        # 过滤无效样本
        self.valid_ids = []
        for sid in self.sample_ids:
            if sid in self.label_dict:
                try:
                    if 'features' in self.h5_data[sid]:
                        features_data = self.h5_data[sid]['features']
                        if features_data.shape[0] > 0:
                            # 检查特征是否包含NaN或Inf
                            features_np = features_data[()]
                            if (not np.any(np.isnan(features_np)) and
                                    not np.any(np.isinf(features_np)) and
                                    np.all(np.isfinite(features_np))):
                                self.valid_ids.append(sid)
                            else:
                                print(f"警告: 样本 {sid} 包含NaN或Inf值，已跳过")
                except (KeyError, AttributeError):
                    continue

        print(f"加载数据集：有效样本数 {len(self.valid_ids)}/{len(self.sample_ids)}")

        # 检查数据统计信息
        self._check_data_statistics()

    def _check_data_statistics(self):
        """检查数据统计信息"""
        if len(self.valid_ids) == 0:
            return

        # 随机检查几个样本
        sample_checks = min(5, len(self.valid_ids))
        print("数据统计检查:")
        for i in range(sample_checks):
            sid = self.valid_ids[i]
            features = self.h5_data[sid]['features'][()]
            print(f"样本 {sid}: 形状={features.shape}, "
                  f"范围=[{features.min():.3f}, {features.max():.3f}], "
                  f"均值={features.mean():.3f}, 标准差={features.std():.3f}, "
                  f"NaN={np.isnan(features).any()}, Inf={np.isinf(features).any()}")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        sid = self.valid_ids[idx]
        features = self.h5_data[sid]['features'][()]
        label = self.label_dict[sid]

        # 最终检查
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print(f"错误: 样本 {sid} 在加载时包含NaN或Inf")
            # 用零替换有问题的值
            features = np.nan_to_num(features)

        return torch.FloatTensor(features), torch.LongTensor([label])[0]

    def get_class_weights(self):
        """计算类别权重"""
        labels = [self.label_dict[sid] for sid in self.valid_ids]
        class_counts = np.bincount(labels)
        total = len(labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)


def create_data_loaders(config, shuffle=True, num_workers=0):
    """创建训练、验证和测试数据加载器（从config读取比例）

    Args:
        config: 完整的配置字典（包含data、training等键）
        shuffle: 是否打乱训练集
        num_workers: 数据加载线程数
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    # 从config获取路径和比例
    h5_path = config['data']['h5_path']
    label_path = config['data']['label_path']
    batch_size = config['training']['batch_size']

    # 从config读取比例
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