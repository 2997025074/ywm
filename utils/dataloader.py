"""创建训练、验证和测试数据加载器"""
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import ABIDEWaveletDataset  # 假设存在该类

def create_data_loaders(h5_path, label_path, batch_size=8,
                        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                        shuffle=True, num_workers=0):
    """创建训练、验证和测试数据加载器"""

    # 完整数据集
    full_dataset = ABIDEWaveletDataset(h5_path, label_path)
    dataset_size = len(full_dataset)

    # 验证比例总和
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"数据集比例总和必须为1，当前为{total_ratio}")

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

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")

    return train_loader, val_loader, test_loader, full_dataset.get_class_weights()