from typing import List, Tuple
from omegaconf import DictConfig, open_dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ABIDE_Dataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        x_real = torch.FloatTensor(x_data.real)
        x_imag = torch.FloatTensor(x_data.imag)
        self.x_data = torch.stack([x_real, x_imag], dim=-1)
        # 关键修改：直接使用原始整数标签，类型为long
        self.y_data = torch.tensor(y_data, dtype=torch.long)  # 形状为[样本数]，如[0,1,0,...]
        self.len = self.y_data.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[index], self.y_data[index]  # 返回1D标签

    def __len__(self) -> int:
        return self.len


def get_dataloader(
        cfg: DictConfig,
        samples: np.ndarray,
        labels: np.ndarray,
        shuffle: bool = True
) -> DataLoader:
    dataset = ABIDE_Dataset(samples, labels)
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers
    )


def init_dataloader(
        cfg: DictConfig,
        train_idx: List[int],
        valid_idx: List[int],
        train_valid_samples: np.ndarray,
        train_valid_labels: np.ndarray,
        test_samples: np.ndarray,
        test_labels: np.ndarray,
) -> List[DataLoader]:
    train_loader = get_dataloader(
        cfg=cfg,
        samples=train_valid_samples[train_idx],
        labels=train_valid_labels[train_idx],
        shuffle=True
    )
    val_loader = get_dataloader(
        cfg=cfg,
        samples=train_valid_samples[valid_idx],
        labels=train_valid_labels[valid_idx],
        shuffle=False
    )
    test_loader = get_dataloader(
        cfg=cfg,
        samples=test_samples,
        labels=test_labels,
        shuffle=False
    )
    with open_dict(cfg):
        cfg.steps_per_epoch = len(train_loader)
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.train_epochs
    return [train_loader, val_loader, test_loader]


def continuous_mixup_data(
        *xs: torch.Tensor,
        y: torch.Tensor,  # 此时y是1D整数标签（torch.long）
        alpha: float = 1.0,
        device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = y.size(0)
    index = torch.randperm(batch_size).to(device)

    # 混合输入特征
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    # 混合标签（对1D整数标签，按比例生成浮点型混合标签）
    y_mixed = lam * y.float() + (1 - lam) * y[index].float()

    return *new_xs, y_mixed
