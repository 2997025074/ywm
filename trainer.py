import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json


class BrainNetworkTrainer:
    """脑网络分类训练器"""

    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = None  # 将在setup_training中设置
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )

        # 训练历史
        self.train_history = {
            'loss': [], 'accuracy': [], 'auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': []
        }

    def setup_training(self, class_weights=None):
        """设置训练组件"""
        from utils.multitask_loss import MultiTaskLoss
        self.criterion = MultiTaskLoss()

        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
        else:
            self.class_weights = None

    def train_epoch(self, dataloader, epoch_idx, progress_bar=True):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []

        if progress_bar:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch_idx + 1} Training')
        else:
            pbar = dataloader

        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # 前向传播
            predictions, model_outputs = self.model(features)

            # 计算损失
            loss, loss_components = self.criterion(predictions, labels, model_outputs)
            total_loss += loss.item()

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 收集预测和标签
            all_predictions.append(predictions.detach().cpu())
            all_labels.append(labels.detach().cpu())

            if progress_bar and batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'CLS': f'{loss_components["cls_loss"]:.4f}'
                })

        # 计算epoch指标
        from utils.metrics import ClassificationMetrics
        metrics_calc = ClassificationMetrics()
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics_calc.update(all_predictions, all_labels)
        metrics = metrics_calc.compute()

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = metrics['accuracy']
        epoch_auc = metrics.get('auc', 0.0)

        # 更新历史
        self.train_history['loss'].append(epoch_loss)
        self.train_history['accuracy'].append(epoch_accuracy)
        self.train_history['auc'].append(epoch_auc)

        return epoch_loss, metrics, loss_components

    def validate(self, dataloader):
        """验证模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_val_loss = 0

        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc='Validation'):
                features = features.to(self.device)
                labels = labels.to(self.device)

                predictions, model_outputs = self.model(features)

                # 计算验证损失
                val_loss, _ = self.criterion(predictions, labels, model_outputs)
                total_val_loss += val_loss.item()

                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        # 计算验证指标
        from utils.metrics import ClassificationMetrics
        metrics_calc = ClassificationMetrics()
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        metrics_calc.update(all_predictions, all_labels)
        metrics = metrics_calc.compute()

        avg_val_loss = total_val_loss / len(dataloader)

        # 更新历史
        self.train_history['val_loss'].append(avg_val_loss)
        self.train_history['val_accuracy'].append(metrics['accuracy'])
        self.train_history['val_auc'].append(metrics.get('auc', 0.0))

        # 更新学习率
        self.scheduler.step(avg_val_loss)

        return avg_val_loss, metrics

    def save_checkpoint(self, epoch, path):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history
        }
        # 确保保存目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']
        print(f"检查点已加载: {path}, 从epoch {checkpoint['epoch']}继续训练")
        return checkpoint['epoch']


def train_complete_model(config):
    """完整训练流程"""

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    from utils.dataloader import create_data_loaders
    train_loader, val_loader, class_weights = create_data_loaders(
        config['h5_path'],
        config['label_path'],
        batch_size=config['batch_size'],
        train_ratio=config.get('train_ratio', 0.8)
    )

    # 创建模型
    from model.classifier import BrainNetworkClassifier
    model = BrainNetworkClassifier(
        num_rois=config['num_rois'],
        num_scales=config['num_scales'],
        num_bands=config['num_bands'],
        feat_dim=config['feat_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes']
    )
    model.enable_attention_saving(True)

    # 创建训练器
    trainer = BrainNetworkTrainer(
        model, device,
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    trainer.setup_training(class_weights)

    # 确保result相关目录存在
    os.makedirs(os.path.dirname(config['best_model_path']), exist_ok=True)
    os.makedirs(config.get('checkpoint_dir', 'result/checkpoints'), exist_ok=True)

    # 训练循环
    best_val_auc = 0
    for epoch in range(config['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        # 训练
        train_loss, train_metrics, loss_components = trainer.train_epoch(
            train_loader, epoch
        )

        # 验证
        val_loss, val_metrics = trainer.validate(val_loader)

        # 打印结果
        print(f"训练损失: {train_loss:.4f}, 准确率: {train_metrics['accuracy']:.4f}")
        print(f"验证损失: {val_loss:.4f}, 准确率: {val_metrics['accuracy']:.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
        print(f"损失组件: {loss_components}")

        # 保存最佳模型
        current_auc = val_metrics.get('auc', 0)
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            trainer.save_checkpoint(epoch, config.get('best_model_path', 'result/best_model.pth'))

        # 定期保存检查点
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            checkpoint_path = os.path.join(
                config.get('checkpoint_dir', 'result/checkpoints'),
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            trainer.save_checkpoint(epoch, checkpoint_path)

    # 保存训练历史
    with open(config['train_history'], 'w') as f:
        # 转换numpy数组为列表以支持JSON序列化
        serializable_history = {
            k: [v.item() if isinstance(v, np.ndarray) else v for v in vals]
            for k, vals in trainer.train_history.items()
        }
        json.dump(serializable_history, f, indent=2)
    print(f"训练历史已保存到: {config['train_history']}")

    print(f"\n训练完成! 最佳验证AUC: {best_val_auc:.4f}")
    return trainer


# 测试代码
if __name__ == "__main__":
    # 配置参数
    config = {
        'h5_path': "data/useful_wavelet_features.h5",
        'label_path': "data/labels.csv",
        'num_rois': 116,
        'num_scales': 64,
        'num_bands': 4,
        'feat_dim': 72,
        'hidden_dim': 64,
        'num_classes': 2,
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'train_ratio': 0.8,
        'best_model_path': 'result/best_model.pth',
        'checkpoint_dir': 'result/checkpoints',
        'train_history': 'result/train_history.json',
        'save_interval': 20
    }

    # 检查数据文件是否存在
    import os
    if os.path.exists(config['h5_path']) and os.path.exists(config['label_path']):
        print("开始训练完整模型...")
        trainer = train_complete_model(config)
    else:
        print("数据文件不存在，请先运行: python main.py --mode extract_features")