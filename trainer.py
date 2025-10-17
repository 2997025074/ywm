import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
import time


class BrainNetworkTrainer:
    """脑网络分类训练器"""

    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.criterion = None
        # 使用更小的学习率
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5, verbose=True
        )

        # 训练历史
        self.train_history = {
            'loss': [], 'accuracy': [], 'auc': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': [],
            'test_loss': [], 'test_accuracy': [], 'test_auc': []
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
        """训练一个epoch（增加数值稳定性检查）"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []

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

            # 检查预测是否包含NaN
            if torch.isnan(predictions).any():
                print(f"警告: 第{epoch_idx}轮第{batch_idx}批次的预测包含NaN，跳过该批次")
                continue

            # 计算损失
            loss, loss_components = self.criterion(predictions, labels, model_outputs)

            # 检查损失是否包含NaN
            if torch.isnan(loss):
                print(f"警告: 第{epoch_idx}轮第{batch_idx}批次的损失为NaN，跳过该批次")
                continue

            total_loss += loss.item()

            # 反向传播
            loss.backward()

            # 梯度裁剪（更严格的裁剪）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            # 检查梯度是否包含NaN
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"警告: 参数 {name} 的梯度包含NaN")
                    has_nan_grad = True
                    break

            if not has_nan_grad:
                self.optimizer.step()
            else:
                print(f"警告: 检测到NaN梯度，跳过参数更新")

            # 收集预测、标签和概率
            all_predictions.append(predictions.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # 计算概率（用于AUC），增加数值稳定性
            with torch.no_grad():
                probabilities = torch.softmax(predictions, dim=1)

                # 检查概率是否包含NaN
                if torch.isnan(probabilities).any():
                    print(f"警告: 概率包含NaN，使用均匀分布替代")
                    probabilities = torch.ones_like(probabilities) / probabilities.shape[1]

                all_probabilities.append(probabilities.detach().cpu())

            if progress_bar and batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'CLS': f'{loss_components["cls_loss"]:.4f}'
                })

        # 计算epoch指标
        from utils.metrics import ClassificationMetrics

        # 安全地获取类别数
        try:
            num_classes = self.model.num_classes
        except AttributeError:
            num_classes = 2

        metrics_calc = ClassificationMetrics(num_classes=num_classes)

        if len(all_predictions) == 0:
            print(f"警告: 第{epoch_idx}轮没有有效批次，使用默认指标")
            return 1.0, {'accuracy': 0.5, 'auc': 0.5}, {'cls_loss': 1.0, 'div_loss': 0.0, 'sparse_loss': 0.0,
                                                        'consistency_loss': 0.0, 'smooth_loss': 0.0}

        # 合并所有批次的数据
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_probabilities = torch.cat(all_probabilities)

        # 更新指标（传入概率）
        metrics_calc.update(all_predictions, all_labels, all_probabilities)
        metrics = metrics_calc.compute()

        epoch_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 1.0
        epoch_accuracy = metrics['accuracy']
        epoch_auc = metrics.get('auc', 0.5)

        # 更新历史
        self.train_history['loss'].append(epoch_loss)
        self.train_history['accuracy'].append(epoch_accuracy)
        self.train_history['auc'].append(epoch_auc)

        return epoch_loss, metrics, loss_components

    def validate(self, dataloader, mode='val'):
        """验证/测试模型（增加数值稳定性）"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_val_loss = 0
        valid_batches = 0

        with torch.no_grad():
            for features, labels in tqdm(dataloader, desc=f'{mode.capitalize()}idation'):
                features = features.to(self.device)
                labels = labels.to(self.device)

                predictions, model_outputs = self.model(features)

                # 跳过包含NaN的预测
                if torch.isnan(predictions).any():
                    print(f"警告: {mode}集中检测到NaN预测，跳过该批次")
                    continue

                # 计算验证损失
                val_loss, _ = self.criterion(predictions, labels, model_outputs)

                if not torch.isnan(val_loss):
                    total_val_loss += val_loss.item()
                    valid_batches += 1

                # 收集数据
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

                # 计算概率，增加稳定性
                probabilities = torch.softmax(predictions, dim=1)
                if torch.isnan(probabilities).any():
                    probabilities = torch.ones_like(probabilities) / probabilities.shape[1]

                all_probabilities.append(probabilities.cpu())

        # 如果没有有效批次，返回默认值
        if valid_batches == 0:
            print(f"警告: {mode}集没有有效批次")
            return 1.0, {'accuracy': 0.5, 'auc': 0.5}

        # 计算验证指标
        from utils.metrics import ClassificationMetrics

        try:
            num_classes = self.model.num_classes
        except AttributeError:
            num_classes = 2

        metrics_calc = ClassificationMetrics(num_classes=num_classes)

        # 合并数据
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_probabilities = torch.cat(all_probabilities)

        # 更新指标（传入概率）
        metrics_calc.update(all_predictions, all_labels, all_probabilities)
        metrics = metrics_calc.compute()

        avg_val_loss = total_val_loss / valid_batches

        # 更新历史
        if mode == 'val':
            self.train_history['val_loss'].append(avg_val_loss)
            self.train_history['val_accuracy'].append(metrics['accuracy'])
            self.train_history['val_auc'].append(metrics.get('auc', 0.5))
            # 更新学习率（仅在验证集上）
            self.scheduler.step(avg_val_loss)
        elif mode == 'test':
            self.train_history['test_loss'].append(avg_val_loss)
            self.train_history['test_accuracy'].append(metrics['accuracy'])
            self.train_history['test_auc'].append(metrics.get('auc', 0.5))

        return avg_val_loss, metrics

    # 其余方法保持不变...
    def save_checkpoint(self, epoch, path):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history
        }
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
    """完整训练流程（使用更保守的参数）"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建数据加载器
    from utils.dataloader import create_data_loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        config=config,
        shuffle=True,
        num_workers=config['training'].get('num_workers', 0)
    )

    # 创建模型
    from model.classifier import BrainNetworkClassifier
    model = BrainNetworkClassifier(
        num_rois=config['model']['num_rois'],
        num_scales=config['model']['num_scales'],
        num_bands=config['model']['num_bands'],
        feat_dim=config['model']['feat_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes']
    )
    model.enable_attention_saving(True)

    # 使用更小的学习率
    learning_rate = config['training'].get('learning_rate', 1e-4)
    # 如果之前有问题，自动降低学习率
    learning_rate = min(learning_rate, 1e-4)

    # 创建训练器
    trainer = BrainNetworkTrainer(
        model, device,
        learning_rate=learning_rate,
        weight_decay=config['training'].get('weight_decay', 1e-4)
    )
    trainer.setup_training(class_weights)

    # 确保result相关目录存在
    os.makedirs(os.path.dirname(config['paths']['best_model']), exist_ok=True)
    os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)

    # 训练循环
    best_val_auc = 0
    num_epochs = config['training']['num_epochs']
    save_interval = config['training']['save_interval']

    print("开始训练...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        # 训练
        train_loss, train_metrics, loss_components = trainer.train_epoch(
            train_loader, epoch
        )

        # 验证
        val_loss, val_metrics = trainer.validate(val_loader, mode='val')

        # 测试
        test_loss, test_metrics = trainer.validate(test_loader, mode='test')

        epoch_time = time.time() - epoch_start_time

        # 打印详细结果
        print(f"Epoch[{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.3f} | Train Accuracy: {train_metrics['accuracy'] * 100:.3f}% | "
              f"Val Loss: {val_loss:.3f} | Val Accuracy: {val_metrics['accuracy'] * 100:.3f}% | Val AUC: {val_metrics.get('auc', 0):.4f} | "
              f"Test Loss: {test_loss:.3f} | Test Accuracy: {test_metrics['accuracy'] * 100:.3f}% | Test AUC: {test_metrics.get('auc', 0):.4f} | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型
        current_auc = val_metrics.get('auc', 0)
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            trainer.save_checkpoint(epoch, config['paths']['best_model'])
            print(f"✨ 新的最佳模型! Val AUC: {best_val_auc:.4f}")

        # 定期保存检查点
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'],
                f"checkpoint_epoch_{epoch + 1}.pth"
            )
            trainer.save_checkpoint(epoch, checkpoint_path)

    # 保存训练历史
    with open(config['paths']['train_history'], 'w') as f:
        serializable_history = {
            k: [v.item() if isinstance(v, np.ndarray) else v for v in vals]
            for k, vals in trainer.train_history.items()
        }
        json.dump(serializable_history, f, indent=2)
    print(f"训练历史已保存到: {config['paths']['train_history']}")

    print(f"\n🎉 训练完成! 最佳验证AUC: {best_val_auc:.4f}")

    return trainer