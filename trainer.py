import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
import matplotlib.pyplot as plt  # 新增：用于可视化训练曲线
from datetime import datetime  # 新增：记录时间


class BrainNetworkTrainer:
    """脑网络分类训练器（优化运行时输出与结果保存）"""

    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-4, log_dir="logs"):
        self.model = model.to(device)
        self.device = device
        self.criterion = None  # 多任务损失函数（在setup_training中初始化）
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5, verbose=True
        )

        # 训练历史（新增更多指标存储）
        self.train_history = {
            'loss': [], 'accuracy': [], 'auc': [], 'precision': [], 'recall': [], 'f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_auc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [],
            'lr': []  # 新增：记录学习率变化
        }

        # 新增：日志与结果保存目录
        self.log_dir = log_dir
        self.checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 用于区分多次实验

    def setup_training(self, class_weights=None):
        """设置训练组件（初始化损失函数）"""
        from utils.multitask_loss import MultiTaskLoss
        self.criterion = MultiTaskLoss()  # 多任务损失（分类+正则化）

        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
            print(f"使用类别权重: {self.class_weights.cpu().numpy()}")  # 新增：打印类别权重
        else:
            self.class_weights = None

    def train_epoch(self, dataloader, epoch_idx, progress_bar=True):
        """训练一个epoch（增强输出信息）"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []  # 新增：存储概率用于计算AUC等指标

        # 新增：获取当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.train_history['lr'].append(current_lr)

        if progress_bar:
            pbar = tqdm(dataloader, desc=f'Epoch {epoch_idx + 1} (LR: {current_lr:.6f})')
        else:
            pbar = dataloader

        for batch_idx, (features, labels) in enumerate(pbar):
            # 确保输入维度正确 [B, N, S, F]
            if features.dim() != 4:
                raise ValueError(f"输入特征维度错误: 预期4维[B,N,S,F]，实际{features.dim()}维")
            features = features.to(self.device)
            labels = labels.to(self.device).long()  # 确保标签为长整型

            self.optimizer.zero_grad()

            # 前向传播
            predictions, model_outputs = self.model(features)

            # 计算损失（多任务损失）
            batch_loss, loss_components = self.criterion(predictions, labels, model_outputs)
            total_loss += batch_loss.item()

            # 反向传播
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪防爆炸
            self.optimizer.step()

            # 收集预测结果（用于计算 epoch 指标）
            all_predictions.append(predictions.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_probabilities.append(F.softmax(predictions, dim=1).detach().cpu())  # 新增：存储概率

            # 批次级日志（每10个batch更新一次进度条）
            if progress_bar and batch_idx % 10 == 0:
                pbar.set_postfix({
                    'Batch Loss': f'{batch_loss.item():.4f}',
                    'CLS Loss': f'{loss_components["cls_loss"]:.4f}',
                    'Sparse Loss': f'{loss_components["sparse_loss"]:.4f}'
                })

        # 计算epoch级指标
        from utils.metrics import ClassificationMetrics
        metrics_calc = ClassificationMetrics()
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_probabilities = torch.cat(all_probabilities)
        metrics_calc.update(all_predictions, all_labels, all_probabilities)  # 新增：传入概率
        metrics = metrics_calc.compute()

        # 记录训练历史
        avg_train_loss = total_loss / len(dataloader)
        self.train_history['loss'].append(avg_train_loss)
        self.train_history['accuracy'].append(metrics['accuracy'])
        self.train_history['auc'].append(metrics['auc'])
        self.train_history['precision'].append(metrics['precision'])
        self.train_history['recall'].append(metrics['recall'])
        self.train_history['f1'].append(metrics['f1'])

        # 打印epoch训练结果（详细版）
        print(f"\n训练集结果 - 损失: {avg_train_loss:.4f} | "
              f"准确率: {metrics['accuracy']:.4f} | "
              f"AUC: {metrics['auc']:.4f} | "
              f"F1: {metrics['f1']:.4f}")

        return avg_train_loss, metrics, loss_components

    def validate(self, dataloader):
        """验证模型（增强输出与指标记录）"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []  # 新增：存储概率
        total_val_loss = 0.0

        with torch.no_grad():  # 关闭梯度计算
            for features, labels in tqdm(dataloader, desc='验证中'):
                features = features.to(self.device)
                labels = labels.to(self.device).long()

                predictions, model_outputs = self.model(features)

                # 计算验证损失
                val_loss, _ = self.criterion(predictions, labels, model_outputs)
                total_val_loss += val_loss.item()

                # 收集结果
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
                all_probabilities.append(F.softmax(predictions, dim=1).cpu())

        # 计算验证指标
        from utils.metrics import ClassificationMetrics
        metrics_calc = ClassificationMetrics()
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_probabilities = torch.cat(all_probabilities)
        metrics_calc.update(all_predictions, all_labels, all_probabilities)
        metrics = metrics_calc.compute()

        # 记录验证历史
        avg_val_loss = total_val_loss / len(dataloader)
        self.train_history['val_loss'].append(avg_val_loss)
        self.train_history['val_accuracy'].append(metrics['accuracy'])
        self.train_history['val_auc'].append(metrics['auc'])
        self.train_history['val_precision'].append(metrics['precision'])
        self.train_history['val_recall'].append(metrics['recall'])
        self.train_history['val_f1'].append(metrics['f1'])

        # 打印验证结果
        print(f"验证集结果 - 损失: {avg_val_loss:.4f} | "
              f"准确率: {metrics['accuracy']:.4f} | "
              f"AUC: {metrics['auc']:.4f} | "
              f"F1: {metrics['f1']:.4f}")

        # 更新学习率调度器
        self.scheduler.step(avg_val_loss)

        return avg_val_loss, metrics

    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点（新增最佳模型标记与元数据）"""
        # 检查点文件名（区分普通检查点和最佳模型）
        if is_best:
            checkpoint_name = f"best_model_epoch_{epoch + 1}.pth"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        # 检查点内容（新增元数据）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'best_auc': max(self.train_history['val_auc']) if self.train_history['val_auc'] else 0,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, path):
        """加载检查点（新增完整性校验）"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"检查点文件不存在: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_history = checkpoint['train_history']

        print(f"加载检查点成功: {path} | "
              f"训练至epoch {checkpoint['epoch'] + 1} | "
              f"最佳验证AUC: {checkpoint['best_auc']:.4f}")
        return checkpoint['epoch']

    def save_training_results(self):
        """新增：保存训练历史与可视化结果"""
        # 1. 保存历史指标为JSON
        history_path = os.path.join(self.log_dir, f"train_history_{self.start_time}.json")
        with open(history_path, 'w') as f:
            # 将numpy数组转为列表（JSON序列化）
            serializable_history = {k: [v.item() if isinstance(v, np.ndarray) else v for v in val]
                                    for k, val in self.train_history.items()}
            json.dump(serializable_history, f, indent=2)
        print(f"训练历史已保存: {history_path}")

        # 2. 绘制训练曲线（损失+准确率）
        self._plot_training_curves()

    def _plot_training_curves(self):
        """新增：绘制并保存训练曲线"""
        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['loss'], label='训练损失')
        plt.plot(self.train_history['val_loss'], label='验证损失')
        plt.title('损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_history['accuracy'], label='训练准确率')
        plt.plot(self.train_history['val_accuracy'], label='验证准确率')
        plt.title('准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # 保存图片
        curve_path = os.path.join(self.log_dir, f"training_curves_{self.start_time}.png")
        plt.tight_layout()
        plt.savefig(curve_path, dpi=300)
        plt.close()
        print(f"训练曲线已保存: {curve_path}")


def train_complete_model(config):
    """完整训练流程（整合新功能）"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device} | CUDA可用: {torch.cuda.is_available()}")

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
    model.enable_attention_saving(True)  # 启用注意力保存（可解释性）
    print(f"模型初始化完成 | 脑区数: {config['num_rois']} | 频带数: {config['num_bands']}")

    # 创建训练器（指定日志目录）
    log_dir = config.get('log_dir', 'experiment_logs')
    trainer = BrainNetworkTrainer(
        model, device,
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4),
        log_dir=log_dir
    )
    trainer.setup_training(class_weights)

    # 加载预训练检查点（如果指定）
    start_epoch = 0
    if config.get('checkpoint'):
        start_epoch = trainer.load_checkpoint(config['checkpoint'])

    # 训练循环
    best_val_auc = max(trainer.train_history['val_auc']) if trainer.train_history['val_auc'] else 0
    print(
        f"开始训练 | 总epochs: {config['num_epochs']} | 起始epoch: {start_epoch + 1} | 初始最佳AUC: {best_val_auc:.4f}")

    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        # 训练
        train_loss, train_metrics, loss_components = trainer.train_epoch(
            train_loader, epoch
        )

        # 验证
        val_loss, val_metrics = trainer.validate(val_loader)

        # 打印损失组件详情
        print("损失组件: " + ", ".join([f"{k}: {v:.4f}" for k, v in loss_components.items()]))

        # 保存最佳模型（以验证AUC为指标）
        current_auc = val_metrics['auc']
        if current_auc > best_val_auc:
            best_val_auc = current_auc
            trainer.save_checkpoint(epoch, is_best=True)  # 标记为最佳模型

        # 定期保存检查点
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            trainer.save_checkpoint(epoch, is_best=False)

    # 训练结束后保存最终结果
    trainer.save_training_results()
    print(f"\n训练完成! 最佳验证AUC: {best_val_auc:.4f} | 结果保存至: {trainer.log_dir}")
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
        'save_interval': 20,
        'log_dir': "experiment_logs"  # 新增：日志目录
    }

    # 检查数据文件
    if os.path.exists(config['h5_path']) and os.path.exists(config['label_path']):
        print("开始训练完整模型...")
        trainer = train_complete_model(config)
    else:
        print("数据文件不存在，请先运行cwt-processor.py生成特征文件")