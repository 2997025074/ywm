import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ClassificationMetrics:
    """分类评估指标计算器"""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []

    def update(self, predictions, labels):
        """
        更新指标状态
        Args:
            predictions: 模型预测 [B, num_classes]
            labels: 真实标签 [B]
        """
        # 转换为numpy
        probs = predictions.detach().cpu().numpy()
        preds = np.argmax(probs, axis=1)
        labels_np = labels.detach().cpu().numpy()

        self.all_predictions.extend(preds)
        self.all_labels.extend(labels_np)
        self.all_probabilities.extend(probs)

    def compute(self):
        """计算所有指标"""
        if len(self.all_predictions) == 0:
            return {}

        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)
        probabilities = np.array(self.all_probabilities)

        metrics = {}

        # 基础分类指标
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['f1_score'] = f1_score(labels, predictions, average='binary')
        metrics['precision'] = precision_score(labels, predictions, average='binary')
        metrics['recall'] = recall_score(labels, predictions, average='binary')

        # AUC（二分类）
        if self.num_classes == 2:
            metrics['auc'] = roc_auc_score(labels, probabilities[:, 1])
        else:
            metrics['auc'] = roc_auc_score(labels, probabilities, multi_class='ovo')

        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm

        # 各类别准确率
        class_accuracy = []
        for i in range(self.num_classes):
            mask = labels == i
            if mask.sum() > 0:
                class_acc = (predictions[mask] == i).mean()
                class_accuracy.append(class_acc)
            else:
                class_accuracy.append(0.0)
        metrics['class_accuracy'] = class_accuracy

        return metrics

    def plot_confusion_matrix(self, class_names=None):
        """绘制混淆矩阵"""
        if class_names is None:
            class_names = [f'Class {i}' for i in range(self.num_classes)]

        cm = confusion_matrix(self.all_labels, self.all_predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        return plt.gcf()


def compute_band_diversity(attn_weights):
    """
    计算频带多样性（用于可解释性分析）
    Args:
        attn_weights: [B, N, S, K] 注意力权重
    Returns:
        diversity: 标量，频带多样性度量
    """
    B, N, S, K = attn_weights.shape
    # 计算每个频带的专一度（避免所有频带学习相同模式）
    band_specificity = 1.0 - torch.sum(attn_weights ** 2, dim=2) / (S + 1e-8)
    diversity = torch.mean(band_specificity)
    return diversity


# 测试代码
if __name__ == "__main__":
    # 测试指标计算
    metrics_calculator = ClassificationMetrics()

    # 模拟数据
    predictions = torch.randn(100, 2)
    labels = torch.randint(0, 2, (100,))

    metrics_calculator.update(predictions, labels)
    results = metrics_calculator.compute()

    print("=== 指标计算结果 ===")
    for key, value in results.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value:.4f}")
    print(f"混淆矩阵:\n{results['confusion_matrix']}")