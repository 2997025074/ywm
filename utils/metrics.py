import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


class ClassificationMetrics:
    """分类任务指标计算（完善多指标支持）"""

    def __init__(self):
        self.all_predictions = []  # 预测类别
        self.all_labels = []  # 真实标签
        self.all_probabilities = []  # 预测概率（用于AUC）

    def update(self, predictions, labels, probabilities=None):
        """
        更新指标状态
        Args:
            predictions: 模型预测类别 [B, num_classes] 或 [B]
            labels: 真实标签 [B]
            probabilities: 模型预测概率 [B, num_classes]（可选，用于AUC）
        """
        # 处理预测类别（如果是logits则取argmax）
        if predictions.ndim == 2:
            preds = np.argmax(predictions.detach().cpu().numpy(), axis=1)
        else:
            preds = predictions.detach().cpu().numpy()

        # 处理标签和概率
        labels_np = labels.detach().cpu().numpy()
        probs_np = probabilities.detach().cpu().numpy() if probabilities is not None else None

        self.all_predictions.extend(preds)
        self.all_labels.extend(labels_np)
        if probs_np is not None:
            self.all_probabilities.extend(probs_np)

    def compute(self):
        """计算所有指标并返回"""
        metrics = {
            'accuracy': accuracy_score(self.all_labels, self.all_predictions),
            'precision': precision_score(self.all_labels, self.all_predictions, average='weighted'),
            'recall': recall_score(self.all_labels, self.all_predictions, average='weighted'),
            'f1': f1_score(self.all_labels, self.all_predictions, average='weighted')
        }

        # 计算AUC（仅二分类且提供概率时）
        if len(self.all_probabilities) > 0 and len(np.unique(self.all_labels)) == 2:
            # 取正类概率
            positive_probs = [p[1] for p in self.all_probabilities]
            metrics['auc'] = roc_auc_score(self.all_labels, positive_probs)
        else:
            metrics['auc'] = 0.0  # 多分类或无概率时AUC设为0

        return metrics