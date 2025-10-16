import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


class ClassificationMetrics:
    """分类任务指标计算（完善多指标支持）"""

    def __init__(self, num_classes=2):  # 新增 num_classes 参数
        self.num_classes = num_classes  # 存储类别数
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
        metrics['sensitivity'] = recall_score(labels, predictions, average='binary')  # 灵敏度即召回率

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