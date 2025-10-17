import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import torch


class ClassificationMetrics:
    """分类任务指标计算（修复版本）"""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.all_predictions = []  # 预测类别
        self.all_labels = []  # 真实标签
        self.all_probabilities = []  # 预测概率（用于AUC）

    def update(self, predictions, labels, probabilities=None):
        """
        更新指标状态（修复概率处理）
        """
        # 处理预测类别（如果是logits则取argmax）
        if predictions.ndim == 2:
            preds = np.argmax(predictions.detach().cpu().numpy(), axis=1)
            # 如果是二分类，自动计算概率
            if probabilities is None and self.num_classes == 2:
                # 使用softmax将logits转换为概率
                probs = torch.softmax(predictions, dim=1).detach().cpu().numpy()
                probabilities = probs
        else:
            preds = predictions.detach().cpu().numpy()

        # 处理标签
        labels_np = labels.detach().cpu().numpy()

        # 处理概率
        if probabilities is not None:
            probs_np = probabilities.detach().cpu().numpy() if hasattr(probabilities, 'detach') else probabilities

            # 检查并修复NaN值
            if np.isnan(probs_np).any():
                print(f"警告: 检测到概率中包含NaN值，进行修复")
                probs_np = np.nan_to_num(probs_np, nan=0.5)  # 将NaN替换为0.5

            # 确保概率在有效范围内
            probs_np = np.clip(probs_np, 1e-8, 1.0 - 1e-8)

            self.all_probabilities.extend(probs_np)
        elif predictions.ndim == 2:
            # 如果没有提供概率但predictions是logits，计算softmax概率
            probs = torch.softmax(predictions, dim=1).detach().cpu().numpy()

            # 检查并修复NaN值
            if np.isnan(probs).any():
                print(f"警告: 检测到概率中包含NaN值，进行修复")
                probs = np.nan_to_num(probs, nan=0.5)

            probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
            self.all_probabilities.extend(probs)

        self.all_predictions.extend(preds)
        self.all_labels.extend(labels_np)

    def compute(self):
        """计算所有指标（修复边界情况处理）"""
        if len(self.all_predictions) == 0:
            return {}

        predictions = np.array(self.all_predictions)
        labels = np.array(self.all_labels)

        # 抑制sklearn的警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            metrics = {}

            # 基础分类指标（处理无预测样本的情况）
            try:
                metrics['accuracy'] = accuracy_score(labels, predictions)

                # 检查是否有预测的正类样本
                if len(np.unique(predictions)) > 1:
                    metrics['f1_score'] = f1_score(labels, predictions, average='binary', zero_division=0)
                    metrics['precision'] = precision_score(labels, predictions, average='binary', zero_division=0)
                    metrics['recall'] = recall_score(labels, predictions, average='binary', zero_division=0)
                else:
                    # 如果所有预测都是同一类别，设置默认值
                    metrics['f1_score'] = 0.0
                    metrics['precision'] = 0.0
                    metrics['recall'] = 0.0

                metrics['sensitivity'] = metrics['recall']  # 灵敏度即召回率
            except Exception as e:
                print(f"计算基础指标时出错: {e}")
                # 设置默认值
                metrics.update({
                    'accuracy': 0.0,
                    'f1_score': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'sensitivity': 0.0
                })

            # AUC计算（修复维度问题）
            if len(self.all_probabilities) > 0:
                probabilities = np.array(self.all_probabilities)

                # 最终检查NaN值
                if np.isnan(probabilities).any():
                    print(f"警告: 概率中仍然包含NaN值，使用默认AUC")
                    metrics['auc'] = 0.5
                else:
                    try:
                        if self.num_classes == 2:
                            # 确保概率是2维的
                            if probabilities.ndim == 1:
                                # 如果是1维，假设是正类的概率
                                metrics['auc'] = roc_auc_score(labels, probabilities)
                            else:
                                # 使用第二列作为正类概率
                                metrics['auc'] = roc_auc_score(labels, probabilities[:, 1])
                        else:
                            # 多分类AUC
                            metrics['auc'] = roc_auc_score(labels, probabilities, multi_class='ovo')
                    except Exception as e:
                        print(f"计算AUC时出错: {e}")
                        metrics['auc'] = 0.5  # 随机猜测的AUC
            else:
                metrics['auc'] = 0.5  # 没有概率数据时使用默认值

            # 混淆矩阵
            try:
                cm = confusion_matrix(labels, predictions)
                metrics['confusion_matrix'] = cm
            except:
                metrics['confusion_matrix'] = np.zeros((self.num_classes, self.num_classes))

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