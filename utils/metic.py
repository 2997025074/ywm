from typing import Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score


def get_pred(predictions):
    """将logits转换为预测标签（numpy数组）"""
    # 拼接所有批次的logits（形状：[total_samples, num_classes]）
    stack = torch.cat(predictions)  # 此时是PyTorch张量
    # 计算softmax并取最大概率对应的索引（预测标签）
    _, preds = torch.max(torch.softmax(stack, dim=-1), dim=-1)  # 形状：[total_samples]
    # 转换为numpy数组（关键：从张量转为numpy）
    return preds.cpu().numpy()


def confusion(g_true, g_pred):
    """计算混淆矩阵和准确率等指标"""
    # 1. 处理预测标签：从logits得到numpy格式的预测标签
    pred = get_pred(g_pred)  # 现在是numpy数组：[total_samples]

    # 2. 处理真实标签：确保是numpy数组（原g_true是Python列表）
    lab = np.array(g_true)  # 转换为numpy数组：[total_samples]

    # 3. 验证长度是否一致（防止批次处理时的漏算）
    assert len(lab) == len(pred), f"真实标签长度 {len(lab)} 与预测标签长度 {len(pred)} 不一致"

    # 4. 计算准确率和混淆矩阵
    acc = accuracy_score(lab, pred)
    tn, fp, fn, tp = confusion_matrix(lab, pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    return acc, tn, fp, fn, tp, sensitivity, specificity
    
