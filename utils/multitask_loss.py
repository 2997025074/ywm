import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """多任务损失函数（增加数值稳定性）"""

    def __init__(self, alpha=0.1, beta=0.1, gamma=0.1, delta=0.1):  # 减小初始权重
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.delta = nn.Parameter(torch.tensor(delta))

        # 主分类损失
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(self, predictions, labels, model_outputs):
        # 检查输入
        if torch.isnan(predictions).any():
            print("警告: 预测包含NaN，使用随机预测")
            predictions = torch.randn_like(predictions)

        # 1. 主分类损失（增加稳定性）
        cls_loss = self.cls_criterion(predictions, labels)

        # 检查损失
        if torch.isnan(cls_loss):
            print("错误: 分类损失为NaN，使用默认值")
            cls_loss = torch.tensor(1.0, device=predictions.device)

        # 2. 频带多样性损失（限制范围）
        div_loss = torch.tensor(0.0, device=predictions.device)
        if 'attn_weights' in model_outputs:
            div_loss = -self.compute_band_diversity(model_outputs['attn_weights'])
            div_loss = torch.clamp(div_loss, -1.0, 1.0)

        # 3. 稀疏性正则化（限制范围）
        sparse_loss = self.compute_sparsity_regularization(model_outputs)
        sparse_loss = torch.clamp(sparse_loss, 0.0, 1.0)

        # 4. 嵌入一致性损失（限制范围）
        consistency_loss = self.compute_embedding_consistency(model_outputs)
        consistency_loss = torch.clamp(consistency_loss, -1.0, 1.0)

        # 5. 注意力平滑损失（限制范围）
        smooth_loss = self.compute_attention_smoothness(model_outputs.get('attn_weights', None))
        smooth_loss = torch.clamp(smooth_loss, 0.0, 1.0)

        # 使用sigmoid限制权重范围
        alpha = torch.sigmoid(self.alpha) * 0.1
        beta = torch.sigmoid(self.beta) * 0.1
        gamma = torch.sigmoid(self.gamma) * 0.1
        delta = torch.sigmoid(self.delta) * 0.1

        total_loss = (cls_loss +
                      alpha * div_loss +
                      beta * sparse_loss +
                      gamma * consistency_loss +
                      delta * smooth_loss)

        # 最终检查
        if torch.isnan(total_loss):
            print("错误: 总损失为NaN，仅使用分类损失")
            total_loss = cls_loss

        loss_components = {
            'cls_loss': cls_loss.item(),
            'div_loss': div_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'smooth_loss': smooth_loss.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, loss_components

    def compute_band_diversity(self, attn_weights):
        """计算频带多样性"""
        if attn_weights is None:
            return torch.tensor(0.0)

        B, N, S, K = attn_weights.shape
        # 增加数值稳定性
        attn_safe = torch.clamp(attn_weights, min=1e-8, max=1.0 - 1e-8)
        band_specificity = 1.0 - torch.sum(attn_safe ** 2, dim=2) / (S + 1e-8)
        diversity = torch.mean(band_specificity)
        return diversity

    def compute_sparsity_regularization(self, model_outputs):
        """计算图稀疏性正则化"""
        if 'sparsity_info' not in model_outputs:
            return torch.tensor(0.0)

        try:
            intra_sparsity = model_outputs['sparsity_info']['intra_band_sparsity']
            inter_sparsity = model_outputs['sparsity_info']['inter_band_sparsity']
            target_sparsity = 0.7
            sparse_loss = (intra_sparsity - target_sparsity) ** 2 + (inter_sparsity - target_sparsity) ** 2
            return torch.clamp(sparse_loss, 0.0, 1.0)
        except:
            return torch.tensor(0.0)

    def compute_embedding_consistency(self, model_outputs):
        """计算嵌入一致性损失"""
        if 'final_embeddings' not in model_outputs:
            return torch.tensor(0.0)

        try:
            final_embeddings = model_outputs['final_embeddings']
            B, N, D = final_embeddings.shape
            if B < 2:
                return torch.tensor(0.0)

            embeddings_flat = final_embeddings.reshape(B, -1)
            # 增加数值稳定性
            embeddings_norm = F.normalize(embeddings_flat, p=2, dim=1, eps=1e-8)
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            consistency_loss = -torch.std(similarity_matrix)
            return torch.clamp(consistency_loss, -1.0, 1.0)
        except:
            return torch.tensor(0.0)

    def compute_attention_smoothness(self, attn_weights):
        """计算注意力平滑损失"""
        if attn_weights is None:
            return torch.tensor(0.0)

        try:
            B, N, S, K = attn_weights.shape
            if S <= 1:
                return torch.tensor(0.0)

            diff = attn_weights[:, :, 1:, :] - attn_weights[:, :, :-1, :]
            smooth_loss = torch.mean(diff ** 2)
            return torch.clamp(smooth_loss, 0.0, 1.0)
        except:
            return torch.tensor(0.0)