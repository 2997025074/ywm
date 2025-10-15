import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """多任务损失函数（符合idea文件9.2节）"""

    def __init__(self, alpha=0.3, beta=0.2, gamma=0.1, delta=0.1):
        super().__init__()
        self.alpha = alpha  # 频带多样性权重
        self.beta = beta  # 稀疏性正则化权重
        self.gamma = gamma  # 一致性损失权重
        self.delta = delta  # 注意力平滑权重

    def forward(self, predictions, labels, model_outputs):
        # 1. 主分类损失
        cls_loss = F.cross_entropy(predictions, labels)

        # 2. 频带多样性损失
        div_loss = -self.compute_band_diversity(model_outputs['attn_weights'])

        # 3. 稀疏性正则化
        sparse_loss = self.compute_sparsity_regularization(model_outputs)

        # 4. 嵌入一致性损失
        consistency_loss = self.compute_embedding_consistency(model_outputs)

        # 5. 注意力平滑损失
        smooth_loss = self.compute_attention_smoothness(model_outputs['attn_weights'])

        total_loss = (cls_loss +
                      self.alpha * div_loss +
                      self.beta * sparse_loss +
                      self.gamma * consistency_loss +
                      self.delta * smooth_loss)

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
        """计算频带多样性（避免频带模式趋同）"""
        # attn_weights: [B, N, S, K]
        B, N, S, K = attn_weights.shape
        # 计算每个频带的专一度
        band_specificity = 1.0 - torch.sum(attn_weights ** 2, dim=2) / (S + 1e-8)
        diversity = torch.mean(band_specificity)
        return diversity

    def compute_sparsity_regularization(self, model_outputs):
        """计算图稀疏性正则化"""
        if 'sparsity_info' in model_outputs:
            intra_sparsity = model_outputs['sparsity_info']['intra_band_sparsity']
            inter_sparsity = model_outputs['sparsity_info']['inter_band_sparsity']
            # 鼓励适度的稀疏性（避免过连接或过稀疏）
            target_sparsity = 0.7  # 目标稀疏度
            sparse_loss = (intra_sparsity - target_sparsity) ** 2 + (inter_sparsity - target_sparsity) ** 2
        else:
            sparse_loss = torch.tensor(0.0, device=next(iter(model_outputs.values())).device)
        return sparse_loss

    def compute_embedding_consistency(self, model_outputs):
        """计算嵌入一致性损失（同类别样本嵌入应该相似）"""
        if 'final_embeddings' in model_outputs:
            final_embeddings = model_outputs['final_embeddings']  # [B, N, D]
            B, N, D = final_embeddings.shape

            # 计算样本间的相似性（简化版本）
            embeddings_flat = final_embeddings.reshape(B, -1)  # [B, N*D]
            similarity_matrix = F.cosine_similarity(
                embeddings_flat.unsqueeze(1),
                embeddings_flat.unsqueeze(0),
                dim=2
            )

            # 鼓励嵌入具有适度的多样性
            consistency_loss = -torch.std(similarity_matrix)  # 避免所有嵌入相同
        else:
            consistency_loss = torch.tensor(0.0, device=next(iter(model_outputs.values())).device)

        return consistency_loss

    def compute_attention_smoothness(self, attn_weights):
        """计算注意力平滑损失（相邻尺度应该有相似的分配）"""
        # attn_weights: [B, N, S, K]
        B, N, S, K = attn_weights.shape

        if S <= 1:
            return torch.tensor(0.0, device=attn_weights.device)

        # 计算相邻尺度的注意力差异
        diff = attn_weights[:, :, 1:, :] - attn_weights[:, :, :-1, :]
        smooth_loss = torch.mean(diff ** 2)

        return smooth_loss


# 测试代码
if __name__ == "__main__":
    # 测试损失函数
    criterion = MultiTaskLoss()

    # 模拟数据
    predictions = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))

    # 模拟模型输出
    model_outputs = {
        'attn_weights': torch.randn(8, 116, 64, 4),
        'final_embeddings': torch.randn(8, 116, 64),
        'sparsity_info': {
            'intra_band_sparsity': torch.tensor(0.8),
            'inter_band_sparsity': torch.tensor(0.7)
        }
    }

    total_loss, loss_components = criterion(predictions, labels, model_outputs)

    print("=== 多任务损失测试 ===")
    print(f"总损失: {total_loss:.4f}")
    for key, value in loss_components.items():
        print(f"{key}: {value:.4f}")