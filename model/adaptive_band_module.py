import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LearnableFrequencyBands(nn.Module):
    def __init__(self, num_scales=64, num_bands=4, feat_dim=72, hidden_dim=32):
        super().__init__()
        self.num_scales = num_scales  # 输入的离散尺度数（64）
        self.num_bands = num_bands    # 目标频带数（4-6，可调）
        self.feat_dim = feat_dim      # 每个尺度的特征维度（72）
        self.hidden_dim = hidden_dim

        # 1. 尺度特征编码器：将每个尺度的72维特征压缩为低维向量
        self.scale_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),  # 72→32
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), # 32→32
            nn.ReLU()
        )

        # 2. 可学习的注意力权重矩阵：计算每个尺度对K个频带的分配权重
        self.scale2band_attn = nn.Parameter(
            torch.randn(num_scales, hidden_dim, num_bands) / math.sqrt(hidden_dim)
        )

        # 3. 频带重要性权重
        self.band_importance = nn.Parameter(torch.ones(1, 1, num_bands, 1))

    def forward(self, scale_features):
        """
        Args:
            scale_features: [B, N, S, F] → 小波特征（B=批大小，N=116，S=64，F=72）
        Returns:
            band_features: [B, N, K, F] → 自适应频带特征
            attn_weights: [B, N, S, K] → 尺度-频带分配权重（可解释性）
        """
        # 将F重命名为feat_dim，避免与torch.nn.functional的F冲突
        B, N, S, feat_dim = scale_features.shape  # 这里修改变量名

        # 步骤1：编码每个尺度的特征 → [B, N, S, hidden_dim]
        encoded_scales = self.scale_encoder(scale_features)

        # 步骤2：计算注意力权重（软分配）→ [B, N, S, K]
        attn_weights = torch.einsum('bnsh,shk->bnsk', encoded_scales, self.scale2band_attn)
        attn_weights = F.softmax(attn_weights, dim=2)  # 现在F正确指向torch.nn.functional

        # 步骤3：按权重融合尺度特征，得到K个频带的特征 → [B, N, K, F]
        # 注意这里的 einsum 中"f"对应原来的F，现在变量名改为feat_dim，但维度含义不变，无需修改einsum表达式
        band_features = torch.einsum('bnsk,bnsf->bnkf', attn_weights, scale_features)

        # 步骤4：应用频带重要性权重
        band_features = band_features * F.softmax(self.band_importance, dim=2)

        return band_features, attn_weights

    def compute_band_diversity(self, attn_weights):
        """
        计算频带多样性（用于正则化损失）
        Args:
            attn_weights: [B, N, S, K] 注意力权重
        Returns:
            diversity: 标量，频带多样性度量
        """
        # 计算每个频带的专一度（避免所有频带学习相同模式）
        band_specificity = 1.0 - torch.sum(attn_weights**2, dim=2) / (self.num_scales + 1e-8)
        diversity = torch.mean(band_specificity)
        return diversity