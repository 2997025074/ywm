import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableFrequencyBands(nn.Module):
    def __init__(self, num_scales=64, num_bands=4, feat_dim=28, hidden_dim=32):  # feat_dim从72改为28
        super().__init__()
        self.num_scales = num_scales
        self.num_bands = num_bands
        self.feat_dim = feat_dim  # 现在为28
        self.hidden_dim = hidden_dim

        # 1. 尺度特征编码器
        self.scale_encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),  # 输入维度改为28
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. 可学习的注意力权重矩阵
        self.scale2band_attn = nn.Parameter(
            torch.randn(num_scales, hidden_dim, num_bands) * 0.01
        )

        # 3. 频带重要性权重
        self.band_importance = nn.Parameter(torch.ones(1, 1, num_bands, 1) * 0.1)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, scale_features):
        """
        Args:
            scale_features: [B, N, S, 28] → 小波特征 (28维)
        Returns:
            band_features: [B, N, K, 28] → 自适应频带特征
            attn_weights: [B, N, S, K] → 尺度-频带分配权重
        """
        B, N, S, feat_dim = scale_features.shape

        # 检查输入维度
        if feat_dim != self.feat_dim:
            print(f"警告: 输入特征维度{feat_dim}与预期{self.feat_dim}不匹配")
        # 检查输入
        if torch.isnan(scale_features).any() or torch.isinf(scale_features).any():
            print("警告: 自适应频带模块输入包含NaN或Inf")
            scale_features = torch.nan_to_num(scale_features)

        # 步骤1：编码每个尺度的特征
        encoded_scales = self.scale_encoder(scale_features)

        # 检查编码输出
        if torch.isnan(encoded_scales).any():
            print("错误: 尺度编码输出包含NaN")
            encoded_scales = torch.nan_to_num(encoded_scales)

        # 步骤2：计算注意力权重
        attn_weights = torch.einsum('bnsh,shk->bnsk', encoded_scales, self.scale2band_attn)

        # 数值稳定性：减去最大值
        max_vals, _ = torch.max(attn_weights, dim=2, keepdim=True)
        attn_weights = attn_weights - max_vals

        attn_weights = F.softmax(attn_weights, dim=2)

        # 检查注意力权重
        if torch.isnan(attn_weights).any():
            print("错误: 注意力权重包含NaN，使用均匀分布")
            attn_weights = torch.ones_like(attn_weights) / attn_weights.shape[2]

        # 步骤3：按权重融合尺度特征
        band_features = torch.einsum('bnsk,bnsf->bnkf', attn_weights, scale_features)

        # 步骤4：应用频带重要性权重
        importance_weights = F.softmax(self.band_importance, dim=2)
        band_features = band_features * importance_weights

        # 最终检查
        if torch.isnan(band_features).any():
            print("错误: 频带特征包含NaN，使用零替换")
            band_features = torch.nan_to_num(band_features)

        return band_features, attn_weights

    def compute_band_diversity(self, attn_weights):
        """计算频带多样性"""
        band_specificity = 1.0 - torch.sum(attn_weights ** 2, dim=2) / (self.num_scales + 1e-8)
        diversity = torch.mean(band_specificity)
        return diversity