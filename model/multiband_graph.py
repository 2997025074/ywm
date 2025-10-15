import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBandGraphBuilder(nn.Module):
    """多频带脑图构建模块（优化版）"""
    def __init__(self, num_rois=116, num_bands=4, feature_dim=72,
                 intra_threshold=0.1, inter_threshold=0.2, top_k=10):
        super().__init__()
        self.num_rois = num_rois
        self.num_bands = num_bands
        self.feature_dim = feature_dim
        self.intra_threshold = intra_threshold
        self.inter_threshold = inter_threshold
        self.top_k = top_k

        self.intra_thresh_adj = nn.Parameter(torch.tensor(0.0))
        self.inter_thresh_adj = nn.Parameter(torch.tensor(0.0))
        self.attn_fuse_weight = nn.Parameter(torch.tensor(0.5))

        self.intra_refiner = nn.Sequential(
            nn.Linear(num_rois, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_rois),
            nn.Sigmoid()
        )
        self.inter_refiner = nn.Sequential(
            nn.Linear(num_bands, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_bands),
            nn.Sigmoid()
        )

        input_dim = feature_dim * 2 + 6
        self.feat_compressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, band_features, attn_weights=None):
        B, N, K, _ = band_features.shape

        intra_adj = self._build_intra_band_graph(band_features)
        inter_adj = self._build_inter_band_graph(band_features, attn_weights)
        unified_adj = self._build_unified_graph(intra_adj, inter_adj)
        node_features = self._prepare_node_features(band_features)
        graph_features = self._compute_graph_level_features(band_features, intra_adj, inter_adj)
        sparsity_info = self._compute_graph_sparsity(intra_adj, inter_adj)

        return {
            'node_features': node_features,
            'unified_adj': unified_adj,
            'intra_band_adj': intra_adj,
            'inter_band_adj': inter_adj,
            'graph_features': graph_features,
            'sparsity_info': sparsity_info,
            'node_mapping': self._get_node_mapping()
        }

    def _safe_compute_correlation(self, features):
        centered = features - features.mean(dim=2, keepdim=True)
        cov = torch.bmm(centered, centered.transpose(1, 2))
        variances = torch.diagonal(cov, dim1=1, dim2=2)
        std = torch.sqrt(torch.clamp(variances, min=1e-8)).unsqueeze(2)
        corr = cov / (torch.bmm(std, std.transpose(1, 2)) + 1e-8)
        return torch.clamp(corr, -1.0, 1.0)

    def _ensure_symmetry(self, matrix):
        return (matrix + matrix.transpose(-1, -2)) / 2

    def _build_intra_band_graph(self, band_features):
        B, N, K, _ = band_features.shape
        batch_band_feats = band_features.permute(0, 2, 1, 3).reshape(B * K, N, self.feature_dim)
        corr = self._safe_compute_correlation(batch_band_feats)

        intra_thresh = self.intra_threshold + torch.sigmoid(self.intra_thresh_adj) * 0.2
        thresh_mask = (corr.abs() > intra_thresh).float()
        topk_val, topk_idx = torch.topk(corr.abs(), min(self.top_k, N), dim=2)
        topk_mask = torch.zeros_like(corr).scatter_(2, topk_idx, 1.0)
        final_mask = thresh_mask * topk_mask
        sparse_corr = corr * final_mask
        sparse_corr = self._ensure_symmetry(sparse_corr)
        sparse_corr = F.relu(sparse_corr)

        batch_size_refine = sparse_corr.shape[0] * sparse_corr.shape[1]
        sparse_corr_flat = sparse_corr.reshape(batch_size_refine, N)
        refined_flat = self.intra_refiner(sparse_corr_flat)
        refined_corr = refined_flat.reshape(B * K, N, N)
        refined_corr = self._ensure_symmetry(refined_corr)

        return refined_corr.reshape(B, K, N, N)

    def _build_inter_band_graph(self, band_features, attn_weights=None):
        B, N, K, _ = band_features.shape
        batch_roi_feats = band_features.reshape(B * N, K, self.feature_dim)
        base_corr = self._safe_compute_correlation(batch_roi_feats)

        if attn_weights is not None:
            batch_attn = attn_weights.reshape(B * N, -1, K)
            attn_corr = self._safe_compute_correlation(batch_attn.transpose(1, 2))
            fuse_w = torch.sigmoid(self.attn_fuse_weight)
            base_corr = fuse_w * base_corr + (1 - fuse_w) * attn_corr

        inter_thresh = self.inter_threshold + torch.sigmoid(self.inter_thresh_adj) * 0.2
        thresh_mask = (base_corr.abs() > inter_thresh).float()
        sparse_corr = base_corr * thresh_mask
        sparse_corr = self._ensure_symmetry(sparse_corr)
        sparse_corr = F.relu(sparse_corr)

        batch_size_refine = sparse_corr.shape[0] * sparse_corr.shape[1]
        sparse_corr_flat = sparse_corr.reshape(batch_size_refine, K)
        refined_flat = self.inter_refiner(sparse_corr_flat)
        refined_corr = refined_flat.reshape(B * N, K, K)
        refined_corr = self._ensure_symmetry(refined_corr)

        return refined_corr.reshape(B, N, K, K)

    def _build_unified_graph(self, intra_adj, inter_adj):
        """修复循环未闭合问题，完整构建统一邻接矩阵"""
        B, K, N, _ = intra_adj.shape
        unified_adj = torch.zeros(B, N*K, N*K, device=intra_adj.device)

        # 1. 填充频带内连接（对角线块）
        for k in range(K):
            node_start = k * N
            node_end = (k + 1) * N
            unified_adj[:, node_start:node_end, node_start:node_end] = intra_adj[:, k]

        # 2. 填充频带间连接（非对角线块：同一脑区的不同频带）
        for n in range(N):
            for k1 in range(K):
                for k2 in range(K):
                    if k1 != k2:  # 跳过对角块（已由频带内连接填充）
                        # 计算节点索引：(脑区n, 频带k1) → 全局索引
                        idx1 = k1 * N + n
                        idx2 = k2 * N + n
                        # 填充频带间连接权重（取inter_adj中对应脑区n的频带k1-k2连接）
                        unified_adj[:, idx1, idx2] = inter_adj[:, n, k1, k2]
                        unified_adj[:, idx2, idx1] = inter_adj[:, n, k1, k2]  # 保证对称

        return unified_adj

    def _prepare_node_features(self, band_features):
        """将频带特征转换为统一节点特征 [B, N*K, F]"""
        B, N, K, F = band_features.shape
        return band_features.permute(0, 2, 1, 3).reshape(B, K*N, F)

    def _compute_graph_level_features(self, band_features, intra_adj, inter_adj):
        """计算图级统计特征（简化实现）"""
        B, N, K, F = band_features.shape
        mean_feat = band_features.mean(dim=(1, 2))  # [B, F]
        std_feat = band_features.std(dim=(1, 2))   # [B, F]
        intra_density = intra_adj.mean(dim=(1, 2, 3))  # [B]
        inter_density = inter_adj.mean(dim=(1, 2, 3))  # [B]
        graph_feats = torch.cat([mean_feat, std_feat,
                                intra_density.unsqueeze(1),
                                inter_density.unsqueeze(1)], dim=1)
        return self.feat_compressor(graph_feats)

    def _compute_graph_sparsity(self, intra_adj, inter_adj):
        """计算图稀疏性（用于正则化）"""
        intra_sparsity = (intra_adj == 0).float().mean()
        inter_sparsity = (inter_adj == 0).float().mean()
        return {
            'intra_band_sparsity': intra_sparsity,
            'inter_band_sparsity': inter_sparsity
        }

    def _get_node_mapping(self):
        """生成节点映射表：全局节点索引 → (脑区ID, 频带ID)"""
        mapping = []
        for k in range(self.num_bands):
            for n in range(self.num_rois):
                mapping.append([n, k])
        return torch.tensor(mapping, dtype=torch.long)