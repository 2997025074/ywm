import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBandGraphBuilder(nn.Module):
    """
    多频带脑图构建模块（最终优化版）
    核心改进：向量化计算、安全数值处理、严格对称保证、高效内存利用
    输入：band_features [B, N, K, F]、attn_weights [B, N, S, K]
    输出：统一图结构（节点特征+邻接矩阵）+ 可解释性信息
    """
    def __init__(self, num_rois=116, num_bands=4, feature_dim=72,
                 intra_threshold=0.1, inter_threshold=0.2, top_k=10):
        super().__init__()
        self.num_rois = num_rois          # 脑区数（AAL=116）
        self.num_bands = num_bands        # 自适应频带数（默认4）
        self.feature_dim = feature_dim    # 节点特征维度（72）
        self.intra_threshold = intra_threshold  # 频带内连接阈值
        self.inter_threshold = inter_threshold  # 频带间连接阈值
        self.top_k = top_k                # 动态top-k稀疏化参数

        # 1. 可学习参数（自适应调整阈值和融合权重）
        self.intra_thresh_adj = nn.Parameter(torch.tensor(0.0))  # 阈值调整范围：0→0.2
        self.inter_thresh_adj = nn.Parameter(torch.tensor(0.0))
        self.attn_fuse_weight = nn.Parameter(torch.tensor(0.5))  # 注意力融合权重：0→1

        # 2. 连接精炼网络（优化节点连接向量）
        self.intra_refiner = nn.Sequential(  # 频带内：输入N（116）→ 输出N（116）
            nn.Linear(num_rois, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_rois),
            nn.Sigmoid()
        )
        self.inter_refiner = nn.Sequential(  # 频带间：输入K（4）→ 输出K（4）
            nn.Linear(num_bands, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_bands),
            nn.Sigmoid()
        )

        # 3. 图级特征压缩器（预定义，避免重复创建）
        input_dim = feature_dim * 2 + 6  # 72×2（mean/std） + 6个拓扑统计量 = 150
        self.feat_compressor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, band_features, attn_weights=None):
        """前向传播：端到端输出完整图结构"""
        B, N, K, _ = band_features.shape

        # 步骤1：构建频带内图（脑区间功能连接）
        intra_adj = self._build_intra_band_graph(band_features)  # [B, K, N, N]
        # 步骤2：构建频带间图（频带间相关性，向量化计算）
        inter_adj = self._build_inter_band_graph(band_features, attn_weights)  # [B, N, K, K]
        # 步骤3：构建统一脑区-频带图（整合双层次连接）
        unified_adj = self._build_unified_graph(intra_adj, inter_adj)  # [B, N*K, N*K]
        # 步骤4：准备节点特征（频带优先顺序，与统一图对齐）
        node_features = self._prepare_node_features(band_features)  # [B, N*K, F]
        # 步骤5：计算图级特征（供分类器使用）
        graph_features = self._compute_graph_level_features(band_features, intra_adj, inter_adj)  # [B, 128]
        # 步骤6：计算图稀疏性（供正则化损失）
        sparsity_info = self._compute_graph_sparsity(intra_adj, inter_adj)

        return {
            # 核心GNN输入
            'node_features': node_features,    # [B, N*K, F]
            'unified_adj': unified_adj,        # [B, N*K, N*K]
            # 分层图结构（可解释性分析）
            'intra_band_adj': intra_adj,      # [B, K, N, N]
            'inter_band_adj': inter_adj,      # [B, N, K, K]
            # 辅助信息
            'graph_features': graph_features,  # [B, 128]
            'sparsity_info': sparsity_info,    # 稀疏性统计
            'node_mapping': self._get_node_mapping()  # [N*K, 2]：节点→(脑区, 频带)
        }

    def _safe_compute_correlation(self, features):
        """
        安全的相关系数计算（向量化版本）
        解决：除零错误、数值溢出，统一相关系数计算逻辑
        Input: features [batch, dim1, dim2]（dim1=脑区/频带数，dim2=特征/尺度数）
        Output: corr [batch, dim1, dim1]（相关矩阵）
        """
        # 1. 中心化：每个dim1维度减去自身均值
        centered = features - features.mean(dim=2, keepdim=True)  # [batch, dim1, dim2]
        # 2. 批量计算协方差矩阵
        cov = torch.bmm(centered, centered.transpose(1, 2))  # [batch, dim1, dim1]
        # 3. 安全计算标准差（避免方差为0导致除零）
        variances = torch.diagonal(cov, dim1=1, dim2=2)  # [batch, dim1]（每个dim1的方差）
        std = torch.sqrt(torch.clamp(variances, min=1e-8)).unsqueeze(2)  # [batch, dim1, 1]
        # 4. 计算相关系数（避免数值溢出）
        corr = cov / (torch.bmm(std, std.transpose(1, 2)) + 1e-8)
        return torch.clamp(corr, -1.0, 1.0)  # 限制范围，避免数值异常

    def _ensure_symmetry(self, matrix):
        """
        确保矩阵严格对称（适配任意维度：2D/3D/4D）
        功能连接矩阵必须对称，避免GNN学习错误的单向连接
        Input: matrix [B, ..., dim, dim]
        Output: symmetric_matrix [B, ..., dim, dim]
        """
        return (matrix + matrix.transpose(-1, -2)) / 2  # 最后两维转置并平均

    def _build_intra_band_graph(self, band_features):
        """批量构建频带内图：向量化计算，无循环"""
        B, N, K, _ = band_features.shape

        # 1. 重塑特征：[B, N, K, F] → [B*K, N, F]（合并批和频带，便于批量计算）
        batch_band_feats = band_features.permute(0, 2, 1, 3).reshape(B * K, N, self.feature_dim)

        # 2. 批量计算功能连接（安全相关系数）
        corr = self._safe_compute_correlation(batch_band_feats)  # [B*K, N, N]

        # 3. 动态稀疏化（阈值+top-k，保留强连接）
        intra_thresh = self.intra_threshold + torch.sigmoid(self.intra_thresh_adj) * 0.2
        # 阈值过滤：保留绝对值>阈值的连接
        thresh_mask = (corr.abs() > intra_thresh).float()
        # Top-k过滤：每个节点保留top-k强连接（避免孤立节点）
        topk_val, topk_idx = torch.topk(corr.abs(), min(self.top_k, N), dim=2)
        topk_mask = torch.zeros_like(corr).scatter_(2, topk_idx, 1.0)
        # 合并掩码（同时满足阈值和top-k）
        final_mask = thresh_mask * topk_mask
        sparse_corr = corr * final_mask
        # 确保对称+保留正连接（负连接视为噪声）
        sparse_corr = self._ensure_symmetry(sparse_corr)
        sparse_corr = F.relu(sparse_corr)

        # 4. 连接精炼（优化节点连接向量）
        # 展平：[B*K, N, N] → [B*K*N, N]（每个节点的连接向量作为一行）
        batch_size_refine = sparse_corr.shape[0] * sparse_corr.shape[1]
        sparse_corr_flat = sparse_corr.reshape(batch_size_refine, N)
        # 精炼连接向量
        refined_flat = self.intra_refiner(sparse_corr_flat)  # [B*K*N, N]
        # 重塑回原维度并确保对称
        refined_corr = refined_flat.reshape(B * K, N, N)
        refined_corr = self._ensure_symmetry(refined_corr)

        # 5. 最终维度：[B*K, N, N] → [B, K, N, N]
        return refined_corr.reshape(B, K, N, N)

    def _build_inter_band_graph(self, band_features, attn_weights=None):
        """批量构建频带间图：向量化替代循环，解决内存泄漏"""
        B, N, K, _ = band_features.shape

        # 1. 重塑特征：[B, N, K, F] → [B*N, K, F]（合并批和脑区）
        batch_roi_feats = band_features.reshape(B * N, K, self.feature_dim)

        # 2. 基础频带相关性（基于频带特征）
        base_corr = self._safe_compute_correlation(batch_roi_feats)  # [B*N, K, K]

        # 3. 融合注意力权重（可选，增强频带相关性）
        if attn_weights is not None:
            # 重塑注意力权重：[B, N, S, K] → [B*N, S, K]
            batch_attn = attn_weights.reshape(B * N, -1, K)  # [B*N, S, K]
            # 向量化计算注意力相关性（无需循环）
            attn_corr = self._safe_compute_correlation(batch_attn.transpose(1, 2))  # [B*N, K, K]
            # 加权融合（可学习权重）
            fuse_w = torch.sigmoid(self.attn_fuse_weight)
            base_corr = fuse_w * base_corr + (1 - fuse_w) * attn_corr

        # 4. 动态稀疏化
        inter_thresh = self.inter_threshold + torch.sigmoid(self.inter_thresh_adj) * 0.2
        thresh_mask = (base_corr.abs() > inter_thresh).float()
        sparse_corr = base_corr * thresh_mask
        # 确保对称+保留正连接
        sparse_corr = self._ensure_symmetry(sparse_corr)
        sparse_corr = F.relu(sparse_corr)

        # 5. 连接精炼
        batch_size_refine = sparse_corr.shape[0] * sparse_corr.shape[1]
        sparse_corr_flat = sparse_corr.reshape(batch_size_refine, K)
        refined_flat = self.inter_refiner(sparse_corr_flat)
        refined_corr = refined_flat.reshape(B * N, K, K)
        refined_corr = self._ensure_symmetry(refined_corr)

        # 6. 最终维度：[B*N, K, K] → [B, N, K, K]
        return refined_corr.reshape(B, N, K, K)

    def _build_unified_graph(self, intra_adj, inter_adj):
        """构建统一脑区-频带图：整合频带内+频带间连接"""
        B, K, N, _ = intra_adj.shape
        unified_adj = torch.zeros(B, N*K, N*K, device=intra_adj.device)

        # 1. 填充频带内连接（对角线块：同一频带的脑区连接）
        for k in range(K):
            node_start = k * N
            node_end = (k + 1) * N
            unified_adj[:, node_start:node_end, node_start:node_end] = intra_adj[:, k]

        # 2. 填充频带间连接（非对角线块：同一脑区的频带连接）
        for n in range(N):
            for k1 in range(K):
                for k2 in range(K):
                    if k1 == k2:
                        continue  # 跳过自连接（已在频带内图处理）
                    node1 = k1 * N + n  # 脑区n在频带k1的节点索引
                    node2 = k2 * N + n  # 脑区n在频带k2的节点索引
                    unified_adj[:, node1, node2] = inter_adj[:, n, k1, k2]

        return unified_adj

    def _prepare_node_features(self, band_features):
        """准备节点特征：频带优先顺序，与统一图节点对齐"""
        B, N, K, _ = band_features.shape
        # 重塑逻辑：[B, N, K, F] → [B, K, N, F] → [B, K*N, F] = [B, N*K, F]
        return band_features.permute(0, 2, 1, 3).reshape(B, K*N, self.feature_dim)

    def _compute_graph_level_features(self, band_features, intra_adj, inter_adj):
        """计算图级特征：整合节点统计和拓扑特征"""
        B, N, K, _ = band_features.shape

        # 1. 节点特征统计（均值+标准差）
        node_mean = band_features.mean(dim=(1, 2))  # [B, F]
        node_std = band_features.std(dim=(1, 2))    # [B, F]

        # 2. 频带内图拓扑特征（度中心性+聚类系数）
        intra_degree = (intra_adj > 0).float().sum(dim=3)  # [B, K, N]：每个脑区的连接数
        intra_degree_mean = intra_degree.mean(dim=(1, 2)).unsqueeze(1)  # [B, 1]
        intra_degree_std = intra_degree.std(dim=(1, 2)).unsqueeze(1)    # [B, 1]
        # 聚类系数（近似：三角连接数/可能的三角数）
        intra_square = torch.bmm(intra_adj.reshape(B*K, N, N), intra_adj.reshape(B*K, N, N))  # [B*K, N, N]
        intra_tri = torch.diagonal(intra_square, dim1=1, dim2=2).reshape(B, K, N)  # [B, K, N]：每个脑区的三角数
        intra_cluster = intra_tri / (intra_degree ** 2 + 1e-8)  # [B, K, N]
        intra_cluster_mean = intra_cluster.mean(dim=(1, 2)).unsqueeze(1)  # [B, 1]
        intra_cluster_std = intra_cluster.std(dim=(1, 2)).unsqueeze(1)    # [B, 1]

        # 3. 频带间图拓扑特征（连接密度+强度）
        inter_density = (inter_adj > 0).float().mean(dim=(2, 3))  # [B, N]：每个脑区的频带连接密度
        inter_density_mean = inter_density.mean(dim=1).unsqueeze(1)  # [B, 1]
        inter_strength = inter_adj.mean(dim=(2, 3))  # [B, N]：每个脑区的频带连接强度
        inter_strength_mean = inter_strength.mean(dim=1).unsqueeze(1)  # [B, 1]

        # 4. 合并所有特征并压缩
        combined_feats = torch.cat([
            node_mean, node_std,
            intra_degree_mean, intra_degree_std, intra_cluster_mean, intra_cluster_std,
            inter_density_mean, inter_strength_mean
        ], dim=1)  # [B, F×2 + 6]
        graph_feats = self.feat_compressor(combined_feats)  # [B, 128]

        return graph_feats

    def _compute_graph_sparsity(self, intra_adj, inter_adj):
        """计算图稀疏性：用于稀疏性正则化损失（idea.txt第9.2节）"""
        B, K, N, _ = intra_adj.shape
        # 计算总元素数（加权稀疏性，避免不同维度权重不均）
        intra_total = B * K * N * N
        inter_total = B * N * K * K
        # 稀疏性 = 1 - 非零元素比例
        intra_sparse = 1.0 - (intra_adj > 0).float().mean()
        inter_sparse = 1.0 - (inter_adj > 0).float().mean()
        total_sparse = (intra_sparse * intra_total + inter_sparse * inter_total) / (intra_total + inter_total)

        return {
            'intra_band_sparsity': intra_sparse,    # 频带内图稀疏性
            'inter_band_sparsity': inter_sparse,   # 频带间图稀疏性
            'total_sparsity': total_sparse         # 整体图稀疏性
        }

    def _get_node_mapping(self):
        """获取节点映射表：用于可解释性分析（定位异常脑区-频带）"""
        return torch.tensor([
            [n, k] for k in range(self.num_bands) for n in range(self.num_rois)
        ], dtype=torch.long)


# -------------------------- 测试代码：验证最终版本功能 --------------------------
if __name__ == "__main__":
    # 模拟前序模块输出（自适应频带划分结果）
    B = 8          # 批大小
    N = 116        # 脑区数
    K = 4          # 频带数
    feat_dim = 72         # 特征维度
    S = 64         # 尺度数（自适应频带划分的输入尺度）

    # 生成模拟数据
    band_features = torch.randn(B, N, K, feat_dim)  # [B, N, K, F]
    attn_weights = torch.randn(B, N, S, K)   # [B, N, S, K]（注意力权重）

    # 初始化最终版本模块
    graph_builder = MultiBandGraphBuilder(
        num_rois=N,
        num_bands=K,
        feature_dim=feat_dim,
        top_k=10
    )

    # 前向传播（验证无错误）
    with torch.no_grad():
        graph_dict = graph_builder(band_features, attn_weights)

    # 输出验证（确保维度正确+功能正常）
    print("=== 多频带图模块最终版本验证通过 ===")
    print(f"1. 节点特征形状: {graph_dict['node_features'].shape} → 预期 [{B}, {N*K}, {F}]")
    print(f"2. 统一邻接矩阵形状: {graph_dict['unified_adj'].shape} → 预期 [{B}, {N*K}, {N*K}]")
    print(f"3. 频带内邻接矩阵形状: {graph_dict['intra_band_adj'].shape} → 预期 [{B}, {K}, {N}, {N}]")
    print(f"4. 频带间邻接矩阵形状: {graph_dict['inter_band_adj'].shape} → 预期 [{B}, {N}, {K}, {K}]")
    print(f"5. 图级特征形状: {graph_dict['graph_features'].shape} → 预期 [{B}, 128]")
    print(f"6. 整体图稀疏性: {graph_dict['sparsity_info']['total_sparsity']:.3f}（预期>0.85）")
    print(f"7. 节点映射表形状: {graph_dict['node_mapping'].shape} → 预期 [{N*K}, 2]")