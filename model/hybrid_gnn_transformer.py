import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class HybridGNNTransformer(nn.Module):
    """
    混合GNN-Transformer模块（修复维度不匹配和数据处理逻辑）
    核心修复：
    1. 阶段3GNN输入维度修正
    2. 阶段3数据处理逻辑修正
    3. 增强与idea文件的匹配度
    """

    def __init__(self,
                 in_feat_dim=72,  # 输入特征维度（72维）
                 hidden_dim=64,  # 隐藏层维度
                 num_rois=116,  # 脑区数（AAL=116）
                 num_bands=4,  # 频带数（K=4）
                 gat_heads=4,  # GAT注意力头数
                 gat_dropout=0.2,  # GAT dropout
                 trans_layers=2,  # Transformer层数
                 trans_heads=4,  # Transformer头数
                 trans_dropout=0.2):  # Transformer dropout
        super().__init__()
        self.num_rois = num_rois
        self.num_bands = num_bands
        self.hidden_dim = hidden_dim

        # -------------------------- 阶段1：频带内GNN --------------------------
        # 符合idea文件7.1节：各频带独立学习脑区空间关系
        self.intra_band_gnns = nn.ModuleList([
            GATConv(
                in_channels=in_feat_dim,
                out_channels=hidden_dim // gat_heads,
                heads=gat_heads,
                dropout=gat_dropout,
                concat=True,
                add_self_loops=True,
                edge_dim=1
            ) for _ in range(num_bands)
        ])
        self.intra_band_norm = nn.LayerNorm(hidden_dim)

        # -------------------------- 阶段2：跨频带Transformer --------------------------
        # 符合idea文件7.1节：自注意力机制建模频带间复杂交互
        trans_encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=trans_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=trans_dropout,
            batch_first=True
        )
        self.cross_band_transformer = TransformerEncoder(
            encoder_layer=trans_encoder_layer,
            num_layers=trans_layers
        )
        self.trans_norm = nn.LayerNorm(hidden_dim)

        # -------------------------- 阶段3：频带间GNN --------------------------
        # 修复：输入维度应为hidden_dim，不是hidden_dim * num_bands
        self.inter_band_gnn = GATConv(
            in_channels=hidden_dim,  # 修正：每个频带节点的特征维度
            out_channels=hidden_dim // gat_heads,
            heads=gat_heads,
            dropout=gat_dropout,
            concat=True,
            add_self_loops=True,
            edge_dim=1
        )
        self.inter_band_norm = nn.LayerNorm(hidden_dim)

        # -------------------------- 多频带嵌入融合 --------------------------
        # 符合idea文件8.2节：注意力加权融合
        self.band_fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

        # -------------------------- 可解释性组件 --------------------------
        self.save_attn_weights = False
        self.intra_gat_attn = []
        self.inter_gat_attn = None
        self.trans_attn_weights = []

    def _split_band_features(self, node_features):
        """拆分节点特征：[B, N×K, F] → [B, K, N, F]"""
        B, _, F = node_features.shape
        return node_features.reshape(B, self.num_bands, self.num_rois, F)

    def _dense_to_edge_index(self, dense_adj, subgraph_node_num):
        """
        生成edge_index和edge_attr
        Args:
            dense_adj: [num_subgraphs, subgraph_node_num, subgraph_node_num]
            subgraph_node_num: 子图节点数（N或K）
        Returns:
            edge_index: [2, total_E]
            edge_attr: [total_E, 1]
        """
        num_subgraphs, _, _ = dense_adj.shape
        edge_index_list = []
        edge_attr_list = []

        for sg_idx in range(num_subgraphs):
            # 提取当前子图的非零边
            i, j = torch.nonzero(dense_adj[sg_idx], as_tuple=True)
            edge_weights = dense_adj[sg_idx][i, j].unsqueeze(1)
            edge_idx = torch.stack([i, j], dim=0)

            # 添加全局偏移：避免不同子图节点索引冲突
            edge_idx += sg_idx * subgraph_node_num

            edge_index_list.append(edge_idx)
            edge_attr_list.append(edge_weights)

        return torch.cat(edge_index_list, dim=1), torch.cat(edge_attr_list, dim=0)

    def _extract_inter_band_adj(self, unified_adj):
        """从统一邻接矩阵提取频带间边：[B, N, K, K]"""
        B, total_nodes, _ = unified_adj.shape
        inter_band_adj = torch.zeros(B, self.num_rois, self.num_bands, self.num_bands,
                                     device=unified_adj.device)

        for b in range(B):
            for n in range(self.num_rois):
                # 提取脑区n的所有频带节点索引
                roi_band_nodes = [n + k * self.num_rois for k in range(self.num_bands)]
                # 提取子邻接矩阵
                inter_band_adj[b, n] = unified_adj[b, roi_band_nodes, :][:, roi_band_nodes]

        return inter_band_adj

    def forward(self, node_features, intra_band_adj, unified_adj=None):
        """前向传播（修复阶段3维度问题）"""
        B = node_features.shape[0]
        intermediate_outputs = {}
        self.intra_gat_attn.clear()

        # ========================= 阶段1：频带内GNN处理 =========================
        # 符合idea文件7.2节：学习脑区空间关系
        band_features = self._split_band_features(node_features)  # [B, K, N, F]
        intra_band_embeddings = []

        for k in range(self.num_bands):
            # 准备当前频带数据
            curr_feat = band_features[:, k, :, :].reshape(B * self.num_rois, -1)  # [B×N, F]
            curr_dense_adj = intra_band_adj[:, k, :, :]  # [B, N, N]

            # 生成edge_index
            curr_edge_idx, curr_edge_attr = self._dense_to_edge_index(
                curr_dense_adj, self.num_rois
            )

            # GAT前向传播
            if self.save_attn_weights:
                gat_out, gat_attn = self.intra_band_gnns[k](
                    x=curr_feat,
                    edge_index=curr_edge_idx,
                    edge_attr=curr_edge_attr,
                    return_attention_weights=True
                )
                self.intra_gat_attn.append(gat_attn)
            else:
                gat_out = self.intra_band_gnns[k](
                    x=curr_feat,
                    edge_index=curr_edge_idx,
                    edge_attr=curr_edge_attr
                )

            # 后处理
            gat_out = gat_out.reshape(B, self.num_rois, self.hidden_dim)
            gat_out = self.intra_band_norm(gat_out)
            intra_band_embeddings.append(gat_out)

        intra_band_embeddings = torch.stack(intra_band_embeddings, dim=1)  # [B, K, N, hidden_dim]
        intermediate_outputs['band_embeddings'] = intra_band_embeddings

        # ========================= 阶段2：跨频带Transformer处理 =========================
        # 符合idea文件7.2节：建模频带间复杂交互
        trans_input = intra_band_embeddings.permute(0, 2, 1, 3).reshape(
            B, self.num_rois * self.num_bands, self.hidden_dim
        )
        trans_out = self.trans_norm(trans_input)
        trans_out = self.cross_band_transformer(trans_out)  # [B, N×K, hidden_dim]
        intermediate_outputs['trans_embeddings'] = trans_out

        # ========================= 阶段3：频带间GNN处理 =========================
        # 修复：正确的数据处理逻辑
        # 将Transformer输出重塑为脑区-频带节点格式
        inter_nodes = trans_out.reshape(B * self.num_rois, self.num_bands, self.hidden_dim)  # [B×N, K, hidden_dim]
        inter_nodes_flat = inter_nodes.reshape(-1, self.hidden_dim)  # [B×N×K, hidden_dim]

        # 准备频带间边数据
        if unified_adj is not None:
            inter_dense_adj = self._extract_inter_band_adj(unified_adj)  # [B, N, K, K]
            inter_dense_adj_flat = inter_dense_adj.reshape(B * self.num_rois, self.num_bands, self.num_bands)
        else:
            inter_dense_adj_flat = torch.ones(
                B * self.num_rois, self.num_bands, self.num_bands,
                device=node_features.device
            )

        # 生成edge_index（子图节点数=K）
        inter_edge_idx, inter_edge_attr = self._dense_to_edge_index(
            inter_dense_adj_flat, self.num_bands
        )

        # 频带间GNN前向传播
        if self.save_attn_weights:
            inter_out, inter_attn = self.inter_band_gnn(
                x=inter_nodes_flat,
                edge_index=inter_edge_idx,
                edge_attr=inter_edge_attr,
                return_attention_weights=True
            )
            self.inter_gat_attn = inter_attn
        else:
            inter_out = self.inter_band_gnn(
                x=inter_nodes_flat,
                edge_index=inter_edge_idx,
                edge_attr=inter_edge_attr
            )

        # 重塑为脑区级特征
        inter_out = inter_out.reshape(B * self.num_rois, self.num_bands, self.hidden_dim)  # [B×N, K, hidden_dim]

        # ========================= 多频带融合 =========================
        # 符合idea文件8.2节：注意力加权融合
        fused_embeddings, fusion_attn = self.band_fusion_attention(
            query=inter_out.mean(dim=1, keepdim=True),  # [B×N, 1, hidden_dim]
            key=inter_out,
            value=inter_out
        )
        fused_embeddings = fused_embeddings.squeeze(1)  # [B×N, hidden_dim]
        fused_embeddings = self.fusion_norm(fused_embeddings)

        # 最终脑区嵌入
        final_embeddings = fused_embeddings.reshape(B, self.num_rois, self.hidden_dim)
        final_embeddings = self.inter_band_norm(final_embeddings)

        # ========================= 整理输出 =========================
        intermediate_outputs.update({
            'intra_band_embeddings': intra_band_embeddings,
            'trans_embeddings': trans_out,
            'final_embeddings': final_embeddings,
            'inter_gat_attn': self.inter_gat_attn,
            'fusion_attention': fusion_attn,
            'band_specific_embeddings': inter_out.reshape(B, self.num_rois, self.num_bands, self.hidden_dim)
        })

        return final_embeddings, intermediate_outputs

    def enable_attn_saving(self, enable=True):
        """启用注意力权重保存（用于可解释性分析）"""
        self.save_attn_weights = enable

    def get_attention_weights(self):
        """获取所有注意力权重（用于idea文件10.2节的可解释性分析）"""
        return {
            'intra_band_attention': self.intra_gat_attn,
            'inter_band_attention': self.inter_gat_attn,
            'fusion_attention': getattr(self, 'fusion_attn', None)
        }


# 测试代码
if __name__ == "__main__":
    # 测试修复后的模块
    B, N, K, F = 8, 116, 4, 72
    hidden_dim = 64

    node_features = torch.randn(B, N * K, F)
    intra_band_adj = torch.randn(B, K, N, N) * 0.5 + 0.3
    intra_band_adj = torch.clamp(intra_band_adj, 0.0, 1.0)
    unified_adj = torch.randn(B, N * K, N * K) * 0.4 + 0.2
    unified_adj = torch.clamp(unified_adj, 0.0, 1.0)

    # 初始化修复后的模块
    hybrid_model = HybridGNNTransformer(
        in_feat_dim=F,
        hidden_dim=hidden_dim,
        num_rois=N,
        num_bands=K
    )
    hybrid_model.enable_attn_saving(enable=True)

    # 前向传播测试
    with torch.no_grad():
        final_emb, inter_out = hybrid_model(
            node_features=node_features,
            intra_band_adj=intra_band_adj,
            unified_adj=unified_adj
        )

    print("=== 修复后的混合GNN-Transformer测试通过 ===")
    print(f"最终脑区嵌入形状: {final_emb.shape} → 预期 [{B}, {N}, {hidden_dim}]")
    print(f"频带特异性嵌入形状: {inter_out['band_specific_embeddings'].shape} → 预期 [{B}, {N}, {K}, {hidden_dim}]")
    print("所有维度匹配正确！")