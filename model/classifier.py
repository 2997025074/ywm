import torch
import torch.nn as nn
from .adaptive_band_module import LearnableFrequencyBands  # 使用相对导入
from .multiband_graph import MultiBandGraphBuilder  # 使用相对导入
from .hybrid_gnn_transformer import HybridGNNTransformer  # 使用相对导入


class BrainNetworkClassifier(nn.Module):
    """完整的端到端脑网络分类模型"""

    def __init__(self, num_rois=116, num_scales=64, num_bands=4,
                 feat_dim=28, hidden_dim=64, num_classes=2):
        super().__init__()

        self.num_rois = num_rois
        self.num_scales = num_scales
        self.num_bands = num_bands
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes  # 添加这行

        # 1. 自适应频带划分模块
        self.adaptive_bands = LearnableFrequencyBands(
            num_scales=num_scales,
            num_bands=num_bands,
            feat_dim=feat_dim
        )

        # 2. 多频带图构建模块
        self.graph_builder = MultiBandGraphBuilder(
            num_rois=num_rois,
            num_bands=num_bands,
            feature_dim=feat_dim
        )

        # 3. 混合GNN-Transformer模块
        self.hybrid_net = HybridGNNTransformer(
            in_feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            num_rois=num_rois,
            num_bands=num_bands
        )

        # 4. 分类器模块（符合idea文件9.1节）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * num_rois, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: 输入小波特征 [B, N, S, F]
        Returns:
            predictions: 分类预测 [B, num_classes]
            model_outputs: 所有中间结果（用于可解释性和损失计算）
        """
        model_outputs = {}

        # 步骤1: 自适应频带划分
        band_features, attn_weights = self.adaptive_bands(x)
        model_outputs['band_features'] = band_features
        model_outputs['attn_weights'] = attn_weights

        # 步骤2: 多频带图构建
        graph_data = self.graph_builder(band_features, attn_weights)
        model_outputs.update(graph_data)

        # 步骤3: 混合网络处理
        final_embeddings, hybrid_outputs = self.hybrid_net(
            node_features=graph_data['node_features'],
            intra_band_adj=graph_data['intra_band_adj'],
            unified_adj=graph_data['unified_adj']
        )
        model_outputs.update(hybrid_outputs)
        model_outputs['final_embeddings'] = final_embeddings

        # 步骤4: 分类预测
        B, N, D = final_embeddings.shape
        graph_representation = final_embeddings.reshape(B, -1)  # [B, N*D]
        predictions = self.classifier(graph_representation)
        model_outputs['predictions'] = predictions

        return predictions, model_outputs

    def enable_attention_saving(self, enable=True):
        """启用注意力权重保存"""
        self.hybrid_net.enable_attn_saving(enable)


# 测试代码
if __name__ == "__main__":
    # 测试完整模型
    B, N, S, F = 8, 116, 64, 72
    num_bands = 4
    hidden_dim = 64
    num_classes = 2

    # 模拟输入
    cwt_features = torch.randn(B, N, S, F)

    # 初始化完整模型
    model = BrainNetworkClassifier(
        num_rois=N,
        num_scales=S,
        num_bands=num_bands,
        feat_dim=F,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )
    model.enable_attention_saving(True)

    # 前向传播
    with torch.no_grad():
        predictions, model_outputs = model(cwt_features)

    print("=== 完整模型测试通过 ===")
    print(f"输入形状: {cwt_features.shape}")
    print(f"预测形状: {predictions.shape} → 预期 [{B}, {num_classes}]")
    print(f"最终嵌入形状: {model_outputs['final_embeddings'].shape} → 预期 [{B}, {N}, {hidden_dim}]")
    print(f"注意力权重数量: {len(model_outputs['intra_gat_attn'])}")
    print(f"模型类别数: {model.num_classes}")  # 测试新属性