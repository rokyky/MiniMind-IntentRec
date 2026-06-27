"""
mlp_intent_head.py - 用于会话嵌入的 MLP 意图分类器。

架构：2-3 层 MLP，带 LayerNorm 和 Dropout。
输入：session_embedding（来自序列编码器）
输出：分类类别上的意图分布。

支持 softmax（单意图）和 sigmoid（多标签）两种模式。
可配置隐藏层维度和 dropout。
"""

import math
import torch
from torch import nn
from typing import List, Optional


class MLPIntentHead(nn.Module):
    """
    基于 MLP 的会话嵌入意图分类器。

    将会话嵌入向量映射到预定义分类类别上的意图分布。

    Args:
        input_dim: 输入会话嵌入的维度。
        num_intents: 分类体系中意图类别的数量。
        hidden_dims: 隐藏层维度的列表。
        dropout: 层之间的 dropout 概率。
        mode: 'softmax' 用于单标签分类（互斥意图），
              'sigmoid' 用于多标签分类。
        use_layer_norm: 是否在每层之前应用 LayerNorm。
    """

    def __init__(
        self,
        input_dim: int = 64,
        num_intents: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        mode: str = "softmax",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        assert mode in ("softmax", "sigmoid"), \
            f"mode must be 'softmax' or 'sigmoid', got '{mode}'"
        self.mode = mode
        self.num_intents = num_intents

        if hidden_dims is None:
            hidden_dims = [input_dim * 2, input_dim]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            if use_layer_norm:
                layers.append(nn.LayerNorm(prev_dim))
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # 最终分类层
        if use_layer_norm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, num_intents))

        self.mlp = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """使用小高斯值初始化线性层权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 形状为 (batch_size, input_dim) 的输入张量。

        Returns:
            输出张量：
              - softmax 模式：类别概率 (batch_size, num_intents)
              - sigmoid 模式：独立概率 (batch_size, num_intents)
        """
        logits = self.mlp(x)
        if self.mode == "softmax":
            return torch.softmax(logits, dim=-1)
        else:
            return torch.sigmoid(logits)

    def predict_top_k(self, x: torch.Tensor, k: int = 5) -> tuple:
        """
        预测 top-k 意图。

        Args:
            x: 形状为 (batch_size, input_dim) 的输入张量。
            k: 返回的 top 意图数量。

        Returns:
            (values, indices): 每个形状均为 (batch_size, k)
        """
        probs = self.forward(x)
        return torch.topk(probs, k=k, dim=-1)

    def predict_threshold(self, x: torch.Tensor, threshold: float = 0.5) -> List[torch.Tensor]:
        """
        预测高于置信度阈值的意图（仅 sigmoid 模式）。

        Args:
            x: 形状为 (batch_size, input_dim) 的输入张量。
            threshold: 置信度阈值。

        Returns:
            每个批次样本的布尔掩码列表。
        """
        assert self.mode == "sigmoid", \
            "predict_threshold only works in sigmoid mode"
        probs = self.forward(x)
        return [probs[i] >= threshold for i in range(probs.size(0))]

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """获取激活之前的原始 logits。"""
        return self.mlp(x)


class MLPIntentHeadConfig:
    """MLPIntentHead 的配置类。"""

    def __init__(
        self,
        input_dim: int = 64,
        num_intents: int = 64,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        mode: str = "softmax",
        use_layer_norm: bool = True,
    ):
        self.input_dim = input_dim
        self.num_intents = num_intents
        self.hidden_dims = hidden_dims or [input_dim * 2, input_dim]
        self.dropout = dropout
        self.mode = mode
        self.use_layer_norm = use_layer_norm

    def build(self) -> MLPIntentHead:
        """从配置构建 MLPIntentHead。"""
        return MLPIntentHead(
            input_dim=self.input_dim,
            num_intents=self.num_intents,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            mode=self.mode,
            use_layer_norm=self.use_layer_norm,
        )


def count_parameters(model: MLPIntentHead) -> int:
    """统计 MLP 头中的可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
