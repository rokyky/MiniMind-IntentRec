"""
mlp_intent_head.py - MLP intent classifier for session embeddings.

Architecture: 2-3 layer MLP with LayerNorm and Dropout.
Input: session_embedding (from sequence encoder)
Output: intent distribution over taxonomy classes.

Supports both softmax (single intent) and sigmoid (multi-label) modes.
Configurable hidden dims and dropout.
"""

import math
import torch
from torch import nn
from typing import List, Optional


class MLPIntentHead(nn.Module):
    """
    MLP-based intent classifier for session embeddings.

    Maps session embedding vector to intent distribution over
    pre-defined taxonomy classes.

    Args:
        input_dim: Dimension of input session embedding.
        num_intents: Number of intent classes in taxonomy.
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability between layers.
        mode: 'softmax' for single-label classification (mutually exclusive
              intents), 'sigmoid' for multi-label classification.
        use_layer_norm: Whether to apply LayerNorm before each layer.
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

        # Final classification layer
        if use_layer_norm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, num_intents))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with small Gaussian weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Output tensor:
              - softmax mode: class probabilities (batch_size, num_intents)
              - sigmoid mode: independent probabilities (batch_size, num_intents)
        """
        logits = self.mlp(x)
        if self.mode == "softmax":
            return torch.softmax(logits, dim=-1)
        else:
            return torch.sigmoid(logits)

    def predict_top_k(self, x: torch.Tensor, k: int = 5) -> tuple:
        """
        Predict top-k intents.

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            k: Number of top intents to return.

        Returns:
            (values, indices): each shape (batch_size, k)
        """
        probs = self.forward(x)
        return torch.topk(probs, k=k, dim=-1)

    def predict_threshold(self, x: torch.Tensor, threshold: float = 0.5) -> List[torch.Tensor]:
        """
        Predict intents above a confidence threshold (sigmoid mode only).

        Args:
            x: Input tensor of shape (batch_size, input_dim).
            threshold: Confidence threshold.

        Returns:
            List of boolean masks per batch item.
        """
        assert self.mode == "sigmoid", \
            "predict_threshold only works in sigmoid mode"
        probs = self.forward(x)
        return [probs[i] >= threshold for i in range(probs.size(0))]

    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits before activation."""
        return self.mlp(x)


class MLPIntentHeadConfig:
    """Configuration for MLPIntentHead."""

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
        """Build MLPIntentHead from config."""
        return MLPIntentHead(
            input_dim=self.input_dim,
            num_intents=self.num_intents,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            mode=self.mode,
            use_layer_norm=self.use_layer_norm,
        )


def count_parameters(model: MLPIntentHead) -> int:
    """Count trainable parameters in the MLP head."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
