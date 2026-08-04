"""
mlp.py

Feed Forward Network (MLP) used inside each Transformer Encoder block.

Input:
    (B, N, D)

Output:
    (B, N, D)

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    MLP_RATIO,
    MLP_DROPOUT,
)


class MLP(nn.Module):
    """
    Feed Forward Network used after Multi-Head Self-Attention.

    Architecture:

        Linear
            ↓
        GELU
            ↓
        Dropout
            ↓
        Linear
            ↓
        Dropout
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        mlp_ratio=MLP_RATIO,
        dropout=MLP_DROPOUT,
    ):
        super().__init__()

        hidden_dim = embed_dim * mlp_ratio

        self.fc1 = nn.Linear(
            embed_dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(
            hidden_dim,
            embed_dim,
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input:
            x : (B, N, D)

        Output:
            (B, N, D)
        """

        x = self.fc1(x)

        x = self.activation(x)

        x = self.dropout1(x)

        x = self.fc2(x)

        x = self.dropout2(x)

        return x


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MLP Test")
    print("=" * 60)

    tokens = torch.randn(
        8,
        196,
        EMBED_DIM,
    )

    model = MLP()

    output = model(tokens)

    print("\nInput Shape:")
    print(tokens.shape)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nExpected:")
    print("(8, 196, 192)")