"""
encoder_block.py

Single Transformer Encoder Block used in ViT-Tiny and MAL-ViT.

Architecture:

Input
    │
LayerNorm
    │
Multi-Head Self Attention
    │
Residual Add
    │
LayerNorm
    │
MLP
    │
Residual Add
    │
Output

"""

import torch
import torch.nn as nn

from config import EMBED_DIM

from models.attention import MultiHeadSelfAttention
from models.mlp import MLP


class TransformerEncoderBlock(nn.Module):
    """
    One Transformer Encoder Block.
    """

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention()

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLP()

    def forward(self, x):
        """
        Input:
            (B, N, D)

        Output:
            (B, N, D)
        """

        # -------------------------------
        # Multi-Head Self Attention
        # -------------------------------

        x = x + self.attention(
            self.norm1(x)
        )

        # -------------------------------
        # Feed Forward Network
        # -------------------------------

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Transformer Encoder Block Test")
    print("=" * 60)

    tokens = torch.randn(
        8,
        196,
        EMBED_DIM,
    )

    model = TransformerEncoderBlock()

    output = model(tokens)

    print("\nInput Shape:")
    print(tokens.shape)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nExpected:")
    print("(8, 196, 192)")