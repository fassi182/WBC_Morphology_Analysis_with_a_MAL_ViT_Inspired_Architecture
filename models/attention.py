"""
attention.py

Multi-Head Self-Attention module used by ViT-Tiny and MAL-ViT.

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
    NUM_HEADS,
    QKV_BIAS,
    ATTENTION_DROPOUT,
    PROJECTION_DROPOUT,
)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention.

    Input:
        (B, N, D)

    Output:
        (B, N, D)

    where

    B = Batch Size
    N = Number of Tokens
    D = Embedding Dimension
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        qkv_bias=QKV_BIAS,
        attention_dropout=ATTENTION_DROPOUT,
        projection_dropout=PROJECTION_DROPOUT,
    ):
        super().__init__()

        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5

        # Linear projection for Query, Key and Value
        self.qkv = nn.Linear(
            embed_dim,
            embed_dim * 3,
            bias=qkv_bias,
        )

        self.attention_dropout = nn.Dropout(attention_dropout)

        self.projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.projection_dropout = nn.Dropout(
            projection_dropout
        )

    def forward(self, x):
        """
        x:
            (B, N, D)
        """

        B, N, D = x.shape

        # -----------------------------------------
        # Compute Q, K, V
        # -----------------------------------------

        qkv = self.qkv(x)

        # (B, N, 3*D)
        # ->
        # (B, N, 3, Heads, Head_Dim)

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.num_heads,
            self.head_dim,
        )

        # ->
        # (3, B, Heads, N, Head_Dim)

        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        # -----------------------------------------
        # Attention Scores
        # -----------------------------------------

        attention = (q @ k.transpose(-2, -1)) * self.scale

        attention = attention.softmax(dim=-1)

        attention = self.attention_dropout(attention)

        # -----------------------------------------
        # Weighted Sum
        # -----------------------------------------

        x = attention @ v

        # (B, Heads, N, Head_Dim)
        # ->
        # (B, N, Heads, Head_Dim)

        x = x.transpose(1, 2)

        x = x.reshape(
            B,
            N,
            self.embed_dim,
        )

        # Final Linear Projection

        x = self.projection(x)

        x = self.projection_dropout(x)

        return x


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Multi-Head Self-Attention Test")
    print("=" * 60)

    tokens = torch.randn(
        8,
        196,
        EMBED_DIM,
    )

    model = MultiHeadSelfAttention()

    output = model(tokens)

    print("\nInput Shape:")
    print(tokens.shape)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nExpected:")
    print("(8, 196, 192)")