"""
transformer.py

Vision Transformer (ViT-Tiny) Backbone.

This module stacks multiple Transformer Encoder Blocks to build
the feature extractor used by MAL-ViT.

Input:
    Patch Tokens
    (B, N, D)

Output:
    Contextualized Tokens
    (B, N, D)

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_TRANSFORMER_LAYERS,
)

from models.encoder_block import TransformerEncoderBlock


class VisionTransformer(nn.Module):
    """
    Standard Vision Transformer Encoder.

    Consists of multiple Transformer Encoder Blocks.
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        depth=NUM_TRANSFORMER_LAYERS,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth

        # Stack of Transformer Encoder Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim)
                for _ in range(depth)
            ]
        )

        # Final Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Input:
            (B, N, D)

        Output:
            (B, N, D)
        """

        # Pass through all encoder blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Vision Transformer Test")
    print("=" * 60)

    tokens = torch.randn(
        8,
        196,
        EMBED_DIM,
    )

    model = VisionTransformer()

    output = model(tokens)

    print("\nNumber of Encoder Blocks:")
    print(len(model.blocks))

    print("\nInput Shape:")
    print(tokens.shape)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nExpected:")
    print("(8, 196, 192)")