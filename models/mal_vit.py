"""
mal_vit.py

Implementation of the MAL-ViT architecture.

Pipeline

Image
    │
Patch Embedding
    │
Patch Tokens
    │
+ Attribute Tokens
    │
+ Positional Embeddings
    │
Transformer Encoder
    │
Attribute Tokens
    │
Attribute Heads
    │
Attribute Predictions

Author: Muhammad Fassi Ur Rehman
"""

import torch
import torch.nn as nn

from config import (
    NUM_PATCHES,
    NUM_ATTRIBUTE_TOKENS,
    EMBED_DIM,
    EMBED_DROPOUT,
)

from models.patch_embedding import PatchEmbedding
from models.attribute_tokens import AttributeTokens
from models.transformer import VisionTransformer
from models.attribute_heads import AttributeHeads


class MALViT(nn.Module):

    def __init__(self):
        super().__init__()

        # ------------------------------------
        # Patch Embedding
        # ------------------------------------

        self.patch_embedding = PatchEmbedding()

        # ------------------------------------
        # Attribute Tokens
        # ------------------------------------

        self.attribute_tokens = AttributeTokens()

        # ------------------------------------
        # Position Embedding
        # ------------------------------------

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                NUM_PATCHES + NUM_ATTRIBUTE_TOKENS,
                EMBED_DIM,
            )
        )

        self.dropout = nn.Dropout(
            EMBED_DROPOUT
        )

        # ------------------------------------
        # Transformer
        # ------------------------------------

        self.transformer = VisionTransformer()

        # ------------------------------------
        # Attribute Heads
        # ------------------------------------

        self.attribute_heads = AttributeHeads()

    def forward(self, images):

        batch_size = images.size(0)

        # ------------------------------------
        # Patch Tokens
        # ------------------------------------

        patch_tokens = self.patch_embedding(images)

        # (B,196,192)

        # ------------------------------------
        # Attribute Tokens
        # ------------------------------------

        attribute_tokens = self.attribute_tokens(batch_size)

        # (B,11,192)

        # ------------------------------------
        # Concatenate
        # ------------------------------------

        tokens = torch.cat(
            [
                attribute_tokens,
                patch_tokens,
            ],
            dim=1,
        )

        # (B,207,192)

        # ------------------------------------
        # Add Position Embedding
        # ------------------------------------

        tokens = tokens + self.position_embedding[:, :tokens.size(1), :]

        tokens = self.dropout(tokens)

        # ------------------------------------
        # Transformer
        # ------------------------------------

        tokens = self.transformer(tokens)

        # ------------------------------------
        # Take Attribute Tokens
        # ------------------------------------

        attribute_features = tokens[
            :,
            :NUM_ATTRIBUTE_TOKENS,
            :
        ]

        # ------------------------------------
        # Attribute Predictions
        # ------------------------------------

        attribute_predictions = self.attribute_heads(
            attribute_features
        )

        return {
            "attribute_features": attribute_features,
            "attribute_predictions": attribute_predictions,
        }


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Test")
    print("=" * 60)

    images = torch.randn(
        4,
        3,
        224,
        224,
    )

    model = MALViT()

    outputs = model(images)

    print("\nAttribute Feature Shape:")
    print(outputs["attribute_features"].shape)

    print("\nAttribute Predictions")

    for name, pred in outputs["attribute_predictions"].items():
        print(f"{name:30} {tuple(pred.shape)}")