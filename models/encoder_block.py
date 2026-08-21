"""
encoder_block.py

MAL-ViT Transformer Encoder Block.

Pipeline:

Input
    │
    ├── LayerNorm
    │
    ├── Multi-Head Self-Attention
    │
    ├── Residual Connection
    │
    ├── LayerNorm
    │
    ├── MLP
    │
    └── Residual Connection
    │
Output
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    LAYER_NORM_EPS,
)

from models.attention import MultiHeadSelfAttention
from models.mlp import MLP


class EncoderBlock(nn.Module):

    def __init__(
        self,
        dim=EMBED_DIM,
    ):
        super().__init__()

        # ==================================================
        # LayerNorm 1
        # ==================================================

        self.norm1 = nn.LayerNorm(
            dim,
            eps=LAYER_NORM_EPS,
        )

        # ==================================================
        # Multi-Head Self-Attention
        # ==================================================

        self.attention = MultiHeadSelfAttention()

        # ==================================================
        # LayerNorm 2
        # ==================================================

        self.norm2 = nn.LayerNorm(
            dim,
            eps=LAYER_NORM_EPS,
        )

        # ==================================================
        # MLP
        # ==================================================

        self.mlp = MLP()

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        x,
        return_attention=False,
    ):
        """
        Parameters
        ----------
        x:
            (B, N, D)

        return_attention:
            If True, return attention weights as well.

        Returns
        -------
        If return_attention=False:

            output

        If return_attention=True:

            output, attention_weights
        """

        # ==================================================
        # 1. Self-Attention
        # ==================================================

        residual = x

        x_norm = self.norm1(x)

        attention_result = self.attention(
            x_norm,
            return_attention=return_attention,
        )

        # ==================================================
        # Attention Result
        # ==================================================

        if return_attention:

            attention_output, attention_weights = (
                attention_result
            )

        else:

            attention_output = attention_result
            attention_weights = None

        # ==================================================
        # 2. Attention Residual
        # ==================================================

        x = residual + attention_output

        # ==================================================
        # 3. MLP
        # ==================================================

        residual = x

        x_norm = self.norm2(x)

        mlp_output = self.mlp(
            x_norm
        )

        # ==================================================
        # 4. MLP Residual
        # ==================================================

        x = residual + mlp_output

        # ==================================================
        # Return
        # ==================================================

        if return_attention:

            return (
                x,
                attention_weights,
            )

        return x


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Transformer Encoder Block Test")
    print("=" * 70)

    print("\nConfiguration")

    print(
        f"Embedding dimension : {EMBED_DIM}"
    )

    print(
        "Total tokens        : 211"
    )

    print(
        f"LayerNorm epsilon   : {LAYER_NORM_EPS}"
    )

    # ------------------------------------------------------
    # Create block
    # ------------------------------------------------------

    block = EncoderBlock()

    print("\nEncoder Block:")
    print(block)

    # ------------------------------------------------------
    # Dummy input
    # ------------------------------------------------------

    x = torch.randn(
        4,
        211,
        EMBED_DIM,
    )

    print("\nInput shape:")
    print(x.shape)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    output, attention = block(
        x,
        return_attention=True,
    )

    print("\nOutput shape:")
    print(output.shape)

    print("\nAttention weight shape:")
    print(attention.shape)

    # ------------------------------------------------------
    # Expected
    # ------------------------------------------------------

    expected_output_shape = (
        4,
        211,
        EMBED_DIM,
    )

    expected_attention_shape = (
        4,
        6,
        211,
        211,
    )

    print("\nExpected output shape:")
    print(expected_output_shape)

    print("\nExpected attention shape:")
    print(expected_attention_shape)

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    assert tuple(output.shape) == (
        expected_output_shape
    ), (
        f"Expected output "
        f"{expected_output_shape}, "
        f"got {tuple(output.shape)}"
    )

    assert tuple(attention.shape) == (
        expected_attention_shape
    ), (
        f"Expected attention "
        f"{expected_attention_shape}, "
        f"got {tuple(attention.shape)}"
    )

    assert torch.isfinite(
        output
    ).all()

    assert torch.isfinite(
        attention
    ).all()

    print(
        "\nEncoder block validation: PASSED"
    )

    print("=" * 70)