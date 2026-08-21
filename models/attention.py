"""
attention.py

Multi-Head Self-Attention for MAL-ViT.

Input:
    (B, N, EMBED_DIM)

Output:
    attention output:
        (B, N, EMBED_DIM)

    attention weights:
        (B, NUM_HEADS, N, N)

The attention weights are explicitly returned because they are
required for:

    - attention-sink diagnostics
    - attention visualization
    - XAI
    - attribute-token localization
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_ATTENTION_HEADS,
    ATTENTION_DROPOUT,
    TRANSFORMER_DROPOUT,
)


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        dim=EMBED_DIM,
        num_heads=NUM_ATTENTION_HEADS,
        attention_dropout=ATTENTION_DROPOUT,
        projection_dropout=TRANSFORMER_DROPOUT,
    ):
        super().__init__()

        assert dim % num_heads == 0, (
            f"Embedding dimension {dim} must be "
            f"divisible by number of heads {num_heads}."
        )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        # --------------------------------------------------
        # QKV Projection
        # --------------------------------------------------

        self.qkv = nn.Linear(
            dim,
            dim * 3,
        )

        # --------------------------------------------------
        # Output Projection
        # --------------------------------------------------

        self.projection = nn.Linear(
            dim,
            dim,
        )

        # --------------------------------------------------
        # Dropout
        # --------------------------------------------------

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )

        self.projection_dropout = nn.Dropout(
            projection_dropout
        )

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
        x : torch.Tensor

            Shape:
                (B, N, D)

        return_attention : bool

            If True:
                returns output and attention weights.

            If False:
                returns only output.

        Returns
        -------
        output

            Shape:
                (B, N, D)

        attention_weights

            Shape:
                (B, H, N, N)
        """

        batch_size, num_tokens, dim = x.shape

        # --------------------------------------------------
        # QKV
        # --------------------------------------------------

        qkv = self.qkv(x)

        # (B, N, 3D)
        qkv = qkv.reshape(
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            self.head_dim,
        )

        # (3, B, H, N, head_dim)
        qkv = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        )

        query, key, value = qkv.unbind(0)

        # --------------------------------------------------
        # Attention Scores
        # --------------------------------------------------

        attention_scores = (
            query @ key.transpose(-2, -1)
        ) * self.scale

        # --------------------------------------------------
        # Softmax
        # --------------------------------------------------

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        # --------------------------------------------------
        # Save RAW attention before dropout
        # --------------------------------------------------
        #
        # This is important for diagnostics.
        #
        # Dropout-modified attention should NOT be used
        # to determine whether an attention sink exists.
        #
        # --------------------------------------------------

        raw_attention_weights = attention_weights

        # --------------------------------------------------
        # Attention Dropout
        # --------------------------------------------------

        attention_weights = self.attention_dropout(
            attention_weights
        )

        # --------------------------------------------------
        # Weighted Value
        # --------------------------------------------------

        output = (
            attention_weights @ value
        )

        # (B, H, N, head_dim)

        # --------------------------------------------------
        # Merge Heads
        # --------------------------------------------------

        output = output.transpose(
            1,
            2,
        )

        # (B, N, H, head_dim)

        output = output.reshape(
            batch_size,
            num_tokens,
            dim,
        )

        # --------------------------------------------------
        # Output Projection
        # --------------------------------------------------

        output = self.projection(
            output
        )

        output = self.projection_dropout(
            output
        )

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        if return_attention:

            return (
                output,
                raw_attention_weights,
            )

        return output


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Multi-Head Self-Attention Test")
    print("=" * 70)

    print("\nConfiguration")

    print(
        f"Embedding dimension : {EMBED_DIM}"
    )

    print(
        f"Attention heads     : {NUM_ATTENTION_HEADS}"
    )

    print(
        f"Head dimension      : "
        f"{EMBED_DIM // NUM_ATTENTION_HEADS}"
    )

    print(
        f"Total tokens        : 211"
    )

    print(
        f"Attention dropout   : "
        f"{ATTENTION_DROPOUT}"
    )

    # ------------------------------------------------------
    # Create module
    # ------------------------------------------------------

    attention = MultiHeadSelfAttention()

    print("\nAttention Module:")
    print(attention)

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

    output, weights = attention(
        x,
        return_attention=True,
    )

    print("\nOutput shape:")
    print(output.shape)

    print("\nAttention weight shape:")
    print(weights.shape)

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
        NUM_ATTENTION_HEADS,
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
    )

    assert tuple(weights.shape) == (
        expected_attention_shape
    )

    assert torch.isfinite(output).all()

    assert torch.isfinite(weights).all()

    # Attention rows should sum approximately to 1
    row_sums = weights.sum(
        dim=-1
    )

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5,
    )

    print("\nAttention validation: PASSED")

    print("=" * 70)