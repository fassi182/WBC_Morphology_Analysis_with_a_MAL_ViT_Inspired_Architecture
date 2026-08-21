"""
mlp.py

Feed-Forward Network (MLP) used inside each MAL-ViT
Transformer Encoder block.

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

Current configuration:

    Input dimension : 192
    Hidden dimension: 768
    Output dimension: 192
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    MLP_HIDDEN_DIM,
    TRANSFORMER_DROPOUT,
)


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):
    """
    Transformer feed-forward network.

    Input:
        (B, N, 192)

    Output:
        (B, N, 192)
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
        hidden_dim=MLP_HIDDEN_DIM,
        dropout=TRANSFORMER_DROPOUT,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # ----------------------------------------------------
        # First projection
        # ----------------------------------------------------

        self.fc1 = nn.Linear(
            embed_dim,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Activation
        # ----------------------------------------------------

        self.activation = nn.GELU()

        # ----------------------------------------------------
        # Dropout
        # ----------------------------------------------------

        self.dropout1 = nn.Dropout(dropout)

        # ----------------------------------------------------
        # Projection back to embedding dimension
        # ----------------------------------------------------

        self.fc2 = nn.Linear(
            hidden_dim,
            embed_dim,
        )

        self.dropout2 = nn.Dropout(dropout)

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:

                (B, N, embed_dim)

        Returns
        -------
        torch.Tensor
            Shape:

                (B, N, embed_dim)
        """

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        if x.ndim != 3:
            raise ValueError(
                "MLP expects input with shape "
                "(B, N, EMBED_DIM). "
                f"Received {tuple(x.shape)}."
            )

        if x.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embed_dim}, "
                f"got {x.size(-1)}."
            )

        # ----------------------------------------------------
        # Feed-forward network
        # ----------------------------------------------------

        x = self.fc1(x)

        # (B, N, 768)

        x = self.activation(x)

        x = self.dropout1(x)

        x = self.fc2(x)

        # (B, N, 192)

        x = self.dropout2(x)

        return x


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT MLP Test")
    print("=" * 70)

    print("\nConfiguration")
    print(f"Embedding dimension : {EMBED_DIM}")
    print(f"MLP hidden dimension: {MLP_HIDDEN_DIM}")
    print(f"Dropout             : {TRANSFORMER_DROPOUT}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    mlp = MLP()

    print("\nMLP:")
    print(mlp)

    # --------------------------------------------------------
    # Dummy transformer sequence
    # --------------------------------------------------------

    batch_size = 4
    num_tokens = 211

    x = torch.randn(
        batch_size,
        num_tokens,
        EMBED_DIM,
    )

    print("\nInput shape:")
    print(x.shape)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    output = mlp(x)

    print("\nOutput shape:")
    print(output.shape)

    expected_shape = (
        batch_size,
        num_tokens,
        EMBED_DIM,
    )

    print("\nExpected shape:")
    print(expected_shape)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert output.shape == expected_shape, (
        f"MLP shape mismatch. "
        f"Expected {expected_shape}, "
        f"got {tuple(output.shape)}"
    )

    assert torch.isfinite(output).all(), (
        "MLP output contains NaN or Inf values."
    )

    # --------------------------------------------------------
    # Parameter validation
    # --------------------------------------------------------

    assert mlp.fc1.in_features == EMBED_DIM
    assert mlp.fc1.out_features == MLP_HIDDEN_DIM

    assert mlp.fc2.in_features == MLP_HIDDEN_DIM
    assert mlp.fc2.out_features == EMBED_DIM

    print("\nMLP validation: PASSED")

    print("=" * 70)