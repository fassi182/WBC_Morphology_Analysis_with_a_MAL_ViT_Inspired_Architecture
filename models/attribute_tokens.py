"""
attribute_tokens.py

Learnable Attribute Tokens for MAL-ViT.

Each morphology attribute has one dedicated learnable token.

Attributes:

    0  cell_size
    1  cell_shape
    2  nucleus_shape
    3  nuclear_cytoplasmic_ratio
    4  chromatin_density
    5  cytoplasm_vacuole
    6  cytoplasm_texture
    7  cytoplasm_colour
    8  granule_type
    9  granule_colour
    10 granularity

For the current configuration:

    Number of attribute tokens = 11
    Embedding dimension         = 192

Input:
    batch_size

Output:
    (B, 11, 192)
"""

import torch
import torch.nn as nn

from config import (
    NUM_ATTRIBUTE_TOKENS,
    EMBED_DIM,
    INIT_STD,
)


# ============================================================
# Attribute Tokens
# ============================================================

class AttributeTokens(nn.Module):
    """
    Learnable tokens representing the morphology attributes.

    These tokens act as attribute-specific query representations
    inside the transformer.
    """

    def __init__(
        self,
        num_tokens=NUM_ATTRIBUTE_TOKENS,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # Learnable attribute token parameters
        # ----------------------------------------------------
        #
        # Shape:
        #
        #     (1, 11, 192)
        #
        # The first dimension is 1 because the same learned
        # attribute-token set is expanded for every image in
        # the batch.
        #
        # ----------------------------------------------------

        self.tokens = nn.Parameter(
            torch.zeros(
                1,
                num_tokens,
                embed_dim,
            )
        )

        # ----------------------------------------------------
        # Initialization
        # ----------------------------------------------------

        nn.init.trunc_normal_(
            self.tokens,
            std=INIT_STD,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, batch_size):
        """
        Expand the learnable attribute tokens for a batch.

        Parameters
        ----------
        batch_size : int
            Number of images in the batch.

        Returns
        -------
        torch.Tensor
            Shape:

                (B, 11, 192)
        """

        if not isinstance(batch_size, int):
            raise TypeError(
                "batch_size must be an integer."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        # ----------------------------------------------------
        # Expand across batch dimension
        # ----------------------------------------------------

        tokens = self.tokens.expand(
            batch_size,
            -1,
            -1,
        )

        return tokens


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Attribute Tokens Test")
    print("=" * 70)

    print("\nConfiguration")
    print(
        f"Number of attribute tokens : "
        f"{NUM_ATTRIBUTE_TOKENS}"
    )

    print(
        f"Embedding dimension        : "
        f"{EMBED_DIM}"
    )

    # --------------------------------------------------------
    # Create module
    # --------------------------------------------------------

    attribute_tokens = AttributeTokens()

    print("\nAttribute Token Module:")
    print(attribute_tokens)

    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

    batch_size = 4

    tokens = attribute_tokens(batch_size)

    print("\nOutput shape:")
    print(tokens.shape)

    expected_shape = (
        batch_size,
        NUM_ATTRIBUTE_TOKENS,
        EMBED_DIM,
    )

    print("\nExpected shape:")
    print(expected_shape)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert tokens.shape == expected_shape, (
        f"Attribute token shape mismatch. "
        f"Expected {expected_shape}, "
        f"got {tuple(tokens.shape)}"
    )

    assert isinstance(
        attribute_tokens.tokens,
        nn.Parameter,
    ), (
        "Attribute tokens must be nn.Parameter."
    )

    assert attribute_tokens.tokens.requires_grad, (
        "Attribute tokens must be learnable."
    )

    assert torch.isfinite(tokens).all(), (
        "Attribute tokens contain NaN or Inf values."
    )

    # --------------------------------------------------------
    # Check that all attribute tokens exist
    # --------------------------------------------------------

    assert (
        attribute_tokens.tokens.shape[1]
        == NUM_ATTRIBUTE_TOKENS
    )

    assert (
        attribute_tokens.tokens.shape[2]
        == EMBED_DIM
    )

    print("\nAttribute token validation: PASSED")

    print("=" * 70)