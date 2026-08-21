"""
register_tokens.py

Learnable Register Tokens for MAL-ViT.

Register tokens are free latent tokens. They do not correspond
to any morphology attribute and have no prediction head.

Their purpose is to provide the transformer with unconstrained
latent storage for global/contextual information.

Current configuration:

    Register tokens : 4
    Embedding dim   : 192

Output:

    (B, 4, 192)
"""

import torch
import torch.nn as nn

from config import (
    NUM_REGISTER_TOKENS,
    EMBED_DIM,
    INIT_STD,
)


# ============================================================
# Register Tokens
# ============================================================

class RegisterTokens(nn.Module):
    """
    Learnable register tokens.

    Unlike attribute tokens, these tokens have no direct
    prediction objective. They are free latent representations
    available to the transformer.
    """

    def __init__(
        self,
        num_tokens=NUM_REGISTER_TOKENS,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # ----------------------------------------------------
        # Learnable register tokens
        # ----------------------------------------------------
        #
        # Shape:
        #
        #     (1, 4, 192)
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
        Expand register tokens for the current batch.

        Parameters
        ----------
        batch_size : int
            Number of images in the batch.

        Returns
        -------
        torch.Tensor
            Shape:

                (B, 4, 192)
        """

        if not isinstance(batch_size, int):
            raise TypeError(
                "batch_size must be an integer."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

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
    print("MAL-ViT Register Tokens Test")
    print("=" * 70)

    print("\nConfiguration")

    print(
        f"Number of register tokens : "
        f"{NUM_REGISTER_TOKENS}"
    )

    print(
        f"Embedding dimension       : "
        f"{EMBED_DIM}"
    )

    # --------------------------------------------------------
    # Create module
    # --------------------------------------------------------

    register_tokens = RegisterTokens()

    print("\nRegister Token Module:")
    print(register_tokens)

    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

    batch_size = 4

    tokens = register_tokens(batch_size)

    print("\nOutput shape:")
    print(tokens.shape)

    expected_shape = (
        batch_size,
        NUM_REGISTER_TOKENS,
        EMBED_DIM,
    )

    print("\nExpected shape:")
    print(expected_shape)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert tokens.shape == expected_shape, (
        f"Register token shape mismatch. "
        f"Expected {expected_shape}, "
        f"got {tuple(tokens.shape)}"
    )

    assert isinstance(
        register_tokens.tokens,
        nn.Parameter,
    ), (
        "Register tokens must be nn.Parameter."
    )

    assert register_tokens.tokens.requires_grad, (
        "Register tokens must be learnable."
    )

    assert torch.isfinite(tokens).all(), (
        "Register tokens contain NaN or Inf values."
    )

    assert (
        register_tokens.tokens.shape[1]
        == NUM_REGISTER_TOKENS
    )

    assert (
        register_tokens.tokens.shape[2]
        == EMBED_DIM
    )

    print("\nRegister token validation: PASSED")

    print("=" * 70)