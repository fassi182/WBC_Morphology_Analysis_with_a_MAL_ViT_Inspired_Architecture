"""
attribute_tokens.py

Learnable Attribute Tokens used in MAL-ViT.

Instead of a CLS token, MAL-ViT introduces one learnable token
for each morphology attribute.

For the WBCAtt dataset:
11 morphology attributes
→ 11 learnable attribute tokens.


"""

import torch
import torch.nn as nn

from config import (
    NUM_ATTRIBUTE_TOKENS,
    EMBED_DIM,
)


class AttributeTokens(nn.Module):
    """
    Learnable Attribute Tokens.

    Initial Shape:
        (1, 11, 192)

    Forward Output:
        (B, 11, 192)
    """

    def __init__(
        self,
        num_tokens=NUM_ATTRIBUTE_TOKENS,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        self.attribute_tokens = nn.Parameter(
            torch.randn(
                1,
                num_tokens,
                embed_dim,
            )
        )

    def forward(self, batch_size):
        """
        Expand attribute tokens for a batch.

        Parameters
        ----------
        batch_size : int

        Returns
        -------
        Tensor
            Shape:
            (B, 11, 192)
        """

        return self.attribute_tokens.expand(
            batch_size,
            -1,
            -1,
        )


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Attribute Tokens Test")
    print("=" * 60)

    batch_size = 8

    model = AttributeTokens()

    tokens = model(batch_size)

    print("\nParameter Shape:")
    print(model.attribute_tokens.shape)

    print("\nExpanded Shape:")
    print(tokens.shape)

    print("\nExpected:")
    print("(8, 11, 192)")