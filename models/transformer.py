"""
transformer.py

MAL-ViT Transformer Encoder.

Pipeline:

Input Tokens
    │
    ├── Attribute Tokens
    ├── Register Tokens
    └── Patch Tokens
    │
    ▼
Transformer Encoder Blocks
    │
    ▼
Final LayerNorm
    │
    ▼
Output Tokens

The transformer preserves attention weights from every block
so that they can later be used for:

    - Attention-sink diagnostics
    - Attention visualization
    - XAI
    - Morphology localization
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_TRANSFORMER_BLOCKS,
    TOTAL_TOKENS,
    LAYER_NORM_EPS,
)

from models.encoder_block import EncoderBlock


class VisionTransformer(nn.Module):
    """
    MAL-ViT Transformer Encoder.

    Input:
        (B, TOTAL_TOKENS, EMBED_DIM)

    Output:
        tokens:
            (B, TOTAL_TOKENS, EMBED_DIM)

        attentions:
            list containing attention matrices from every block.

            Each attention matrix:
                (B, NUM_HEADS, TOTAL_TOKENS, TOTAL_TOKENS)
    """

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Transformer Blocks
        # --------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                EncoderBlock()
                for _ in range(NUM_TRANSFORMER_BLOCKS)
            ]
        )

        # --------------------------------------------------
        # Final Layer Normalization
        # --------------------------------------------------

        self.norm = nn.LayerNorm(
            EMBED_DIM,
            eps=LAYER_NORM_EPS,
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        x,
        return_attention=True,
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape:
                (B, TOTAL_TOKENS, EMBED_DIM)

        return_attention : bool
            If True, attention weights from every transformer
            block are returned.

        Returns
        -------
        x : torch.Tensor
            Final transformer representation.

        attentions : list or None
            Attention weights from each transformer block.
        """

        # --------------------------------------------------
        # Validate Input
        # --------------------------------------------------

        if x.ndim != 3:

            raise ValueError(
                "Transformer expected a 3D tensor "
                "(B, tokens, embedding_dim), "
                f"but received shape {tuple(x.shape)}"
            )

        if x.size(1) != TOTAL_TOKENS:

            raise ValueError(
                f"Expected {TOTAL_TOKENS} tokens, "
                f"but received {x.size(1)}"
            )

        if x.size(2) != EMBED_DIM:

            raise ValueError(
                f"Expected embedding dimension {EMBED_DIM}, "
                f"but received {x.size(2)}"
            )

        # --------------------------------------------------
        # Store Attention
        # --------------------------------------------------

        attentions = []

        # --------------------------------------------------
        # Transformer Encoder
        # --------------------------------------------------

        for block in self.blocks:

            if return_attention:

                x, attention = block(
                    x,
                    return_attention=True,
                )

                attentions.append(attention)

            else:

                x = block(
                    x,
                    return_attention=False,
                )

        # --------------------------------------------------
        # Final LayerNorm
        # --------------------------------------------------

        x = self.norm(x)

        # --------------------------------------------------
        # Return
        # --------------------------------------------------

        if return_attention:

            return x, attentions

        return x


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Transformer Encoder Test")
    print("=" * 70)

    print("\nConfiguration")
    print(f"Embedding dimension : {EMBED_DIM}")
    print(f"Transformer blocks  : {NUM_TRANSFORMER_BLOCKS}")
    print(f"Total tokens        : {TOTAL_TOKENS}")

    # ------------------------------------------------------
    # Create Model
    # ------------------------------------------------------

    transformer = VisionTransformer()

    print("\nTransformer:")
    print(transformer)

    # ------------------------------------------------------
    # Dummy Input
    # ------------------------------------------------------

    batch_size = 4

    x = torch.randn(
        batch_size,
        TOTAL_TOKENS,
        EMBED_DIM,
    )

    print("\nInput shape:")
    print(x.shape)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    output, attentions = transformer(
        x,
        return_attention=True,
    )

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    print("\nOutput shape:")
    print(output.shape)

    print("\nExpected output shape:")
    print(
        (
            batch_size,
            TOTAL_TOKENS,
            EMBED_DIM,
        )
    )

    # ------------------------------------------------------
    # Attention Information
    # ------------------------------------------------------

    print("\nNumber of attention matrices:")
    print(len(attentions))

    print("\nAttention shapes:")

    for index, attention in enumerate(attentions):

        print(
            f"  Block {index + 1}: "
            f"{tuple(attention.shape)}"
        )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    assert output.shape == (
        batch_size,
        TOTAL_TOKENS,
        EMBED_DIM,
    )

    assert len(attentions) == NUM_TRANSFORMER_BLOCKS

    for attention in attentions:

        assert attention.ndim == 4

        assert attention.shape[0] == batch_size

        assert attention.shape[2] == TOTAL_TOKENS

        assert attention.shape[3] == TOTAL_TOKENS

    print("\nTransformer validation: PASSED")

    print("=" * 70)