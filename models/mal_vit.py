"""
models/mal_vit.py

MAL-ViT
------

Morphology-Aware Learning Vision Transformer.

This module implements the main MAL-ViT backbone.

Architecture
------------

Input Image
    ↓
Patch Embedding
    ↓
196 Patch Tokens
    +
11 Attribute Tokens
    +
4 Register Tokens
    ↓
211 Tokens
    ↓
Positional Embedding
    ↓
Transformer Encoder × 6
    ↓
Final Normalized Tokens
    ├── Attribute Tokens → Attribute Heads
    ├── Register Tokens  → WBC Classifier
    └── Patch Tokens     → XAI / Grad-CAM

Responsibilities
----------------
- Convert images into patch tokens.
- Add learnable morphology attribute tokens.
- Add learnable register tokens.
- Add positional embeddings.
- Run the complete Transformer encoder.
- Separate attribute/register/patch features.
- Produce morphology attribute predictions.
- Optionally return attention matrices.

The WBC classification head is intentionally kept outside this
module and will be connected in models/complete_model.py.
"""

import torch
import torch.nn as nn

from config import (
    NUM_PATCHES,
    NUM_ATTRIBUTE_TOKENS,
    NUM_REGISTER_TOKENS,
    EMBED_DIM,
    EMBED_DROPOUT,
    INIT_STD,
)

from models.patch_embedding import PatchEmbedding
from models.attribute_tokens import AttributeTokens
from models.register_tokens import RegisterTokens
from models.transformer import VisionTransformer
from models.attribute_heads import AttributeHeads


class MALViT(nn.Module):
    """
    Morphology-Aware Learning Vision Transformer.

    This class represents the main MAL-ViT feature extractor.

    Outputs
    -------
    attribute_features
        Transformer representations of the 11 attribute tokens.

    register_features
        Transformer representations of the 4 register tokens.

    patch_features
        Transformer representations of the 196 image patch tokens.

    attribute_predictions
        Predictions from the 11 morphology attribute heads.

    attention
        Optional attention matrices from all Transformer blocks.
    """

    def __init__(self):

        super().__init__()

        # ==========================================================
        # Configuration Validation
        # ==========================================================

        self.num_patches = NUM_PATCHES

        self.num_attribute_tokens = (
            NUM_ATTRIBUTE_TOKENS
        )

        self.num_register_tokens = (
            NUM_REGISTER_TOKENS
        )

        self.embed_dim = EMBED_DIM

        self.num_tokens = (
            self.num_attribute_tokens
            + self.num_register_tokens
            + self.num_patches
        )

        # Expected:
        #
        # 11 attribute tokens
        # + 4 register tokens
        # + 196 patch tokens
        #
        # = 211

        # ==========================================================
        # Patch Embedding
        # ==========================================================

        self.patch_embedding = PatchEmbedding()

        # Output:
        #
        # (B, 3, 224, 224)
        #        ↓
        # (B, 196, 192)

        # ==========================================================
        # Attribute Tokens
        # ==========================================================

        self.attribute_tokens = AttributeTokens()

        # Output:
        #
        # (B, 11, 192)

        # ==========================================================
        # Register Tokens
        # ==========================================================

        self.register_tokens = RegisterTokens()

        # Output:
        #
        # (B, 4, 192)

        # ==========================================================
        # Positional Embedding
        # ==========================================================

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_tokens,
                self.embed_dim,
            )
        )

        # ==========================================================
        # Token Dropout
        # ==========================================================

        self.dropout = nn.Dropout(
            p=EMBED_DROPOUT
        )

        # ==========================================================
        # Transformer Encoder
        # ==========================================================

        self.transformer = VisionTransformer()

        # Output:
        #
        # (B, 211, 192)
        #
        # Attention:
        #
        # 6 × (B, 6, 211, 211)

        # ==========================================================
        # Attribute Prediction Heads
        # ==========================================================

        self.attribute_heads = AttributeHeads()

        # ==========================================================
        # Weight Initialization
        # ==========================================================

        self._initialize_weights()

    # ==============================================================
    # Weight Initialization
    # ==============================================================

    def _initialize_weights(self):
        """
        Initialize MAL-ViT parameters.

        Learnable positional embeddings use a small normal
        distribution controlled by INIT_STD.
        """

        nn.init.trunc_normal_(
            self.position_embedding,
            std=INIT_STD,
        )

    # ==============================================================
    # Forward
    # ==============================================================

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Forward pass through MAL-ViT.

        Parameters
        ----------
        images : torch.Tensor
            Input images.

            Shape:
                (B, 3, 224, 224)

        return_attention : bool
            If True, return attention matrices from every
            Transformer block.

        Returns
        -------
        dict
            Dictionary containing MAL-ViT representations and
            morphology predictions.
        """

        # ==========================================================
        # Input Validation
        # ==========================================================

        if images.ndim != 4:

            raise ValueError(
                "MALViT expects a 4D image tensor "
                "(B, C, H, W), "
                f"but received shape {tuple(images.shape)}."
            )

        batch_size = images.size(0)

        # ==========================================================
        # Patch Tokens
        # ==========================================================

        patch_tokens = self.patch_embedding(
            images
        )

        # Expected:
        #
        # (B, 196, 192)

        if patch_tokens.shape[1] != self.num_patches:

            raise RuntimeError(
                "Unexpected number of patch tokens. "
                f"Expected {self.num_patches}, "
                f"got {patch_tokens.shape[1]}."
            )

        # ==========================================================
        # Attribute Tokens
        # ==========================================================

        attribute_tokens = self.attribute_tokens(
            batch_size
        )

        # Expected:
        #
        # (B, 11, 192)

        # ==========================================================
        # Register Tokens
        # ==========================================================

        register_tokens = self.register_tokens(
            batch_size
        )

        # Expected:
        #
        # (B, 4, 192)

        # ==========================================================
        # Concatenate Tokens
        # ==========================================================

        tokens = torch.cat(
            [
                attribute_tokens,
                register_tokens,
                patch_tokens,
            ],
            dim=1,
        )

        # Expected:
        #
        # (B, 211, 192)

        if tokens.shape[1] != self.num_tokens:

            raise RuntimeError(
                "Unexpected total token count. "
                f"Expected {self.num_tokens}, "
                f"got {tokens.shape[1]}."
            )

        # ==========================================================
        # Positional Embedding
        # ==========================================================

        tokens = (
            tokens
            + self.position_embedding
        )

        # ==========================================================
        # Embedding Dropout
        # ==========================================================

        tokens = self.dropout(
            tokens
        )

        # ==========================================================
        # Transformer Encoder
        # ==========================================================

        transformer_output = self.transformer(
            tokens,
            return_attention=return_attention,
        )

        # VisionTransformer returns:
        #
        # if return_attention=False:
        #
        #     tokens
        #
        # if return_attention=True:
        #
        #     tokens, attention

        if return_attention:

            tokens, attention = (
                transformer_output
            )

        else:

            tokens = transformer_output

            attention = None

        # ==========================================================
        # Attribute Token Features
        # ==========================================================

        attribute_features = tokens[
            :,
            :self.num_attribute_tokens,
            :,
        ]

        # Expected:
        #
        # (B, 11, 192)

        # ==========================================================
        # Register Token Features
        # ==========================================================

        register_start = (
            self.num_attribute_tokens
        )

        register_end = (
            self.num_attribute_tokens
            + self.num_register_tokens
        )

        register_features = tokens[
            :,
            register_start:register_end,
            :,
        ]

        # Expected:
        #
        # (B, 4, 192)

        # ==========================================================
        # Patch Token Features
        # ==========================================================

        patch_features = tokens[
            :,
            register_end:,
            :,
        ]

        # Expected:
        #
        # (B, 196, 192)

        # ==========================================================
        # Attribute Predictions
        # ==========================================================

        attribute_predictions = (
            self.attribute_heads(
                attribute_features
            )
        )

        # ==========================================================
        # Output Dictionary
        # ==========================================================

        outputs = {

            # Complete Transformer representation
            "tokens": tokens,

            # Morphology token representations
            "attribute_features":
                attribute_features,

            # Global/contextual register representations
            "register_features":
                register_features,

            # Spatial image representations
            "patch_features":
                patch_features,

            # Eleven morphology predictions
            "attribute_predictions":
                attribute_predictions,
        }

        # ==========================================================
        # Optional Attention
        # ==========================================================

        if return_attention:

            outputs["attention"] = attention

        return outputs


# ==================================================================
# Quick Test
# ==================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Main Model Test")
    print("=" * 70)

    print("\nConfiguration")

    print(
        f"Image size       : 224 x 224"
    )

    print(
        f"Patch tokens     : {NUM_PATCHES}"
    )

    print(
        f"Attribute tokens : "
        f"{NUM_ATTRIBUTE_TOKENS}"
    )

    print(
        f"Register tokens  : "
        f"{NUM_REGISTER_TOKENS}"
    )

    print(
        f"Total tokens     : "
        f"{NUM_PATCHES + NUM_ATTRIBUTE_TOKENS + NUM_REGISTER_TOKENS}"
    )

    print(
        f"Embedding dim    : "
        f"{EMBED_DIM}"
    )

    # ==============================================================
    # Create Test Input
    # ==============================================================

    images = torch.randn(
        4,
        3,
        224,
        224,
    )

    print("\nInput shape:")
    print(images.shape)

    # ==============================================================
    # Create Model
    # ==============================================================

    model = MALViT()

    print("\nMAL-ViT:")
    print(model)

    # ==============================================================
    # Evaluation Mode
    # ==============================================================

    model.eval()

    # ==============================================================
    # Forward Pass
    # ==============================================================

    with torch.no_grad():

        outputs = model(
            images,
            return_attention=True,
        )

    # ==============================================================
    # Feature Shapes
    # ==============================================================

    print("\nOutput Shapes")

    print(
        "Complete tokens:",
        outputs["tokens"].shape,
    )

    print(
        "Attribute features:",
        outputs["attribute_features"].shape,
    )

    print(
        "Register features:",
        outputs["register_features"].shape,
    )

    print(
        "Patch features:",
        outputs["patch_features"].shape,
    )

    # ==============================================================
    # Expected Shapes
    # ==============================================================

    expected_tokens = (
        4,
        211,
        192,
    )

    expected_attributes = (
        4,
        11,
        192,
    )

    expected_registers = (
        4,
        4,
        192,
    )

    expected_patches = (
        4,
        196,
        192,
    )

    print("\nExpected Shapes")

    print(
        "Complete tokens:",
        expected_tokens,
    )

    print(
        "Attribute features:",
        expected_attributes,
    )

    print(
        "Register features:",
        expected_registers,
    )

    print(
        "Patch features:",
        expected_patches,
    )

    # ==============================================================
    # Validate Shapes
    # ==============================================================

    assert tuple(
        outputs["tokens"].shape
    ) == expected_tokens

    assert tuple(
        outputs["attribute_features"].shape
    ) == expected_attributes

    assert tuple(
        outputs["register_features"].shape
    ) == expected_registers

    assert tuple(
        outputs["patch_features"].shape
    ) == expected_patches

    # ==============================================================
    # Validate Attribute Predictions
    # ==============================================================

    print("\nAttribute Predictions")

    for name, prediction in (
        outputs[
            "attribute_predictions"
        ].items()
    ):

        print(
            f"  {name:30s}"
            f"{tuple(prediction.shape)}"
        )

    # ==============================================================
    # Validate Attention
    # ==============================================================

    attention = outputs.get(
        "attention"
    )

    assert attention is not None

    print(
        "\nNumber of attention matrices:",
        len(attention),
    )

    for index, matrix in enumerate(
        attention,
        start=1,
    ):

        print(
            f"  Block {index}: "
            f"{tuple(matrix.shape)}"
        )

        assert tuple(
            matrix.shape
        ) == (
            4,
            6,
            211,
            211,
        )

    # ==============================================================
    # Final Validation
    # ==============================================================

    print("\n" + "=" * 70)
    print("MAL-ViT validation: PASSED")
    print("=" * 70)