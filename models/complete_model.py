"""
complete_model.py

Complete MAL-ViT model.

Architecture:

Image
  ↓
Patch Embedding
  ↓
196 Patch Tokens
  +
11 Attribute Tokens
  +
4 Register Tokens
  ↓
Positional Embedding
  ↓
Transformer Encoder
  ↓
 ┌───────────────────────┬──────────────────────┐
 │                       │                      │
 Attribute Tokens    Register Tokens       Patch Tokens
 │                       │
 ↓                       ↓
Attribute Heads      WBC Classifier
 │                       │
 ↓                       ↓
Morphology          WBC Classification
Predictions
"""

import torch
import torch.nn as nn

from config import (
    NUM_PATCHES,
    NUM_ATTRIBUTE_TOKENS,
    NUM_REGISTER_TOKENS,
    EMBED_DIM,
    EMBED_DROPOUT,
    NUM_WBC_CLASSES,
)

from models.patch_embedding import PatchEmbedding
from models.attribute_tokens import AttributeTokens
from models.register_tokens import RegisterTokens
from models.transformer import VisionTransformer
from models.attribute_heads import AttributeHeads
from models.wbc_classifier import WBCClassifier


class CompleteMALViT(nn.Module):
    """
    Complete MAL-ViT model.

    Performs:

    1. WBC classification
    2. Morphology attribute prediction
    3. Feature extraction
    4. Attention extraction
    """

    def __init__(self):

        super().__init__()

        # ==================================================
        # Patch Embedding
        # ==================================================

        self.patch_embedding = PatchEmbedding()

        # ==================================================
        # Attribute Tokens
        # ==================================================

        self.attribute_tokens = AttributeTokens()

        # ==================================================
        # Register Tokens
        # ==================================================

        self.register_tokens = RegisterTokens()

        # ==================================================
        # Total Tokens
        # ==================================================

        self.num_tokens = (
            NUM_ATTRIBUTE_TOKENS
            + NUM_REGISTER_TOKENS
            + NUM_PATCHES
        )

        # ==================================================
        # Positional Embedding
        # ==================================================

        self.position_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_tokens,
                EMBED_DIM,
            ) * 0.02
        )

        # ==================================================
        # Embedding Dropout
        # ==================================================

        self.dropout = nn.Dropout(
            EMBED_DROPOUT
        )

        # ==================================================
        # Transformer
        # ==================================================

        self.transformer = VisionTransformer()

        # ==================================================
        # Attribute Heads
        # ==================================================

        self.attribute_heads = AttributeHeads()

        # ==================================================
        # WBC Classifier
        # ==================================================

        self.wbc_classifier = WBCClassifier()

    # ======================================================
    # Forward
    # ======================================================

    def forward(
        self,
        images,
        return_attention=False,
    ):
        """
        Forward pass.

        Parameters
        ----------
        images : torch.Tensor
            Shape:

                (B, 3, 224, 224)

        return_attention : bool
            Whether transformer attention matrices
            should be returned.

        Returns
        -------
        dict
            Complete model outputs.
        """

        batch_size = images.size(0)

        # ==================================================
        # 1. PATCH TOKENS
        # ==================================================

        patch_tokens = self.patch_embedding(
            images
        )

        # Expected:
        #
        # (B, 196, 192)

        # ==================================================
        # 2. ATTRIBUTE TOKENS
        # ==================================================

        attribute_tokens = self.attribute_tokens(
            batch_size
        )

        # Expected:
        #
        # (B, 11, 192)

        # ==================================================
        # 3. REGISTER TOKENS
        # ==================================================

        register_tokens = self.register_tokens(
            batch_size
        )

        # Expected:
        #
        # (B, 4, 192)

        # ==================================================
        # 4. CONCATENATE TOKENS
        # ==================================================

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

        # ==================================================
        # 5. POSITIONAL EMBEDDING
        # ==================================================

        tokens = (
            tokens
            + self.position_embedding
        )

        tokens = self.dropout(tokens)

        # ==================================================
        # 6. TRANSFORMER
        # ==================================================

        if return_attention:

            transformer_output = self.transformer(
                tokens,
                return_attention=True,
            )

            # VisionTransformer returns:
            #
            # (tokens, attention)

            tokens, attention = transformer_output

        else:

            transformer_output = self.transformer(
                tokens
            )

            # Depending on the current transformer
            # implementation, normal forward may return
            # either tokens or (tokens, attention).

            if isinstance(
                transformer_output,
                tuple
            ):

                tokens = transformer_output[0]

                attention = transformer_output[1]

            else:

                tokens = transformer_output

                attention = None

        # ==================================================
        # 7. ATTRIBUTE FEATURES
        # ==================================================

        attribute_features = tokens[
            :,
            :NUM_ATTRIBUTE_TOKENS,
            :
        ]

        # Expected:
        #
        # (B, 11, 192)

        # ==================================================
        # 8. REGISTER FEATURES
        # ==================================================

        register_start = NUM_ATTRIBUTE_TOKENS

        register_end = (
            NUM_ATTRIBUTE_TOKENS
            + NUM_REGISTER_TOKENS
        )

        register_features = tokens[
            :,
            register_start:register_end,
            :
        ]

        # Expected:
        #
        # (B, 4, 192)

        # ==================================================
        # 9. PATCH FEATURES
        # ==================================================

        patch_features = tokens[
            :,
            register_end:,
            :
        ]

        # Expected:
        #
        # (B, 196, 192)

        # ==================================================
        # 10. ATTRIBUTE PREDICTIONS
        # ==================================================

        attribute_predictions = (
            self.attribute_heads(
                attribute_features
            )
        )

        # ==================================================
        # 11. WBC CLASSIFICATION
        # ==================================================

        wbc_logits = self.wbc_classifier(
            register_features
        )

        # Expected:
        #
        # (B, 5)

        # ==================================================
        # 12. WBC PREDICTION
        # ==================================================

        wbc_prediction = torch.argmax(
            wbc_logits,
            dim=1,
        )

        # Expected:
        #
        # (B,)

        # ==================================================
        # 13. OUTPUT DICTIONARY
        # ==================================================

        outputs = {

            "wbc_logits":
                wbc_logits,

            "wbc_prediction":
                wbc_prediction,

            "attribute_predictions":
                attribute_predictions,

            "attribute_features":
                attribute_features,

            "register_features":
                register_features,

            "patch_features":
                patch_features,

            "tokens":
                tokens,
        }

        # ==================================================
        # 14. ATTENTION
        # ==================================================

        if return_attention:

            outputs["attention"] = attention

        return outputs


# ==========================================================
# QUICK TEST
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Complete MAL-ViT Model Test")
    print("=" * 70)

    # ======================================================
    # Configuration
    # ======================================================

    from config import (
        IMAGE_SIZE,
        NUM_PATCHES,
        NUM_ATTRIBUTE_TOKENS,
        NUM_REGISTER_TOKENS,
        TOTAL_TOKENS,
        EMBED_DIM,
        NUM_WBC_CLASSES,
    )

    print("\nConfiguration")

    print(
        f"Image size       : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Patch tokens     : "
        f"{NUM_PATCHES}"
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
        f"{TOTAL_TOKENS}"
    )

    print(
        f"Embedding dim    : "
        f"{EMBED_DIM}"
    )

    print(
        f"WBC classes      : "
        f"{NUM_WBC_CLASSES}"
    )

    # ======================================================
    # Model
    # ======================================================

    model = CompleteMALViT()

    print("\nComplete MAL-ViT:")
    print(model)

    # ======================================================
    # Parameter Count
    # ======================================================

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ======================================================
    # Input
    # ======================================================

    images = torch.randn(
        4,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    print("\nInput shape:")
    print(images.shape)

    # ======================================================
    # Forward
    # ======================================================

    outputs = model(
        images,
        return_attention=True,
    )

    # ======================================================
    # Output Shapes
    # ======================================================

    print("\nOutput Shapes")

    print(
        "WBC logits:",
        outputs["wbc_logits"].shape
    )

    print(
        "WBC prediction:",
        outputs["wbc_prediction"].shape
    )

    print(
        "Attribute features:",
        outputs["attribute_features"].shape
    )

    print(
        "Register features:",
        outputs["register_features"].shape
    )

    print(
        "Patch features:",
        outputs["patch_features"].shape
    )

    print(
        "Complete tokens:",
        outputs["tokens"].shape
    )

    # ======================================================
    # Expected Shapes
    # ======================================================

    print("\nExpected Shapes")

    print(
        "WBC logits:        ",
        f"(4, {NUM_WBC_CLASSES})"
    )

    print(
        "WBC prediction:    ",
        "(4,)"
    )

    print(
        "Attribute features:",
        f"(4, {NUM_ATTRIBUTE_TOKENS}, {EMBED_DIM})"
    )

    print(
        "Register features: ",
        f"(4, {NUM_REGISTER_TOKENS}, {EMBED_DIM})"
    )

    print(
        "Patch features:    ",
        f"(4, {NUM_PATCHES}, {EMBED_DIM})"
    )

    print(
        "Complete tokens:   ",
        f"(4, {TOTAL_TOKENS}, {EMBED_DIM})"
    )

    # ======================================================
    # Attribute Predictions
    # ======================================================

    print("\nAttribute Predictions")

    for (
        name,
        prediction
    ) in outputs[
        "attribute_predictions"
    ].items():

        print(
            f"  {name:30s}"
            f"{tuple(prediction.shape)}"
        )

    # ======================================================
    # Attention
    # ======================================================

    attention = outputs.get(
        "attention"
    )

    if attention is not None:

        print(
            "\nNumber of attention matrices:",
            len(attention)
        )

        for index, matrix in enumerate(
            attention
        ):

            print(
                f"  Block {index + 1}: "
                f"{tuple(matrix.shape)}"
            )

    # ======================================================
    # Validation
    # ======================================================

    assert outputs[
        "wbc_logits"
    ].shape == (
        4,
        NUM_WBC_CLASSES,
    )

    assert outputs[
        "wbc_prediction"
    ].shape == (4,)

    assert outputs[
        "attribute_features"
    ].shape == (
        4,
        NUM_ATTRIBUTE_TOKENS,
        EMBED_DIM,
    )

    assert outputs[
        "register_features"
    ].shape == (
        4,
        NUM_REGISTER_TOKENS,
        EMBED_DIM,
    )

    assert outputs[
        "patch_features"
    ].shape == (
        4,
        NUM_PATCHES,
        EMBED_DIM,
    )

    assert outputs[
        "tokens"
    ].shape == (
        4,
        TOTAL_TOKENS,
        EMBED_DIM,
    )

    assert attention is not None

    assert len(attention) == 6

    print(
        "\nComplete MAL-ViT validation: PASSED"
    )

    print("=" * 70)