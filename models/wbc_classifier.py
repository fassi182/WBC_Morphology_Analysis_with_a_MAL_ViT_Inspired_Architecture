"""
wbc_classifier.py

WBC classification head for MAL-ViT.

The model uses the register tokens as a compact representation
of global/contextual information for the final WBC classification.

Input:
    Register token features
    (B, NUM_REGISTER_TOKENS, EMBED_DIM)

The register tokens are pooled and passed through a small
classification head.

Output:
    WBC logits
    (B, NUM_WBC_CLASSES)

Classes:
    0 - Neutrophil
    1 - Eosinophil
    2 - Monocyte
    3 - Basophil
    4 - Lymphocyte
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_REGISTER_TOKENS,
    NUM_WBC_CLASSES,
    WBC_CLASSES,
)


class WBCClassifier(nn.Module):
    """
    Global WBC classification head.

    Register tokens are used instead of an individual patch token
    because register tokens are specifically intended to store
    global/contextual information.
    """

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Register Token Pooling
        # --------------------------------------------------

        self.register_pool = nn.AdaptiveAvgPool1d(1)

        # --------------------------------------------------
        # Classification Head
        # --------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                EMBED_DIM
            ),

            nn.Linear(
                EMBED_DIM,
                EMBED_DIM,
            ),

            nn.GELU(),

            nn.Dropout(
                0.1
            ),

            nn.Linear(
                EMBED_DIM,
                NUM_WBC_CLASSES,
            ),
        )

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, register_features):
        """
        Parameters
        ----------
        register_features : torch.Tensor

            Shape:
                (B, NUM_REGISTER_TOKENS, EMBED_DIM)

        Returns
        -------
        logits : torch.Tensor

            Shape:
                (B, NUM_WBC_CLASSES)
        """

        # --------------------------------------------------
        # Validate Input
        # --------------------------------------------------

        if register_features.ndim != 3:

            raise ValueError(
                "Register features must have shape "
                "(B, register_tokens, embedding_dim). "
                f"Received: "
                f"{tuple(register_features.shape)}"
            )

        if register_features.size(1) != NUM_REGISTER_TOKENS:

            raise ValueError(
                f"Expected {NUM_REGISTER_TOKENS} "
                f"register tokens, "
                f"received {register_features.size(1)}"
            )

        if register_features.size(2) != EMBED_DIM:

            raise ValueError(
                f"Expected embedding dimension "
                f"{EMBED_DIM}, "
                f"received {register_features.size(2)}"
            )

        # --------------------------------------------------
        # Pool Register Tokens
        # --------------------------------------------------

        # (B, R, D)
        #
        # AdaptiveAvgPool1d expects:
        # (B, D, R)

        pooled = register_features.transpose(
            1,
            2,
        )

        # (B, D, R)
        pooled = self.register_pool(
            pooled
        )

        # (B, D, 1)
        pooled = pooled.squeeze(
            -1
        )

        # (B, D)

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        logits = self.classifier(
            pooled
        )

        return logits


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT WBC Classifier Test")
    print("=" * 70)

    print("\nConfiguration")

    print(
        f"Number of WBC classes : "
        f"{NUM_WBC_CLASSES}"
    )

    print(
        f"Register tokens       : "
        f"{NUM_REGISTER_TOKENS}"
    )

    print(
        f"Embedding dimension   : "
        f"{EMBED_DIM}"
    )

    print("\nWBC Classes:")

    for index, name in enumerate(
        WBC_CLASSES
    ):

        print(
            f"  {index}: {name}"
        )

    # ------------------------------------------------------
    # Create Model
    # ------------------------------------------------------

    classifier = WBCClassifier()

    print("\nClassifier:")
    print(classifier)

    # ------------------------------------------------------
    # Dummy Input
    # ------------------------------------------------------

    batch_size = 4

    register_features = torch.randn(
        batch_size,
        NUM_REGISTER_TOKENS,
        EMBED_DIM,
    )

    print("\nInput shape:")
    print(
        register_features.shape
    )

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    logits = classifier(
        register_features
    )

    print("\nOutput shape:")
    print(
        logits.shape
    )

    print("\nExpected shape:")
    print(
        (
            batch_size,
            NUM_WBC_CLASSES,
        )
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    assert logits.shape == (
        batch_size,
        NUM_WBC_CLASSES,
    )

    assert torch.isfinite(
        logits
    ).all()

    print(
        "\nWBC classifier validation: PASSED"
    )

    print("=" * 70)