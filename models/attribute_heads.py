"""
attribute_heads.py

Attribute classification heads for MAL-ViT.

Each morphology attribute has its own classification head.

Input:
    Attribute token features
    (B, 11, 192)

Output:
    Dictionary containing logits for each attribute.

Example:
    cell_size        -> (B, 2)
    cell_shape       -> (B, 2)
    nucleus_shape    -> (B, 6)
    cytoplasm_colour -> (B, 3)
    granule_type     -> (B, 4)
    ...

The number of output classes is determined by the WBCAtt
attribute configuration.
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_ATTRIBUTE_TOKENS,
    ATTRIBUTE_NAMES,
    NUM_ATTRIBUTES,
)

from data.encoders import ATTRIBUTE_ENCODERS


class AttributeHeads(nn.Module):
    """
    Collection of independent classification heads.

    One head corresponds to one morphology attribute/token.
    """

    def __init__(self):

        super().__init__()

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        if NUM_ATTRIBUTES != NUM_ATTRIBUTE_TOKENS:

            raise ValueError(
                "Number of attributes and attribute tokens "
                "must be identical. "
                f"NUM_ATTRIBUTES={NUM_ATTRIBUTES}, "
                f"NUM_ATTRIBUTE_TOKENS={NUM_ATTRIBUTE_TOKENS}"
            )

        # --------------------------------------------------
        # Create One Head Per Attribute
        # --------------------------------------------------

        self.heads = nn.ModuleDict()

        for attribute in ATTRIBUTE_NAMES:

            if attribute not in ATTRIBUTE_ENCODERS:

                raise KeyError(
                    f"Missing encoder for attribute: {attribute}"
                )

            num_classes = len(
                ATTRIBUTE_ENCODERS[attribute]
            )

            self.heads[attribute] = nn.Linear(
                EMBED_DIM,
                num_classes,
            )

    # ======================================================
    # Forward
    # ======================================================

    def forward(self, attribute_features):
        """
        Parameters
        ----------
        attribute_features : torch.Tensor

            Shape:
                (B, NUM_ATTRIBUTES, EMBED_DIM)

        Returns
        -------
        predictions : dict[str, torch.Tensor]

            Each dictionary value contains logits.

            Example:

                {
                    "cell_size": (B, 2),
                    "cell_shape": (B, 2),
                    "nucleus_shape": (B, 6),
                    ...
                }
        """

        # --------------------------------------------------
        # Validate Input
        # --------------------------------------------------

        if attribute_features.ndim != 3:

            raise ValueError(
                "Attribute features must have shape "
                "(B, attributes, embedding_dim). "
                f"Received: "
                f"{tuple(attribute_features.shape)}"
            )

        if attribute_features.size(1) != NUM_ATTRIBUTE_TOKENS:

            raise ValueError(
                f"Expected {NUM_ATTRIBUTE_TOKENS} "
                f"attribute tokens, "
                f"received {attribute_features.size(1)}"
            )

        if attribute_features.size(2) != EMBED_DIM:

            raise ValueError(
                f"Expected embedding dimension "
                f"{EMBED_DIM}, "
                f"received {attribute_features.size(2)}"
            )

        # --------------------------------------------------
        # Generate Predictions
        # --------------------------------------------------

        predictions = {}

        for index, attribute in enumerate(
            ATTRIBUTE_NAMES
        ):

            token = attribute_features[
                :,
                index,
                :,
            ]

            predictions[attribute] = self.heads[
                attribute
            ](token)

        return predictions


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Attribute Heads Test")
    print("=" * 70)

    print("\nConfiguration")
    print(
        f"Number of attributes : "
        f"{NUM_ATTRIBUTES}"
    )

    print(
        f"Attribute tokens     : "
        f"{NUM_ATTRIBUTE_TOKENS}"
    )

    print(
        f"Embedding dimension  : "
        f"{EMBED_DIM}"
    )

    # ------------------------------------------------------
    # Create Model
    # ------------------------------------------------------

    heads = AttributeHeads()

    print("\nAttribute Heads:")
    print(heads)

    # ------------------------------------------------------
    # Dummy Input
    # ------------------------------------------------------

    batch_size = 4

    attribute_features = torch.randn(
        batch_size,
        NUM_ATTRIBUTE_TOKENS,
        EMBED_DIM,
    )

    print("\nInput shape:")
    print(attribute_features.shape)

    # ------------------------------------------------------
    # Forward
    # ------------------------------------------------------

    predictions = heads(
        attribute_features
    )

    # ------------------------------------------------------
    # Print Predictions
    # ------------------------------------------------------

    print("\nAttribute predictions:")

    for attribute in ATTRIBUTE_NAMES:

        logits = predictions[attribute]

        num_classes = len(
            ATTRIBUTE_ENCODERS[attribute]
        )

        print(
            f"  {attribute:30} "
            f"{tuple(logits.shape)} "
            f"expected="
            f"({batch_size}, {num_classes})"
        )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    assert len(predictions) == NUM_ATTRIBUTES

    for attribute in ATTRIBUTE_NAMES:

        logits = predictions[attribute]

        expected_classes = len(
            ATTRIBUTE_ENCODERS[attribute]
        )

        assert logits.shape == (
            batch_size,
            expected_classes,
        )

    print("\nAttribute heads validation: PASSED")

    print("=" * 70)