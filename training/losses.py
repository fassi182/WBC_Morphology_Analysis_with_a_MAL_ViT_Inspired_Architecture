"""
MAL-ViT Loss Functions
----------------------

Multi-task loss for:

1. WBC classification
2. Morphological attribute classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_CLASS_COUNTS,
    NUM_WBC_CLASSES,
    WBC_LOSS_WEIGHT,
    ATTRIBUTE_LOSS_WEIGHT,
)


# ============================================================
# WBC Classification Loss
# ============================================================

def compute_wbc_loss(
    wbc_logits,
    wbc_targets,
):
    """
    Cross-entropy loss for WBC classification.

    wbc_logits:
        (B, 5)

    wbc_targets:
        (B,)
    """

    return F.cross_entropy(
        wbc_logits,
        wbc_targets,
    )


# ============================================================
# Attribute Classification Loss
# ============================================================

def compute_attribute_loss(
    attribute_predictions,
    attribute_targets,
):
    """
    Computes average cross-entropy loss
    across all 11 morphological attributes.

    attribute_predictions:
        Dictionary:
            attribute name -> (B, num_classes)

    attribute_targets:
        Tensor:
            (B, 11)
    """

    total_loss = 0.0

    for index, name in enumerate(
        ATTRIBUTE_NAMES
    ):

        logits = attribute_predictions[
            name
        ]

        targets = attribute_targets[
            :, index
        ]

        loss = F.cross_entropy(
            logits,
            targets,
        )

        total_loss += loss

    total_loss = (
        total_loss
        / len(ATTRIBUTE_NAMES)
    )

    return total_loss


# ============================================================
# Complete Multi-Task Loss
# ============================================================

def compute_total_loss(
    outputs,
    wbc_targets,
    attribute_targets,
    wbc_weight=WBC_LOSS_WEIGHT,
    attribute_weight=ATTRIBUTE_LOSS_WEIGHT,
):
    """
    Computes the complete MAL-ViT loss.

    Total Loss =
        WBC Loss
        +
        Attribute Loss

    Parameters
    ----------
    outputs:
        Output dictionary from CompleteMALViT.

    wbc_targets:
        Tensor of shape (B,)

    attribute_targets:
        Tensor of shape (B, 11)
    """

    wbc_logits = outputs[
        "wbc_logits"
    ]

    attribute_predictions = outputs[
        "attribute_predictions"
    ]

    # --------------------------------------------------------
    # WBC loss
    # --------------------------------------------------------

    wbc_loss = compute_wbc_loss(
        wbc_logits,
        wbc_targets,
    )

    # --------------------------------------------------------
    # Attribute loss
    # --------------------------------------------------------

    attribute_loss = (
        compute_attribute_loss(
            attribute_predictions,
            attribute_targets,
        )
    )

    # --------------------------------------------------------
    # Total loss
    # --------------------------------------------------------

    total_loss = (
        wbc_weight * wbc_loss
        +
        attribute_weight * attribute_loss
    )

    return {
        "total_loss": total_loss,
        "wbc_loss": wbc_loss,
        "attribute_loss": attribute_loss,
    }


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Loss Test")
    print("=" * 60)

    batch_size = 4

    # ========================================================
    # WBC
    # ========================================================

    wbc_logits = torch.randn(
        batch_size,
        NUM_WBC_CLASSES,
    )

    wbc_targets = torch.randint(
        0,
        NUM_WBC_CLASSES,
        (batch_size,),
    )

    # ========================================================
    # Attribute class counts
    #
    # Use the values directly from the configuration dictionary.
    # ========================================================

    attribute_class_counts = {
        "cell_size": 2,
        "cell_shape": 2,
        "nucleus_shape": 6,
        "nuclear_cytoplasmic_ratio": 2,
        "chromatin_density": 2,
        "cytoplasm_vacuole": 2,
        "cytoplasm_texture": 2,
        "cytoplasm_colour": 3,
        "granule_type": 4,
        "granule_colour": 4,
        "granularity": 2,
    }

    # ========================================================
    # Attribute Predictions
    # ========================================================

    attribute_predictions = {}

    for name in ATTRIBUTE_NAMES:

        num_classes = attribute_class_counts[
            name
        ]

        attribute_predictions[name] = torch.randn(
            batch_size,
            num_classes,
        )

    # ========================================================
    # Attribute Targets
    # ========================================================

    attribute_targets = torch.stack(
        [
            torch.randint(
                0,
                attribute_class_counts[name],
                (batch_size,),
            )
            for name in ATTRIBUTE_NAMES
        ],
        dim=1,
    )

    # ========================================================
    # Complete Outputs
    # ========================================================

    outputs = {
        "wbc_logits": wbc_logits,
        "attribute_predictions":
            attribute_predictions,
    }

    # ========================================================
    # Compute Loss
    # ========================================================

    losses = compute_total_loss(
        outputs=outputs,
        wbc_targets=wbc_targets,
        attribute_targets=attribute_targets,
    )

    # ========================================================
    # Results
    # ========================================================

    print("\nTotal loss:")
    print(
        losses["total_loss"].item()
    )

    print("\nWBC loss:")
    print(
        losses["wbc_loss"].item()
    )

    print("\nAttribute loss:")
    print(
        losses["attribute_loss"].item()
    )

    # ========================================================
    # Validation
    # ========================================================

    assert torch.isfinite(
        losses["total_loss"]
    )

    assert torch.isfinite(
        losses["wbc_loss"]
    )

    assert torch.isfinite(
        losses["attribute_loss"]
    )

    print(
        "\nLoss validation: PASSED"
    )
