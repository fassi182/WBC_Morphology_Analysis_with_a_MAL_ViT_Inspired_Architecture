"""
losses.py

Loss functions 

Computes

1. Morphology Attribute Loss
2. WBC Classification Loss
3. Combined Multi-Task Loss

"""

import torch
import torch.nn as nn

from config import (
    ATTRIBUTE_CLASSES,
    ATTRIBUTE_LOSS_WEIGHT,
    WBC_LOSS_WEIGHT,
)


class ExplainableWBCLoss(nn.Module):
    """
    Joint loss for morphology prediction and
    WBC classification.

    Total Loss

        L = λ_attr * L_attr + λ_wbc * L_wbc
    """

    def __init__(
        self,
        attribute_weight=ATTRIBUTE_LOSS_WEIGHT,
        wbc_weight=WBC_LOSS_WEIGHT,
    ):
        super().__init__()

        self.attribute_weight = attribute_weight
        self.wbc_weight = wbc_weight

        self.cross_entropy = nn.CrossEntropyLoss()

        self.attribute_names = list(
            ATTRIBUTE_CLASSES.keys()
        )

    def forward(
        self,
        outputs,
        targets,
    ):
        """
        Parameters
        ----------
        outputs : dict

            {
                attribute_predictions,
                wbc_logits
            }

        targets : dict

            {
                attributes,
                cell_label
            }

        Returns
        -------
        dict
        """

        attribute_predictions = outputs["attribute_predictions"]
        attribute_targets = targets["attributes"]

        wbc_logits = outputs["wbc_logits"]
        wbc_targets = targets["cell_label"]

        # --------------------------------------------------
        # Attribute Loss
        # --------------------------------------------------

        attribute_loss = 0.0

        for idx, attribute_name in enumerate(self.attribute_names):

            prediction = attribute_predictions[attribute_name]

            target = attribute_targets[:, idx]

            attribute_loss += self.cross_entropy(
                prediction,
                target,
            )

        attribute_loss /= len(self.attribute_names)

        # --------------------------------------------------
        # WBC Classification Loss
        # --------------------------------------------------

        wbc_loss = self.cross_entropy(
            wbc_logits,
            wbc_targets,
        )

        # --------------------------------------------------
        # Total Loss
        # --------------------------------------------------

        total_loss = (

            self.attribute_weight * attribute_loss

            +

            self.wbc_weight * wbc_loss

        )

        return {

            "total_loss": total_loss,

            "attribute_loss": attribute_loss,

            "wbc_loss": wbc_loss,

        }


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    batch_size = 4

    outputs = {

        "attribute_predictions": {

            "cell_size": torch.randn(batch_size, 2),

            "cell_shape": torch.randn(batch_size, 2),

            "nucleus_shape": torch.randn(batch_size, 6),

            "nuclear_cytoplasmic_ratio": torch.randn(batch_size, 2),

            "chromatin_density": torch.randn(batch_size, 2),

            "cytoplasm_vacuole": torch.randn(batch_size, 2),

            "cytoplasm_texture": torch.randn(batch_size, 2),

            "cytoplasm_colour": torch.randn(batch_size, 3),

            "granule_type": torch.randn(batch_size, 4),

            "granule_colour": torch.randn(batch_size, 4),

            "granularity": torch.randn(batch_size, 2),

        },

        "wbc_logits": torch.randn(batch_size, 8),

    }

    targets = {

        "attributes": torch.randint(
            0,
            2,
            (batch_size, 11),
        ),

        "cell_label": torch.randint(
            0,
            8,
            (batch_size,),
        ),

    }

    # Multi-class attribute targets

    targets["attributes"][:, 2] = torch.randint(
        0, 6, (batch_size,)
    )

    targets["attributes"][:, 7] = torch.randint(
        0, 3, (batch_size,)
    )

    targets["attributes"][:, 8] = torch.randint(
        0, 4, (batch_size,)
    )

    targets["attributes"][:, 9] = torch.randint(
        0, 4, (batch_size,)
    )

    criterion = ExplainableWBCLoss()

    losses = criterion(outputs, targets)

    print("=" * 60)
    print("Explainable WBC Loss Test")
    print("=" * 60)

    print(f"Attribute Weight : {criterion.attribute_weight}")
    print(f"WBC Weight       : {criterion.wbc_weight}")

    print()

    for name, value in losses.items():
        print(f"{name:20}: {value.item():.4f}")