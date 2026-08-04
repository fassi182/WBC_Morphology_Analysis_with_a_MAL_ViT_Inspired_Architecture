"""
losses.py

Loss functions for Explainable WBC Classification using MAL-ViT.

This module computes:

1. Multi-Attribute Classification Loss
2. WBC Classification Loss
3. Combined Training Loss

Author: Muhammad Fassi Ur Rehman
"""

import torch
import torch.nn as nn

from config import ATTRIBUTE_CLASSES


class ExplainableWBCLoss(nn.Module):
    """
    Joint loss for attribute prediction and WBC classification.
    """

    def __init__(
        self,
        attribute_weight=1.0,
        wbc_weight=1.0,
    ):
        super().__init__()

        self.attribute_weight = attribute_weight
        self.wbc_weight = wbc_weight

        self.criterion = nn.CrossEntropyLoss()

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

            Model outputs.

        targets : dict

            {
                "attributes": Tensor(B,11),
                "cell_label": Tensor(B)
            }

        Returns
        -------
        dict
        """

        attribute_predictions = outputs["attribute_predictions"]

        attribute_targets = targets["attributes"]

        wbc_predictions = outputs["wbc_logits"]

        wbc_targets = targets["cell_label"]

        # ----------------------------------------
        # Attribute Loss
        # ----------------------------------------

        attribute_loss = 0.0

        for idx, attribute_name in enumerate(
            self.attribute_names
        ):

            prediction = attribute_predictions[
                attribute_name
            ]

            target = attribute_targets[:, idx]

            attribute_loss += self.criterion(
                prediction,
                target,
            )

        attribute_loss = (
            attribute_loss
            / len(self.attribute_names)
        )

        # ----------------------------------------
        # WBC Loss
        # ----------------------------------------

        wbc_loss = self.criterion(
            wbc_predictions,
            wbc_targets,
        )

        # ----------------------------------------
        # Total Loss
        # ----------------------------------------

        total_loss = (

            self.attribute_weight
            * attribute_loss

            +

            self.wbc_weight
            * wbc_loss

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

    batch = 4

    outputs = {

        "attribute_predictions": {

            "cell_size": torch.randn(batch,2),

            "cell_shape": torch.randn(batch,2),

            "nucleus_shape": torch.randn(batch,6),

            "nuclear_cytoplasmic_ratio": torch.randn(batch,2),

            "chromatin_density": torch.randn(batch,2),

            "cytoplasm_vacuole": torch.randn(batch,2),

            "cytoplasm_texture": torch.randn(batch,2),

            "cytoplasm_colour": torch.randn(batch,3),

            "granule_type": torch.randn(batch,4),

            "granule_colour": torch.randn(batch,4),

            "granularity": torch.randn(batch,2),

        },

        "wbc_logits": torch.randn(batch,8)

    }

    targets = {

        "attributes": torch.randint(
            0,
            2,
            (batch,11)
        ),

        "cell_label": torch.randint(
            0,
            8,
            (batch,)
        )

    }

    # Fix multi-class attribute targets
    targets["attributes"][:,2] = torch.randint(0,6,(batch,))
    targets["attributes"][:,7] = torch.randint(0,3,(batch,))
    targets["attributes"][:,8] = torch.randint(0,4,(batch,))
    targets["attributes"][:,9] = torch.randint(0,4,(batch,))

    criterion = ExplainableWBCLoss()

    losses = criterion(outputs, targets)

    print("="*60)
    print("Loss Test")
    print("="*60)

    for name, value in losses.items():
        print(f"{name:20}: {value.item():.4f}")