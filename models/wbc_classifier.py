"""
wbc_classifier.py

Predicts the WBC subtype from the morphology attribute logits.

This is our research extension and is NOT part of the original
MAL-ViT paper.

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

import torch
import torch.nn as nn

from config import (
    NUM_CELL_CLASSES,
    CLASSIFIER_HIDDEN_DIM,
    CLASSIFIER_DROPOUT,
    ATTRIBUTE_CLASSES,
)


class WBCClassifier(nn.Module):
    """
    WBC classifier built on top of attribute predictions.
    """

    def __init__(self):
        super().__init__()

        input_dim = sum(ATTRIBUTE_CLASSES.values())

        self.classifier = nn.Sequential(

            nn.Linear(input_dim, CLASSIFIER_HIDDEN_DIM),

            nn.ReLU(),

            nn.Dropout(CLASSIFIER_DROPOUT),

            nn.Linear(
                CLASSIFIER_HIDDEN_DIM,
                NUM_CELL_CLASSES
            ),
        )

    def forward(self, attribute_predictions):
        """
        Parameters
        ----------
        attribute_predictions : dict

        Returns
        -------
        logits : Tensor
            (B,8)
        """

        logits = []

        for prediction in attribute_predictions.values():
            logits.append(prediction)

        x = torch.cat(logits, dim=1)

        return self.classifier(x)


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WBC Classifier Test")
    print("=" * 60)

    batch = 4

    predictions = {
        "cell_size": torch.randn(batch, 2),
        "cell_shape": torch.randn(batch, 2),
        "nucleus_shape": torch.randn(batch, 6),
        "nuclear_cytoplasmic_ratio": torch.randn(batch, 2),
        "chromatin_density": torch.randn(batch, 2),
        "cytoplasm_vacuole": torch.randn(batch, 2),
        "cytoplasm_texture": torch.randn(batch, 2),
        "cytoplasm_colour": torch.randn(batch, 3),
        "granule_type": torch.randn(batch, 4),
        "granule_colour": torch.randn(batch, 4),
        "granularity": torch.randn(batch, 2),
    }

    model = WBCClassifier()

    output = model(predictions)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nExpected:")
    print("(4, 8)")