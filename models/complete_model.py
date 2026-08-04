"""
complete_model.py

Complete Explainable WBC Classification Model.

This model combines

1. MAL-ViT
2. WBC Classification Head

Pipeline

Image
    ↓
MAL-ViT
    ↓
11 Morphology Predictions
    ↓
WBC Classifier
    ↓
8 WBC Classes

Author:
Muhammad Fassi Ur Rehman
"""

import torch
import torch.nn as nn

from models.mal_vit import MALViT
from models.wbc_classifier import WBCClassifier


class ExplainableWBCModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.mal_vit = MALViT()

        self.wbc_classifier = WBCClassifier()

    def forward(self, images):

        # -----------------------------
        # MAL-ViT
        # -----------------------------

        outputs = self.mal_vit(images)

        attribute_predictions = outputs["attribute_predictions"]

        attribute_features = outputs["attribute_features"]

        # -----------------------------
        # WBC Prediction
        # -----------------------------

        wbc_logits = self.wbc_classifier(
            attribute_predictions
        )

        return {

            "attribute_predictions": attribute_predictions,

            "attribute_features": attribute_features,

            "wbc_logits": wbc_logits,

        }


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Complete Explainable WBC Model Test")
    print("=" * 60)

    images = torch.randn(
        4,
        3,
        224,
        224,
    )

    model = ExplainableWBCModel()

    outputs = model(images)

    print("\nAttribute Predictions")

    for name, pred in outputs["attribute_predictions"].items():
        print(f"{name:30} {tuple(pred.shape)}")

    print("\nAttribute Features")

    print(outputs["attribute_features"].shape)

    print("\nFinal WBC Logits")

    print(outputs["wbc_logits"].shape)