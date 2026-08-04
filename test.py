"""
test.py

Evaluate the trained Explainable WBC Classification model
on the test dataset.

Author:
Muhammad Fassi Ur Rehman
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

from data.dataloader import create_dataloaders
from models.complete_model import ExplainableWBCModel


def main():

    print("=" * 70)
    print("Testing Explainable WBC Classification Model")
    print("=" * 70)

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    _, _, test_loader = create_dataloaders()

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = ExplainableWBCModel().to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / BEST_MODEL_NAME

    model.load_state_dict(

        torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    )

    model.eval()

    print("\nBest model loaded.")

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    y_true = []

    y_pred = []

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(DEVICE)

            labels = batch["cell_label"].to(DEVICE)

            outputs = model(images)

            predictions = outputs["wbc_logits"].argmax(dim=1)

            y_true.extend(labels.cpu().numpy())

            y_pred.extend(predictions.cpu().numpy())

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print("\nOverall Accuracy")

    print(f"{accuracy*100:.2f}%")

    print("\nClassification Report\n")

    print(

        classification_report(
            y_true,
            y_pred,
            digits=4,
        )

    )

    print("\nConfusion Matrix\n")

    print(

        confusion_matrix(
            y_true,
            y_pred,
        )

    )


if __name__ == "__main__":

    main()