"""
inference.py

Inference script 

Pipeline
--------
Image
    ↓
MAL-ViT
    ↓
11 Morphology Predictions
    ↓
WBC Prediction

"""

from pathlib import Path

import torch
from PIL import Image

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

from data.transforms import test_transform

from data.encoders import (
    CELL_LABELS_INV,
    ATTRIBUTE_ENCODERS_INV,
    ATTRIBUTE_NAMES,
)

from models.complete_model import ExplainableWBCModel


# ==========================================================
# Load Model
# ==========================================================

model = ExplainableWBCModel().to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_DIR / BEST_MODEL_NAME,
    map_location=DEVICE,
)

model.load_state_dict(checkpoint)

model.eval()


# ==========================================================
# Predict
# ==========================================================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = test_transform(image)

    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(image)

    # ------------------------------------------------------
    # Attribute Prediction
    # ------------------------------------------------------

    predicted_attributes = {}

    for attribute_name in ATTRIBUTE_NAMES:

        logits = outputs["attribute_predictions"][attribute_name]

        prediction = logits.argmax(dim=1).item()

        predicted_attributes[attribute_name] = \
            ATTRIBUTE_ENCODERS_INV[
                attribute_name
            ][prediction]

    # ------------------------------------------------------
    # WBC Prediction
    # ------------------------------------------------------

    prediction = outputs["wbc_logits"].argmax(dim=1).item()

    predicted_cell = CELL_LABELS_INV[prediction]

    return predicted_cell, predicted_attributes


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    image_path = input(
        "\nEnter image path : "
    )

    image_path = Path(image_path)

    cell, attributes = predict(image_path)

    print("\n" + "=" * 70)

    print("Predicted WBC Type")

    print("=" * 70)

    print(cell)

    print("\n")

    print("=" * 70)

    print("Predicted Morphology")

    print("=" * 70)

    for key, value in attributes.items():

        print(f"{key:35}: {value}")