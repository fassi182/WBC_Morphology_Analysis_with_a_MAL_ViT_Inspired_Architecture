"""
smoke_test.py

End-to-end integration test 

This script verifies the complete training pipeline:

Dataset
→ DataLoader
→ ExplainableWBCModel
→ Loss
→ Backward
→ Optimizer


"""

import torch
import torch.optim as optim

from config import DEVICE
from data.dataloader import create_dataloaders
from models.complete_model import ExplainableWBCModel
from training.losses import ExplainableWBCLoss


def main():

    print("=" * 70)
    print("Explainable WBC Smoke Test")
    print("=" * 70)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    train_loader, _, _ = create_dataloaders(batch_size=4)

    batch = next(iter(train_loader))

    images = batch["image"].to(DEVICE)
    cell_labels = batch["cell_label"].to(DEVICE)
    attributes = batch["attributes"].to(DEVICE)

    print("\n✓ Batch Loaded")

    print("Images:", images.shape)
    print("Attributes:", attributes.shape)
    print("Cell Labels:", cell_labels.shape)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = ExplainableWBCModel().to(DEVICE)

    print("\n✓ Model Created")

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    outputs = model(images)

    print("\n✓ Forward Pass")

    print("\nAttribute Features")

    print(outputs["attribute_features"].shape)

    print("\nAttribute Heads")

    for name, pred in outputs["attribute_predictions"].items():
        print(f"{name:30} {tuple(pred.shape)}")

    print("\nWBC Logits")

    print(outputs["wbc_logits"].shape)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = ExplainableWBCLoss()

    losses = criterion(

        outputs,

        {

            "attributes": attributes,

            "cell_label": cell_labels,

        }

    )

    print("\n✓ Loss Computed")

    print(f"Total Loss      : {losses['total_loss'].item():.4f}")
    print(f"Attribute Loss  : {losses['attribute_loss'].item():.4f}")
    print(f"WBC Loss        : {losses['wbc_loss'].item():.4f}")

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    optimizer.zero_grad()

    losses["total_loss"].backward()

    optimizer.step()

    print("\n✓ Backward Pass Successful")
    print("✓ Optimizer Step Successful")

    print("\n" + "=" * 70)
    print("🎉 SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()