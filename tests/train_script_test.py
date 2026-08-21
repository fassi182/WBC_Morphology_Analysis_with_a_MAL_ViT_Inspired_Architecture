"""
MAL-ViT Training Script Integration Test
"""

import torch

from config import DEVICE

from data.dataloader import create_dataloaders
from models.complete_model import CompleteMALViT
from training.losses import compute_total_loss
from training.engine import train_step, validation_step


def main():

    print("=" * 70)
    print("MAL-ViT TRAINING SCRIPT INTEGRATION TEST")
    print("=" * 70)

    device = torch.device(DEVICE)

    print("\nDevice:", device)

    # ==========================================================
    # Data
    # ==========================================================

    print("\n[1] Loading data...")

    train_loader, val_loader, _ = create_dataloaders()

    batch = next(iter(train_loader))

    images = batch["images"].to(device)
    labels = batch["labels"].to(device)
    attributes = batch["attributes"].to(device)

    print("Images:", images.shape)
    print("Labels:", labels.shape)
    print("Attributes:", attributes.shape)

    # ==========================================================
    # Model
    # ==========================================================

    print("\n[2] Creating model...")

    model = CompleteMALViT().to(device)

    # ==========================================================
    # Optimizer
    # ==========================================================

    print("\n[3] Creating optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # ==========================================================
    # Train Step
    # ==========================================================

    print("\n[4] Train step...")

    result = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        loss_fn=compute_total_loss,
        device=device,
    )

    print("Train step completed.")

    if isinstance(result, dict):

        for key, value in result.items():

            if isinstance(value, (float, int)):
                print(
                    f"{key:20s}: {value}"
                )

    # ==========================================================
    # Validation Step
    # ==========================================================

    print("\n[5] Validation step...")

    val_batch = next(iter(val_loader))

    result = validation_step(
        model=model,
        batch=val_batch,
        loss_fn=compute_total_loss,
        device=device,
    )

    print("Validation step completed.")

    # ==========================================================
    # Done
    # ==========================================================

    print("\n" + "=" * 70)
    print("TRAINING SCRIPT INTEGRATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()