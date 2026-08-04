"""
trainer.py

Main training script for Explainable WBC Classification
using MAL-ViT.

Responsibilities
----------------
1. Create DataLoaders
2. Build model
3. Create optimizer
4. Create loss function
5. Train for multiple epochs
6. Validate after every epoch
7. Save best model

Author:
Muhammad Fassi Ur Rehman
"""

from pathlib import Path

import torch
import torch.optim as optim

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_DIR,
)

from data.dataloader import create_dataloaders

from models.complete_model import ExplainableWBCModel

from training.losses import ExplainableWBCLoss

from training.train_one_epoch import train_one_epoch

from training.validate import validate


def train():

    print("=" * 70)
    print("Explainable WBC Classification using MAL-ViT")
    print("=" * 70)

    # ----------------------------------------------------
    # Data
    # ----------------------------------------------------

    train_loader, val_loader, _ = create_dataloaders()

    print(f"\nTraining Samples   : {len(train_loader.dataset)}")
    print(f"Validation Samples : {len(val_loader.dataset)}")

    # ----------------------------------------------------
    # Model
    # ----------------------------------------------------

    model = ExplainableWBCModel().to(DEVICE)

    # ----------------------------------------------------
    # Loss
    # ----------------------------------------------------

    criterion = ExplainableWBCLoss()

    # ----------------------------------------------------
    # Optimizer
    # ----------------------------------------------------

    optimizer = optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )

    # ----------------------------------------------------
    # Best Validation Loss
    # ----------------------------------------------------

    best_loss = float("inf")

    # ----------------------------------------------------
    # Training Loop
    # ----------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        print("\n" + "=" * 70)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        print("=" * 70)

        # -------------------------
        # Train
        # -------------------------

        train_metrics = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            DEVICE,

        )

        # -------------------------
        # Validate
        # -------------------------

        val_metrics = validate(

            model,

            val_loader,

            criterion,

            DEVICE,

        )

        # -------------------------
        # Print Results
        # -------------------------

        print("\nTraining")

        print(f"Loss            : {train_metrics['loss']:.4f}")

        print(f"Attribute Loss  : {train_metrics['attribute_loss']:.4f}")

        print(f"WBC Loss        : {train_metrics['wbc_loss']:.4f}")

        print(f"Accuracy        : {train_metrics['accuracy']:.2f}%")

        print("\nValidation")

        print(f"Loss            : {val_metrics['loss']:.4f}")

        print(f"Attribute Loss  : {val_metrics['attribute_loss']:.4f}")

        print(f"WBC Loss        : {val_metrics['wbc_loss']:.4f}")

        print(f"Accuracy        : {val_metrics['accuracy']:.2f}%")

        # -------------------------
        # Save Best Model
        # -------------------------

        if val_metrics["loss"] < best_loss:

            best_loss = val_metrics["loss"]

            save_path = MODEL_DIR / "best_model.pth"

            torch.save(

                model.state_dict(),

                save_path,

            )

            print("\n✅ Best model saved.")

    print("\n" + "=" * 70)

    print("Training Finished")

    print("=" * 70)


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    train()