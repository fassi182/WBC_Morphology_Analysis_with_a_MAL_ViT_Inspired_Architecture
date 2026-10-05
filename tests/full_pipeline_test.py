"""
MAL-ViT Full Pipeline Test

Data
 â†“
DataLoader
 â†“
Model
 â†“
Forward
 â†“
Loss
 â†“
Backward
 â†“
Optimizer
 â†“
Validation
 â†“
Checkpoint
"""

import os
import torch

from config import DEVICE

from data.dataloader import create_dataloaders
from models.complete_model import CompleteMALViT
from training.losses import compute_total_loss
from training.engine import train_step, validation_step
from training.evaluator import evaluate
from utils.checkpoint_manager import CheckpointManager


def main():

    print("=" * 70)
    print("MAL-ViT FULL PIPELINE TEST")
    print("=" * 70)

    device = torch.device(DEVICE)

    print("\nDevice:", device)

    # ==========================================================
    # Data
    # ==========================================================

    print("\n[1] DataLoader")

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    print(
        "Train:",
        len(train_loader),
        "batches"
    )

    print(
        "Val:",
        len(val_loader),
        "batches"
    )

    print(
        "Test:",
        len(test_loader),
        "batches"
    )

    # ==========================================================
    # Model
    # ==========================================================

    print("\n[2] Model")

    model = CompleteMALViT().to(device)

    print("Model created.")

    # ==========================================================
    # Optimizer
    # ==========================================================

    print("\n[3] Optimizer")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    print("Optimizer created.")

    # ==========================================================
    # Train
    # ==========================================================

    print("\n[4] Training step")

    batch = next(iter(train_loader))

    train_result = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        loss_fn=compute_total_loss,
        device=device,
    )

    print("Training step passed.")

    if isinstance(train_result, dict):

        for key, value in train_result.items():

            if isinstance(value, (float, int)):
                print(
                    f"{key:20s}: {value:.4f}"
                )

    # ==========================================================
    # Validation
    # ==========================================================

    print("\n[5] Validation step")

    val_batch = next(iter(val_loader))

    validation_result = validation_step(
        model=model,
        batch=val_batch,
        loss_fn=compute_total_loss,
        device=device,
    )

    print("Validation step passed.")

    if isinstance(validation_result, dict):

        for key, value in validation_result.items():

            if isinstance(value, (float, int)):
                print(
                    f"{key:20s}: {value:.4f}"
                )

    # ==========================================================
    # Evaluation
    # ==========================================================

    print("\n[6] Evaluator")

    evaluation = evaluate(
        model=model,
        dataloader=val_loader,
        loss_fn=compute_total_loss,
        device=device,
    )

    print("Evaluator passed.")

    # ==========================================================
    # Checkpoint
    # ==========================================================

    print("\n[7] Checkpoint")

    manager = CheckpointManager(
        directory="checkpoints"
    )

    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        metric=float(
            evaluation.get(
                "loss",
                0.0
            )
        ),
        filename="pipeline_test.pt",
    )

    checkpoint_path = os.path.join(
        "checkpoints",
        "pipeline_test.pt",
    )

    if not os.path.exists(checkpoint_path):

        raise RuntimeError(
            "Checkpoint was not created."
        )

    print(
        "Checkpoint created:",
        checkpoint_path
    )

    print("\n" + "=" * 70)
    print("FULL PIPELINE VALIDATION: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()