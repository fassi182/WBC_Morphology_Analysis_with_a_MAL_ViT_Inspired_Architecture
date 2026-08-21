"""
MAL-ViT Training Pipeline Integration Test
"""

import torch
from torch.optim import AdamW

from config import DEVICE

from data.dataloader import create_dataloaders

from models.complete_model import CompleteMALViT

from training.engine import (
    train_step,
    validation_step,
)

from training.losses import compute_total_loss


def main():

    print("=" * 70)
    print("MAL-ViT TRAINING PIPELINE TEST")
    print("=" * 70)

    device = torch.device(
        DEVICE
    )

    print(
        "\nDevice:",
        device,
    )

    # ==================================================
    # DATA
    # ==================================================

    print(
        "\n[1] Loading data..."
    )

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    print(
        "Train:",
        len(train_loader),
        "batches",
    )

    print(
        "Val:",
        len(val_loader),
        "batches",
    )

    print(
        "Test:",
        len(test_loader),
        "batches",
    )

    # ==================================================
    # MODEL
    # ==================================================

    print(
        "\n[2] Creating model..."
    )

    model = CompleteMALViT()

    model.to(device)

    print(
        "Model created."
    )

    # ==================================================
    # OPTIMIZER
    # ==================================================

    print(
        "\n[3] Creating optimizer..."
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )

    # ==================================================
    # TRAINING BATCH
    # ==================================================

    print(
        "\n[4] Training batch..."
    )

    batch = next(
        iter(train_loader)
    )

    train_result = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        loss_fn=compute_total_loss,
    )

    print(
        "Train loss:",
        train_result["total_loss"],
    )

    print(
        "WBC loss:",
        train_result["wbc_loss"],
    )

    print(
        "Attribute loss:",
        train_result["attribute_loss"],
    )

    print(
        "Accuracy:",
        train_result["accuracy"],
    )

    # ==================================================
    # VALIDATION
    # ==================================================

    print(
        "\n[5] Validation batch..."
    )

    val_batch = next(
        iter(val_loader)
    )

    val_result = validation_step(
        model=model,
        batch=val_batch,
        device=device,
        loss_fn=compute_total_loss,
    )

    print(
        "Validation loss:",
        val_result["total_loss"],
    )

    print(
        "Validation accuracy:",
        val_result["accuracy"],
    )

    # ==================================================
    # OUTPUT VALIDATION
    # ==================================================

    assert (
        train_result["total_loss"]
        >= 0
    )

    assert (
        val_result["total_loss"]
        >= 0
    )

    assert (
        0 <= train_result["accuracy"] <= 1
    )

    assert (
        0 <= val_result["accuracy"] <= 1
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TRAINING PIPELINE VALIDATION: PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()