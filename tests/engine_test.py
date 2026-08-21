"""
MAL-ViT Training Engine Test
"""

import torch
from torch.optim import AdamW

from config import DEVICE

from data.dataloader import create_dataloaders

from models.complete_model import CompleteMALViT

from training.losses import compute_total_loss

from training.engine import (
    train_step,
    validation_step,
    train_epoch,
    validate_epoch,
)


def main():

    print("=" * 70)
    print("MAL-ViT TRAINING ENGINE TEST")
    print("=" * 70)

    device = torch.device(
        DEVICE
    )

    print(
        "\nDevice:",
        device,
    )

    # ==================================================
    # Data
    # ==================================================

    print(
        "\n[1] Creating DataLoader..."
    )

    train_loader, val_loader, _ = (
        create_dataloaders()
    )

    batch = next(
        iter(train_loader)
    )

    print(
        "Images:",
        batch["images"].shape,
    )

    print(
        "Labels:",
        batch["labels"].shape,
    )

    print(
        "Attributes:",
        batch["attributes"].shape,
    )

    # ==================================================
    # Model
    # ==================================================

    print(
        "\n[2] Creating model..."
    )

    model = CompleteMALViT()

    model = model.to(device)

    # ==================================================
    # Optimizer
    # ==================================================

    print(
        "\n[3] Creating optimizer..."
    )

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
    )

    # ==================================================
    # Train step
    # ==================================================

    print(
        "\n[4] Testing train_step..."
    )

    result = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        device=device,
        loss_fn=compute_total_loss,
    )

    print(
        "Train loss:",
        result["total_loss"],
    )

    print(
        "WBC loss:",
        result["wbc_loss"],
    )

    print(
        "Attribute loss:",
        result["attribute_loss"],
    )

    print(
        "Accuracy:",
        result["accuracy"],
    )

    # ==================================================
    # Validation step
    # ==================================================

    print(
        "\n[5] Testing validation_step..."
    )

    val_batch = next(
        iter(val_loader)
    )

    result = validation_step(
        model=model,
        batch=val_batch,
        device=device,
        loss_fn=compute_total_loss,
    )

    print(
        "Validation loss:",
        result["total_loss"],
    )

    print(
        "Validation accuracy:",
        result["accuracy"],
    )

    # ==================================================
    # Full epoch
    # ==================================================

    print(
        "\n[6] Training epoch interface..."
    )

    print(
        "train_epoch imported successfully."
    )

    print(
        "validate_epoch imported successfully."
    )

    # ==================================================
    # Validation
    # ==================================================

    assert isinstance(
        result,
        dict,
    )

    required = {
        "total_loss",
        "wbc_loss",
        "attribute_loss",
        "accuracy",
    }

    assert required.issubset(
        result.keys()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Training engine validation: PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()