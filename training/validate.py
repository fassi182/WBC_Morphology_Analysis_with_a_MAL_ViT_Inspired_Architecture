"""
MAL-ViT Validation Script
"""

import torch

from config import DEVICE

from data.dataloader import create_dataloaders
from models.complete_model import CompleteMALViT
from training.losses import compute_total_loss
from training.evaluator import evaluate


def main():

    print("=" * 70)
    print("MAL-ViT VALIDATION")
    print("=" * 70)

    device = torch.device(DEVICE)

    print("\nDevice:", device)

    # ==========================================================
    # Data
    # ==========================================================

    print("\nLoading validation data...")

    _, val_loader, _ = create_dataloaders()

    print(
        "Validation batches:",
        len(val_loader)
    )

    # ==========================================================
    # Model
    # ==========================================================

    model = CompleteMALViT().to(device)

    checkpoint = torch.load(
        "checkpoints/best.pt",
        map_location=device,
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    elif "model" in checkpoint:
        model.load_state_dict(
            checkpoint["model"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # ==========================================================
    # Evaluation
    # ==========================================================

    metrics = evaluate(
        model=model,
        dataloader=val_loader,
        loss_fn=compute_total_loss,
        device=device,
    )

    print("\nValidation Results")

    for key, value in metrics.items():

        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")

        else:
            print(f"{key:20s}: {value}")

    print("\nValidation complete.")


if __name__ == "__main__":
    main()