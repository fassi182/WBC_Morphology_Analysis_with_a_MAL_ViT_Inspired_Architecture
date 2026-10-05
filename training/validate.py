"""Validate the image -> attributes -> WBC pipeline."""

from config import DEVICE
from data.dataloader import create_dataloaders
from training.losses import compute_total_loss
from training.evaluator import evaluate
from utils.model_loading import load_model


def main():
    _, val_loader, _ = create_dataloaders()
    model = load_model(device=DEVICE)
    metrics = evaluate(model, val_loader, compute_total_loss, DEVICE)
    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    main()
