"""
checkpoint.py

Utilities for saving and loading training checkpoints.

A checkpoint stores:
- Model weights
- Optimizer state
- Current epoch
- Best validation accuracy
- Training history

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

from pathlib import Path
import torch

from config import MODEL_DIR


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_accuracy: float,
    history: dict,
    filename: str = "checkpoint.pth",
):
    """
    Save a training checkpoint.

    Parameters
    ----------
    model : nn.Module
        Model to save.

    optimizer : torch.optim.Optimizer
        Optimizer.

    epoch : int
        Current epoch.

    best_accuracy : float
        Best validation accuracy achieved so far.

    history : dict
        Training history.

    filename : str
        Output checkpoint filename.
    """

    checkpoint_path = MODEL_DIR / filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_accuracy": best_accuracy,
        "history": history,
    }

    torch.save(checkpoint, checkpoint_path)

    print("=" * 60)
    print("Checkpoint Saved")
    print("=" * 60)
    print(f"Path : {checkpoint_path}")


def load_checkpoint(
    model,
    optimizer=None,
    filename: str = "checkpoint.pth",
    device: str = "cpu",
):
    """
    Load a training checkpoint.

    Parameters
    ----------
    model : nn.Module
        Model.

    optimizer : Optimizer, optional
        Optimizer to restore.

    filename : str
        Checkpoint filename.

    device : str
        cpu or cuda.

    Returns
    -------
    model
    optimizer
    epoch
    best_accuracy
    history
    """

    checkpoint_path = MODEL_DIR / filename

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    print("=" * 60)
    print("Checkpoint Loaded")
    print("=" * 60)
    print(f"Path : {checkpoint_path}")

    return (
        model,
        optimizer,
        checkpoint["epoch"],
        checkpoint["best_accuracy"],
        checkpoint["history"],
    )


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    import torch.optim as optim

    from models.complete_model import ExplainableWBCModel

    model = ExplainableWBCModel()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        best_accuracy=0.0,
        history=history,
        filename="test_checkpoint.pth",
    )

    (
        model,
        optimizer,
        epoch,
        best_accuracy,
        history,
    ) = load_checkpoint(
        model,
        optimizer,
        filename="test_checkpoint.pth",
    )

    print("\nLoaded Values")
    print(f"Epoch          : {epoch}")
    print(f"Best Accuracy  : {best_accuracy}")
    print(f"History Keys   : {list(history.keys())}")