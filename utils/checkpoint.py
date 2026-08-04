"""
checkpoint.py

Utilities for saving and loading training checkpoints.

"""

from pathlib import Path
import torch

from config import CHECKPOINT_DIR


# ==========================================================
# Save Checkpoint
# ==========================================================

def save_checkpoint(
    checkpoint: dict,
    save_path,
):
    """
    Save checkpoint dictionary.

    Parameters
    ----------
    checkpoint : dict
        Complete training state.

    save_path : Path
        Output checkpoint path.
    """

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        save_path,
    )

    print("=" * 60)
    print("Checkpoint Saved")
    print("=" * 60)
    print(f"Path : {save_path}")


# ==========================================================
# Load Checkpoint
# ==========================================================

def load_checkpoint(load_path):
    """
    Load checkpoint dictionary.

    Parameters
    ----------
    load_path : Path

    Returns
    -------
    dict
    """

    load_path = Path(load_path)

    if not load_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{load_path}"
        )

    checkpoint = torch.load(
        load_path,
        map_location="cpu",
    )

    print("=" * 60)
    print("Checkpoint Loaded")
    print("=" * 60)
    print(f"Path : {load_path}")

    return checkpoint


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    checkpoint = {

        "epoch": 10,

        "best_loss": 0.73,

        "history": {

            "train_loss": [1.2, 0.9],

            "val_loss": [1.0, 0.8],

        },

    }

    path = CHECKPOINT_DIR / "test_checkpoint.pth"

    save_checkpoint(
        checkpoint,
        path,
    )

    loaded = load_checkpoint(path)

    print("\nLoaded Keys")

    print(loaded.keys())

    print("\nEpoch")

    print(loaded["epoch"])

    print("\nBest Loss")

    print(loaded["best_loss"])