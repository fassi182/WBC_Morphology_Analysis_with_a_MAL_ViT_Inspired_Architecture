"""
MAL-ViT Checkpoint Manager Integration Test
"""

import os
import torch

from models.complete_model import CompleteMALViT
from utils.checkpoint_manager import CheckpointManager


def main():

    print("=" * 60)
    print("MAL-ViT CHECKPOINT MANAGER TEST")
    print("=" * 60)

    print(
        "\n[1] Creating model..."
    )

    model = CompleteMALViT()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    manager = CheckpointManager(
        "checkpoints"
    )

    print(
        "\n[2] Saving checkpoint..."
    )

    path = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=3,
        best_metric=1.25,
        filename="test_checkpoint.pt",
    )

    print(
        "Saved:",
        path,
    )

    assert os.path.exists(path)

    print(
        "\n[3] Loading checkpoint..."
    )

    new_model = CompleteMALViT()

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=1e-4,
    )

    checkpoint = manager.load(
        path=path,
        model=new_model,
        optimizer=new_optimizer,
    )

    print(
        "Loaded epoch:",
        checkpoint["epoch"],
    )

    print(
        "Loaded metric:",
        checkpoint["best_metric"],
    )

    assert checkpoint["epoch"] == 3

    print(
        "\nCheckpoint manager validation: PASSED"
    )


if __name__ == "__main__":

    main()