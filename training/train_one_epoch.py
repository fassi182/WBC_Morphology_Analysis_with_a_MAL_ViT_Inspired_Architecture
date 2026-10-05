"""
MAL-ViT One Epoch Training Test
"""

import torch

from config import DEVICE

from data.dataloader import (
    create_dataloaders,
)

from models.complete_model import (
    CompleteMALViT,
)

from training.optimizer import (
    create_optimizer,
)

from training.trainer import (
    Trainer,
)


def main():

    print("=" * 60)
    print("MAL-ViT ONE EPOCH TRAINING TEST")
    print("=" * 60)

    print("\n[1] Creating DataLoaders...")

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    print(
        "Train batches:",
        len(train_loader),
    )

    print("\n[2] Creating model...")

    model = CompleteMALViT().to(DEVICE)

    print("\n[3] Creating optimizer...")

    optimizer = create_optimizer(
        model
    )

    print("\n[4] Creating trainer...")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=DEVICE,
    )

    print("\n[5] Training one epoch...")

    metrics = trainer.train_epoch(
        train_loader
    )

    print("\nTraining results:")

    for name, value in metrics.items():

        print(
            f"{name:20s}: {value:.6f}"
        )

    print(
        "\nOne epoch training: PASSED"
    )


if __name__ == "__main__":

    main()
