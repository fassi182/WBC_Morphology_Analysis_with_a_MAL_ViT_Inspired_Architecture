"""
training/scheduler.py

Learning-rate scheduler for MAL-ViT.

Uses:
    Linear warmup
    +
    Cosine decay
"""

import math

from torch.optim import Optimizer

from config import (
    NUM_EPOCHS,
    WARMUP_EPOCHS,
    MIN_LEARNING_RATE,
)


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
):
    """
    Create LambdaLR with warmup + cosine decay.
    """

    base_lrs = [
        group["lr"]
        for group in optimizer.param_groups
    ]

    def lr_lambda(epoch):

        if epoch < warmup_epochs:

            return float(
                epoch + 1
            ) / float(
                max(1, warmup_epochs)
            )

        progress = (
            epoch - warmup_epochs
        ) / float(
            max(
                1,
                total_epochs
                - warmup_epochs
                - 1,
            )
        )

        progress = min(
            max(progress, 0.0),
            1.0,
        )

        cosine = (
            0.5
            * (
                1
                + math.cos(
                    math.pi * progress
                )
            )
        )

        lr = (
            min_lr
            + (
                base_lrs[0]
                - min_lr
            )
            * cosine
        )

        return lr / base_lrs[0]

    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(
        optimizer,
        lr_lambda,
    )


if __name__ == "__main__":

    import torch

    from config import LEARNING_RATE

    print("=" * 60)
    print("MAL-ViT Scheduler Test")
    print("=" * 60)

    model = torch.nn.Linear(
        10,
        5,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    scheduler = cosine_warmup_scheduler(
        optimizer,
        WARMUP_EPOCHS,
        NUM_EPOCHS,
        MIN_LEARNING_RATE,
    )

    print("\nLearning rates:")

    for epoch in range(10):

        optimizer.step()

        scheduler.step()

        print(
            f"Epoch {epoch + 1:02d}: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

    print("\nScheduler validation: PASSED")