"""
MAL-ViT Optimizer
"""

import torch

from config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
)


def create_optimizer(
    model,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
):
    """
    Creates AdamW optimizer.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    return optimizer


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    from models.complete_model import CompleteMALViT

    print("=" * 60)
    print("MAL-ViT Optimizer Test")
    print("=" * 60)

    model = CompleteMALViT()

    optimizer = create_optimizer(
        model
    )

    print("\nOptimizer:")
    print(optimizer)

    print("\nLearning rate:")

    for group in optimizer.param_groups:

        print(
            group["lr"]
        )

    print("\nOptimizer validation: PASSED")