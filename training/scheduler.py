"""
scheduler.py

Learning Rate Scheduler for MAL-ViT.

Uses Cosine Annealing Learning Rate scheduling, which is commonly
used for Vision Transformer training.

Author:
Muhammad Fassi Ur Rehman
Project:
Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

import torch
import torch.nn as nn
import torch.optim as optim

from config import NUM_EPOCHS


def create_scheduler(optimizer):
    """
    Create the learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    Returns
    -------
    torch.optim.lr_scheduler.CosineAnnealingLR
    """

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    return scheduler


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Cosine Annealing Scheduler Test")
    print("=" * 60)

    model = nn.Linear(10, 2)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    scheduler = create_scheduler(optimizer)

    criterion = nn.MSELoss()

    for epoch in range(10):

        optimizer.zero_grad()

        x = torch.randn(8, 10)
        y = torch.randn(8, 2)

        prediction = model(x)

        loss = criterion(prediction, y)

        loss.backward()

        optimizer.step()

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:2d} | "
            f"Loss: {loss.item():.4f} | "
            f"LR: {current_lr:.8f}"
        )