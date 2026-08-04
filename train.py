"""
train.py

Entry point for training Explainable WBC Classification
using MAL-ViT.

Author:
Muhammad Fassi Ur Rehman
"""

import random
import numpy as np
import torch

from config import (
    RANDOM_SEED,
    DEVICE,
)

from data.dataloader import create_dataloaders

from models.complete_model import ExplainableWBCModel

from training.losses import ExplainableWBCLoss

from training.trainer import Trainer


# ==========================================================
# Set Random Seed
# ==========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("Explainable WBC Classification using MAL-ViT")
    print("=" * 70)

    print(f"\nDevice : {DEVICE}")

    # ------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------

    set_seed(RANDOM_SEED)

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    train_loader, val_loader, test_loader = create_dataloaders()

    print(f"\nTraining Samples   : {len(train_loader.dataset)}")

    print(f"Validation Samples : {len(val_loader.dataset)}")

    print(f"Testing Samples    : {len(test_loader.dataset)}")

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = ExplainableWBCModel()

    print("\nModel Created Successfully.")

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    criterion = ExplainableWBCLoss()

    # ------------------------------------------------------
    # Trainer
    # ------------------------------------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

    )

    # ------------------------------------------------------
    # Resume (Optional)
    # ------------------------------------------------------

    # trainer.resume()

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------

    trainer.fit()


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    main()