"""
seed.py

Utility functions for reproducible experiments.

This module sets random seeds for:
- Python
- NumPy
- PyTorch
- CUDA

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("=" * 60)
    print("Random Seed Initialized")
    print("=" * 60)
    print(f"Seed : {seed}")


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    set_seed(42)

    print("\nRandom Tensor")

    print(torch.randn(3))