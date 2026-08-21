"""
Reproducibility utilities for MAL-ViT.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible experiments.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Seed Test")
    print("=" * 60)

    set_seed(42)

    a = torch.randn(5)

    set_seed(42)

    b = torch.randn(5)

    print("\nFirst tensor:")
    print(a)

    print("\nSecond tensor:")
    print(b)

    print("\nEqual:", torch.equal(a, b))

    assert torch.equal(a, b)

    print("\nSeed validation: PASSED")