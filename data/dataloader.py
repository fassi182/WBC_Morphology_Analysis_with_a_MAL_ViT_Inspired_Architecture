"""
dataloader.py

Creates PyTorch DataLoaders for the WBCAtt dataset.


"""

from torch.utils.data import DataLoader

from data.dataset import WBCDataset
from data.transforms import (
    train_transform,
    val_transform,
    test_transform,
)

# ==========================================================
# Dataset Paths
# ==========================================================

from config import (
    DATASET_ROOT,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)
# ==========================================================
# DataLoader Factory
# ==========================================================

def create_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """
    Create training, validation and testing dataloaders.

    Parameters
    ----------
    batch_size : int
        Number of images per batch.

    num_workers : int
        Number of worker processes.

    Returns
    -------
    tuple
        train_loader, val_loader, test_loader
    """

    train_dataset = WBCDataset(
        annotation_file=TRAIN_CSV,
        dataset_root=DATASET_ROOT,
        transform=train_transform,
    )

    val_dataset = WBCDataset(
        annotation_file=VAL_CSV,
        dataset_root=DATASET_ROOT,
        transform=val_transform,
    )

    test_dataset = WBCDataset(
        annotation_file=TEST_CSV,
        dataset_root=DATASET_ROOT,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=8
    )

    print("=" * 60)
    print("WBCAtt DataLoader Test")
    print("=" * 60)

    print(f"Training batches   : {len(train_loader)}")
    print(f"Validation batches : {len(val_loader)}")
    print(f"Testing batches    : {len(test_loader)}")

    print("\nLoading one batch...\n")

    batch = next(iter(train_loader))

    print("Batch Keys")
    print(batch.keys())

    print("\nImage Batch Shape")
    print(batch["image"].shape)

    print("\nCell Labels Shape")
    print(batch["cell_label"].shape)

    print("\nAttribute Tensor Shape")
    print(batch["attributes"].shape)

    print("\nImage Names")
    print(batch["image_name"][:3])

    print("\nDataset Indices")
    print(batch["index"][:3])