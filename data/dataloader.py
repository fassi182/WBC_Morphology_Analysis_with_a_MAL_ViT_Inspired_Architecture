# data/dataloader.py

import torch

from torch.utils.data import DataLoader

from config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    DATASET_ROOT,
    BATCH_SIZE,
    NUM_WORKERS,
)

from data.dataset import WBCDataset

from data.transforms import (
    train_transform,
    val_transform,
    test_transform,
)


# ==========================================================
# Collate Function
# ==========================================================

def wbca_collate_fn(batch):

    images = torch.stack(
        [
            sample["image"]
            for sample in batch
        ]
    )

    labels = torch.stack(
        [
            sample["cell_label"]
            for sample in batch
        ]
    )

    attributes = torch.stack(
        [
            sample["attributes"]
            for sample in batch
        ]
    )

    return {
        "images": images,
        "labels": labels,
        "attributes": attributes,

        "indices": [
            sample["index"]
            for sample in batch
        ],

        "image_names": [
            sample["image_name"]
            for sample in batch
        ],

        "image_paths": [
            sample["image_path"]
            for sample in batch
        ],
    }


# ==========================================================
# Create DataLoaders
# ==========================================================

def create_dataloaders():

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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=wbca_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=wbca_collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=wbca_collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("WBCAtt DataLoader Test")
    print("=" * 70)

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print("\nDataset sizes:")

    print(
        f"Train: {len(train_loader.dataset)}"
    )

    print(
        f"Val  : {len(val_loader.dataset)}"
    )

    print(
        f"Test : {len(test_loader.dataset)}"
    )

    print("\nNumber of batches:")

    print(
        f"Train: {len(train_loader)}"
    )

    print(
        f"Val  : {len(val_loader)}"
    )

    print(
        f"Test : {len(test_loader)}"
    )

    batch = next(
        iter(train_loader)
    )

    print("\nFirst training batch:")

    print(
        "Images:",
        batch["images"].shape
    )

    print(
        "WBC labels:",
        batch["labels"].shape
    )

    print(
        "Attributes:",
        batch["attributes"].shape
    )

    # ------------------------------------------------------
    # Validation
    # ------------------------------------------------------

    assert batch["images"].shape == (
        BATCH_SIZE,
        3,
        224,
        224,
    )

    assert batch["labels"].shape == (
        BATCH_SIZE,
    )

    assert batch["attributes"].shape == (
        BATCH_SIZE,
        11,
    )

    assert batch["images"].dtype == torch.float32
    assert batch["labels"].dtype == torch.long
    assert batch["attributes"].dtype == torch.long

    print(
        "\nDataLoader validation: PASSED"
    )

    print("=" * 70)