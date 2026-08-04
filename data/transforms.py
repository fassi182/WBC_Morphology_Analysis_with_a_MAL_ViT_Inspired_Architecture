"""
transforms.py

Image preprocessing and augmentation for the WBCAtt dataset.

This module defines image transformations for:
1. Training
2. Validation
3. Testing


"""

from torchvision import transforms

# ==========================================================
# ImageNet Normalization
# ==========================================================
# Since MAL-ViT is based on Vision Transformers that are
# pretrained on ImageNet, we use the same normalization.

from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
# ==========================================================
# Input Image Size
# ==========================================================
# ViT models commonly use 224x224 input images.



# ==========================================================
# Training Transform
# ==========================================================
# Includes data augmentation to improve generalization.

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=15),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ==========================================================
# Validation Transform
# ==========================================================
# No random augmentation.
# We only resize and normalize.

val_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ==========================================================
# Test Transform
# ==========================================================

test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ==========================================================
# Utility Function
# ==========================================================

def get_transforms(split: str):
    """
    Returns the appropriate transform for the dataset split.

    Args:
        split (str): "train", "val", or "test"

    Returns:
        torchvision.transforms.Compose
    """

    split = split.lower()

    if split == "train":
        return train_transform

    elif split == "val":
        return val_transform

    elif split == "test":
        return test_transform

    else:
        raise ValueError(
            f"Unknown split '{split}'. "
            "Expected one of ['train', 'val', 'test']"
        )


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("WBCAtt Transform Pipeline")
    print("=" * 60)

    print("\nTraining Transform:\n")
    print(train_transform)

    print("\nValidation Transform:\n")
    print(val_transform)

    print("\nTesting Transform:\n")
    print(test_transform)