"""
transforms.py

Image preprocessing and augmentation for WBCAtt.

Three pipelines are provided:

    train_transform
    val_transform
    test_transform

Training:
    Resize
    Horizontal Flip
    Rotation
    Color Jitter
    Normalize

Validation/Test:
    Resize
    Normalize
"""

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    TRAIN_RANDOM_HORIZONTAL_FLIP,
    TRAIN_RANDOM_ROTATION,
)


# ============================================================
# TRAINING TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=InterpolationMode.BICUBIC,
    ),

    # --------------------------------------------------------
    # Horizontal flip
    # --------------------------------------------------------

    transforms.RandomHorizontalFlip(
        p=0.5
        if TRAIN_RANDOM_HORIZONTAL_FLIP
        else 0.0
    ),

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    transforms.RandomRotation(
        degrees=TRAIN_RANDOM_ROTATION
    ),

    # --------------------------------------------------------
    # Mild color augmentation
    # --------------------------------------------------------
    #
    # Important for microscopy images:
    # keep augmentation relatively conservative so that
    # morphology/color information is not destroyed.
    #

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02,
    ),

    # --------------------------------------------------------
    # Convert PIL -> Tensor
    # --------------------------------------------------------

    transforms.ToTensor(),

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=InterpolationMode.BICUBIC,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=InterpolationMode.BICUBIC,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# TRANSFORM SELECTOR
# ============================================================

def get_transforms(
    split: str,
):
    """
    Return the correct transformation pipeline.

    Parameters
    ----------
    split : str
        One of:
            "train"
            "val"
            "test"

    Returns
    -------
    torchvision.transforms.Compose
    """

    split = split.strip().lower()

    if split == "train":

        return train_transform

    elif split in (
        "val",
        "validation",
    ):

        return val_transform

    elif split == "test":

        return test_transform

    else:

        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected 'train', 'val', or 'test'."
        )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("WBCAtt Transform Pipeline")
    print("=" * 70)

    print(
        f"\nImage size       : "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"ImageNet mean    : "
        f"{IMAGENET_MEAN}"
    )

    print(
        f"ImageNet std     : "
        f"{IMAGENET_STD}"
    )

    print("\nTraining Transform:")
    print(train_transform)

    print("\nValidation Transform:")
    print(val_transform)

    print("\nTest Transform:")
    print(test_transform)

    print("\nTransform validation:")

    assert get_transforms("train") is train_transform

    assert get_transforms("val") is val_transform

    assert get_transforms("validation") is val_transform

    assert get_transforms("test") is test_transform

    print("PASSED")