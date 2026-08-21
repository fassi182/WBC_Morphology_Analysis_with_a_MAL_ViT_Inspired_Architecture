"""
patch_embedding.py

Patch Embedding for MAL-ViT.

Pipeline:

    Input Image
        │
        ▼
    Conv2D
        │
        ▼
    Non-overlapping Image Patches
        │
        ▼
    Flatten
        │
        ▼
    Patch Embeddings

For the current configuration:

    Image size      = 224 x 224
    Patch size      = 16 x 16
    Patch grid      = 14 x 14
    Number patches  = 196
    Embedding dim   = 192

Input:
    (B, 3, 224, 224)

Output:
    (B, 196, 192)
"""

import torch
import torch.nn as nn

from config import (
    IMAGE_SIZE,
    PATCH_SIZE,
    EMBED_DIM,
    NUM_PATCHES,
)


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Convert an image into a sequence of patch embeddings.

    Each 16x16 image patch becomes one 192-dimensional token.
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=3,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Validate image / patch configuration
        # ----------------------------------------------------

        if image_size % patch_size != 0:
            raise ValueError(
                f"Image size ({image_size}) must be divisible "
                f"by patch size ({patch_size})."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.grid_size = image_size // patch_size

        self.num_patches = (
            self.grid_size * self.grid_size
        )

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------
        #
        # Conv2D with:
        #
        # kernel_size = patch_size
        # stride      = patch_size
        #
        # creates non-overlapping patches.
        #
        # Input:
        #     (B, 3, 224, 224)
        #
        # Output:
        #     (B, 192, 14, 14)
        #
        # ----------------------------------------------------

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ----------------------------------------------------
        # Safety check against config.py
        # ----------------------------------------------------

        if self.num_patches != NUM_PATCHES:
            raise ValueError(
                "Patch configuration mismatch.\n"
                f"Calculated patches: {self.num_patches}\n"
                f"Config patches:     {NUM_PATCHES}"
            )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, images):
        """
        Parameters
        ----------
        images : torch.Tensor
            Shape:
                (B, 3, 224, 224)

        Returns
        -------
        torch.Tensor
            Shape:
                (B, 196, 192)
        """

        # ----------------------------------------------------
        # Validate input dimensions
        # ----------------------------------------------------

        if images.ndim != 4:
            raise ValueError(
                "Expected image tensor with shape "
                "(B, C, H, W), "
                f"got {tuple(images.shape)}"
            )

        batch_size, channels, height, width = images.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {channels}."
            )

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected images of size "
                f"{self.image_size}x{self.image_size}, "
                f"got {height}x{width}."
            )

        # ----------------------------------------------------
        # Patch projection
        # ----------------------------------------------------

        x = self.projection(images)

        # Shape:
        # (B, 192, 14, 14)

        # ----------------------------------------------------
        # Flatten spatial dimensions
        # ----------------------------------------------------

        x = x.flatten(2)

        # Shape:
        # (B, 192, 196)

        # ----------------------------------------------------
        # Move embedding dimension to last position
        # ----------------------------------------------------

        x = x.transpose(1, 2)

        # Shape:
        # (B, 196, 192)

        return x


# ============================================================
# Quick Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Patch Embedding Test")
    print("=" * 70)

    print("\nConfiguration")
    print(f"Image size     : {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Patch size     : {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Patch grid     : {IMAGE_SIZE // PATCH_SIZE} x "
          f"{IMAGE_SIZE // PATCH_SIZE}")
    print(f"Number patches : {NUM_PATCHES}")
    print(f"Embedding dim  : {EMBED_DIM}")

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    patch_embedding = PatchEmbedding()

    print("\nPatch Embedding:")
    print(patch_embedding)

    # --------------------------------------------------------
    # Dummy image batch
    # --------------------------------------------------------

    images = torch.randn(
        4,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    patch_tokens = patch_embedding(images)

    print("\nInput shape:")
    print(images.shape)

    print("\nOutput shape:")
    print(patch_tokens.shape)

    # --------------------------------------------------------
    # Expected shape
    # --------------------------------------------------------

    expected_shape = (
        4,
        NUM_PATCHES,
        EMBED_DIM,
    )

    print("\nExpected shape:")
    print(expected_shape)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert patch_tokens.shape == expected_shape, (
        f"Patch embedding shape mismatch. "
        f"Expected {expected_shape}, "
        f"got {tuple(patch_tokens.shape)}"
    )

    assert torch.isfinite(patch_tokens).all(), (
        "Patch embeddings contain NaN or Inf values."
    )

    print("\nPatch embedding validation: PASSED")
    print("=" * 70)