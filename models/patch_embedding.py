import torch
import torch.nn as nn

from config import (
    IMAGE_SIZE,
    PATCH_SIZE,
    IN_CHANNELS,
    EMBED_DIM
)


class PatchEmbedding(nn.Module):
    """
    Converts input images into patch tokens.

    Input:
        Image tensor:
        (B, C, H, W)

    Output:
        Patch tokens:
        (B, Num_Patches, Embed_Dim)

    Example:
        (8,3,224,224)
            ->
        (8,196,192)
    """

    def __init__(
        self,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = (
            image_size // patch_size
        ) ** 2


        # Patch projection layer
        #
        # Equivalent operation:
        #
        # 16x16x3 patch
        #
        #       |
        #
        # Flatten
        #
        #       |
        #
        # Linear projection
        #
        #       |
        #
        # 192 dimensional token
        #

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


    def forward(self, x):

        """
        Forward pass

        Input:
            x:
            (B,C,H,W)

        Output:
            tokens:
            (B,N,D)

        """

        B, C, H, W = x.shape


        # Check image size
        assert (
            H == self.image_size
            and W == self.image_size
        ), (
            f"Expected image size "
            f"{self.image_size}, "
            f"but got {H}x{W}"
        )


        # Create patches
        #
        # Before:
        # (B,3,224,224)
        #
        # After Conv:
        # (B,192,14,14)

        x = self.projection(x)


        # Rearrange:
        #
        # (B,192,14,14)
        #
        # ->
        #
        # (B,196,192)

        x = x.flatten(2)

        x = x.transpose(1,2)


        return x



if __name__ == "__main__":


    print("="*60)
    print("Patch Embedding Test")
    print("="*60)


    image = torch.randn(
        8,
        3,
        224,
        224
    )


    model = PatchEmbedding()


    output = model(image)


    print("Input Shape:")
    print(image.shape)


    print("\nOutput Shape:")
    print(output.shape)


    print("\nExpected:")
    print("(8,196,192)")