"""
attribute_heads.py

Independent attribute classification heads for MAL-ViT.

Each attribute token is passed to its own classification head.

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    ATTRIBUTE_CLASSES,
)


class AttributeHeads(nn.Module):
    """
    Independent classification heads for each morphology attribute.
    """

    def __init__(
        self,
        embed_dim=EMBED_DIM,
    ):
        super().__init__()

        self.attribute_names = list(
            ATTRIBUTE_CLASSES.keys()
        )

        self.heads = nn.ModuleDict()

        for attribute_name, num_classes in ATTRIBUTE_CLASSES.items():

            self.heads[attribute_name] = nn.Linear(
                embed_dim,
                num_classes,
            )

    def forward(self, attribute_tokens):
        """
        Parameters
        ----------
        attribute_tokens : Tensor

            Shape:
            (B, 11, 192)

        Returns
        -------
        dict

            {
                "cell_size": logits,
                ...
            }
        """

        outputs = {}

        for i, attribute_name in enumerate(
            self.attribute_names
        ):

            token = attribute_tokens[:, i]

            logits = self.heads[attribute_name](
                token
            )

            outputs[attribute_name] = logits

        return outputs


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Attribute Heads Test")
    print("=" * 60)

    batch_size = 8

    attribute_tokens = torch.randn(
        batch_size,
        11,
        EMBED_DIM,
    )

    model = AttributeHeads()

    outputs = model(attribute_tokens)

    for name, logits in outputs.items():

        print(f"{name:30} {tuple(logits.shape)}")