"""
dataset.py

Custom PyTorch Dataset for the WBCAtt dataset.

Responsibilities:
- Read annotation Excel files
- Load RGB images
- Apply preprocessing transforms
- Encode WBC subtype labels
- Encode morphology attribute labels

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.encoders import (
    CELL_LABELS,
    ATTRIBUTE_ENCODERS,
    ATTRIBUTE_NAMES,
)


class WBCDataset(Dataset):
    """
    Custom Dataset for WBCAtt.
    """

    def __init__(
        self,
        annotation_file: str,
        dataset_root: str,
        transform=None,
    ):
        """
        Parameters
        ----------
        annotation_file : str
            Path to Excel annotation file.

        dataset_root : str
            Root folder containing PBC_dataset_normal_DIB.

        transform : torchvision transform
            Image preprocessing pipeline.
        """

        self.data = pd.read_csv(annotation_file)

        self.dataset_root = dataset_root

        self.transform = transform

    def __len__(self):
        """Return number of samples."""
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # --------------------------------------------------
        # Load Image
        # --------------------------------------------------

        image_path = self.dataset_root / row["path"]

        image = Image.open(image_path).convert("RGB")

        # --------------------------------------------------
        # Apply Transform
        # --------------------------------------------------

        if self.transform is not None:
            image = self.transform(image)

        # --------------------------------------------------
        # Encode WBC Label
        # --------------------------------------------------

        cell_label = CELL_LABELS[row["label"].lower()]

        # --------------------------------------------------
        # Encode Morphology Attributes
        # --------------------------------------------------

        attributes = []

        for attribute in ATTRIBUTE_NAMES:

            value = str(row[attribute]).lower()

            encoded = ATTRIBUTE_ENCODERS[attribute][value]

            attributes.append(encoded)

        attributes = torch.tensor(
            attributes,
            dtype=torch.long
        )

        cell_label = torch.tensor(
            cell_label,
            dtype=torch.long
        )

        # --------------------------------------------------
        # Return Sample
        # --------------------------------------------------

        sample = {

    "index": index,

    "image": image,

    "cell_label": cell_label,

    "attributes": attributes,

    "image_name": row["img_name"],

    # Convert Path object to string
    "image_path": str(image_path),

}

        return sample


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    from data.transforms import train_transform

    from config import (
    TRAIN_CSV,
    DATASET_ROOT,
)

    dataset = WBCDataset(

    annotation_file=TRAIN_CSV,

    dataset_root=DATASET_ROOT,

    transform=train_transform,

)

    print("=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print("Dataset Size :", len(dataset))

    sample = dataset[0]

    print("\nReturned Keys")

    print(sample.keys())

    print("\nImage Shape")

    print(sample["image"].shape)

    print("\nCell Label")

    print(sample["cell_label"])

    print("\nMorphology Attributes")

    print(sample["attributes"])
    print("++++++++++++++++++/n/n")
    print("\nOriginal CSV Row")
    print(dataset.data.iloc[0][
    ["label",
     "cell_size",
     "cell_shape",
     "nucleus_shape",
     "nuclear_cytoplasmic_ratio",
     "chromatin_density",
     "cytoplasm_vacuole",
     "cytoplasm_texture",
     "cytoplasm_colour",
     "granule_type",
     "granule_colour",
     "granularity"]
])