# data/dataset.py

"""
WBCAtt PyTorch Dataset
======================

Responsibilities
----------------
1. Read WBCAtt CSV annotations.
2. Resolve image paths safely.
3. Load RGB images.
4. Apply train/validation/test transforms.
5. Encode WBC class labels.
6. Encode all 11 morphology attributes.
7. Return a consistent dictionary.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.encoders import (
    CELL_LABELS,
    ATTRIBUTE_ENCODERS,
    ATTRIBUTE_NAMES,
)


# ==========================================================
# Dataset
# ==========================================================

class WBCDataset(Dataset):

    def __init__(
        self,
        annotation_file,
        dataset_root,
        transform=None,
    ):

        self.annotation_file = Path(annotation_file)
        self.dataset_root = Path(dataset_root)
        self.transform = transform

        # --------------------------------------------------
        # Validate paths
        # --------------------------------------------------

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"\nAnnotation file not found:\n"
                f"{self.annotation_file}"
            )

        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"\nDataset root not found:\n"
                f"{self.dataset_root}"
            )

        # --------------------------------------------------
        # Read CSV
        # --------------------------------------------------

        self.data = pd.read_csv(
            self.annotation_file
        )

        # --------------------------------------------------
        # Validate columns
        # --------------------------------------------------

        self._validate_columns()

        self.num_samples = len(self.data)

    # ======================================================
    # Validate CSV
    # ======================================================

    def _validate_columns(self):

        required_columns = [
            "path",
            "img_name",
            "label",
        ] + ATTRIBUTE_NAMES

        missing_columns = [
            column
            for column in required_columns
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                "\nMissing required CSV columns:\n"
                f"{missing_columns}\n\n"
                f"Available columns:\n"
                f"{list(self.data.columns)}"
            )

    # ======================================================
    # Length
    # ======================================================

    def __len__(self):

        return self.num_samples

    # ======================================================
    # Resolve image path
    # ======================================================

    def _resolve_image_path(self, relative_path):

        relative_path = Path(
            str(relative_path).strip()
        )

        # Remove accidental leading slash
        relative_path = Path(
            str(relative_path).lstrip("/\\")
        )

        candidates = [
            self.dataset_root / relative_path,

            self.dataset_root
            / "PBC_dataset_normal_DIB"
            / relative_path,
        ]

        for path in candidates:

            if path.exists():
                return path

        raise FileNotFoundError(
            "\nImage could not be found.\n"
            f"CSV path     : {relative_path}\n"
            f"Dataset root : {self.dataset_root}\n\n"
            "Tried:\n"
            + "\n".join(
                f"  {path}"
                for path in candidates
            )
        )

    # ======================================================
    # Encode WBC label
    # ======================================================

    @staticmethod
    def _encode_cell_label(value):

        # Convert input to normalized string
        normalized_value = (
            str(value)
            .strip()
            .lower()
        )

        # --------------------------------------------------
        # Case 1:
        # CELL_LABELS uses lowercase string keys
        # --------------------------------------------------

        normalized_labels = {
            str(key).strip().lower(): int(index)
            for key, index in CELL_LABELS.items()
        }

        if normalized_value in normalized_labels:

            return normalized_labels[
                normalized_value
            ]

        # --------------------------------------------------
        # Case 2:
        # CSV contains integer labels
        # --------------------------------------------------

        try:

            numeric_value = int(
                float(str(value).strip())
            )

            if numeric_value in normalized_labels.values():

                return numeric_value

        except (ValueError, TypeError):

            pass

        # --------------------------------------------------
        # Failed
        # --------------------------------------------------

        raise ValueError(
            "\nUnknown WBC label:\n"
            f"'{value}'\n\n"
            f"Normalized value:\n"
            f"'{normalized_value}'\n\n"
            f"Valid labels:\n"
            f"{list(CELL_LABELS.keys())}"
        )

    # ======================================================
    # Encode attributes
    # ======================================================

    @staticmethod
    def _encode_attributes(row):

        encoded_attributes = []

        for attribute in ATTRIBUTE_NAMES:

            value = str(
                row[attribute]
            ).strip().lower()

            encoder = ATTRIBUTE_ENCODERS[
                attribute
            ]

            # Normalize encoder keys too
            normalized_encoder = {
                str(key).strip().lower(): int(index)
                for key, index in encoder.items()
            }

            if value not in normalized_encoder:

                raise ValueError(
                    "\nUnknown attribute value.\n"
                    f"Attribute : {attribute}\n"
                    f"Value     : '{value}'\n"
                    f"Valid     : "
                    f"{list(encoder.keys())}"
                )

            encoded_attributes.append(
                normalized_encoder[value]
            )

        return torch.tensor(
            encoded_attributes,
            dtype=torch.long,
        )

    # ======================================================
    # Get item
    # ======================================================

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # --------------------------------------------------
        # Image path
        # --------------------------------------------------

        image_path = self._resolve_image_path(
            row["path"]
        )

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:

            raise RuntimeError(
                "\nFailed to load image.\n"
                f"Image : {image_path}\n"
                f"Error : {exc}"
            ) from exc

        # --------------------------------------------------
        # Transform
        # --------------------------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        # --------------------------------------------------
        # WBC class
        # --------------------------------------------------

        cell_label = self._encode_cell_label(
            row["label"]
        )

        cell_label = torch.tensor(
            cell_label,
            dtype=torch.long,
        )

        # --------------------------------------------------
        # Morphology attributes
        # --------------------------------------------------

        attributes = self._encode_attributes(
            row
        )

        # --------------------------------------------------
        # Return dictionary
        # --------------------------------------------------

        return {

            "index": index,

            "image": image,

            "cell_label": cell_label,

            "attributes": attributes,

            "image_name": str(
                row["img_name"]
            ),

            "image_path": str(
                image_path
            ),
        }


# ==========================================================
# Dataset statistics
# ==========================================================

def print_dataset_statistics(dataset):

    print("=" * 70)
    print("WBCAtt DATASET")
    print("=" * 70)

    print(
        f"\nSamples          : {len(dataset)}"
    )

    print(
        f"Annotation file  : "
        f"{dataset.annotation_file}"
    )

    print(
        f"Dataset root     : "
        f"{dataset.dataset_root}"
    )

    print(
        f"Transform        : "
        f"{dataset.transform}"
    )

    print("\nCSV columns:")

    for column in dataset.data.columns:

        print(
            f"  - {column}"
        )

    print(
        "\n" + "=" * 70
    )


# ==========================================================
# Validate sample
# ==========================================================

def validate_sample(sample):

    required_keys = {
        "index",
        "image",
        "cell_label",
        "attributes",
        "image_name",
        "image_path",
    }

    actual_keys = set(
        sample.keys()
    )

    missing = (
        required_keys - actual_keys
    )

    if missing:

        raise RuntimeError(
            f"Missing sample keys: {missing}"
        )

    # ------------------------------------------------------
    # Image
    # ------------------------------------------------------

    if not isinstance(
        sample["image"],
        torch.Tensor,
    ):

        raise TypeError(
            "sample['image'] must be torch.Tensor"
        )

    if sample["image"].ndim != 3:

        raise ValueError(
            "Image must have shape (C,H,W)"
        )

    if sample["image"].shape[0] != 3:

        raise ValueError(
            "Image must contain 3 RGB channels"
        )

    # ------------------------------------------------------
    # Cell label
    # ------------------------------------------------------

    if not isinstance(
        sample["cell_label"],
        torch.Tensor,
    ):

        raise TypeError(
            "cell_label must be torch.Tensor"
        )

    if sample["cell_label"].dtype != torch.long:

        raise TypeError(
            "cell_label must use torch.long"
        )

    # ------------------------------------------------------
    # Attributes
    # ------------------------------------------------------

    if not isinstance(
        sample["attributes"],
        torch.Tensor,
    ):

        raise TypeError(
            "attributes must be torch.Tensor"
        )

    if sample["attributes"].shape != (
        len(ATTRIBUTE_NAMES),
    ):

        raise ValueError(
            "Attribute tensor must have shape "
            f"({len(ATTRIBUTE_NAMES)},)"
        )

    if sample["attributes"].dtype != torch.long:

        raise TypeError(
            "attributes must use torch.long"
        )


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    from config import (
        TRAIN_CSV,
        DATASET_ROOT,
    )

    from data.transforms import (
        train_transform,
    )

    print("=" * 70)
    print("WBCAtt Dataset Test")
    print("=" * 70)

    print(
        "\nCreating training dataset..."
    )

    dataset = WBCDataset(
        annotation_file=TRAIN_CSV,
        dataset_root=DATASET_ROOT,
        transform=train_transform,
    )

    print_dataset_statistics(
        dataset
    )

    print(
        "\nDataset size:"
    )

    print(
        len(dataset)
    )

    print(
        "\nLoading first sample..."
    )

    sample = dataset[0]

    validate_sample(
        sample
    )

    print(
        "Sample validation: PASSED"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FIRST SAMPLE"
    )

    print(
        "=" * 70
    )

    print(
        "\nIndex:"
    )
    print(
        sample["index"]
    )

    print(
        "\nImage name:"
    )
    print(
        sample["image_name"]
    )

    print(
        "\nImage path:"
    )
    print(
        sample["image_path"]
    )

    print(
        "\nImage tensor shape:"
    )
    print(
        sample["image"].shape
    )

    print(
        "\nImage tensor dtype:"
    )
    print(
        sample["image"].dtype
    )

    print(
        "\nWBC label:"
    )
    print(
        sample["cell_label"]
    )

    print(
        "\nAttribute tensor:"
    )
    print(
        sample["attributes"]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ATTRIBUTE ENCODING"
    )

    print(
        "=" * 70
    )

    for index, attribute in enumerate(
        ATTRIBUTE_NAMES
    ):

        value = sample[
            "attributes"
        ][index].item()

        print(
            f"{attribute:30} -> {value}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "Dataset validation: PASSED"
    )

    print(
        "=" * 70
    )