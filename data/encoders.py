"""
encoders.py

Label encoders for the WBCAtt dataset.

This module contains mappings from categorical labels to integer IDs
used throughout the project for training, evaluation, and inference.

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

# ==========================================================
# White Blood Cell Subtype Labels
# ==========================================================

CELL_LABELS = {
    "basophil": 0,
    "eosinophil": 1,
    "erythroblast": 2,
    "ig": 3,
    "lymphocyte": 4,
    "monocyte": 5,
    "neutrophil": 6,
    "platelet": 7,
}

# Reverse mapping
CELL_LABELS_INV = {v: k for k, v in CELL_LABELS.items()}


# ==========================================================
# Morphology Attribute Encoders
# ==========================================================

ATTRIBUTE_ENCODERS = {

    "cell_size": {
        "big": 0,
        "small": 1,
    },

    "cell_shape": {
        "irregular": 0,
        "round": 1,
    },

    "nucleus_shape": {
        "irregular": 0,
        "segmented-bilobed": 1,
        "segmented-multilobed": 2,
        "unsegmented-band": 3,
        "unsegmented-indented": 4,
        "unsegmented-round": 5,
    },

    "nuclear_cytoplasmic_ratio": {
        "high": 0,
        "low": 1,
    },

    "chromatin_density": {
        "densely": 0,
        "loosely": 1,
    },

    "cytoplasm_vacuole": {
        "no": 0,
        "yes": 1,
    },

    "cytoplasm_texture": {
        "clear": 0,
        "frosted": 1,
    },

    "cytoplasm_colour": {
        "blue": 0,
        "light blue": 1,
        "purple blue": 2,
    },

    "granule_type": {
        "coarse": 0,
        "nil": 1,
        "round": 2,
        "small": 3,
    },

    "granule_colour": {
        "nil": 0,
        "pink": 1,
        "purple": 2,
        "red": 3,
    },

    "granularity": {
        "no": 0,
        "yes": 1,
    },
}


# ==========================================================
# Reverse Attribute Encoders
# ==========================================================

ATTRIBUTE_ENCODERS_INV = {
    attribute: {v: k for k, v in mapping.items()}
    for attribute, mapping in ATTRIBUTE_ENCODERS.items()
}


# ==========================================================
# Attribute Names
# ==========================================================

ATTRIBUTE_NAMES = [
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
    "granularity",
]


# ==========================================================
# Number of Classes
# ==========================================================

NUM_CELL_CLASSES = len(CELL_LABELS)

NUM_ATTRIBUTE_CLASSES = {
    attribute: len(mapping)
    for attribute, mapping in ATTRIBUTE_ENCODERS.items()
}


# ==========================================================
# Utility Functions
# ==========================================================

def encode_cell_label(label: str) -> int:
    """
    Convert a WBC subtype label into an integer.

    Example:
        "neutrophil" -> 6
    """
    return CELL_LABELS[label.lower()]


def decode_cell_label(index: int) -> str:
    """
    Convert an integer prediction back to its WBC subtype.

    Example:
        6 -> "neutrophil"
    """
    return CELL_LABELS_INV[index]


def encode_attribute(attribute_name: str, value: str) -> int:
    """
    Encode a morphology attribute.

    Example:
        encode_attribute("cell_size", "big") -> 0
    """
    return ATTRIBUTE_ENCODERS[attribute_name][value.lower()]


def decode_attribute(attribute_name: str, index: int) -> str:
    """
    Decode an integer prediction back into a morphology label.

    Example:
        decode_attribute("cell_size", 0) -> "big"
    """
    return ATTRIBUTE_ENCODERS_INV[attribute_name][index]


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("White Blood Cell Labels")
    print("=" * 60)

    print("Neutrophil ->", encode_cell_label("neutrophil"))
    print("6 ->", decode_cell_label(6))

    print("\n" + "=" * 60)
    print("Morphology Attributes")
    print("=" * 60)

    print("cell_size (big) ->",
          encode_attribute("cell_size", "big"))

    print("0 ->",
          decode_attribute("cell_size", 0))

    print("\n" + "=" * 60)
    print("Number of Classes")
    print("=" * 60)

    print("Cell Classes:", NUM_CELL_CLASSES)

    print("\nMorphology Attribute Classes:")
    for attribute, num_classes in NUM_ATTRIBUTE_CLASSES.items():
        print(f"{attribute:35} : {num_classes}")