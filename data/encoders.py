"""
encoders.py

Label encoders for the WBCAtt dataset.

This module defines:

    1. WBC class encoding
    2. Morphology attribute encoding
    3. Number of classes per attribute

The ordering here is important.

The attribute token index in MAL-ViT corresponds directly to
ATTRIBUTE_NAMES.

Token 0 -> cell_size
Token 1 -> cell_shape
Token 2 -> nucleus_shape
...
Token 10 -> granularity
"""


# ============================================================
# WBC CLASS ENCODER
# ============================================================

CELL_LABELS = {

    "neutrophil": 0,

    "eosinophil": 1,

    "monocyte": 2,

    "basophil": 3,

    "lymphocyte": 4,
}


# Human-readable WBC classes.

WBC_LABEL_NAMES = {

    index: name

    for name, index in CELL_LABELS.items()
}


NUM_WBC_CLASSES = len(
    CELL_LABELS
)


# ============================================================
# ATTRIBUTE NAMES
# ============================================================

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


# ============================================================
# ATTRIBUTE ENCODERS
# ============================================================

ATTRIBUTE_ENCODERS = {

    # --------------------------------------------------------
    # Token 0
    # --------------------------------------------------------

    "cell_size": {

        "big": 0,

        "small": 1,
    },


    # --------------------------------------------------------
    # Token 1
    # --------------------------------------------------------

    "cell_shape": {

        "round": 0,

        "irregular": 1,
    },


    # --------------------------------------------------------
    # Token 2
    # --------------------------------------------------------

    "nucleus_shape": {

        "segmented-bilobed": 0,

        "unsegmented-band": 1,

        "unsegmented-indented": 2,

        "segmented-multilobed": 3,

        "unsegmented-round": 4,

        "irregular": 5,
    },


    # --------------------------------------------------------
    # Token 3
    # --------------------------------------------------------

    "nuclear_cytoplasmic_ratio": {

        "low": 0,

        "high": 1,
    },


    # --------------------------------------------------------
    # Token 4
    # --------------------------------------------------------

    "chromatin_density": {

        "densely": 0,

        "loosely": 1,
    },


    # --------------------------------------------------------
    # Token 5
    # --------------------------------------------------------

    "cytoplasm_vacuole": {

        "no": 0,

        "yes": 1,
    },


    # --------------------------------------------------------
    # Token 6
    # --------------------------------------------------------

    "cytoplasm_texture": {

        "clear": 0,

        "frosted": 1,
    },


    # --------------------------------------------------------
    # Token 7
    # --------------------------------------------------------

    "cytoplasm_colour": {

        "light blue": 0,

        "blue": 1,

        "purple blue": 2,
    },


    # --------------------------------------------------------
    # Token 8
    # --------------------------------------------------------

    "granule_type": {

        "small": 0,

        "round": 1,

        "nil": 2,

        "coarse": 3,
    },


    # --------------------------------------------------------
    # Token 9
    # --------------------------------------------------------

    "granule_colour": {

        "pink": 0,

        "red": 1,

        "nil": 2,

        "purple": 3,
    },


    # --------------------------------------------------------
    # Token 10
    # --------------------------------------------------------

    "granularity": {

        "yes": 0,

        "no": 1,
    },
}


# ============================================================
# NUMBER OF CLASSES PER ATTRIBUTE
# ============================================================

ATTRIBUTE_CLASS_COUNTS = {

    attribute: len(
        encoder
    )

    for attribute, encoder
    in ATTRIBUTE_ENCODERS.items()
}


# ============================================================
# REVERSE ATTRIBUTE ENCODERS
# ============================================================

ATTRIBUTE_DECODERS = {

    attribute: {

        index: label

        for label, index
        in encoder.items()
    }

    for attribute, encoder
    in ATTRIBUTE_ENCODERS.items()
}


# ============================================================
# VALIDATION
# ============================================================

def validate_encoders():

    # --------------------------------------------------------
    # Validate WBC classes
    # --------------------------------------------------------

    assert len(CELL_LABELS) == 5

    assert set(
        CELL_LABELS.values()
    ) == {0, 1, 2, 3, 4}


    # --------------------------------------------------------
    # Validate attribute count
    # --------------------------------------------------------

    assert len(ATTRIBUTE_NAMES) == 11

    assert len(
        ATTRIBUTE_ENCODERS
    ) == len(
        ATTRIBUTE_NAMES
    )


    # --------------------------------------------------------
    # Validate attribute ordering
    # --------------------------------------------------------

    assert (
        list(ATTRIBUTE_ENCODERS.keys())
        == ATTRIBUTE_NAMES
    )


    # --------------------------------------------------------
    # Validate each encoder
    # --------------------------------------------------------

    for attribute in ATTRIBUTE_NAMES:

        encoder = ATTRIBUTE_ENCODERS[
            attribute
        ]

        indices = set(
            encoder.values()
        )

        expected = set(
            range(len(encoder))
        )

        assert indices == expected, (
            f"Invalid encoding for "
            f"{attribute}"
        )


    return True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def encode_wbc(label: str) -> int:
    """
    Convert a WBC label into an integer.
    """

    label = str(label).strip().lower()

    if label not in CELL_LABELS:

        raise ValueError(
            f"Unknown WBC label: '{label}'. "
            f"Expected one of: "
            f"{list(CELL_LABELS.keys())}"
        )

    return CELL_LABELS[label]


def decode_wbc(index: int) -> str:
    """
    Convert WBC integer index back to label.
    """

    if index not in WBC_LABEL_NAMES:

        raise ValueError(
            f"Unknown WBC index: {index}"
        )

    return WBC_LABEL_NAMES[index]


def encode_attribute(
    attribute: str,
    value: str,
) -> int:
    """
    Encode one morphology attribute.
    """

    if attribute not in ATTRIBUTE_ENCODERS:

        raise ValueError(
            f"Unknown attribute: "
            f"'{attribute}'"
        )

    value = str(value).strip().lower()

    encoder = ATTRIBUTE_ENCODERS[
        attribute
    ]

    if value not in encoder:

        raise ValueError(
            f"Unknown value '{value}' "
            f"for attribute '{attribute}'. "
            f"Expected one of: "
            f"{list(encoder.keys())}"
        )

    return encoder[value]


def decode_attribute(
    attribute: str,
    index: int,
) -> str:
    """
    Decode one morphology attribute.
    """

    if attribute not in ATTRIBUTE_DECODERS:

        raise ValueError(
            f"Unknown attribute: "
            f"'{attribute}'"
        )

    decoder = ATTRIBUTE_DECODERS[
        attribute
    ]

    if index not in decoder:

        raise ValueError(
            f"Unknown index {index} "
            f"for attribute '{attribute}'"
        )

    return decoder[index]


# ============================================================
# RUN VALIDATION
# ============================================================

validate_encoders()


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("WBCAtt Encoder Configuration")
    print("=" * 70)

    print("\nWBC Classes:")

    for index, label in WBC_LABEL_NAMES.items():

        print(
            f"  {index}: {label.title()}"
        )


    print("\nAttributes:")

    for token_index, attribute in enumerate(
        ATTRIBUTE_NAMES
    ):

        print(
            f"\n  Token {token_index}: "
            f"{attribute}"
        )

        encoder = ATTRIBUTE_ENCODERS[
            attribute
        ]

        for label, index in encoder.items():

            print(
                f"      {index}: {label}"
            )


    print("\nNumber of classes per attribute:")

    for attribute in ATTRIBUTE_NAMES:

        print(
            f"  {attribute:30} "
            f"{ATTRIBUTE_CLASS_COUNTS[attribute]}"
        )


    print("\n" + "=" * 70)
    print("Encoder validation: PASSED")
    print("=" * 70)