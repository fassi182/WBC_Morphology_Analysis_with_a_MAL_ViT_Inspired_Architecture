"""
config.py

Central configuration for:
    WBC Morphology Analysis using MAL-ViT

Dataset:
    WBCAtt

Training samples:
    6,169

Architecture:
    MAL-ViT

Token sequence:
    11 Attribute Tokens
    + 4 Register Tokens
    + 196 Patch Tokens
    = 211 Tokens
"""

from pathlib import Path

import torch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "WBCAtt"
)

ANNOTATION_DIR = (
    DATASET_ROOT
    / "annotations"
)

IMAGE_DIR = (
    DATASET_ROOT
    / "PBC_dataset_normal_DIB"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

LOG_DIR = (
    OUTPUT_DIR
    / "logs"
)

PLOT_DIR = (
    OUTPUT_DIR
    / "plots"
)


# ============================================================
# ANNOTATION FILES
# ============================================================

TRAIN_CSV = (
    ANNOTATION_DIR
    / "pbc_attr_v1_train.csv"
)

VAL_CSV = (
    ANNOTATION_DIR
    / "pbc_attr_v1_val.csv"
)

TEST_CSV = (
    ANNOTATION_DIR
    / "test.csv"
)


# ============================================================
# CHECKPOINTS
# ============================================================

BEST_MODEL_NAME = "best_model.pth"

LAST_CHECKPOINT_NAME = "last_checkpoint.pth"

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / BEST_MODEL_NAME
)

LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / LAST_CHECKPOINT_NAME
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

PATCH_SIZE = 16

NUM_PATCHES_SIDE = (
    IMAGE_SIZE // PATCH_SIZE
)

NUM_PATCHES = (
    NUM_PATCHES_SIDE ** 2
)

# 224 / 16 = 14
# 14 x 14 = 196

assert IMAGE_SIZE % PATCH_SIZE == 0


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

# ImageNet normalization.
#
# This is appropriate for the standard ViT-style input
# preprocessing used by this architecture.

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================
# MAL-ViT TOKEN CONFIGURATION
# ============================================================

# One learnable token per morphology attribute.

NUM_ATTRIBUTE_TOKENS = 11


# Register tokens are free latent tokens.
#
# They are NOT associated with a morphology attribute.
#
# Their purpose is to provide unconstrained latent capacity
# and reduce pressure on patch tokens.

NUM_REGISTER_TOKENS = 4


# Total sequence length:
#
# 11 attribute tokens
# + 4 register tokens
# + 196 patch tokens
# = 211

TOTAL_TOKENS = (
    NUM_ATTRIBUTE_TOKENS
    + NUM_REGISTER_TOKENS
    + NUM_PATCHES
)


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBED_DIM = 192

EMBED_DROPOUT = 0.10


# ============================================================
# TRANSFORMER CONFIGURATION
# ============================================================

NUM_TRANSFORMER_BLOCKS = 6

NUM_ATTENTION_HEADS = 6

MLP_RATIO = 4.0

MLP_HIDDEN_DIM = int(
    EMBED_DIM * MLP_RATIO
)

TRANSFORMER_DROPOUT = 0.10

ATTENTION_DROPOUT = 0.10

LAYER_NORM_EPS = 1e-6


# ============================================================
# WBC CLASSIFICATION
# ============================================================

WBC_CLASSES = [
    "Neutrophil",
    "Eosinophil",
    "Monocyte",
    "Basophil",
    "Lymphocyte",
]

NUM_WBC_CLASSES = len(
    WBC_CLASSES
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

NUM_ATTRIBUTES = len(
    ATTRIBUTE_NAMES
)


# ============================================================
# ATTRIBUTE CLASS COUNTS
# ============================================================

ATTRIBUTE_CLASS_COUNTS = {

    "cell_size": {
        "big": 3293,
        "small": 2876,
    },

    "cell_shape": {
        "round": 4774,
        "irregular": 1395,
    },

    "nucleus_shape": {
        "segmented-bilobed": 1899,
        "unsegmented-band": 1587,
        "unsegmented-indented": 769,
        "segmented-multilobed": 725,
        "unsegmented-round": 662,
        "irregular": 527,
    },

    "nuclear_cytoplasmic_ratio": {
        "low": 5424,
        "high": 745,
    },

    "chromatin_density": {
        "densely": 5631,
        "loosely": 538,
    },

    "cytoplasm_vacuole": {
        "no": 5705,
        "yes": 464,
    },

    "cytoplasm_texture": {
        "clear": 4970,
        "frosted": 1199,
    },

    "cytoplasm_colour": {
        "light blue": 4693,
        "blue": 832,
        "purple blue": 644,
    },

    "granule_type": {
        "small": 1987,
        "round": 1875,
        "nil": 1564,
        "coarse": 743,
    },

    "granule_colour": {
        "pink": 1936,
        "red": 1876,
        "nil": 1564,
        "purple": 793,
    },

    "granularity": {
        "yes": 4606,
        "no": 1563,
    },
}


# ============================================================
# WBC TRAINING DISTRIBUTION
# ============================================================

WBC_CLASS_COUNTS = {

    "Neutrophil": 1984,

    "Eosinophil": 1876,

    "Monocyte": 829,

    "Basophil": 744,

    "Lymphocyte": 736,
}


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

NUM_EPOCHS = 30

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

GRADIENT_CLIP_NORM = 1.0


# ============================================================
# OPTIMIZER
# ============================================================

OPTIMIZER_NAME = "AdamW"


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

SCHEDULER_NAME = "cosine"

WARMUP_EPOCHS = 5

MIN_LEARNING_RATE = 1e-6


# ============================================================
# EARLY STOPPING
# ============================================================

EARLY_STOPPING_PATIENCE = 10

EARLY_STOPPING_MIN_DELTA = 1e-4


# ============================================================
# LOSS WEIGHTS
# ============================================================

WBC_LOSS_WEIGHT = 1.0

ATTRIBUTE_LOSS_WEIGHT = 1.0


# ============================================================
# DATA LOADER
# ============================================================

NUM_WORKERS = 0

PIN_MEMORY = (
    DEVICE.type == "cuda"
)

DROP_LAST = False


# ============================================================
# DATA AUGMENTATION
# ============================================================

TRAIN_RANDOM_HORIZONTAL_FLIP = True

TRAIN_RANDOM_VERTICAL_FLIP = False

TRAIN_RANDOM_ROTATION = 15


# ============================================================
# MODEL INITIALIZATION
# ============================================================

INIT_STD = 0.02


# ============================================================
# XAI CONFIGURATION
# ============================================================

XAI_TARGET_LAYER = -1

XAI_PATCH_SIZE = PATCH_SIZE

XAI_IMAGE_SIZE = IMAGE_SIZE


# ============================================================
# DIAGNOSTIC CONFIGURATION
# ============================================================

ATTENTION_SINK_WARNING_THRESHOLD = 25.0


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

for directory in [
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    LOG_DIR,
    PLOT_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

assert NUM_ATTRIBUTE_TOKENS == NUM_ATTRIBUTES

assert NUM_PATCHES == 196

assert NUM_ATTRIBUTE_TOKENS == 11

assert NUM_REGISTER_TOKENS == 4

assert TOTAL_TOKENS == 211

assert NUM_WBC_CLASSES == 5


# ============================================================
# CONFIGURATION SUMMARY
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("WBC MORPHOLOGY ANALYSIS USING MAL-ViT")
    print("=" * 70)

    print(
        f"\nProject root       : {PROJECT_ROOT}"
    )

    print(
        f"Dataset root       : {DATASET_ROOT}"
    )

    print(
        f"Training CSV       : {TRAIN_CSV}"
    )

    print(
        f"Validation CSV     : {VAL_CSV}"
    )

    print(
        f"Test CSV           : {TEST_CSV}"
    )

    print(
        f"\nDevice             : {DEVICE}"
    )

    print("\nImage configuration")

    print(
        f"Image size         : {IMAGE_SIZE}"
    )

    print(
        f"Patch size         : {PATCH_SIZE}"
    )

    print(
        f"Patch grid         : "
        f"{NUM_PATCHES_SIDE} x "
        f"{NUM_PATCHES_SIDE}"
    )

    print(
        f"Number of patches  : {NUM_PATCHES}"
    )

    print("\nMAL-ViT tokens")

    print(
        f"Attribute tokens   : "
        f"{NUM_ATTRIBUTE_TOKENS}"
    )

    print(
        f"Register tokens    : "
        f"{NUM_REGISTER_TOKENS}"
    )

    print(
        f"Patch tokens       : "
        f"{NUM_PATCHES}"
    )

    print(
        f"Total tokens       : "
        f"{TOTAL_TOKENS}"
    )

    print("\nTransformer")

    print(
        f"Embedding dimension: "
        f"{EMBED_DIM}"
    )

    print(
        f"Transformer blocks : "
        f"{NUM_TRANSFORMER_BLOCKS}"
    )

    print(
        f"Attention heads    : "
        f"{NUM_ATTENTION_HEADS}"
    )

    print(
        f"MLP hidden dim     : "
        f"{MLP_HIDDEN_DIM}"
    )

    print("\nTasks")

    print(
        f"WBC classes        : "
        f"{NUM_WBC_CLASSES}"
    )

    print(
        f"Attributes         : "
        f"{NUM_ATTRIBUTES}"
    )

    print("\nTraining")

    print(
        f"Batch size         : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs             : "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Learning rate      : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay       : "
        f"{WEIGHT_DECAY}"
    )

    print(
        "\n" + "=" * 70
    )