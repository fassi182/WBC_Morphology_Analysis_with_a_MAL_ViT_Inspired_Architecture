"""
config.py

Central configuration file for the MAL-ViT project.

Every module in the project should import configuration values
from this file.

Author: Muhammad Fassi Ur Rehman
Project: Explainable White Blood Cell Morphology Analysis using MAL-ViT
"""

from pathlib import Path
import torch

# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "datasets" / "WBCAtt"

ANNOTATION_DIR = DATASET_ROOT / "annotations"

TRAIN_CSV = ANNOTATION_DIR / "pbc_attr_v1_train.csv"
VAL_CSV = ANNOTATION_DIR / "pbc_attr_v1_val.csv"
TEST_CSV = ANNOTATION_DIR / "test.csv"

# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Model checkpoints
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Backward compatibility
MODEL_DIR = CHECKPOINT_DIR

# Logs
LOG_DIR = OUTPUT_DIR / "logs"

# Training curves / confusion matrices / Grad-CAM
PLOT_DIR = OUTPUT_DIR / "plots"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Image Settings
# ==========================================================

IMAGE_SIZE = 224
PATCH_SIZE = 16
IN_CHANNELS = 3

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# ==========================================================
# Dataset Information
# ==========================================================

NUM_CELL_CLASSES = 8

NUM_ATTRIBUTES = 11

ATTRIBUTE_CLASSES = {

    "cell_size": 2,

    "cell_shape": 2,

    "nucleus_shape": 6,

    "nuclear_cytoplasmic_ratio": 2,

    "chromatin_density": 2,

    "cytoplasm_vacuole": 2,

    "cytoplasm_texture": 2,

    "cytoplasm_colour": 3,

    "granule_type": 4,

    "granule_colour": 4,

    "granularity": 2,
}

# ==========================================================
# DataLoader
# ==========================================================

BATCH_SIZE = 32

NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

# ==========================================================
# Training
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42
# ==========================================================
# Loss Weights
# ==========================================================

ATTRIBUTE_LOSS_WEIGHT = 0.5
WBC_LOSS_WEIGHT = 1.0

# ==========================================================
# Learning Rate Scheduler
# ==========================================================

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 5

MIN_LEARNING_RATE = 1e-6

# ==========================================================
# Checkpoint Names
# ==========================================================

BEST_MODEL_NAME = "best_model.pth"

LAST_CHECKPOINT_NAME = "last_checkpoint.pth"
# ==========================================================
# ViT-Tiny Backbone (Paper Configuration)
# ==========================================================

EMBED_DIM = 192

NUM_HEADS = 3

NUM_TRANSFORMER_LAYERS = 12

MLP_RATIO = 4

HEAD_DIM = EMBED_DIM // NUM_HEADS

QKV_BIAS = True

ATTENTION_DROPOUT = 0.0

PROJECTION_DROPOUT = 0.0

MLP_DROPOUT = 0.0

EMBED_DROPOUT = 0.0

# ==========================================================
# MAL-ViT
# ==========================================================

NUM_ATTRIBUTE_TOKENS = NUM_ATTRIBUTES

# ==========================================================
# WBC Classification Head (Our Extension)
# ==========================================================

CLASSIFIER_HIDDEN_DIM = 128

CLASSIFIER_DROPOUT = 0.30

# ==========================================================
# Checkpoint Settings
# ==========================================================

BEST_MODEL_NAME = "best_model.pth"

LAST_MODEL_NAME = "last_model.pth"

SAVE_BEST_ONLY = True

# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Project Configuration")
    print("=" * 60)

    print(f"Project Root        : {PROJECT_ROOT}")
    print(f"Dataset Root        : {DATASET_ROOT}")
    print(f"Device              : {DEVICE}")

    print("\nImage")
    print("-" * 60)
    print(f"Image Size          : {IMAGE_SIZE}")
    print(f"Patch Size          : {PATCH_SIZE}")
    print(f"Number of Patches   : {NUM_PATCHES}")

    print("\nViT-Tiny")
    print("-" * 60)
    print(f"Embedding Dimension : {EMBED_DIM}")
    print(f"Attention Heads     : {NUM_HEADS}")
    print(f"Transformer Layers  : {NUM_TRANSFORMER_LAYERS}")

    print("\nDataset")
    print("-" * 60)
    print(f"WBC Classes         : {NUM_CELL_CLASSES}")
    print(f"Attributes          : {NUM_ATTRIBUTES}")

    print("\nOutput Directories")
    print("-" * 60)
    print(f"Checkpoints         : {CHECKPOINT_DIR}")
    print(f"Logs                : {LOG_DIR}")
    print(f"Plots               : {PLOT_DIR}")