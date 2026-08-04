"""
visualization.py

Visualization utilities for Explainable WBC Classification
using MAL-ViT.

Functions
---------
1. Plot training/validation loss
2. Plot training/validation accuracy
3. Plot confusion matrix

Author:
Muhammad Fassi Ur Rehman
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

from config import OUTPUT_DIR


# ==========================================================
# Plot Directory
# ==========================================================

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Loss Curve
# ==========================================================

def plot_loss(history, save=True):
    """
    Plot training and validation loss.

    Parameters
    ----------
    history : dict
        {
            "train_loss": [...],
            "val_loss": [...]
        }
    """

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()

    if save:
        plt.savefig(
            PLOT_DIR / "loss_curve.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ==========================================================
# Accuracy Curve
# ==========================================================

def plot_accuracy(history, save=True):
    """
    Plot WBC accuracy.

    Parameters
    ----------
    history : dict
        {
            "train_accuracy": [...],
            "val_accuracy": [...]
        }
    """

    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy",
        linewidth=2,
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("WBC Classification Accuracy")
    plt.grid(True)
    plt.legend()

    if save:
        plt.savefig(
            PLOT_DIR / "accuracy_curve.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ==========================================================
# Mean Attribute Accuracy
# ==========================================================

def plot_attribute_accuracy(history, save=True):
    """
    Plot mean attribute accuracy.
    """

    epochs = range(
        1,
        len(history["attribute_accuracy"]) + 1,
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["attribute_accuracy"],
        linewidth=2,
        label="Mean Attribute Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Morphology Attribute Accuracy")
    plt.grid(True)
    plt.legend()

    if save:
        plt.savefig(
            PLOT_DIR / "attribute_accuracy.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ==========================================================
# Confusion Matrix
# ==========================================================

def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    save=True,
):
    """
    Plot confusion matrix.
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=False,
    )

    plt.title("Confusion Matrix")

    if save:
        plt.savefig(
            PLOT_DIR / "confusion_matrix.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Visualization Test")
    print("=" * 60)

    history = {

        "train_loss": [
            2.4,
            1.8,
            1.4,
            1.1,
            0.9,
        ],

        "val_loss": [
            2.6,
            2.0,
            1.6,
            1.3,
            1.0,
        ],

        "train_accuracy": [
            42,
            58,
            69,
            77,
            84,
        ],

        "val_accuracy": [
            40,
            54,
            66,
            74,
            81,
        ],

        "attribute_accuracy": [
            55,
            61,
            68,
            74,
            80,
        ],
    }

    plot_loss(history)

    plot_accuracy(history)

    plot_attribute_accuracy(history)

    cm = np.array([
        [8, 1, 0],
        [2, 7, 1],
        [0, 1, 9],
    ])

    plot_confusion_matrix(
        cm,
        ["A", "B", "C"],
    )

    print("\nPlots saved to:")
    print(PLOT_DIR)