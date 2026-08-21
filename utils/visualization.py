"""
visualization.py

Training visualization utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from config import PLOT_DIR


def plot_training_history(
    history,
    save=True,
):

    if not history:
        return

    epochs = [
        item["epoch"]
        for item in history
    ]

    train_loss = [
        item["train_loss"]
        for item in history
    ]

    val_loss = [
        item["val_loss"]
        for item in history
    ]

    plt.figure()

    plt.plot(
        epochs,
        train_loss,
        label="Train Loss",
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title(
        "MAL-ViT Training Loss"
    )

    plt.legend()

    plt.grid(True)

    if save:

        output_path = (
            PLOT_DIR
            / "training_loss.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Saved: {output_path}"
        )

    plt.close()


def plot_accuracy(
    history,
    save=True,
):

    if not history:
        return

    epochs = [
        item["epoch"]
        for item in history
    ]

    train_accuracy = [
        item["train_accuracy"]
        for item in history
    ]

    val_accuracy = [
        item["val_accuracy"]
        for item in history
    ]

    plt.figure()

    plt.plot(
        epochs,
        train_accuracy,
        label="Train Accuracy",
    )

    plt.plot(
        epochs,
        val_accuracy,
        label="Validation Accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.title(
        "MAL-ViT Classification Accuracy"
    )

    plt.legend()

    plt.grid(True)

    if save:

        output_path = (
            PLOT_DIR
            / "training_accuracy.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"Saved: {output_path}"
        )

    plt.close()