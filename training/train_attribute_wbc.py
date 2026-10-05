"""Train the attribute-to-WBC head on one-hot ground-truth CSV attributes.

This stage trains independently and leaves the image backbone unchanged.
Automatic image inference supplies predicted attribute probabilities to the
shared WBC MLP; the GUI does not require human approval or correction.
"""

from utils.seed import set_seed
from data.encoders import ATTRIBUTE_ENCODERS, ATTRIBUTE_NAMES, CELL_LABELS
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DEVICE,
    NUM_WBC_CLASSES,
    CHECKPOINT_DIR,
)

from data.dataloader import create_dataloaders

from models.attribute_wbc_classifier import (
    AttributeWBCClassifier,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

NUM_EPOCHS = 50

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

PATIENCE = 8

CHECKPOINT_NAME = (
    "best_attribute_wbc.pth"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / CHECKPOINT_NAME
)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    DEVICE
)


# ==========================================================
# ACCURACY
# ==========================================================

def calculate_accuracy(
    logits,
    labels,
):
    """
    Calculate classification accuracy.
    """

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    correct = (
        predictions == labels
    ).sum().item()

    total = labels.size(0)

    return correct / total


# ==========================================================
# TRAIN ONE EPOCH
# ==========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):
    """
    Train for one epoch.
    """

    model.train()

    running_loss = 0.0

    running_correct = 0

    running_total = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for batch in progress:

        attributes = batch[
            "attributes"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        logits = model(
            attributes
        )

        loss = criterion(
            logits,
            labels,
        )

        # --------------------------------------------------
        # Backward
        # --------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        # --------------------------------------------------
        # Statistics
        # --------------------------------------------------

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        running_correct += (
            predictions == labels
        ).sum().item()

        running_total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = (
        running_loss
        / running_total
    )

    epoch_accuracy = (
        running_correct
        / running_total
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ==========================================================
# VALIDATION
# ==========================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
):
    """
    Evaluate the model.
    """

    model.eval()

    running_loss = 0.0

    running_correct = 0

    running_total = 0

    for batch in loader:

        attributes = batch[
            "attributes"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        logits = model(
            attributes
        )

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        running_correct += (
            predictions == labels
        ).sum().item()

        running_total += batch_size

    epoch_loss = (
        running_loss
        / running_total
    )

    epoch_accuracy = (
        running_correct
        / running_total
    )

    return (
        epoch_loss,
        epoch_accuracy,
    )


# ==========================================================
# TEST
# ==========================================================

@torch.no_grad()
def test_model(
    model,
    loader,
):
    """
    Evaluate final model on test set.
    """

    model.eval()

    running_correct = 0

    running_total = 0

    predictions_all = []

    labels_all = []

    for batch in loader:

        attributes = batch[
            "attributes"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        logits = model(
            attributes
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        running_correct += (
            predictions == labels
        ).sum().item()

        running_total += (
            labels.size(0)
        )

        predictions_all.extend(
            predictions.cpu().tolist()
        )

        labels_all.extend(
            labels.cpu().tolist()
        )

    accuracy = (
        running_correct
        / running_total
    )

    return (
        accuracy,
        predictions_all,
        labels_all,
    )


# ==========================================================
# SAVE CHECKPOINT
# ==========================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_accuracy,
):
    """
    Save best model.
    """

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "val_loss":
                val_loss,

            "val_accuracy":
                val_accuracy,
            "attribute_encoders": ATTRIBUTE_ENCODERS,
            "attribute_names": ATTRIBUTE_NAMES,
            "cell_labels": CELL_LABELS,
        },
        CHECKPOINT_PATH,
    )


# ==========================================================
# MAIN TRAINING
# ==========================================================

def main():

    set_seed()

    print("=" * 70)

    print(
        "ATTRIBUTE → WBC CLASSIFIER TRAINING"
    )

    print("=" * 70)

    print(
        f"\nDevice       : {device}"
    )

    print(
        f"Epochs       : {NUM_EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Batch size   : "
        f"from config"
    )

    print(
        f"WBC classes  : "
        f"{NUM_WBC_CLASSES}"
    )

    # ======================================================
    # Data
    # ======================================================

    print(
        "\nCreating dataloaders..."
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Val samples  : "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples : "
        f"{len(test_loader.dataset)}"
    )

    # ======================================================
    # Model
    # ======================================================

    model = (
        AttributeWBCClassifier()
        .to(device)
    )

    print(
        "\nModel:"
    )

    print(model)

    # ======================================================
    # Loss
    # ======================================================

    criterion = nn.CrossEntropyLoss()

    # ======================================================
    # Optimizer
    # ======================================================

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ======================================================
    # Training
    # ======================================================

    best_val_accuracy = -1.0

    best_val_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    print(
        "\nStarting training..."
    )

    print(
        "-" * 70
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print(
            f"\nEpoch "
            f"{epoch}/{NUM_EPOCHS}"
        )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
            )
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        val_loss, val_accuracy = (
            evaluate(
                model,
                val_loader,
                criterion,
            )
        )

        # --------------------------------------------------
        # Print
        # --------------------------------------------------

        print(
            f"Train Loss     : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss       : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy   : "
            f"{val_accuracy * 100:.2f}%"
        )

        # --------------------------------------------------
        # Save best
        # --------------------------------------------------

        improved = (
            val_accuracy
            > best_val_accuracy
        )

        if improved:

            best_val_accuracy = (
                val_accuracy
            )

            best_val_loss = (
                val_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )

            print(
                "\n✓ Best model saved"
            )

            print(
                f"  Path: "
                f"{CHECKPOINT_PATH}"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

        # --------------------------------------------------
        # Early stopping
        # --------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break

    # ======================================================
    # Load best checkpoint
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "LOADING BEST MODEL"
    )

    print(
        "=" * 70
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"\nBest validation accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    # ======================================================
    # Test
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TEST SET EVALUATION"
    )

    print(
        "=" * 70
    )

    (
        test_accuracy,
        predictions,
        labels,
    ) = test_model(
        model,
        test_loader,
    )

    print(
        f"\nTest Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # ======================================================
    # Per-class accuracy
    # ======================================================

    print(
        "\nPer-class accuracy:"
    )

    for class_index in range(
        NUM_WBC_CLASSES
    ):

        class_correct = 0

        class_total = 0

        for prediction, label in zip(
            predictions,
            labels,
        ):

            if label == class_index:

                class_total += 1

                if prediction == label:

                    class_correct += 1

        if class_total > 0:

            class_accuracy = (
                class_correct
                / class_total
            )

            print(
                f"  Class "
                f"{class_index}: "
                f"{class_accuracy * 100:.2f}% "
                f"({class_correct}/"
                f"{class_total})"
            )

    # ======================================================
    # Final
    # ======================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ATTRIBUTE → WBC TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nBest checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        f"\nValidation accuracy:"
        f" {best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Test accuracy:"
        f" {test_accuracy * 100:.2f}%"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    main()
