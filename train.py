"""
MAL-ViT Main Training Script
============================

Complete training pipeline:

    WBCAtt Dataset
         |
         v
    DataLoaders
         |
         v
    CompleteMALViT
         |
         v
    Training
         |
         v
    Validation
         |
         +----> Best Model Checkpoint
         |
         +----> Last Checkpoint
         |
         +----> Scheduler
         |
         +----> Early Stopping

Run from project root:

    python train.py
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WBC_LOSS_WEIGHT,
    ATTRIBUTE_LOSS_WEIGHT,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    BEST_MODEL_PATH,
    LAST_CHECKPOINT_PATH,
    ATTRIBUTE_NAMES,
    SEED,
)

from data.dataloader import create_dataloaders

from models.complete_model import CompleteMALViT

from training.runner import TrainingRunner

from training.early_stopping import EarlyStopping

from utils.checkpoint_manager import CheckpointManager


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):
    """
    Set random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# ============================================================
# LOSS FUNCTION
# ============================================================

class MALViTLoss(nn.Module):
    """
    Multi-task loss for MAL-ViT.

    Total Loss:

        total_loss =
            wbc_weight * WBC_loss
            +
            attribute_weight * Attribute_loss

    WBC:
        5-class classification

    Attributes:
        11 morphological attribute classification heads
    """

    def __init__(
        self,
        wbc_weight=1.0,
        attribute_weight=1.0,
    ):

        super().__init__()

        self.wbc_weight = float(
            wbc_weight
        )

        self.attribute_weight = float(
            attribute_weight
        )

        self.wbc_loss_fn = (
            nn.CrossEntropyLoss()
        )

        self.attribute_loss_fns = (
            nn.ModuleDict()
        )

        for name in ATTRIBUTE_NAMES:

            self.attribute_loss_fns[name] = (
                nn.CrossEntropyLoss()
            )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        outputs,
        wbc_targets=None,
        attribute_targets=None,
        labels=None,
        attributes=None,
        **kwargs,
    ):
        """
        Calculate multi-task loss.

        Supports:

            wbc_targets
            attribute_targets

        Also supports:

            labels
            attributes

        for compatibility with other project modules.
        """

        # ----------------------------------------------------
        # Target aliases
        # ----------------------------------------------------

        if wbc_targets is None:

            wbc_targets = labels

        if attribute_targets is None:

            attribute_targets = attributes

        # ----------------------------------------------------
        # Validate WBC target
        # ----------------------------------------------------

        if wbc_targets is None:

            raise ValueError(
                "WBC targets were not provided."
            )

        # ----------------------------------------------------
        # Validate attribute target
        # ----------------------------------------------------

        if attribute_targets is None:

            raise ValueError(
                "Attribute targets were not provided."
            )

        # ----------------------------------------------------
        # WBC output
        # ----------------------------------------------------

        if "wbc_logits" not in outputs:

            raise KeyError(
                "Model output does not contain "
                "'wbc_logits'."
            )

        wbc_logits = outputs[
            "wbc_logits"
        ]

        # ----------------------------------------------------
        # WBC loss
        # ----------------------------------------------------

        wbc_loss = self.wbc_loss_fn(
            wbc_logits,
            wbc_targets,
        )

        # ----------------------------------------------------
        # Attribute outputs
        # ----------------------------------------------------

        if (
            "attribute_predictions"
            not in outputs
        ):

            raise KeyError(
                "Model output does not contain "
                "'attribute_predictions'."
            )

        attribute_predictions = outputs[
            "attribute_predictions"
        ]

        # ----------------------------------------------------
        # Validate attribute tensor
        # ----------------------------------------------------

        if attribute_targets.ndim != 2:

            raise ValueError(
                "attribute_targets must have shape "
                "(B, NUM_ATTRIBUTES). "
                f"Received: "
                f"{tuple(attribute_targets.shape)}"
            )

        if (
            attribute_targets.shape[1]
            != len(ATTRIBUTE_NAMES)
        ):

            raise ValueError(
                "Unexpected number of attributes. "
                f"Expected "
                f"{len(ATTRIBUTE_NAMES)}, "
                f"got "
                f"{attribute_targets.shape[1]}."
            )

        # ----------------------------------------------------
        # Attribute loss
        # ----------------------------------------------------

        attribute_loss = torch.zeros(
            (),
            device=wbc_logits.device,
        )

        for index, name in enumerate(
            ATTRIBUTE_NAMES
        ):

            if name not in attribute_predictions:

                raise KeyError(
                    "Missing attribute prediction: "
                    f"{name}"
                )

            logits = attribute_predictions[
                name
            ]

            targets = attribute_targets[
                :, index
            ]

            attribute_loss = (
                attribute_loss
                + self.attribute_loss_fns[name](
                    logits,
                    targets,
                )
            )

        # Average over all 11 attributes.

        attribute_loss = (
            attribute_loss
            / len(ATTRIBUTE_NAMES)
        )

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        total_loss = (
            self.wbc_weight * wbc_loss
            +
            self.attribute_weight
            * attribute_loss
        )

        return {
            "total_loss": total_loss,
            "wbc_loss": wbc_loss,
            "attribute_loss": attribute_loss,
        }


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """
    Create directories required for checkpoints.
    """

    best_directory = os.path.dirname(
        str(BEST_MODEL_PATH)
    )

    last_directory = os.path.dirname(
        str(LAST_CHECKPOINT_PATH)
    )

    if best_directory:

        os.makedirs(
            best_directory,
            exist_ok=True,
        )

    if last_directory:

        os.makedirs(
            last_directory,
            exist_ok=True,
        )


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

def create_checkpoint_manager():

    checkpoint_directory = os.path.dirname(
        str(BEST_MODEL_PATH)
    )

    return CheckpointManager(
        directory=checkpoint_directory
    )


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

def print_data_information(
    train_loader,
    val_loader,
    test_loader,
):

    print(
        f"Train: {len(train_loader)} batches"
    )

    print(
        f"Val  : {len(val_loader)} batches"
    )

    print(
        f"Test : {len(test_loader)} batches"
    )


# ============================================================
# PRINT EPOCH RESULTS
# ============================================================

def print_epoch_results(
    epoch,
    train_result,
    val_result,
    learning_rate,
):

    print()

    print(
        f"Epoch {epoch}"
    )

    print("-" * 60)

    print(
        f"Train loss      : "
        f"{train_result['total_loss']:.4f}"
    )

    print(
        f"Train WBC loss  : "
        f"{train_result['wbc_loss']:.4f}"
    )

    print(
        f"Train attr loss : "
        f"{train_result['attribute_loss']:.4f}"
    )

    print(
        f"Train accuracy  : "
        f"{train_result['accuracy'] * 100:.2f}%"
    )

    print()

    print(
        f"Val loss        : "
        f"{val_result['total_loss']:.4f}"
    )

    print(
        f"Val WBC loss    : "
        f"{val_result['wbc_loss']:.4f}"
    )

    print(
        f"Val attr loss   : "
        f"{val_result['attribute_loss']:.4f}"
    )

    print(
        f"Val accuracy    : "
        f"{val_result['accuracy'] * 100:.2f}%"
    )

    print()

    print(
        f"Learning rate   : "
        f"{learning_rate:.8f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "MAL-ViT TRAINING"
    )

    print("=" * 70)

    # ========================================================
    # Seed
    # ========================================================

    print()

    print(
        "[0] Setting random seed..."
    )

    set_seed()

    print(
        f"Seed: {SEED}"
    )

    # ========================================================
    # Directories
    # ========================================================

    create_directories()

    # ========================================================
    # Device
    # ========================================================

    print()

    print(
        "Device:"
    )

    print(
        DEVICE
    )

    # ========================================================
    # Data
    # ========================================================

    print()

    print(
        "[1] Loading data..."
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print_data_information(
        train_loader,
        val_loader,
        test_loader,
    )

    # ========================================================
    # Model
    # ========================================================

    print()

    print(
        "[2] Creating model..."
    )

    model = CompleteMALViT()

    model = model.to(
        DEVICE
    )

    print(
        "Model created."
    )

    # ========================================================
    # Optimizer
    # ========================================================

    print()

    print(
        "[3] Creating optimizer..."
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    # ========================================================
    # Loss
    # ========================================================

    loss_fn = MALViTLoss(
        wbc_weight=WBC_LOSS_WEIGHT,
        attribute_weight=ATTRIBUTE_LOSS_WEIGHT,
    )

    print(
        f"WBC loss weight: "
        f"{WBC_LOSS_WEIGHT}"
    )

    print(
        f"Attribute loss weight: "
        f"{ATTRIBUTE_LOSS_WEIGHT}"
    )

    # ========================================================
    # Scheduler
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(
                NUM_EPOCHS,
                1,
            ),
            eta_min=1e-6,
        )
    )

    # ========================================================
    # Early stopping
    # ========================================================

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )

    print()

    print(
        "Early stopping:"
    )

    print(
        f"Patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Min delta: "
        f"{EARLY_STOPPING_MIN_DELTA}"
    )

    # ========================================================
    # Checkpoint manager
    # ========================================================

    checkpoint_manager = (
        create_checkpoint_manager()
    )

    # ========================================================
    # Training runner
    # ========================================================

    print()

    print(
        "[4] Creating training runner..."
    )

    runner = TrainingRunner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        scheduler=scheduler,
        checkpoint_manager=None,
        early_stopping=early_stopping,
    )

    print(
        "Training runner created."
    )

    # ========================================================
    # Training
    # ========================================================

    print()

    print(
        "[5] Starting training..."
    )

    print()

    print(
        f"Total epochs: "
        f"{NUM_EPOCHS}"
    )

    print(
        "=" * 70
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    completed_epochs = 0

    history = []

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()

        print(
            "=" * 70
        )

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_result = (
            runner.train_one_epoch()
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_result = (
            runner.validate_one_epoch()
        )

        completed_epochs = epoch

        # ----------------------------------------------------
        # Current learning rate
        # ----------------------------------------------------

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print_epoch_results(
            epoch=epoch,
            train_result=train_result,
            val_result=val_result,
            learning_rate=current_lr,
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append(
            {
                "epoch": epoch,

                "train_loss": float(
                    train_result["total_loss"]
                ),

                "train_wbc_loss": float(
                    train_result["wbc_loss"]
                ),

                "train_attribute_loss": float(
                    train_result["attribute_loss"]
                ),

                "train_accuracy": float(
                    train_result["accuracy"]
                ),

                "val_loss": float(
                    val_result["total_loss"]
                ),

                "val_wbc_loss": float(
                    val_result["wbc_loss"]
                ),

                "val_attribute_loss": float(
                    val_result["attribute_loss"]
                ),

                "val_accuracy": float(
                    val_result["accuracy"]
                ),

                "learning_rate": current_lr,
            }
        )

        # ----------------------------------------------------
        # Validation loss
        # ----------------------------------------------------

        val_loss = float(
            val_result["total_loss"]
        )

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            print()

            print(
                "New best model!"
            )

            print(
                f"Validation loss: "
                f"{best_val_loss:.6f}"
            )

            print(
                f"Saving: "
                f"{BEST_MODEL_PATH}"
            )

            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metric=val_loss,
                filename=os.path.basename(
                    str(BEST_MODEL_PATH)
                ),
                history=history,
            )

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metric=val_loss,
            filename=os.path.basename(
                str(LAST_CHECKPOINT_PATH)
            ),
            history=history,
        )

        print()

        print(
            "Checkpoint saved."
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        should_stop = (
            early_stopping.step(
                val_loss
            )
        )

        if should_stop:

            print()

            print(
                "=" * 70
            )

            print(
                "EARLY STOPPING"
            )

            print(
                "=" * 70
            )

            print(
                "Validation loss stopped improving."
            )

            break

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()

    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Epochs completed : "
        f"{completed_epochs}"
    )

    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Best val loss    : "
        f"{best_val_loss:.6f}"
    )

    print()

    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()

    print(
        "Last checkpoint:"
    )

    print(
        LAST_CHECKPOINT_PATH
    )

    print()

    print(
        "=" * 70
    )

    print(
        "Training finished successfully."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()