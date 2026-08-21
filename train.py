"""
MAL-ViT Main Training Script
============================

Complete training pipeline:

    WBCAtt
       |
       v
    DataLoader
       |
       v
    CompleteMALViT
       |
       v
    Train Epoch
       |
       v
    Validation Epoch
       |
       +----> Best Checkpoint
       |
       +----> Last Checkpoint
       |
       +----> Early Stopping

The loss function is defined centrally in:

    training/losses.py

This keeps training and test evaluation consistent.
"""

import os
import random

import numpy as np
import torch

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
    SEED,
)

from data.dataloader import create_dataloaders

from models.complete_model import CompleteMALViT

from training.runner import TrainingRunner

from training.early_stopping import EarlyStopping

from training.losses import compute_total_loss

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

def loss_fn(
    outputs,
    wbc_targets,
    attribute_targets,
):
    """
    Wrapper around the centralized MAL-ViT loss.

    The actual loss implementation is in:

        training/losses.py

    Total loss:

        WBC loss
        +
        Attribute loss

    The weights are controlled through config.py.
    """

    return compute_total_loss(
        outputs=outputs,
        wbc_targets=wbc_targets,
        attribute_targets=attribute_targets,
        wbc_weight=WBC_LOSS_WEIGHT,
        attribute_weight=ATTRIBUTE_LOSS_WEIGHT,
    )


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
    """
    Create the checkpoint manager using the
    directory configured for the best model.
    """

    checkpoint_directory = os.path.dirname(
        str(BEST_MODEL_PATH)
    )

    return CheckpointManager(
        directory=checkpoint_directory
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def main():

    print("=" * 70)
    print("MAL-ViT TRAINING")
    print("=" * 70)

    # ========================================================
    # [0] RANDOM SEED
    # ========================================================

    print("\n[0] Setting random seed...")

    set_seed()

    print(f"Seed: {SEED}")

    # ========================================================
    # DIRECTORIES
    # ========================================================

    create_directories()

    # ========================================================
    # DEVICE
    # ========================================================

    print("\nDevice:")
    print(DEVICE)

    # ========================================================
    # [1] DATA
    # ========================================================

    print("\n[1] Loading data...")

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print(
        f"Train: {len(train_loader)} batches"
    )

    print(
        f"Val  : {len(val_loader)} batches"
    )

    print(
        f"Test : {len(test_loader)} batches"
    )

    # ========================================================
    # [2] MODEL
    # ========================================================

    print("\n[2] Creating model...")

    model = CompleteMALViT()

    model = model.to(DEVICE)

    print("Model created.")

    # ========================================================
    # [3] OPTIMIZER
    # ========================================================

    print("\n[3] Creating optimizer...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("Optimizer: AdamW")

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
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
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(NUM_EPOCHS, 1),
            eta_min=1e-6,
        )
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )

    print("\nEarly stopping:")

    print(
        f"Patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Min delta: "
        f"{EARLY_STOPPING_MIN_DELTA}"
    )

    # ========================================================
    # CHECKPOINT MANAGER
    # ========================================================

    checkpoint_manager = (
        create_checkpoint_manager()
    )

    # ========================================================
    # [4] TRAINING RUNNER
    # ========================================================

    print(
        "\n[4] Creating training runner..."
    )

    runner = TrainingRunner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=DEVICE,
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
        early_stopping=early_stopping,
    )

    print("Training runner created.")

    # ========================================================
    # [5] TRAINING
    # ========================================================

    print(
        "\n[5] Starting training..."
    )

    print(
        f"\nTotal epochs: {NUM_EPOCHS}"
    )

    print("=" * 70)

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_val_loss = float("inf")

    best_epoch = 0

    completed_epochs = 0

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print("\n" + "=" * 70)

        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_result = (
            runner.train_one_epoch()
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_result = (
            runner.validate_one_epoch()
        )

        completed_epochs = epoch

        # ----------------------------------------------------
        # TRAIN METRICS
        # ----------------------------------------------------

        train_loss = float(
            train_result["total_loss"]
        )

        train_wbc_loss = float(
            train_result["wbc_loss"]
        )

        train_attr_loss = float(
            train_result["attribute_loss"]
        )

        train_accuracy = float(
            train_result["accuracy"]
        )

        # ----------------------------------------------------
        # VALIDATION METRICS
        # ----------------------------------------------------

        val_loss = float(
            val_result["total_loss"]
        )

        val_wbc_loss = float(
            val_result["wbc_loss"]
        )

        val_attr_loss = float(
            val_result["attribute_loss"]
        )

        val_accuracy = float(
            val_result["accuracy"]
        )

        # ----------------------------------------------------
        # LEARNING RATE
        # ----------------------------------------------------

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            f"\nEpoch {epoch}"
        )

        print("-" * 60)

        print(
            f"Train loss      : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train WBC loss  : "
            f"{train_wbc_loss:.4f}"
        )

        print(
            f"Train attr loss : "
            f"{train_attr_loss:.4f}"
        )

        print(
            f"Train accuracy  : "
            f"{train_accuracy:.2%}"
        )

        print()

        print(
            f"Val loss        : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val WBC loss    : "
            f"{val_wbc_loss:.4f}"
        )

        print(
            f"Val attr loss   : "
            f"{val_attr_loss:.4f}"
        )

        print(
            f"Val accuracy    : "
            f"{val_accuracy:.2%}"
        )

        print()

        print(
            f"Learning rate   : "
            f"{current_lr:.8f}"
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            print(
                "\nNew best model!"
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
                filename="best_model.pth",
            )

            print(
                "\nCheckpoint saved."
            )

        # ====================================================
        # LAST CHECKPOINT
        # ====================================================

        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metric=val_loss,
            filename="last_checkpoint.pth",
        )

        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step()

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        stop_training = (
            early_stopping.step(
                val_loss
            )
        )

        if stop_training:

            print(
                "\n" + "=" * 70
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

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nEpochs completed : "
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

    print(
        "\nBest model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nLast checkpoint:"
    )

    print(
        LAST_CHECKPOINT_PATH
    )

    print(
        "\n" + "=" * 70
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