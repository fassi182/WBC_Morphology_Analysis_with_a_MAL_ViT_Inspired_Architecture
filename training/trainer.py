"""
trainer.py

Trainer class 

Responsibilities
----------------
1. Build optimizer
2. Build scheduler
3. Load checkpoint
4. Train model
5. Validate model
6. Save best model
7. Resume training


"""

from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
    LAST_CHECKPOINT_NAME,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    MIN_LEARNING_RATE,
)

from utils.logger import TrainingLogger

from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)

from training.train_one_epoch import train_one_epoch
from training.validate import validate


class Trainer:
    """
    Trainer for Explainable WBC Classification.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
    ):

        self.device = DEVICE

        self.model = model.to(self.device)

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.criterion = criterion

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = optim.AdamW(

            self.model.parameters(),

            lr=LEARNING_RATE,

            weight_decay=WEIGHT_DECAY,

        )

        # --------------------------------------------------
        # LR Scheduler
        # --------------------------------------------------

        self.scheduler = ReduceLROnPlateau(

            self.optimizer,

            mode="min",

            factor=SCHEDULER_FACTOR,

            patience=SCHEDULER_PATIENCE,

            min_lr=MIN_LEARNING_RATE,

        )

        # --------------------------------------------------
        # Training State
        # --------------------------------------------------

        self.start_epoch = 0

        self.best_loss = float("inf")

        self.history = {

            "train_loss": [],

            "train_attribute_loss": [],

            "train_wbc_loss": [],

            "train_accuracy": [],

            "val_loss": [],

            "val_attribute_loss": [],

            "val_wbc_loss": [],

            "val_accuracy": [],

        }

        # --------------------------------------------------
        # Logger
        # --------------------------------------------------

        self.logger = TrainingLogger()

        # --------------------------------------------------
        # Paths
        # --------------------------------------------------

        self.best_model_path = (
            CHECKPOINT_DIR / BEST_MODEL_NAME
        )

        self.last_checkpoint_path = (
            CHECKPOINT_DIR / LAST_CHECKPOINT_NAME
        )
        # ======================================================
    # Resume Training
    # ======================================================

    def resume(self):

        if not self.last_checkpoint_path.exists():

            print("\nNo checkpoint found. Starting fresh training.")

            return

        checkpoint = load_checkpoint(
            self.last_checkpoint_path
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        self.start_epoch = checkpoint["epoch"] + 1

        self.best_loss = checkpoint["best_loss"]

        self.history = checkpoint["history"]

        print(f"\nResumed training from epoch {self.start_epoch}")


    # ======================================================
    # Save Checkpoint
    # ======================================================

    def save(self, epoch):

        checkpoint = {

            "epoch": epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                self.scheduler.state_dict(),

            "best_loss":
                self.best_loss,

            "history":
                self.history,

        }

        save_checkpoint(
            checkpoint,
            self.last_checkpoint_path,
        )


    # ======================================================
    # Train
    # ======================================================

    def fit(self):

        print("=" * 70)
        print("Starting Training")
        print("=" * 70)

        for epoch in range(
            self.start_epoch,
            NUM_EPOCHS,
        ):

            print("\n")
            print("=" * 70)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
            print("=" * 70)

            # ----------------------------------------
            # Train
            # ----------------------------------------

            train_metrics = train_one_epoch(

                self.model,

                self.train_loader,

                self.optimizer,

                self.criterion,

                self.device,

            )

            # ----------------------------------------
            # Validation
            # ----------------------------------------

            val_metrics = validate(

                self.model,

                self.val_loader,

                self.criterion,

                self.device,

            )

            # ----------------------------------------
            # Scheduler
            # ----------------------------------------

            self.scheduler.step(
                val_metrics["loss"]
            )

            # ----------------------------------------
            # History
            # ----------------------------------------

            self.history["train_loss"].append(
                train_metrics["loss"]
            )

            self.history["train_attribute_loss"].append(
                train_metrics["attribute_loss"]
            )

            self.history["train_wbc_loss"].append(
                train_metrics["wbc_loss"]
            )

            self.history["train_accuracy"].append(
                train_metrics["accuracy"]
            )

            self.history["val_loss"].append(
                val_metrics["loss"]
            )

            self.history["val_attribute_loss"].append(
                val_metrics["attribute_loss"]
            )

            self.history["val_wbc_loss"].append(
                val_metrics["wbc_loss"]
            )

            self.history["val_accuracy"].append(
                val_metrics["accuracy"]
            )

            # ----------------------------------------
            # Logger
            # ----------------------------------------

            self.logger.log(

                epoch + 1,

                train_metrics,

                val_metrics,

            )

            # ----------------------------------------
            # Save Last Checkpoint
            # ----------------------------------------

            self.save(epoch)

            # ----------------------------------------
            # Save Best Model
            # ----------------------------------------

            if val_metrics["loss"] < self.best_loss:

                self.best_loss = val_metrics["loss"]

                torch.save(

                    self.model.state_dict(),

                    self.best_model_path,

                )

                print("\nBest model updated.")

        print("\n")
        print("=" * 70)
        print("Training Complete")
        print("=" * 70)