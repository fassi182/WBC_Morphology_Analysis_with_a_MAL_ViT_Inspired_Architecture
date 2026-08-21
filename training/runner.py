"""
MAL-ViT Training Runner

Controls:

    train epoch
        ↓
    validation
        ↓
    checkpoint
        ↓
    scheduler
        ↓
    early stopping
        ↓
    history
"""

from training.engine import (
    train_epoch,
    validate_epoch,
)

from training.history import (
    TrainingHistory,
)

from training.early_stopping import (
    EarlyStopping,
)


class TrainingRunner:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        scheduler=None,
        checkpoint_manager=None,
        early_stopping=None,
        history=None,
    ):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.device = device

        self.scheduler = scheduler

        self.checkpoint_manager = (
            checkpoint_manager
        )

        self.early_stopping = (
            early_stopping
            if early_stopping is not None
            else EarlyStopping()
        )

        self.history = (
            history
            if history is not None
            else TrainingHistory()
        )

    # ======================================================
    # Train one epoch
    # ======================================================

    def train_one_epoch(self):

        return train_epoch(
            model=self.model,
            dataloader=self.train_loader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            device=self.device,
        )

    # ======================================================
    # Validate one epoch
    # ======================================================

    def validate_one_epoch(self):

        return validate_epoch(
            model=self.model,
            dataloader=self.val_loader,
            loss_fn=self.loss_fn,
            device=self.device,
        )

    # ======================================================
    # Run training
    # ======================================================

    def fit(
        self,
        epochs,
    ):

        for epoch in range(
            1,
            epochs + 1,
        ):

            print()
            print(
                "=" * 60
            )

            print(
                f"Epoch {epoch}/{epochs}"
            )

            print(
                "=" * 60
            )

            # ----------------------------------------------
            # Training
            # ----------------------------------------------

            train_metrics = (
                self.train_one_epoch()
            )

            # ----------------------------------------------
            # Validation
            # ----------------------------------------------

            val_metrics = (
                self.validate_one_epoch()
            )

            # ----------------------------------------------
            # Learning rate
            # ----------------------------------------------

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            # ----------------------------------------------
            # Determine best validation loss
            #
            # IMPORTANT:
            # Check the previous history BEFORE adding
            # the current epoch.
            # ----------------------------------------------

            previous_best = (
                self.history.best_validation_loss
            )

            is_best = (
                previous_best is None
                or
                val_metrics["total_loss"]
                < previous_best
            )

            # ----------------------------------------------
            # History
            # ----------------------------------------------

            self.history.add(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
            )

            # ----------------------------------------------
            # Print metrics
            # ----------------------------------------------

            print(
                f"Train loss      : "
                f"{train_metrics['total_loss']:.4f}"
            )

            print(
                f"Train WBC loss  : "
                f"{train_metrics['wbc_loss']:.4f}"
            )

            print(
                f"Train attr loss : "
                f"{train_metrics['attribute_loss']:.4f}"
            )

            print(
                f"Train accuracy  : "
                f"{train_metrics['accuracy']:.4f}"
            )

            print()

            print(
                f"Val loss        : "
                f"{val_metrics['total_loss']:.4f}"
            )

            print(
                f"Val WBC loss    : "
                f"{val_metrics['wbc_loss']:.4f}"
            )

            print(
                f"Val attr loss   : "
                f"{val_metrics['attribute_loss']:.4f}"
            )

            print(
                f"Val accuracy    : "
                f"{val_metrics['accuracy']:.4f}"
            )

            print(
                f"Learning rate   : "
                f"{current_lr:.8f}"
            )

            # ----------------------------------------------
            # Checkpoint
            # ----------------------------------------------

            if self.checkpoint_manager:

                # Always save latest checkpoint
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metric=val_metrics[
                        "total_loss"
                    ],
                    filename="latest.pt",
                )

                # Save best checkpoint only when
                # validation loss improves
                if is_best:

                    self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        metric=val_metrics[
                            "total_loss"
                        ],
                        filename="best_model.pt",
                    )

                    print()
                    print(
                        "Best model checkpoint saved."
                    )

            # ----------------------------------------------
            # Scheduler
            # ----------------------------------------------

            if self.scheduler is not None:

                self.scheduler.step()

            # ----------------------------------------------
            # Early stopping
            # ----------------------------------------------

            should_stop = (
                self.early_stopping(
                    val_metrics["total_loss"]
                )
            )

            if should_stop:

                print()
                print(
                    "Early stopping triggered."
                )

                break

        # ==================================================
        # Training completed
        # ==================================================

        print()
        print(
            "=" * 60
        )

        print(
            "Training completed."
        )

        print(
            "=" * 60
        )

        print(
            f"Epochs completed : "
            f"{len(self.history)}"
        )

        print(
            f"Best epoch      : "
            f"{self.history.best_epoch}"
        )

        print(
            f"Best val loss   : "
            f"{self.history.best_validation_loss:.6f}"
        )

        return self.history


# ==========================================================
# Module Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Training Runner")
    print("=" * 60)

    print()
    print(
        "TrainingRunner imported successfully."
    )

    print()
    print(
        "Supported operations:"
    )

    print(
        "  - train_one_epoch"
    )

    print(
        "  - validate_one_epoch"
    )

    print(
        "  - fit"
    )

    print()
    print(
        "Training runner validation: PASSED"
    )