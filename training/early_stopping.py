"""
early_stopping.py

Early stopping utility 

Stops training when validation loss does not improve.


"""

import torch


class EarlyStopping:
    """
    Early stopping based on validation loss.
    """

    def __init__(
        self,
        patience=10,
        min_delta=0.0,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")

        self.counter = 0

        self.early_stop = False

    def __call__(self, val_loss):

        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss

            self.counter = 0

        else:

            self.counter += 1

            print(
                f"EarlyStopping "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:

                self.early_stop = True

        return self.early_stop


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Early Stopping Test")
    print("=" * 60)

    losses = [
        1.20,
        1.10,
        1.00,
        0.95,
        0.95,
        0.96,
        0.97,
        0.98,
        0.99,
    ]

    stopper = EarlyStopping(
        patience=3,
        min_delta=0.001,
    )

    for epoch, loss in enumerate(losses, start=1):

        stop = stopper(loss)

        print(
            f"Epoch {epoch:2d} | "
            f"Loss {loss:.3f}"
        )

        if stop:

            print("\nTraining stopped.")

            break