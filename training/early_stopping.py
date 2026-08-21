"""
MAL-ViT Early Stopping
----------------------

Stops training when validation metric stops improving.
"""

class EarlyStopping:

    def __init__(
        self,
        patience=10,
        min_delta=0.0001,
        mode="min",
    ):
        if mode not in ("min", "max"):
            raise ValueError(
                "mode must be either 'min' or 'max'"
            )

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def step(self, value):
        """
        Update early stopping state.

        Returns:
            bool: True if training should stop.
        """

        if self.best_value is None:
            self.best_value = value
            self.counter = 0
            return False

        if self.mode == "min":
            improved = value < (
                self.best_value - self.min_delta
            )
        else:
            improved = value > (
                self.best_value + self.min_delta
            )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""

        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value):
        """
        Allow:

            early_stopping(value)

        for compatibility with the training runner.
        """

        return self.step(value)


if __name__ == "__main__":

    print("=" * 60)
    print("Early Stopping Test")
    print("=" * 60)

    early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.0001,
        mode="min",
    )

    values = [
        1.0,
        0.8,
        0.7,
        0.7,
        0.7,
        0.7,
    ]

    for value in values:

        stop = early_stopping(value)

        print(
            f"value={value:.4f} "
            f"counter={early_stopping.counter} "
            f"stop={stop}"
        )

    assert early_stopping.should_stop is True

    print()
    print("Early stopping validation: PASSED")