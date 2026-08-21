"""
MAL-ViT Training History
"""

class TrainingHistory:

    def __init__(self):
        self.epochs = []

    def add(
        self,
        epoch,
        train_metrics,
        val_metrics,
        learning_rate=None,
    ):
        entry = {
            "epoch": epoch,
            "train": dict(train_metrics),
            "val": dict(val_metrics),
            "learning_rate": learning_rate,
        }

        self.epochs.append(entry)

    @property
    def best_validation_loss(self):
        if not self.epochs:
            return None

        return min(
            entry["val"]["total_loss"]
            for entry in self.epochs
        )

    @property
    def best_epoch(self):
        if not self.epochs:
            return None

        best = min(
            self.epochs,
            key=lambda x: x["val"]["total_loss"],
        )

        return best["epoch"]

    def latest(self):
        if not self.epochs:
            return None

        return self.epochs[-1]

    def __len__(self):
        return len(self.epochs)


if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Training History Test")
    print("=" * 60)

    history = TrainingHistory()

    history.add(
        epoch=1,
        train_metrics={
            "total_loss": 2.0,
            "wbc_loss": 1.2,
            "attribute_loss": 0.8,
            "accuracy": 0.5,
        },
        val_metrics={
            "total_loss": 1.8,
            "wbc_loss": 1.0,
            "attribute_loss": 0.8,
            "accuracy": 0.6,
        },
        learning_rate=0.0003,
    )

    print()
    print("History entries:", len(history))
    print(
        "Best validation loss:",
        history.best_validation_loss,
    )

    assert len(history) == 1
    assert history.best_validation_loss == 1.8
    assert history.best_epoch == 1

    print()
    print("Training history validation: PASSED")