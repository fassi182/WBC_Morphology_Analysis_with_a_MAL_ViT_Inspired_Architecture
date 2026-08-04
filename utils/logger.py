"""
logger.py

Training logger for MAL-ViT.

Logs training and validation metrics after each epoch.

Author:
Muhammad Fassi Ur Rehman
"""

import csv

from config import LOG_DIR


class TrainingLogger:

    def __init__(self, filename="training_log.csv"):

        self.log_path = LOG_DIR / filename

        self.header = [

            "epoch",

            "train_loss",
            "train_attribute_loss",
            "train_wbc_loss",
            "train_accuracy",

            "val_loss",
            "val_attribute_loss",
            "val_wbc_loss",
            "val_accuracy",

        ]

        self._create_file()

    # =====================================================

    def _create_file(self):

        if not self.log_path.exists():

            with open(
                self.log_path,
                "w",
                newline=""
            ) as f:

                writer = csv.writer(f)

                writer.writerow(self.header)

    # =====================================================

    def log(

        self,

        epoch,

        train_metrics,

        val_metrics,

    ):

        with open(

            self.log_path,

            "a",

            newline=""

        ) as f:

            writer = csv.writer(f)

            writer.writerow([

                epoch,

                round(train_metrics["loss"], 4),
                round(train_metrics["attribute_loss"], 4),
                round(train_metrics["wbc_loss"], 4),
                round(train_metrics["accuracy"], 2),

                round(val_metrics["loss"], 4),
                round(val_metrics["attribute_loss"], 4),
                round(val_metrics["wbc_loss"], 4),
                round(val_metrics["accuracy"], 2),

            ])

    # =====================================================

    @property
    def path(self):

        return self.log_path


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    logger = TrainingLogger()

    train = {

        "loss": 1.24,

        "attribute_loss": 0.62,

        "wbc_loss": 0.62,

        "accuracy": 81.3,

    }

    val = {

        "loss": 1.11,

        "attribute_loss": 0.55,

        "wbc_loss": 0.56,

        "accuracy": 83.7,

    }

    logger.log(

        1,

        train,

        val,

    )

    print("Logger Test Passed")

    print(logger.path)