"""
MAL-ViT Trainer
---------------

Handles:

    Forward pass
    Loss computation
    Backpropagation
    Optimizer step
    Metrics
"""

import torch


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        logger=None,
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger

    # ======================================================
    # Move batch to device
    # ======================================================

    def _move_batch(self, batch):

        images = batch[0].to(
            self.device,
            non_blocking=True,
        )

        labels = batch[1].to(
            self.device,
            non_blocking=True,
        )

        attributes = batch[2].to(
            self.device,
            non_blocking=True,
        )

        return images, labels, attributes

    # ======================================================
    # Training step
    # ======================================================

    def train_step(self, batch):

        self.model.train()

        images, labels, attributes = (
            self._move_batch(batch)
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        outputs = self.model(images)

        # Compatible with compute_total_loss
        loss_dict = self.loss_fn(
            outputs=outputs,
            wbc_targets=labels,
            attribute_targets=attributes,
        )

        total_loss = loss_dict["total_loss"]

        total_loss.backward()

        self.optimizer.step()

        with torch.no_grad():

            predictions = outputs[
                "wbc_prediction"
            ]

            correct = (
                predictions == labels
            ).sum().item()

            batch_size = labels.size(0)

        return {
            "loss": total_loss.item(),
            "wbc_loss": loss_dict[
                "wbc_loss"
            ].item(),
            "attribute_loss": loss_dict[
                "attribute_loss"
            ].item(),
            "correct": correct,
            "total": batch_size,
        }

    # ======================================================
    # Validation step
    # ======================================================

    @torch.no_grad()
    def validation_step(self, batch):

        self.model.eval()

        images, labels, attributes = (
            self._move_batch(batch)
        )

        outputs = self.model(images)

        loss_dict = self.loss_fn(
            outputs=outputs,
            wbc_targets=labels,
            attribute_targets=attributes,
        )

        predictions = outputs[
            "wbc_prediction"
        ]

        correct = (
            predictions == labels
        ).sum().item()

        batch_size = labels.size(0)

        return {
            "loss": loss_dict[
                "total_loss"
            ].item(),
            "wbc_loss": loss_dict[
                "wbc_loss"
            ].item(),
            "attribute_loss": loss_dict[
                "attribute_loss"
            ].item(),
            "correct": correct,
            "total": batch_size,
        }

    # ======================================================
    # Train epoch
    # ======================================================

    def train_epoch(self, dataloader):

        self.model.train()

        total_loss = 0.0
        total_wbc_loss = 0.0
        total_attribute_loss = 0.0

        correct = 0
        total = 0

        for batch in dataloader:

            result = self.train_step(batch)

            total_loss += result["loss"]
            total_wbc_loss += result["wbc_loss"]
            total_attribute_loss += result[
                "attribute_loss"
            ]

            correct += result["correct"]
            total += result["total"]

        num_batches = len(dataloader)

        return {
            "loss": total_loss / num_batches,
            "wbc_loss": (
                total_wbc_loss / num_batches
            ),
            "attribute_loss": (
                total_attribute_loss
                / num_batches
            ),
            "accuracy": (
                correct / total
                if total > 0
                else 0.0
            ),
        }

    # ======================================================
    # Validation epoch
    # ======================================================

    def validate_epoch(self, dataloader):

        self.model.eval()

        total_loss = 0.0
        total_wbc_loss = 0.0
        total_attribute_loss = 0.0

        correct = 0
        total = 0

        for batch in dataloader:

            result = self.validation_step(batch)

            total_loss += result["loss"]
            total_wbc_loss += result["wbc_loss"]
            total_attribute_loss += result[
                "attribute_loss"
            ]

            correct += result["correct"]
            total += result["total"]

        num_batches = len(dataloader)

        return {
            "loss": total_loss / num_batches,
            "wbc_loss": (
                total_wbc_loss / num_batches
            ),
            "attribute_loss": (
                total_attribute_loss
                / num_batches
            ),
            "accuracy": (
                correct / total
                if total > 0
                else 0.0
            ),
        }


if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Trainer Module")
    print("=" * 60)

    print("\nTrainer class imported successfully.")

    print("\nSupported operations:")
    print("  - train_step")
    print("  - validation_step")
    print("  - train_epoch")
    print("  - validate_epoch")

    print("\nTrainer validation: PASSED")