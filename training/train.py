"""
training/train.py

MAL-ViT Training Module
=======================

Reusable training entry point.

This module provides:

    train_model()

The actual epoch operations are delegated to TrainingRunner.

This module does NOT call TrainingRunner.fit() directly because
the root training script controls checkpointing and early stopping
explicitly.
"""

from typing import Optional

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
)

from models.complete_model import CompleteMALViT

from training.runner import TrainingRunner

from training.early_stopping import EarlyStopping


# ============================================================
# LOSS
# ============================================================

class MALViTLoss(torch.nn.Module):
    """
    Combined WBC + morphology attribute loss.

    Compatible with training.engine.py.

    Expected keyword arguments:

        outputs
        wbc_targets
        attribute_targets
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
            torch.nn.CrossEntropyLoss()
        )

        from config import ATTRIBUTE_NAMES

        self.attribute_names = (
            ATTRIBUTE_NAMES
        )

        self.attribute_loss_fns = (
            torch.nn.ModuleDict()
        )

        for name in self.attribute_names:

            self.attribute_loss_fns[name] = (
                torch.nn.CrossEntropyLoss()
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

        # Compatibility with alternative
        # argument names.

        if wbc_targets is None:

            wbc_targets = labels

        if attribute_targets is None:

            attribute_targets = attributes

        if wbc_targets is None:

            raise ValueError(
                "Missing WBC targets."
            )

        if attribute_targets is None:

            raise ValueError(
                "Missing attribute targets."
            )

        # ----------------------------------------------------
        # WBC
        # ----------------------------------------------------

        wbc_logits = outputs[
            "wbc_logits"
        ]

        wbc_loss = self.wbc_loss_fn(
            wbc_logits,
            wbc_targets,
        )

        # ----------------------------------------------------
        # Attributes
        # ----------------------------------------------------

        predictions = outputs[
            "attribute_predictions"
        ]

        attribute_loss = torch.zeros(
            (),
            device=wbc_logits.device,
        )

        for index, name in enumerate(
            self.attribute_names
        ):

            prediction = predictions[
                name
            ]

            target = attribute_targets[
                :, index
            ]

            attribute_loss = (
                attribute_loss
                + self.attribute_loss_fns[name](
                    prediction,
                    target,
                )
            )

        attribute_loss = (
            attribute_loss
            / len(self.attribute_names)
        )

        # ----------------------------------------------------
        # Total
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
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    device=DEVICE,
    epochs=NUM_EPOCHS,
    scheduler=None,
    checkpoint_manager=None,
    early_stopping=None,
    history=None,
):
    """
    Train a MAL-ViT model.

    Parameters
    ----------
    model:
        CompleteMALViT model.

    train_loader:
        Training DataLoader.

    val_loader:
        Validation DataLoader.

    optimizer:
        PyTorch optimizer.

    loss_fn:
        MALViTLoss instance.

    device:
        CPU or CUDA device.

    epochs:
        Number of epochs.

    scheduler:
        Optional learning-rate scheduler.

    checkpoint_manager:
        Optional CheckpointManager.

    early_stopping:
        Optional EarlyStopping.

    history:
        Optional history object.

    Returns
    -------
    list
        Training history.
    """

    # --------------------------------------------------------
    # Defaults
    # --------------------------------------------------------

    if optimizer is None:

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    if loss_fn is None:

        loss_fn = MALViTLoss(
            wbc_weight=WBC_LOSS_WEIGHT,
            attribute_weight=(
                ATTRIBUTE_LOSS_WEIGHT
            ),
        )

    if early_stopping is None:

        early_stopping = EarlyStopping(
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
            min_delta=(
                EARLY_STOPPING_MIN_DELTA
            ),
            mode="min",
        )

    # --------------------------------------------------------
    # Runner
    # --------------------------------------------------------

    runner = TrainingRunner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scheduler=scheduler,
        checkpoint_manager=(
            checkpoint_manager
        ),
        early_stopping=(
            early_stopping
        ),
        history=history,
    )

    # --------------------------------------------------------
    # Manual training loop
    #
    # We deliberately use the stable methods:
    #
    #     train_one_epoch()
    #     validate_one_epoch()
    #
    # rather than runner.fit().
    # --------------------------------------------------------

    results = []

    best_val_loss = float("inf")

    for epoch in range(
        1,
        epochs + 1,
    ):

        train_result = (
            runner.train_one_epoch()
        )

        val_result = (
            runner.validate_one_epoch()
        )

        train_loss = float(
            train_result["total_loss"]
        )

        train_wbc_loss = float(
            train_result["wbc_loss"]
        )

        train_attribute_loss = float(
            train_result["attribute_loss"]
        )

        train_accuracy = float(
            train_result["accuracy"]
        )

        val_loss = float(
            val_result["total_loss"]
        )

        val_wbc_loss = float(
            val_result["wbc_loss"]
        )

        val_attribute_loss = float(
            val_result["attribute_loss"]
        )

        val_accuracy = float(
            val_result["accuracy"]
        )

        learning_rate = float(
            optimizer.param_groups[0][
                "lr"
            ]
        )

        epoch_result = {
            "epoch": epoch,

            "train_loss": train_loss,
            "train_wbc_loss":
                train_wbc_loss,
            "train_attribute_loss":
                train_attribute_loss,
            "train_accuracy":
                train_accuracy,

            "val_loss": val_loss,
            "val_wbc_loss":
                val_wbc_loss,
            "val_attribute_loss":
                val_attribute_loss,
            "val_accuracy":
                val_accuracy,

            "learning_rate":
                learning_rate,
        }

        results.append(
            epoch_result
        )

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        if (
            checkpoint_manager is not None
            and val_loss < best_val_loss
        ):

            best_val_loss = val_loss

            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metric=val_loss,
                filename="best_model.pth",
            )

        elif val_loss < best_val_loss:

            best_val_loss = val_loss

        # ----------------------------------------------------
        # Last checkpoint
        # ----------------------------------------------------

        if checkpoint_manager is not None:

            checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metric=val_loss,
                filename="last_checkpoint.pth",
            )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        if scheduler is not None:

            scheduler.step()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        stop = False

        if early_stopping is not None:

            if hasattr(
                early_stopping,
                "step",
            ):

                stop = (
                    early_stopping.step(
                        val_loss
                    )
                )

            elif hasattr(
                early_stopping,
                "update",
            ):

                stop = (
                    early_stopping.update(
                        val_loss
                    )
                )

            elif hasattr(
                early_stopping,
                "check",
            ):

                stop = (
                    early_stopping.check(
                        val_loss
                    )
                )

        if stop:

            break

    return results


# ============================================================
# MODULE TEST
# ============================================================

def _module_test():

    print("=" * 60)
    print("MAL-ViT Training Module")
    print("=" * 60)

    print("\nSupported operation:")
    print("  - train_model")

    # Validate loss construction.

    loss = MALViTLoss()

    assert isinstance(
        loss,
        torch.nn.Module,
    )

    # Validate model construction.

    model = CompleteMALViT()

    model = model.to("cpu")

    x = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():

        outputs = model(x)

    assert "wbc_logits" in outputs

    assert (
        "attribute_predictions"
        in outputs
    )

    print(
        "\nTraining module validation: PASSED"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    _module_test()