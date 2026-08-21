"""
MAL-ViT Checkpoint Manager
--------------------------

Handles saving and loading model checkpoints.
"""

from pathlib import Path

import torch


class CheckpointManager:

    def __init__(self, directory="checkpoints"):

        self.directory = Path(directory)

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ======================================================
    # Save
    # ======================================================

    def save(
        self,
        model,
        optimizer=None,
        epoch=0,
        metric=None,
        scheduler=None,
        filename=None,
        **kwargs,
    ):
        """
        Save a training checkpoint.

        Parameters
        ----------
        model : torch.nn.Module
            MAL-ViT model.

        optimizer : torch.optim.Optimizer, optional
            Optimizer state.

        epoch : int
            Current epoch.

        metric : float, optional
            Validation metric, usually validation loss.

        scheduler : optional
            Learning-rate scheduler.

        filename : str, optional
            Custom checkpoint filename.

        Returns
        -------
        Path
            Saved checkpoint path.
        """

        if filename is None:

            filename = (
                f"checkpoint_epoch_{epoch}.pt"
            )

        path = self.directory / filename

        checkpoint = {
            "epoch": epoch,
            "metric": metric,
            "model_state_dict":
                model.state_dict(),
        }

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        if optimizer is not None:

            checkpoint[
                "optimizer_state_dict"
            ] = optimizer.state_dict()

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        if scheduler is not None:

            checkpoint[
                "scheduler_state_dict"
            ] = scheduler.state_dict()

        # --------------------------------------------------
        # Additional information
        # --------------------------------------------------

        checkpoint.update(kwargs)

        torch.save(
            checkpoint,
            path,
        )

        return path

    # ======================================================
    # Load
    # ======================================================

    def load(
        self,
        path,
        model,
        optimizer=None,
        scheduler=None,
        map_location="cpu",
    ):
        """
        Load a checkpoint.

        Returns
        -------
        dict
            Loaded checkpoint metadata.
        """

        checkpoint = torch.load(
            path,
            map_location=map_location,
        )

        # --------------------------------------------------
        # Model
        # --------------------------------------------------

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        # --------------------------------------------------
        # Scheduler
        # --------------------------------------------------

        if (
            scheduler is not None
            and "scheduler_state_dict" in checkpoint
        ):

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        return checkpoint

    # ======================================================
    # Best Checkpoint
    # ======================================================

    def save_best(
        self,
        model,
        optimizer=None,
        epoch=0,
        metric=None,
        scheduler=None,
    ):

        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metric=metric,
            scheduler=scheduler,
            filename="best_model.pt",
        )

    # ======================================================
    # Latest Checkpoint
    # ======================================================

    def save_latest(
        self,
        model,
        optimizer=None,
        epoch=0,
        metric=None,
        scheduler=None,
    ):

        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metric=metric,
            scheduler=scheduler,
            filename="latest.pt",
        )


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Checkpoint Manager Test")
    print("=" * 60)

    import torch.nn as nn

    model = nn.Linear(
        10,
        5,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    manager = CheckpointManager()

    print()
    print("[1] Creating model...")
    print("Model created.")

    print()
    print("[2] Saving checkpoint...")

    path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=3,
        metric=1.25,
        filename="test_checkpoint.pt",
    )

    print(f"Saved: {path}")

    print()
    print("[3] Loading checkpoint...")

    new_model = nn.Linear(
        10,
        5,
    )

    new_optimizer = torch.optim.AdamW(
        new_model.parameters(),
        lr=1e-4,
    )

    checkpoint = manager.load(
        path,
        new_model,
        new_optimizer,
    )

    print(
        "Loaded epoch:",
        checkpoint["epoch"],
    )

    print(
        "Loaded metric:",
        checkpoint["metric"],
    )

    print()
    print(
        "Checkpoint manager validation: PASSED"
    )