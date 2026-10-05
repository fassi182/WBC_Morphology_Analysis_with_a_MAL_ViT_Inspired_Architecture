"""Compatibility helpers using the common checkpoint format."""

from pathlib import Path
from utils.checkpoint_manager import CheckpointManager


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, metric=None):
    path = Path(path)
    return CheckpointManager(path.parent).save(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=epoch, metric=metric, filename=path.name,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    return CheckpointManager(Path(path).parent).load(
        path, model, optimizer, scheduler, map_location,
    )
