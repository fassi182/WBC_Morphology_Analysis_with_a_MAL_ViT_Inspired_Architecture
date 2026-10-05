"""Sample-weighted evaluation using the shared validation engine."""

from training.engine import validate_epoch


def evaluate(model, dataloader, loss_fn, device):
    return validate_epoch(model, dataloader, device, loss_fn)
