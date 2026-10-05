"""Compatibility wrapper around the shared dictionary-batch training engine."""

from training.engine import train_step, validation_step, train_epoch, validate_epoch
from training.losses import compute_total_loss
from config import DEVICE


class Trainer:
    def __init__(self, model, optimizer, loss_fn=compute_total_loss, device=DEVICE, logger=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.logger = logger

    def train_step(self, batch):
        return train_step(self.model, batch, self.optimizer, self.device, self.loss_fn)

    def validation_step(self, batch):
        return validation_step(self.model, batch, self.device, self.loss_fn)

    def train_epoch(self, dataloader):
        return train_epoch(self.model, dataloader, self.optimizer, self.device, self.loss_fn)

    def validate_epoch(self, dataloader):
        return validate_epoch(self.model, dataloader, self.device, self.loss_fn)
