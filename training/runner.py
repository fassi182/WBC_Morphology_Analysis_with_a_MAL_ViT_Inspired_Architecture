"""Shared epoch orchestration, history, early stopping, and checkpoint saving."""

from config import BEST_MODEL_NAME, LAST_CHECKPOINT_NAME
from training.engine import train_epoch, validate_epoch
from training.history import TrainingHistory
from training.early_stopping import EarlyStopping


class TrainingRunner:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, device,
                 scheduler=None, checkpoint_manager=None, early_stopping=None, history=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.early_stopping = early_stopping if early_stopping is not None else EarlyStopping()
        self.history = history if history is not None else TrainingHistory()

    def train_one_epoch(self):
        return train_epoch(self.model, self.train_loader, self.optimizer, self.device, self.loss_fn)

    def validate_one_epoch(self):
        return validate_epoch(self.model, self.val_loader, self.device, self.loss_fn)

    def fit(self, epochs):
        if epochs < 1:
            raise ValueError("epochs must be positive")
        for epoch in range(1, epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate_one_epoch()
            previous_best = self.history.best_validation_loss
            is_best = previous_best is None or val_metrics["total_loss"] < previous_best
            self.history.add(epoch, train_metrics, val_metrics, current_lr)
            print(f"Epoch {epoch}/{epochs} | train loss={train_metrics['total_loss']:.4f} "
                  f"accuracy={train_metrics['accuracy']:.2%} | "
                  f"val loss={val_metrics['total_loss']:.4f} "
                  f"accuracy={val_metrics['accuracy']:.2%} | lr={current_lr:.8f}")
            if self.scheduler is not None:
                self.scheduler.step()
            should_stop = self.early_stopping(val_metrics["total_loss"])
            if self.checkpoint_manager is not None:
                arguments = dict(
                    model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
                    epoch=epoch, metric=val_metrics["total_loss"], history=self.history.epochs,
                )
                self.checkpoint_manager.save(filename=LAST_CHECKPOINT_NAME, **arguments)
                if is_best:
                    self.checkpoint_manager.save(filename=BEST_MODEL_NAME, **arguments)
            if should_stop:
                print("Early stopping triggered.")
                break
        print(f"Best epoch: {self.history.best_epoch}; "
              f"validation loss: {self.history.best_validation_loss:.6f}")
        return self.history
