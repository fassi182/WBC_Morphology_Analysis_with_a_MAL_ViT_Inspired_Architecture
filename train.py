"""Train image -> eleven attribute distributions -> WBC type end to end."""

import argparse
from pathlib import Path

from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, CHECKPOINT_DIR,
    WBC_LOSS_WEIGHT, ATTRIBUTE_LOSS_WEIGHT, WARMUP_EPOCHS, MIN_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
)
from data.dataloader import create_dataloaders
from models.complete_model import CompleteMALViT
from training.runner import TrainingRunner
from training.optimizer import create_optimizer
from training.scheduler import cosine_warmup_scheduler
from training.early_stopping import EarlyStopping
from training.losses import compute_total_loss
from utils.checkpoint_manager import CheckpointManager
from utils.model_loading import load_model
from utils.seed import set_seed


def loss_fn(outputs, wbc_targets, attribute_targets):
    return compute_total_loss(
        outputs, wbc_targets, attribute_targets,
        wbc_weight=WBC_LOSS_WEIGHT, attribute_weight=ATTRIBUTE_LOSS_WEIGHT,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--initialize-from-existing", action="store_true",
                        help="Fine-tune loaded concept weights, or the two existing trained stages")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    set_seed()
    train_loader, val_loader, _ = create_dataloaders(batch_size=args.batch_size)
    model = load_model(device=DEVICE) if args.initialize_from_existing else CompleteMALViT().to(DEVICE)
    optimizer = create_optimizer(model)
    scheduler = cosine_warmup_scheduler(
        optimizer, min(WARMUP_EPOCHS, max(args.epochs - 1, 0)), args.epochs,
        min_lr=MIN_LEARNING_RATE,
    )
    runner = TrainingRunner(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, loss_fn=loss_fn, device=DEVICE, scheduler=scheduler,
        checkpoint_manager=CheckpointManager(args.output_dir),
        early_stopping=EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA,
        ),
    )
    print(f"Device: {DEVICE}; flow: image -> 11 attributes -> WBC")
    return runner.fit(args.epochs)


if __name__ == "__main__":
    main()
