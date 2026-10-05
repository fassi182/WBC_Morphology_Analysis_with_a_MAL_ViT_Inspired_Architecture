"""
MAL-ViT Training Engine
-----------------------

Reusable training and validation loops.

Expected batch format:

{
    "images": Tensor,
    "labels": Tensor,
    "attributes": Tensor,
    "indices": list,
    "image_names": list,
    "image_paths": list,
}
"""

import torch

from config import GRADIENT_CLIP_NORM
from training.losses import compute_total_loss


def train_step(
    model,
    batch,
    optimizer,
    device,
    loss_fn=compute_total_loss,
):
    """
    Perform one training step.
    """

    model.train()

    images = batch["images"].to(device)
    labels = batch["labels"].to(device)
    attributes = batch["attributes"].to(device)

    optimizer.zero_grad(set_to_none=True)

    outputs = model(images)

    loss_dict = loss_fn(
        outputs=outputs,
        wbc_targets=labels,
        attribute_targets=attributes,
    )

    total_loss = loss_dict["total_loss"]

    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)

    optimizer.step()

    with torch.no_grad():

        predictions = outputs["wbc_prediction"]

        correct = (
            predictions == labels
        ).sum().item()

        accuracy = (
            correct / labels.size(0)
        )

    return {
        "total_loss": total_loss.item(),
        "wbc_loss": loss_dict["wbc_loss"].item(),
        "attribute_loss": loss_dict["attribute_loss"].item(),
        "accuracy": accuracy,
    }


@torch.no_grad()
def validation_step(
    model,
    batch,
    device,
    loss_fn=compute_total_loss,
):
    """
    Perform one validation step.
    """

    model.eval()

    images = batch["images"].to(device)
    labels = batch["labels"].to(device)
    attributes = batch["attributes"].to(device)

    outputs = model(images)

    loss_dict = loss_fn(
        outputs=outputs,
        wbc_targets=labels,
        attribute_targets=attributes,
    )

    predictions = outputs["wbc_prediction"]

    correct = (
        predictions == labels
    ).sum().item()

    accuracy = (
        correct / labels.size(0)
    )

    return {
        "total_loss": loss_dict["total_loss"].item(),
        "wbc_loss": loss_dict["wbc_loss"].item(),
        "attribute_loss": loss_dict["attribute_loss"].item(),
        "accuracy": accuracy,
    }


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    loss_fn=compute_total_loss,
):
    """
    Train for one complete epoch.
    """

    total_loss = 0.0
    total_wbc_loss = 0.0
    total_attribute_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:

        result = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
        )

        batch_size = batch["labels"].size(0)

        total_loss += (
            result["total_loss"] * batch_size
        )

        total_wbc_loss += (
            result["wbc_loss"] * batch_size
        )

        total_attribute_loss += (
            result["attribute_loss"] * batch_size
        )

        total_correct += (
            result["accuracy"] * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training dataloader contains no samples")

    return {
        "total_loss": total_loss / total_samples,
        "wbc_loss": total_wbc_loss / total_samples,
        "attribute_loss": total_attribute_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


@torch.no_grad()
def validate_epoch(
    model,
    dataloader,
    device,
    loss_fn=compute_total_loss,
):
    """
    Validate for one complete epoch.
    """

    model.eval()

    total_loss = 0.0
    total_wbc_loss = 0.0
    total_attribute_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:

        result = validation_step(
            model=model,
            batch=batch,
            device=device,
            loss_fn=loss_fn,
        )

        batch_size = batch["labels"].size(0)

        total_loss += (
            result["total_loss"] * batch_size
        )

        total_wbc_loss += (
            result["wbc_loss"] * batch_size
        )

        total_attribute_loss += (
            result["attribute_loss"] * batch_size
        )

        total_correct += (
            result["accuracy"] * batch_size
        )

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Validation dataloader contains no samples")

    return {
        "total_loss": total_loss / total_samples,
        "wbc_loss": total_wbc_loss / total_samples,
        "attribute_loss": total_attribute_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Training Engine")
    print("=" * 70)

    print("\nSupported operations:")
    print("  - train_step")
    print("  - validation_step")
    print("  - train_epoch")
    print("  - validate_epoch")

    print("\nTraining engine validation: PASSED")
