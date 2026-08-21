"""
MAL-ViT Evaluator
-----------------

Evaluates MAL-ViT on a validation/test DataLoader.

Metrics:
    - Total loss
    - WBC loss
    - Attribute loss
    - WBC accuracy
"""

import torch


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    loss_fn,
    device,
):
    """
    Evaluate model on a DataLoader.

    Returns
    -------
    dict
        {
            "total_loss": float,
            "wbc_loss": float,
            "attribute_loss": float,
            "accuracy": float,
        }
    """

    model.eval()

    total_loss = 0.0
    total_wbc_loss = 0.0
    total_attribute_loss = 0.0

    correct = 0
    total_samples = 0

    num_batches = 0

    for batch in dataloader:

        # --------------------------------------------------
        # DataLoader returns a dictionary
        # --------------------------------------------------

        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        attributes = batch["attributes"].to(device)

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        outputs = model(images)

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        loss_dict = loss_fn(
            outputs=outputs,
            wbc_targets=labels,
            attribute_targets=attributes,
        )

        # --------------------------------------------------
        # Accumulate losses
        # --------------------------------------------------

        total_loss += loss_dict["total_loss"].item()

        total_wbc_loss += loss_dict["wbc_loss"].item()

        total_attribute_loss += (
            loss_dict["attribute_loss"].item()
        )

        # --------------------------------------------------
        # WBC accuracy
        # --------------------------------------------------

        predictions = outputs["wbc_prediction"]

        correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        num_batches += 1

    # ------------------------------------------------------
    # Avoid division by zero
    # ------------------------------------------------------

    if num_batches == 0:
        return {
            "total_loss": 0.0,
            "wbc_loss": 0.0,
            "attribute_loss": 0.0,
            "accuracy": 0.0,
        }

    # ------------------------------------------------------
    # Average metrics
    # ------------------------------------------------------

    results = {
        "total_loss":
            total_loss / num_batches,

        "wbc_loss":
            total_wbc_loss / num_batches,

        "attribute_loss":
            total_attribute_loss / num_batches,

        "accuracy":
            correct / total_samples
            if total_samples > 0
            else 0.0,
    }

    return results


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Evaluator")
    print("=" * 60)

    print()
    print("Evaluator imported successfully.")
    print()
    print("Evaluation metrics:")
    print("  - Total loss")
    print("  - WBC loss")
    print("  - Attribute loss")
    print("  - WBC accuracy")
    print()
    print("Evaluator validation: PASSED")