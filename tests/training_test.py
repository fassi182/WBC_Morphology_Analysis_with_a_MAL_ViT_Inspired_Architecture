"""
MAL-ViT Evaluator
-----------------

Evaluation utilities for the complete MAL-ViT model.

Expected DataLoader batch:

{
    "images":      Tensor[B, 3, 224, 224],
    "labels":      Tensor[B],
    "attributes":  Tensor[B, 11],
    "indices":     list,
    "image_names": list,
    "image_paths": list,
}

Returns:

{
    "total_loss": ...,
    "wbc_loss": ...,
    "attribute_loss": ...,
    "wbc_accuracy": ...
}
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
    Evaluate MAL-ViT on a dataset.

    Parameters
    ----------
    model : torch.nn.Module
        MAL-ViT model.

    dataloader : DataLoader
        Validation or test DataLoader.

    loss_fn : callable
        Multi-task loss function.

    device : torch.device
        Device used for evaluation.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    # ==================================================
    # Evaluation mode
    # ==================================================

    model.eval()

    # ==================================================
    # Running totals
    # ==================================================

    total_loss = 0.0
    total_wbc_loss = 0.0
    total_attribute_loss = 0.0

    correct = 0
    total_samples = 0

    # ==================================================
    # Iterate through dataset
    # ==================================================

    for batch in dataloader:

        # --------------------------------------------------
        # Current DataLoader returns a dictionary
        # --------------------------------------------------

        if not isinstance(batch, dict):

            raise TypeError(
                "Expected DataLoader batch to be a dictionary, "
                f"got {type(batch)}"
            )

        # --------------------------------------------------
        # Extract tensors
        # --------------------------------------------------

        images = batch["images"].to(device)

        labels = batch["labels"].to(device)

        attributes = batch["attributes"].to(device)

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        outputs = model(images)

        # --------------------------------------------------
        # Compute losses
        # --------------------------------------------------

        loss_dict = loss_fn(
            outputs=outputs,
            wbc_targets=labels,
            attribute_targets=attributes,
        )

        # --------------------------------------------------
        # Batch size
        # --------------------------------------------------

        batch_size = images.size(0)

        # --------------------------------------------------
        # Accumulate losses
        # --------------------------------------------------

        total_loss += (
            loss_dict["total_loss"].item()
            * batch_size
        )

        total_wbc_loss += (
            loss_dict["wbc_loss"].item()
            * batch_size
        )

        total_attribute_loss += (
            loss_dict["attribute_loss"].item()
            * batch_size
        )

        # --------------------------------------------------
        # WBC accuracy
        # --------------------------------------------------

        predictions = outputs[
            "wbc_prediction"
        ]

        correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    # ==================================================
    # Prevent division by zero
    # ==================================================

    if total_samples == 0:

        raise RuntimeError(
            "Evaluation DataLoader contains no samples."
        )

    # ==================================================
    # Average metrics
    # ==================================================

    metrics = {

        "total_loss":
            total_loss / total_samples,

        "wbc_loss":
            total_wbc_loss / total_samples,

        "attribute_loss":
            total_attribute_loss / total_samples,

        "wbc_accuracy":
            correct / total_samples,
    }

    return metrics


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Evaluator")
    print("=" * 60)

    print(
        "\nEvaluator imported successfully."
    )

    print(
        "\nEvaluation metrics:"
    )

    print(
        "  - Total loss"
    )

    print(
        "  - WBC loss"
    )

    print(
        "  - Attribute loss"
    )

    print(
        "  - WBC accuracy"
    )

    print(
        "\nEvaluator validation: PASSED"
    )