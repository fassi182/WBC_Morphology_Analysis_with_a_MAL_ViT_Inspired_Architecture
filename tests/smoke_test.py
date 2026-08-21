# tests/smoke_test.py

"""
MAL-ViT End-to-End Smoke Test
"""

import torch

from config import (
    DEVICE,
    NUM_WBC_CLASSES,
)

from data.dataloader import create_dataloaders
from models.complete_model import CompleteMALViT
from training.losses import compute_total_loss


# ============================================================
# Extract tensors from DataLoader batch
# ============================================================

def extract_batch(batch):

    tensor_items = [
        item for item in batch
        if torch.is_tensor(item)
    ]

    if len(tensor_items) < 3:
        raise RuntimeError(
            "DataLoader batch does not contain enough "
            "tensor elements."
        )

    images = None
    labels = None
    attributes = None

    # Image: (B, 3, H, W)
    for item in tensor_items:

        if (
            item.ndim == 4
            and item.shape[1] == 3
        ):
            images = item
            break

    # WBC labels: (B,)
    for item in tensor_items:

        if item.ndim == 1:

            # Avoid accidentally selecting another 1D tensor
            # with a different batch dimension.
            if images is None or item.shape[0] == images.shape[0]:

                labels = item
                break

    # Attributes: (B, 11)
    for item in tensor_items:

        if (
            item.ndim == 2
            and item.shape[1] == 11
        ):
            attributes = item
            break

    if images is None:
        raise RuntimeError(
            "Could not identify image tensor."
        )

    if labels is None:
        raise RuntimeError(
            "Could not identify WBC label tensor."
        )

    if attributes is None:
        raise RuntimeError(
            "Could not identify attribute tensor."
        )

    return images, labels, attributes


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MAL-ViT END-TO-END SMOKE TEST")
    print("=" * 70)

    # ========================================================
    # 1. DataLoader
    # ========================================================

    print("\n[1] Creating DataLoader...")

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    batch = next(iter(train_loader))

    print(
        "Number of batch elements:",
        len(batch)
    )

    for index, item in enumerate(batch):

        print(
            f"Batch item {index}:",
            type(item),
            end=""
        )

        if torch.is_tensor(item):

            print(
                f" shape={tuple(item.shape)} "
                f"dtype={item.dtype}"
            )

        else:

            print()

    # ========================================================
    # Extract actual tensors
    # ========================================================

    images, labels, attributes = extract_batch(batch)

    print("\nExtracted tensors:")

    print(
        "Images:",
        images.shape
    )

    print(
        "Labels:",
        labels.shape
    )

    print(
        "Attributes:",
        attributes.shape
    )

    # ========================================================
    # Device
    # ========================================================

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    attributes = attributes.to(DEVICE)

    # ========================================================
    # 2. Model
    # ========================================================

    print("\n[2] Creating model...")

    model = CompleteMALViT()

    model = model.to(DEVICE)

    model.train()

    # ========================================================
    # 3. Forward pass
    # ========================================================

    print("\n[3] Forward pass...")

    outputs = model(images)

    print(
        "WBC logits:",
        outputs["wbc_logits"].shape
    )

    print("\nAttribute predictions:")

    for name, prediction in (
        outputs["attribute_predictions"].items()
    ):

        print(
            f"  {name:30s}"
            f"{tuple(prediction.shape)}"
        )

    # ========================================================
    # Validate WBC output
    # ========================================================

    expected_wbc_shape = (
        images.size(0),
        NUM_WBC_CLASSES,
    )

    if tuple(
        outputs["wbc_logits"].shape
    ) != expected_wbc_shape:

        raise RuntimeError(
            "Unexpected WBC logits shape: "
            f"{outputs['wbc_logits'].shape}, "
            f"expected {expected_wbc_shape}"
        )

    print(
        "\nWBC output validation: PASSED"
    )

    # ========================================================
    # 4. Loss
    # ========================================================

    print("\n[4] Loss...")

    loss_dict = compute_total_loss(
        outputs=outputs,
        wbc_targets=labels,
        attribute_targets=attributes,
    )

    print(
        "Total loss:",
        float(loss_dict["total_loss"])
    )

    print(
        "WBC loss:",
        float(loss_dict["wbc_loss"])
    )

    print(
        "Attribute loss:",
        float(loss_dict["attribute_loss"])
    )

    # ========================================================
    # 5. Backward
    # ========================================================

    print("\n[5] Backward pass...")

    total_loss = loss_dict["total_loss"]

    total_loss.backward()

    parameters_with_grad = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            parameters_with_grad += 1

    print(
        "Parameters with gradients:",
        parameters_with_grad
    )

    if parameters_with_grad == 0:

        raise RuntimeError(
            "No gradients were generated."
        )

    print(
        "Backward pass: PASSED"
    )

    # ========================================================
    # 6. Optimizer
    # ========================================================

    print("\n[6] Optimizer step...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    optimizer.step()

    print(
        "Optimizer step: PASSED"
    )

    # ========================================================
    # Final
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "MAL-ViT END-TO-END SMOKE TEST: PASSED"
    )
    print("=" * 70)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()