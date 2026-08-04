"""
train_one_epoch.py

Train the model for one epoch.

Author:
Muhammad Fassi Ur Rehman
"""

from tqdm import tqdm
import torch


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
):
    """
    Train model for one epoch.

    Returns
    -------
    dict
    """

    model.train()

    running_total = 0.0
    running_attribute = 0.0
    running_wbc = 0.0

    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for batch in progress_bar:

        images = batch["image"].to(device)

        cell_labels = batch["cell_label"].to(device)

        attributes = batch["attributes"].to(device)

        optimizer.zero_grad()

        outputs = model(images)

        losses = criterion(

            outputs,

            {
                "attributes": attributes,
                "cell_label": cell_labels,
            }

        )

        loss = losses["total_loss"]

        loss.backward()

        optimizer.step()

        running_total += loss.item()

        running_attribute += losses[
            "attribute_loss"
        ].item()

        running_wbc += losses[
            "wbc_loss"
        ].item()

        predictions = outputs[
            "wbc_logits"
        ].argmax(dim=1)

        correct += (
            predictions == cell_labels
        ).sum().item()

        total += cell_labels.size(0)

        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}",

            acc=f"{100*correct/total:.2f}%"

        )

    return {

        "loss":

            running_total / len(dataloader),

        "attribute_loss":

            running_attribute / len(dataloader),

        "wbc_loss":

            running_wbc / len(dataloader),

        "accuracy":

            100 * correct / total,

    }