"""
validate.py

Validation loop 

This module evaluates the model on the validation dataset without
updating the model parameters.

"""

from tqdm import tqdm
import torch


def validate(
    model,
    dataloader,
    criterion,
    device,
):
    """
    Validate the model for one epoch.

    Parameters
    ----------
    model : nn.Module

    dataloader : DataLoader

    criterion : ExplainableWBCLoss

    device : torch.device

    Returns
    -------
    dict
        Validation statistics.
    """

    model.eval()

    running_total = 0.0
    running_attribute = 0.0
    running_wbc = 0.0

    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    with torch.no_grad():

        for batch in progress_bar:

            images = batch["image"].to(device)

            cell_labels = batch["cell_label"].to(device)

            attributes = batch["attributes"].to(device)

            outputs = model(images)

            losses = criterion(

                outputs,

                {
                    "attributes": attributes,
                    "cell_label": cell_labels,
                }

            )

            running_total += losses["total_loss"].item()

            running_attribute += losses["attribute_loss"].item()

            running_wbc += losses["wbc_loss"].item()

            predictions = outputs["wbc_logits"].argmax(dim=1)

            correct += (predictions == cell_labels).sum().item()

            total += cell_labels.size(0)

            progress_bar.set_postfix(

                loss=f"{losses['total_loss'].item():.4f}",

                acc=f"{100 * correct / total:.2f}%"

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


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Validation Module")
    print("=" * 60)

    print("This module is intended to be called from trainer.py")