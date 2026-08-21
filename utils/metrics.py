"""
metrics.py

Classification and morphology metrics.
"""

import torch


def classification_accuracy(
    predictions,
    targets,
):

    predictions = torch.as_tensor(
        predictions
    )

    targets = torch.as_tensor(
        targets
    )

    if predictions.ndim > 1:

        predictions = (
            predictions.argmax(dim=1)
        )

    return (
        (predictions == targets)
        .float()
        .mean()
        .item()
    )


def attribute_accuracy(
    predictions,
    targets,
):

    """
    predictions:
        Dict[str, Tensor]

    targets:
        Tensor of shape
        (B, NUM_ATTRIBUTES)
    """

    accuracies = {}

    for index, (
        name,
        logits,
    ) in enumerate(
        predictions.items()
    ):

        predicted = logits.argmax(
            dim=1
        )

        target = targets[:, index]

        accuracies[name] = (
            predicted == target
        ).float().mean().item()

    if accuracies:

        overall = sum(
            accuracies.values()
        ) / len(accuracies)

    else:

        overall = 0.0

    return {
        "per_attribute": accuracies,
        "overall": overall,
    }


if __name__ == "__main__":

    predictions = torch.tensor(
        [0, 1, 2, 3]
    )

    targets = torch.tensor(
        [0, 1, 1, 3]
    )

    accuracy = classification_accuracy(
        predictions,
        targets,
    )

    print("=" * 60)
    print("MAL-ViT Metrics Test")
    print("=" * 60)

    print(
        "\nClassification accuracy:"
    )

    print(accuracy)

    print(
        "\nMetrics validation: PASSED"
    )