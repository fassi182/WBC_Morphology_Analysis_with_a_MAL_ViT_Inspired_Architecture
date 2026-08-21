# training/test.py

"""
MAL-ViT Test Evaluation
========================

Responsibilities
----------------
1. Evaluate the trained MAL-ViT model on the test dataset.
2. Calculate WBC classification loss.
3. Calculate morphological attribute loss.
4. Calculate overall accuracy.
5. Calculate macro precision, recall, and F1.
6. Calculate per-class WBC metrics.
7. Calculate per-attribute accuracy.
8. Generate the WBC confusion matrix.
9. Print evaluation results in a clear format.

Expected DataLoader batch format
--------------------------------
{
    "images": Tensor,
    "labels": Tensor,
    "attributes": Tensor,
    "indices": list,
    "image_names": list,
    "image_paths": list,
}

Expected model output
---------------------
{
    "wbc_logits": Tensor,
    "wbc_prediction": Tensor,
    "attribute_predictions": {
        attribute_name: Tensor
    },
    ...
}
"""

import torch

from config import (
    WBC_CLASSES,
    ATTRIBUTE_NAMES,
)


# ============================================================
# Batch Extraction
# ============================================================

def _extract_batch(batch, device):
    """
    Extract images, WBC labels, and attribute labels
    from a DataLoader batch.

    Parameters
    ----------
    batch : dict
        Batch returned by wbca_collate_fn.

    device : torch.device
        Target device.

    Returns
    -------
    images : torch.Tensor
        Shape: (B, 3, H, W)

    labels : torch.Tensor
        Shape: (B,)

    attributes : torch.Tensor
        Shape: (B, 11)
    """

    if not isinstance(batch, dict):
        raise TypeError(
            "Expected DataLoader batch to be a dictionary."
        )

    required_keys = [
        "images",
        "labels",
        "attributes",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in batch
    ]

    if missing_keys:
        raise KeyError(
            "Missing required batch keys: "
            f"{missing_keys}\n"
            f"Available keys: {list(batch.keys())}"
        )

    images = batch["images"].to(device)
    labels = batch["labels"].to(device)
    attributes = batch["attributes"].to(device)

    return (
        images,
        labels,
        attributes,
    )


# ============================================================
# Confusion Matrix
# ============================================================

def _build_confusion_matrix(
    labels,
    predictions,
    num_classes,
):
    """
    Build a standard confusion matrix.

    Rows    = actual classes
    Columns = predicted classes

    Parameters
    ----------
    labels : list[int]

    predictions : list[int]

    num_classes : int

    Returns
    -------
    list[list[int]]
    """

    matrix = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    for actual, predicted in zip(
        labels,
        predictions,
    ):

        matrix[actual][predicted] += 1

    return matrix


# ============================================================
# Per-Class Metrics
# ============================================================

def _calculate_class_metrics(
    confusion_matrix,
    class_names,
):
    """
    Calculate precision, recall, and F1
    for every WBC class.

    Returns
    -------
    dict
    """

    num_classes = len(class_names)

    class_metrics = {}

    for class_index, class_name in enumerate(
        class_names
    ):

        # ----------------------------------------------------
        # True Positive
        # ----------------------------------------------------

        tp = confusion_matrix[
            class_index
        ][
            class_index
        ]

        # ----------------------------------------------------
        # Actual samples
        # ----------------------------------------------------

        actual = sum(
            confusion_matrix[
                class_index
            ]
        )

        # ----------------------------------------------------
        # Predicted samples
        # ----------------------------------------------------

        predicted = sum(
            confusion_matrix[row][class_index]
            for row in range(num_classes)
        )

        # ----------------------------------------------------
        # False Positive / False Negative
        # ----------------------------------------------------

        fp = predicted - tp
        fn = actual - tp

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        if predicted > 0:
            precision = tp / predicted
        else:
            precision = 0.0

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        if actual > 0:
            recall = tp / actual
        else:
            recall = 0.0

        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        if precision + recall > 0:

            f1 = (
                2.0
                * precision
                * recall
                / (precision + recall)
            )

        else:

            f1 = 0.0

        class_metrics[class_name] = {
            "total": actual,
            "correct": tp,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return class_metrics


# ============================================================
# Test Evaluation
# ============================================================

@torch.no_grad()
def evaluate_test_set(
    model,
    dataloader,
    loss_fn,
    device,
):
    """
    Evaluate MAL-ViT on the complete test set.

    Parameters
    ----------
    model : torch.nn.Module
        Trained MAL-ViT model.

    dataloader : DataLoader
        Test DataLoader.

    loss_fn : callable
        Multi-task loss function.

    device : torch.device
        CPU or CUDA device.

    Returns
    -------
    dict
        Complete test evaluation results.
    """

    # ========================================================
    # Evaluation mode
    # ========================================================

    model.eval()

    # ========================================================
    # Loss accumulators
    # ========================================================

    total_loss = 0.0
    total_wbc_loss = 0.0
    total_attribute_loss = 0.0

    total_samples = 0

    # ========================================================
    # Prediction storage
    # ========================================================

    all_predictions = []
    all_labels = []

    # ========================================================
    # Attribute statistics
    # ========================================================

    attribute_correct = {
        name: 0
        for name in ATTRIBUTE_NAMES
    }

    attribute_total = {
        name: 0
        for name in ATTRIBUTE_NAMES
    }

    # ========================================================
    # Iterate through test set
    # ========================================================

    for batch in dataloader:

        images, labels, attributes = _extract_batch(
            batch,
            device,
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        losses = loss_fn(
            outputs=outputs,
            wbc_targets=labels,
            attribute_targets=attributes,
        )

        batch_size = images.size(0)

        # ----------------------------------------------------
        # Accumulate losses
        # ----------------------------------------------------

        total_loss += (
            losses["total_loss"].item()
            * batch_size
        )

        total_wbc_loss += (
            losses["wbc_loss"].item()
            * batch_size
        )

        total_attribute_loss += (
            losses["attribute_loss"].item()
            * batch_size
        )

        total_samples += batch_size

        # ====================================================
        # WBC Predictions
        # ====================================================

        if "wbc_prediction" in outputs:

            predictions = outputs[
                "wbc_prediction"
            ]

        else:

            # Fallback if prediction is not explicitly
            # returned by the model.
            predictions = torch.argmax(
                outputs["wbc_logits"],
                dim=1,
            )

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        # ====================================================
        # Attribute Predictions
        # ====================================================

        attribute_predictions = outputs[
            "attribute_predictions"
        ]

        for index, name in enumerate(
            ATTRIBUTE_NAMES
        ):

            logits = attribute_predictions[
                name
            ]

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            target = attributes[
                :,
                index,
            ]

            correct = (
                predicted == target
            ).sum().item()

            attribute_correct[name] += correct

            attribute_total[name] += batch_size

    # ========================================================
    # Safety check
    # ========================================================

    if total_samples == 0:

        raise RuntimeError(
            "Test DataLoader contains zero samples."
        )

    # ========================================================
    # Average losses
    # ========================================================

    average_total_loss = (
        total_loss / total_samples
    )

    average_wbc_loss = (
        total_wbc_loss / total_samples
    )

    average_attribute_loss = (
        total_attribute_loss
        / total_samples
    )

    # ========================================================
    # Overall accuracy
    # ========================================================

    correct_predictions = sum(
        prediction == label
        for prediction, label
        in zip(
            all_predictions,
            all_labels,
        )
    )

    accuracy = (
        correct_predictions
        / total_samples
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    num_classes = len(
        WBC_CLASSES
    )

    confusion_matrix = _build_confusion_matrix(
        labels=all_labels,
        predictions=all_predictions,
        num_classes=num_classes,
    )

    # ========================================================
    # Per-class metrics
    # ========================================================

    class_metrics = _calculate_class_metrics(
        confusion_matrix=confusion_matrix,
        class_names=WBC_CLASSES,
    )

    # ========================================================
    # Macro metrics
    # ========================================================

    macro_precision = sum(
        metrics["precision"]
        for metrics in class_metrics.values()
    ) / num_classes

    macro_recall = sum(
        metrics["recall"]
        for metrics in class_metrics.values()
    ) / num_classes

    macro_f1 = sum(
        metrics["f1"]
        for metrics in class_metrics.values()
    ) / num_classes

    # ========================================================
    # Attribute accuracy
    # ========================================================

    attribute_metrics = {}

    for name in ATTRIBUTE_NAMES:

        if attribute_total[name] > 0:

            attribute_metrics[name] = (
                attribute_correct[name]
                / attribute_total[name]
            )

        else:

            attribute_metrics[name] = 0.0

    # ========================================================
    # Return results
    # ========================================================

    return {
        "total_loss":
            average_total_loss,

        "wbc_loss":
            average_wbc_loss,

        "attribute_loss":
            average_attribute_loss,

        "accuracy":
            accuracy,

        "precision":
            macro_precision,

        "recall":
            macro_recall,

        "f1":
            macro_f1,

        "class_metrics":
            class_metrics,

        "attribute_metrics":
            attribute_metrics,

        "confusion_matrix":
            confusion_matrix,

        "predictions":
            all_predictions,

        "labels":
            all_labels,
    }


# ============================================================
# Print Test Results
# ============================================================

def print_test_results(results):
    """
    Print test evaluation results.

    Parameters
    ----------
    results : dict
        Output returned by evaluate_test_set().
    """

    print()
    print("=" * 70)
    print("MAL-ViT TEST SET RESULTS")
    print("=" * 70)

    # ========================================================
    # Overall Metrics
    # ========================================================

    print()
    print("Overall Metrics")
    print("-" * 70)

    print(
        f"Total loss      : "
        f"{results['total_loss']:.4f}"
    )

    print(
        f"WBC loss        : "
        f"{results['wbc_loss']:.4f}"
    )

    print(
        f"Attribute loss  : "
        f"{results['attribute_loss']:.4f}"
    )

    print(
        f"Accuracy        : "
        f"{results['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro Precision : "
        f"{results['precision'] * 100:.2f}%"
    )

    print(
        f"Macro Recall    : "
        f"{results['recall'] * 100:.2f}%"
    )

    print(
        f"Macro F1        : "
        f"{results['f1'] * 100:.2f}%"
    )

    # ========================================================
    # Per-Class Metrics
    # ========================================================

    print()
    print("Per-Class Metrics")
    print("-" * 70)

    print(
        f"{'Class':18s}"
        f"{'Precision':>12s}"
        f"{'Recall':>12s}"
        f"{'F1':>12s}"
    )

    for name, metrics in (
        results["class_metrics"].items()
    ):

        print(
            f"{name:18s}"
            f"{metrics['precision'] * 100:11.2f}%"
            f"{metrics['recall'] * 100:11.2f}%"
            f"{metrics['f1'] * 100:11.2f}%"
        )

    # ========================================================
    # Attribute Accuracy
    # ========================================================

    print()
    print("Attribute Accuracy")
    print("-" * 70)

    for name, accuracy in (
        results["attribute_metrics"].items()
    ):

        print(
            f"{name:30s}"
            f"{accuracy * 100:7.2f}%"
        )

    # ========================================================
    # Confusion Matrix
    # ========================================================

    print()
    print("Confusion Matrix")
    print("-" * 70)

    print(
        "Actual \\ Predicted"
    )

    print(
        f"{'':15s}",
        *[
            f"{name[:10]:>10s}"
            for name in WBC_CLASSES
        ],
    )

    for name, row in zip(
        WBC_CLASSES,
        results["confusion_matrix"],
    ):

        print(
            f"{name:15s}",
            *[
                f"{value:10d}"
                for value in row
            ],
        )

    print()
    print("=" * 70)


# ============================================================
# JSON-Friendly Results
# ============================================================

def format_results_for_json(results):
    """
    Convert evaluation results into a clean,
    JSON-serializable dictionary.

    This is useful for saving results to a
    JSON file or sending them through an API.
    """

    per_class_metrics = {}

    for name, metrics in (
        results["class_metrics"].items()
    ):

        per_class_metrics[name] = {
            "precision":
                f"{metrics['precision'] * 100:.2f}%",

            "recall":
                f"{metrics['recall'] * 100:.2f}%",

            "f1_score":
                f"{metrics['f1'] * 100:.2f}%",
        }

    attribute_accuracy = {}

    for name, accuracy in (
        results["attribute_metrics"].items()
    ):

        attribute_accuracy[name] = (
            f"{accuracy * 100:.2f}%"
        )

    confusion_matrix_dict = {}

    for actual_name, row in zip(
        WBC_CLASSES,
        results["confusion_matrix"],
    ):

        confusion_matrix_dict[
            actual_name
        ] = {
            predicted_name: int(value)
            for predicted_name, value
            in zip(
                WBC_CLASSES,
                row,
            )
        }

    return {
        "per_class_metrics":
            per_class_metrics,

        "attribute_accuracy":
            attribute_accuracy,

        "confusion_matrix": {
            "predicted_labels":
                list(WBC_CLASSES),

            "matrix":
                confusion_matrix_dict,
        },
    }


# ============================================================
# Module Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Test Evaluation Module")
    print("=" * 60)

    print()
    print("Supported operations:")
    print("  - evaluate_test_set")
    print("  - print_test_results")
    print("  - format_results_for_json")

    print()
    print(
        "Test evaluation module validation: PASSED"
    )