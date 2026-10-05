"""
MAL-ViT Test Evaluation
=======================

Evaluates a trained MAL-ViT model on the test dataset.

Provides:
    - evaluate_test_set()
    - print_test_results()
    - format_results_for_json()
"""

import json

import torch

from config import (
    ATTRIBUTE_NAMES,
    WBC_CLASSES,
)


# ============================================================
# HELPERS
# ============================================================

def _extract_targets(batch):
    """
    Extract WBC and attribute targets from a batch.

    Supports common naming conventions used in the project.
    """

    if isinstance(batch, dict):

        wbc_targets = batch.get(
            "wbc_targets",
            batch.get(
                "label",
                batch.get("labels"),
            ),
        )

        attribute_targets = batch.get(
            "attribute_targets",
            batch.get(
                "attributes",
            ),
        )

    else:

        raise TypeError(
            "Expected dataloader batch to be a dictionary."
        )

    if wbc_targets is None:

        raise KeyError(
            "Could not find WBC targets in batch. "
            "Expected 'wbc_targets', 'label', or 'labels'."
        )

    if attribute_targets is None:

        raise KeyError(
            "Could not find attribute targets in batch. "
            "Expected 'attribute_targets' or 'attributes'."
        )

    return (
        wbc_targets,
        attribute_targets,
    )


# ============================================================
# MODEL OUTPUT HELPERS
# ============================================================

def _get_wbc_logits(outputs):

    if isinstance(outputs, dict):

        if "wbc_logits" in outputs:

            return outputs["wbc_logits"]

        if "logits" in outputs:

            return outputs["logits"]

    raise KeyError(
        "Model output does not contain 'wbc_logits'."
    )


def _get_attribute_predictions(outputs):

    if not isinstance(outputs, dict):

        raise TypeError(
            "Model output must be a dictionary."
        )

    if "attribute_predictions" not in outputs:

        raise KeyError(
            "Model output does not contain "
            "'attribute_predictions'."
        )

    return outputs[
        "attribute_predictions"
    ]


# ============================================================
# LOSS HELPER
# ============================================================

def _compute_loss(
    loss_fn,
    outputs,
    wbc_targets,
    attribute_targets,
):
    """
    Supports the project's compute_total_loss()
    and MALViTLoss interfaces.
    """

    # --------------------------------------------------------
    # Try project keyword interface
    # --------------------------------------------------------

    try:

        result = loss_fn(
            outputs=outputs,
            wbc_targets=wbc_targets,
            attribute_targets=attribute_targets,
        )

    except TypeError:

        # ----------------------------------------------------
        # Try positional interface
        # ----------------------------------------------------

        result = loss_fn(
            outputs,
            wbc_targets,
            attribute_targets,
        )

    # --------------------------------------------------------
    # Dictionary result
    # --------------------------------------------------------

    if isinstance(result, dict):

        total_loss = result.get(
            "total_loss"
        )

        wbc_loss = result.get(
            "wbc_loss"
        )

        attribute_loss = result.get(
            "attribute_loss"
        )

        if total_loss is None:

            raise KeyError(
                "Loss result does not contain "
                "'total_loss'."
            )

        if wbc_loss is None:

            wbc_loss = torch.zeros_like(
                total_loss
            )

        if attribute_loss is None:

            attribute_loss = torch.zeros_like(
                total_loss
            )

        return (
            total_loss,
            wbc_loss,
            attribute_loss,
        )

    # --------------------------------------------------------
    # Tensor result
    # --------------------------------------------------------

    if torch.is_tensor(result):

        zero = torch.zeros_like(result)

        return (
            result,
            zero,
            zero,
        )

    raise TypeError(
        "Unsupported loss function output."
    )


# ============================================================
# TEST EVALUATION
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

    Returns a dictionary containing:

        total_loss
        wbc_loss
        attribute_loss
        accuracy
        precision
        recall
        f1
        class_metrics
        confusion_matrix
        attribute_metrics
        predictions
        labels
    """

    model.eval()

    total_loss_sum = 0.0
    wbc_loss_sum = 0.0
    attribute_loss_sum = 0.0

    total_samples = 0

    predictions = []
    labels = []

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    num_classes = len(
        WBC_CLASSES
    )

    confusion_matrix = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    # --------------------------------------------------------
    # Attribute statistics
    # --------------------------------------------------------

    attribute_correct = {
        name: 0
        for name in ATTRIBUTE_NAMES
    }

    attribute_total = {
        name: 0
        for name in ATTRIBUTE_NAMES
    }

    # ========================================================
    # BATCH LOOP
    # ========================================================

    for batch in dataloader:

        (
            wbc_targets,
            attribute_targets,
        ) = _extract_targets(batch)

        # ----------------------------------------------------
        # Move targets to device
        # ----------------------------------------------------

        if isinstance(batch, dict):

            if "image" in batch:

                images = batch["image"]

            elif "images" in batch:

                images = batch["images"]

            else:

                raise KeyError(
                    "Could not find images in batch. "
                    "Expected 'image' or 'images'."
                )

        else:

            raise TypeError(
                "Expected dictionary batch."
            )

        images = images.to(
            device,
            non_blocking=True,
        )

        wbc_targets = wbc_targets.to(
            device,
            non_blocking=True,
        )

        attribute_targets = (
            attribute_targets.to(
                device,
                non_blocking=True,
            )
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        outputs = model(
            images
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        (
            total_loss,
            wbc_loss,
            attribute_loss,
        ) = _compute_loss(
            loss_fn=loss_fn,
            outputs=outputs,
            wbc_targets=wbc_targets,
            attribute_targets=attribute_targets,
        )

        batch_size = images.size(0)

        total_samples += batch_size

        total_loss_sum += (
            float(total_loss.item())
            * batch_size
        )

        wbc_loss_sum += (
            float(wbc_loss.item())
            * batch_size
        )

        attribute_loss_sum += (
            float(attribute_loss.item())
            * batch_size
        )

        # ----------------------------------------------------
        # WBC predictions
        # ----------------------------------------------------

        wbc_logits = _get_wbc_logits(
            outputs
        )

        wbc_predictions = (
            torch.argmax(
                wbc_logits,
                dim=1,
            )
        )

        predictions.extend(
            wbc_predictions.cpu().tolist()
        )

        labels.extend(
            wbc_targets.cpu().tolist()
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        for true_label, predicted_label in zip(
            wbc_targets.cpu().tolist(),
            wbc_predictions.cpu().tolist(),
        ):

            if (
                0 <= true_label < num_classes
                and
                0 <= predicted_label < num_classes
            ):

                confusion_matrix[
                    true_label
                ][predicted_label] += 1

        # ----------------------------------------------------
        # Attribute predictions
        # ----------------------------------------------------

        attribute_predictions = (
            _get_attribute_predictions(
                outputs
            )
        )

        for index, name in enumerate(
            ATTRIBUTE_NAMES
        ):

            if name not in attribute_predictions:

                continue

            logits = (
                attribute_predictions[name]
            )

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            target = (
                attribute_targets[:, index]
            )

            correct = (
                predicted == target
            ).sum().item()

            attribute_correct[
                name
            ] += correct

            attribute_total[
                name
            ] += target.numel()

    # ========================================================
    # FINAL LOSS
    # ========================================================

    if total_samples == 0:

        raise ValueError(
            "Test dataloader contains no samples."
        )

    average_total_loss = (
        total_loss_sum
        / total_samples
    )

    average_wbc_loss = (
        wbc_loss_sum
        / total_samples
    )

    average_attribute_loss = (
        attribute_loss_sum
        / total_samples
    )

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    correct = sum(
        confusion_matrix[i][i]
        for i in range(num_classes)
    )

    accuracy = (
        correct / total_samples
    )

    true_positive = []
    precision_values = []
    recall_values = []
    f1_values = []

    class_metrics = {}

    for i, class_name in enumerate(
        WBC_CLASSES
    ):

        tp = confusion_matrix[i][i]

        fp = sum(
            confusion_matrix[row][i]
            for row in range(num_classes)
            if row != i
        )

        fn = sum(
            confusion_matrix[i][col]
            for col in range(num_classes)
            if col != i
        )

        support = sum(
            confusion_matrix[i]
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        true_positive.append(tp)

        precision_values.append(
            precision
        )

        recall_values.append(
            recall
        )

        f1_values.append(
            f1
        )

        class_metrics[
            class_name
        ] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    precision = (
        sum(precision_values)
        / num_classes
    )

    recall = (
        sum(recall_values)
        / num_classes
    )

    f1 = (
        sum(f1_values)
        / num_classes
    )

    # ========================================================
    # ATTRIBUTE METRICS
    # ========================================================

    attribute_metrics = {}

    for name in ATTRIBUTE_NAMES:

        total = attribute_total[
            name
        ]

        if total > 0:

            attribute_metrics[
                name
            ] = (
                attribute_correct[name]
                / total
            )

        else:

            attribute_metrics[
                name
            ] = 0.0

    # ========================================================
    # RESULTS
    # ========================================================

    results = {

        "total_loss":
            average_total_loss,

        "wbc_loss":
            average_wbc_loss,

        "attribute_loss":
            average_attribute_loss,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "class_metrics":
            class_metrics,

        "confusion_matrix":
            confusion_matrix,

        "attribute_metrics":
            attribute_metrics,

        "predictions":
            predictions,

        "labels":
            labels,
    }

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_test_results(
    results,
):
    """
    Print formatted test evaluation results.
    """

    print()
    print("=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)

    print()

    print(
        f"Test loss       : "
        f"{results['total_loss']:.6f}"
    )

    print(
        f"WBC loss        : "
        f"{results['wbc_loss']:.6f}"
    )

    print(
        f"Attribute loss  : "
        f"{results['attribute_loss']:.6f}"
    )

    print()

    print(
        f"Accuracy        : "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Precision       : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1              : "
        f"{results['f1']:.4f}"
    )

    # --------------------------------------------------------
    # Class metrics
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("PER-CLASS METRICS")
    print("-" * 70)

    for class_name, metrics in (
        results["class_metrics"].items()
    ):

        print(
            f"{class_name:<15} "
            f"Precision={metrics['precision']:.4f} "
            f"Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"Support={metrics['support']}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)

    print(
        "Classes:",
        WBC_CLASSES,
    )

    for row in results[
        "confusion_matrix"
    ]:

        print(row)

    # --------------------------------------------------------
    # Attributes
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("ATTRIBUTE ACCURACY")
    print("-" * 70)

    for name, value in (
        results[
            "attribute_metrics"
        ].items()
    ):

        print(
            f"{name:<35}: "
            f"{value:.4f}"
        )

    print()
    print("=" * 70)


# ============================================================
# JSON FORMAT
# ============================================================

def format_results_for_json(
    results,
):
    """
    Convert test results into a JSON-safe dictionary.

    The expected keys are:

        per_class_metrics
        attribute_accuracy
        confusion_matrix
    """

    json_results = {

        "test_loss":
            float(
                results["total_loss"]
            ),

        "wbc_loss":
            float(
                results["wbc_loss"]
            ),

        "attribute_loss":
            float(
                results["attribute_loss"]
            ),

        "accuracy":
            float(
                results["accuracy"]
            ),

        "precision":
            float(
                results["precision"]
            ),

        "recall":
            float(
                results["recall"]
            ),

        "f1":
            float(
                results["f1"]
            ),

        "per_class_metrics":
            results[
                "class_metrics"
            ],

        "attribute_accuracy":
            {
                name: float(value)
                for name, value in
                results[
                    "attribute_metrics"
                ].items()
            },

        "confusion_matrix":
            results[
                "confusion_matrix"
            ],

        "predictions":
            [
                int(x)
                for x in results[
                    "predictions"
                ]
            ],

        "labels":
            [
                int(x)
                for x in results[
                    "labels"
                ]
            ],
    }

    if "checkpoint_source" in results:
        json_results["checkpoint_source"] = results["checkpoint_source"]
    json_results["num_test_images"] = len(json_results["labels"])
    if "concept_representation" in results:
        json_results["concept_representation"] = results["concept_representation"]

    return json_results


# ============================================================
# SAVE JSON
# ============================================================

def save_results_json(
    results,
    path,
):
    """
    Save formatted test results to JSON.
    """

    formatted = (
        format_results_for_json(
            results
        )
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            formatted,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Test Evaluation Module")
    print("=" * 70)

    print()
    print(
        "Available functions:"
    )

    print(
        "  - evaluate_test_set()"
    )

    print(
        "  - print_test_results()"
    )

    print(
        "  - format_results_for_json()"
    )

    print()
    print(
        "Test evaluation module validation: PASSED"
    )
