

"""
evaluate_attributes.py

Diagnostic #2: Per-Attribute Accuracy, Confusion Matrix, and
Majority-Class Baseline.

WHY THIS SCRIPT EXISTS
-----------------------
`test.py` in this repo only reports accuracy for the final 8-class
WBC prediction. It never checks whether each of the 11 morphology
attribute heads (cell_size, cell_shape, nucleus_shape, ...) is
actually learning anything, or just predicting the majority class
for that attribute.

If, say, 92% of "cell_shape" labels in your training data are
"round", a head that always outputs "round" gets 92% accuracy while
learning nothing — and its Grad-CAM will latch onto whatever
spurious correlate happens to exist in the image (background shade,
an unrelated blob, an "attention sink" patch, etc.) rather than the
actual cell boundary. That would explain a centrally-located,
non-boundary-tracing heatmap for "Cell Shape".

This script reports, for every attribute:
- accuracy
- per-class precision/recall/F1
- confusion matrix
- the majority-class baseline accuracy (what you'd get by always
  guessing the most common label)
- the gap between your model and that baseline

A head whose accuracy is close to its majority-class baseline is a
red flag: it likely hasn't learned the attribute at all.

USAGE
-----
    python evaluate_attributes.py

Runs on the test split by default. Uses the same checkpoint/config
as the rest of the repo.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

from data.dataloader import create_dataloaders
from data.encoders import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_ENCODERS,
)

from utils.model_loading import load_model as load_pipeline_model

# ==========================================================
# Build Reverse Label Maps (index -> human-readable name)
# ==========================================================
#
# ATTRIBUTE_ENCODERS[attribute] is {label_string: index}.
# We invert it so confusion matrices are readable.
#
# ==========================================================

def build_index_to_label():

    mapping = {}

    for attribute, encoder in ATTRIBUTE_ENCODERS.items():

        mapping[attribute] = {
            index: label
            for label, index in encoder.items()
        }

    return mapping


# ==========================================================
# Load Model
# ==========================================================

def load_model():
    return load_pipeline_model(device=DEVICE)


# ==========================================================
# Collect Predictions
# ==========================================================

def collect_predictions(model, loader):
    """
    Returns
    -------
    y_true : dict[attribute_name] -> list[int]
    y_pred : dict[attribute_name] -> list[int]
    """

    y_true = {name: [] for name in ATTRIBUTE_NAMES}
    y_pred = {name: [] for name in ATTRIBUTE_NAMES}
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():

        for batch in loader:

            images = batch["images"].to(device)

            attributes = batch["attributes"].to(device)
            # (B, 11)

            outputs = model(images)

            predictions = outputs["attribute_predictions"]
            # dict[name] -> (B, num_classes) logits

            for idx, name in enumerate(ATTRIBUTE_NAMES):

                pred_labels = predictions[name].argmax(dim=1)

                y_true[name].extend(
                    attributes[:, idx].cpu().numpy().tolist()
                )

                y_pred[name].extend(
                    pred_labels.cpu().numpy().tolist()
                )

    return y_true, y_pred


# ==========================================================
# Report
# ==========================================================

def majority_baseline_accuracy(labels):
    """
    Accuracy of always predicting the single most frequent
    label in `labels`.
    """

    labels = np.array(labels)

    values, counts = np.unique(labels, return_counts=True)

    majority_count = counts.max()

    return majority_count / len(labels)


def main():

    print("=" * 70)
    print("Per-Attribute Diagnostic: Accuracy vs. Majority-Class Baseline")
    print("=" * 70)

    _, _, test_loader = create_dataloaders()

    model = load_model()

    print("\nModel loaded. Running inference on test set...\n")

    y_true, y_pred = collect_predictions(model, test_loader)

    index_to_label = build_index_to_label()

    summary_rows = []

    for name in ATTRIBUTE_NAMES:

        true_labels = y_true[name]
        pred_labels = y_pred[name]

        acc = accuracy_score(true_labels, pred_labels)

        baseline = majority_baseline_accuracy(true_labels)

        gap = acc - baseline

        summary_rows.append(
            (name, acc, baseline, gap)
        )

        label_map = index_to_label[name]

        class_indices = sorted(label_map.keys())

        target_names = [
            label_map[i] for i in class_indices
        ]

        print("-" * 70)
        print(f"Attribute: {name}")
        print("-" * 70)

        print(f"Model accuracy       : {acc*100:6.2f}%")
        print(f"Majority-class guess  : {baseline*100:6.2f}%  "
              f"(always predicting '{label_map[max(set(true_labels), key=true_labels.count)]}')")
        print(f"Gap over baseline     : {gap*100:+6.2f} points")

        if gap < 0.05:
            print("  Accuracy is less than 5 percentage points above the majority "
                  "baseline. Inspect per-class recall and the confusion matrix; "
                  "this gap alone does not establish whether the head learned useful features.")

        print()

        print(
            classification_report(
                true_labels,
                pred_labels,
                labels=class_indices,
                target_names=target_names,
                digits=3,
                zero_division=0,
            )
        )

        print("Confusion Matrix "
              f"(rows = true, cols = predicted, order = {target_names})")
        print(
            confusion_matrix(
                true_labels,
                pred_labels,
                labels=class_indices,
            )
        )
        print()

    # ------------------------------------------------------
    # Final Summary Table
    # ------------------------------------------------------

    print("=" * 70)
    print("SUMMARY (sorted by gap over majority-class baseline)")
    print("=" * 70)

    summary_rows.sort(key=lambda row: row[3])

    print(f"{'Attribute':30} {'Accuracy':>10} {'Baseline':>10} {'Gap':>8}")

    for name, acc, baseline, gap in summary_rows:

        flag = " <-- inspect per-class metrics" if gap < 0.05 else ""

        print(
            f"{name:30} {acc*100:9.2f}% {baseline*100:9.2f}% "
            f"{gap*100:+7.2f}{flag}"
        )


if __name__ == "__main__":
    main()
