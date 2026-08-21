# tests/test_evaluation_test.py

"""
MAL-ViT Test Evaluation Integration Test
"""

import torch

from config import (
    DEVICE,
    BEST_MODEL_PATH,
    WBC_CLASSES,
)

from data.dataloader import create_dataloaders
from models.complete_model import CompleteMALViT

from training.losses import compute_total_loss

from training.test import (
    evaluate_test_set,
    print_test_results,
    format_results_for_json,
)


def main():

    print("=" * 70)
    print("MAL-ViT TEST EVALUATION INTEGRATION TEST")
    print("=" * 70)

    device = torch.device(DEVICE)

    print()
    print("Device:", device)

    # ======================================================
    # Data
    # ======================================================

    print()
    print("[1] Loading data...")

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders()

    print(
        f"Train: {len(train_loader)} batches"
    )

    print(
        f"Val  : {len(val_loader)} batches"
    )

    print(
        f"Test : {len(test_loader)} batches"
    )

    # ======================================================
    # Model
    # ======================================================

    print()
    print("[2] Creating model...")

    model = CompleteMALViT()

    model = model.to(device)

    print("Model created.")

    # ======================================================
    # Load checkpoint
    # ======================================================

    print()
    print("[3] Loading trained checkpoint...")

    print(
        "Checkpoint path:"
    )

    print(BEST_MODEL_PATH)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )

    # ------------------------------------------------------
    # Validate checkpoint
    # ------------------------------------------------------

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            "Checkpoint must be a dictionary."
        )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    # ------------------------------------------------------
    # Load model weights
    # ------------------------------------------------------

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        "Checkpoint loaded successfully."
    )

    if "epoch" in checkpoint:

        print(
            f"Checkpoint epoch : "
            f"{checkpoint['epoch']}"
        )

    if "metric" in checkpoint:

        print(
            f"Checkpoint metric: "
            f"{checkpoint['metric']:.6f}"
        )

    # ======================================================
    # Evaluate
    # ======================================================

    print()
    print("[4] Evaluating test set...")

    results = evaluate_test_set(
        model=model,
        dataloader=test_loader,
        loss_fn=compute_total_loss,
        device=device,
    )

    print(
        "Test evaluation completed."
    )

    # ======================================================
    # Print results
    # ======================================================

    print_test_results(
        results
    )

    # ======================================================
    # Validation
    # ======================================================

    assert "total_loss" in results
    assert "wbc_loss" in results
    assert "attribute_loss" in results
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "class_metrics" in results
    assert "confusion_matrix" in results
    assert "attribute_metrics" in results
    assert "predictions" in results
    assert "labels" in results

    # ------------------------------------------------------
    # Confusion matrix shape
    # ------------------------------------------------------

    assert len(
        results["confusion_matrix"]
    ) == len(WBC_CLASSES)

    assert all(
        len(row) == len(WBC_CLASSES)
        for row in results[
            "confusion_matrix"
        ]
    )

    # ------------------------------------------------------
    # Prediction count
    # ------------------------------------------------------

    assert len(
        results["predictions"]
    ) == len(
        results["labels"]
    )

    # ------------------------------------------------------
    # Class metrics
    # ------------------------------------------------------

    assert set(
        results["class_metrics"].keys()
    ) == set(WBC_CLASSES)

    # ------------------------------------------------------
    # Metric ranges
    # ------------------------------------------------------

    for metric_name in [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:

        value = results[
            metric_name
        ]

        assert 0.0 <= value <= 1.0, (
            f"{metric_name} outside "
            f"[0,1]: {value}"
        )

    # ------------------------------------------------------
    # Attribute metrics
    # ------------------------------------------------------

    for name, value in (
        results["attribute_metrics"].items()
    ):

        assert 0.0 <= value <= 1.0, (
            f"Attribute metric outside "
            f"[0,1]: {name}={value}"
        )

    # ======================================================
    # JSON result
    # ======================================================

    json_results = format_results_for_json(
        results
    )

    assert "per_class_metrics" in json_results
    assert "attribute_accuracy" in json_results
    assert "confusion_matrix" in json_results

    # ======================================================
    # Final validation
    # ======================================================

    print()
    print(
        "=" * 70
    )

    print(
        "TEST EVALUATION VALIDATION: PASSED"
    )

    print(
        "=" * 70)


if __name__ == "__main__":

    main()