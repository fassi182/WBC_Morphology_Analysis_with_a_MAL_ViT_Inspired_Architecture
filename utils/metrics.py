"""
metrics.py

Evaluation metrics for

This module computes:

1. WBC Classification Accuracy
2. Attribute Prediction Accuracy
3. Confusion Matrix
4. Classification Report


"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from config import ATTRIBUTE_CLASSES


# ==========================================================
# WBC Accuracy
# ==========================================================

def compute_wbc_accuracy(
    predictions,
    targets,
):
    """
    Compute WBC classification accuracy.

    Parameters
    ----------
    predictions : array-like

    targets : array-like

    Returns
    -------
    float
    """

    return accuracy_score(
        targets,
        predictions,
    ) * 100


# ==========================================================
# Attribute Accuracy
# ==========================================================

def compute_attribute_accuracy(
    prediction_dict,
    target_matrix,
):
    """
    Computes accuracy for every morphology attribute.

    Parameters
    ----------
    prediction_dict : dict

        {
            attribute_name : ndarray(B,)
        }

    target_matrix : ndarray

        Shape (B,11)

    Returns
    -------
    dict
    """

    results = {}

    for index, attribute in enumerate(
        ATTRIBUTE_CLASSES.keys()
    ):

        accuracy = accuracy_score(

            target_matrix[:, index],

            prediction_dict[attribute],

        )

        results[attribute] = accuracy * 100

    return results


# ==========================================================
# Mean Attribute Accuracy
# ==========================================================

def mean_attribute_accuracy(
    attribute_results,
):
    """
    Average accuracy across all attributes.
    """

    return float(
        np.mean(
            list(attribute_results.values())
        )
    )


# ==========================================================
# Confusion Matrix
# ==========================================================

def compute_confusion_matrix(
    predictions,
    targets,
):
    """
    Returns confusion matrix.
    """

    return confusion_matrix(
        targets,
        predictions,
    )


# ==========================================================
# Classification Report
# ==========================================================

def compute_classification_report(
    predictions,
    targets,
    class_names=None,
):
    """
    Returns sklearn classification report.
    """

    return classification_report(

        targets,

        predictions,

        target_names=class_names,

        digits=4,

        zero_division=0,

    )


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Metrics Test")
    print("=" * 60)

    # -----------------------------
    # WBC Accuracy
    # -----------------------------

    y_true = np.array(
        [0, 1, 2, 1, 0, 2]
    )

    y_pred = np.array(
        [0, 1, 2, 0, 0, 2]
    )

    accuracy = compute_wbc_accuracy(
        y_pred,
        y_true,
    )

    print(f"\nWBC Accuracy : {accuracy:.2f}%")

    # -----------------------------
    # Attribute Accuracy
    # -----------------------------

    prediction_dict = {}

    target_matrix = np.random.randint(
        0,
        2,
        (10, 11),
    )

    for attribute, classes in ATTRIBUTE_CLASSES.items():

        prediction_dict[attribute] = np.random.randint(
            0,
            classes,
            10,
        )

    # Fix targets for multi-class attributes

    target_matrix[:, 2] = np.random.randint(
        0,
        6,
        10,
    )

    target_matrix[:, 7] = np.random.randint(
        0,
        3,
        10,
    )

    target_matrix[:, 8] = np.random.randint(
        0,
        4,
        10,
    )

    target_matrix[:, 9] = np.random.randint(
        0,
        4,
        10,
    )

    attribute_results = compute_attribute_accuracy(
        prediction_dict,
        target_matrix,
    )

    print("\nAttribute Accuracy")

    for key, value in attribute_results.items():

        print(f"{key:30} {value:.2f}%")

    print(
        f"\nMean Attribute Accuracy : "
        f"{mean_attribute_accuracy(attribute_results):.2f}%"
    )

    print("\nConfusion Matrix")

    print(
        compute_confusion_matrix(
            y_pred,
            y_true,
        )
    )

    print("\nClassification Report")

    print(
        compute_classification_report(
            y_pred,
            y_true,
        )
    )