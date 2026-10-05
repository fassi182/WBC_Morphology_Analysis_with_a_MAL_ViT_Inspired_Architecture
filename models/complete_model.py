"""Image -> 11 morphology predictions -> attribute-only WBC classification."""

import torch

from data.encoders import ATTRIBUTE_NAMES, ATTRIBUTE_ENCODERS, CELL_LABELS
from models.mal_vit import MALViT
from models.wbc_classifier import WBCClassifier
from models.attribute_wbc_classifier import attributes_to_one_hot

ARCHITECTURE = "mal_vit_attribute_bottleneck_v1"


class CompleteMALViT(MALViT):
    """Jointly train image-to-attributes and attributes-to-WBC stages.

    Training and automatic inference use the same soft bottleneck: eleven
    probability distributions concatenated into 31 values. Categorical
    attribute interventions are available as a separate method.
    """

    def __init__(self):
        super().__init__()
        self.wbc_classifier = WBCClassifier()

    def checkpoint_metadata(self):
        return {
            "architecture": ARCHITECTURE,
            "attribute_encoders": ATTRIBUTE_ENCODERS,
            "attribute_names": ATTRIBUTE_NAMES,
            "cell_labels": CELL_LABELS,
            "concept_representation": "probabilities",
        }

    def extract_attributes(self, images, return_attention=False):
        """Stage 1: predict morphology without invoking the WBC head."""
        return super().forward(images, return_attention=return_attention)

    def attribute_probabilities_to_vector(self, attribute_predictions):
        return torch.cat([
            attribute_predictions[name].softmax(dim=-1) for name in ATTRIBUTE_NAMES
        ], dim=-1)

    def attributes_to_vector(self, attribute_indices):
        if isinstance(attribute_indices, dict):
            attribute_indices = torch.stack([
                attribute_indices[name] for name in ATTRIBUTE_NAMES
            ], dim=1)
        return attributes_to_one_hot(attribute_indices)

    def classify_attribute_predictions(self, attribute_predictions):
        """Stage 2: classify using only the eleven predicted distributions."""
        vector = self.attribute_probabilities_to_vector(attribute_predictions)
        return self._classify_vector(vector)

    def _classify_vector(self, vector):
        logits = self.wbc_classifier(vector)
        return {
            "attribute_vector": vector,
            "wbc_logits": logits,
            "wbc_prediction": logits.argmax(dim=1),
        }

    def forward(self, images, return_attention=False):
        results = self.extract_attributes(images, return_attention=return_attention)
        results.update(self.classify_attribute_predictions(results["attribute_predictions"]))
        return results

    def classify_approved_attributes(self, attribute_indices):
        """Optional intervention: classify supplied categorical attributes only."""
        return self._classify_vector(self.attributes_to_vector(attribute_indices))
