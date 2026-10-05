"""Train the shared WBC head independently on categorical morphology labels."""

import torch
import torch.nn.functional as F

from data.encoders import ATTRIBUTE_NAMES, ATTRIBUTE_CLASS_COUNTS
from models.wbc_classifier import WBCClassifier, ATTRIBUTE_INPUT_DIM

HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
DROPOUT = 0.20


def attributes_to_one_hot(attributes):
    """Encode (B, 11) categorical indices as (B, 31) concept values."""
    if attributes.ndim != 2 or attributes.size(1) != len(ATTRIBUTE_NAMES):
        raise ValueError(f"Expected (B, {len(ATTRIBUTE_NAMES)}) attribute indices")
    if attributes.is_floating_point() or attributes.is_complex() or attributes.dtype == torch.bool:
        raise ValueError("Attribute indices must use an integer dtype")
    vectors = []
    for index, name in enumerate(ATTRIBUTE_NAMES):
        values = attributes[:, index].long()
        count = ATTRIBUTE_CLASS_COUNTS[name]
        if ((values < 0) | (values >= count)).any():
            raise ValueError(f"Invalid category index for {name}; expected 0..{count - 1}")
        vectors.append(F.one_hot(values, num_classes=count).float())
    return torch.cat(vectors, dim=1)


class AttributeWBCClassifier(WBCClassifier):
    """Same head/checkpoint keys, with categorical input instead of probabilities."""

    def forward(self, attributes):
        return super().forward(attributes_to_one_hot(attributes))


@torch.no_grad()
def predict_wbc_from_attributes(model, attributes):
    model.eval()
    return model(attributes).argmax(dim=1)
