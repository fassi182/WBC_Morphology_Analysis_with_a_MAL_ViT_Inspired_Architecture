"""WBC classification using only the 31 values encoding 11 attributes."""

import torch.nn as nn

from config import NUM_WBC_CLASSES
from data.encoders import ATTRIBUTE_CLASS_COUNTS, ATTRIBUTE_NAMES

ATTRIBUTE_INPUT_DIM = sum(ATTRIBUTE_CLASS_COUNTS[name] for name in ATTRIBUTE_NAMES)


class WBCClassifier(nn.Module):
    """Shared head for attribute probabilities or one-hot attributes.

    No patch tokens, register tokens, or image features enter this head.
    """

    def __init__(self, hidden_dim_1=64, hidden_dim_2=32, dropout=0.2):
        super().__init__()
        self.input_dim = ATTRIBUTE_INPUT_DIM
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim_1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, NUM_WBC_CLASSES),
        )

    def forward(self, attribute_vector):
        if attribute_vector.ndim != 2 or attribute_vector.size(1) != self.input_dim:
            raise ValueError(f"Expected (B, {self.input_dim}) attribute values, "
                             f"got {tuple(attribute_vector.shape)}")
        return self.network(attribute_vector)
