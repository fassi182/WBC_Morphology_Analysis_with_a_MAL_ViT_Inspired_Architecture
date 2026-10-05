"""Reusable application API. Loads trained weights once; never trains a model."""

from config import DEVICE
from inference import predict_details
from utils.model_loading import load_model


class WBCPredictor:
    """Image -> eleven morphology distributions -> WBC type.

    Create one instance at application startup and reuse predict() for images.
    Inputs can be image paths, PIL images, or encoded JPG/PNG bytes.
    Results contain ordinary Python values and can be serialized to JSON.
    """

    def __init__(self, checkpoint_path=None, *, attribute_checkpoint_path=None, device=None):
        self.model = load_model(
            checkpoint_path, device=DEVICE if device is None else device,
            attribute_checkpoint_path=attribute_checkpoint_path,
        )

    def predict(self, image):
        return predict_details(image, model=self.model)
