"""Image -> eleven morphology distributions -> WBC prediction."""

import argparse
from io import BytesIO
from os import PathLike
from pathlib import Path

import torch
from PIL import Image

from config import DEVICE
from data.transforms import test_transform
from data.encoders import ATTRIBUTE_NAMES, ATTRIBUTE_DECODERS, decode_wbc
from utils.model_loading import load_model


def load_rgb_image(image):
    """Accept a filename, PIL image, or encoded JPG/PNG bytes from an upload."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        image = BytesIO(image)
    elif not isinstance(image, (str, PathLike)):
        raise TypeError("Expected an image path, PIL.Image, or encoded image bytes")
    with Image.open(image) as source:
        return source.convert("RGB")


def predict_details(image, model=None):
    """Shared image prediction for the CLI and explanation/app pipeline."""
    image = load_rgb_image(image)
    if model is None:
        model = load_model()
    model.eval()
    device = next(model.parameters()).device
    tensor = test_transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
    attributes = {}
    probabilities = {}
    for name in ATTRIBUTE_NAMES:
        values = outputs["attribute_predictions"][name].softmax(dim=-1)[0].cpu()
        labels = ATTRIBUTE_DECODERS[name]
        attributes[name] = labels[int(values.argmax())]
        probabilities[name] = {labels[i]: float(p) for i, p in enumerate(values)}
    wbc_probabilities = outputs["wbc_logits"].softmax(dim=-1)[0].cpu()
    wbc_index = int(wbc_probabilities.argmax())
    return {
        "wbc": decode_wbc(wbc_index),
        "wbc_index": wbc_index,
        "wbc_confidence": float(wbc_probabilities[wbc_index]),
        "wbc_probabilities": {decode_wbc(i): float(p) for i, p in enumerate(wbc_probabilities)},
        "attributes": attributes,
        "attribute_probabilities": probabilities,
        "checkpoint_source": getattr(model, "checkpoint_source", {}),
    }


def predict(image_path, model=None):
    """Backward-compatible tuple result, without loading a model at import time."""
    result = predict_details(image_path, model=model)
    return result["wbc"], result["attributes"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--attribute-checkpoint", type=Path,
                        help="Separate WBC head when --checkpoint is a legacy image model")
    args = parser.parse_args(argv)
    model = load_model(args.checkpoint, device=DEVICE,
                       attribute_checkpoint_path=args.attribute_checkpoint)
    result = predict_details(args.image, model=model)
    print("Morphology predictions:")
    for name, value in result["attributes"].items():
        print(f"  {name:30s}: {value}")
    print(f"WBC type from these attribute distributions: {result['wbc']}")
    print(f"WBC probability: {result['wbc_confidence']:.2%}")


if __name__ == "__main__":
    main()
