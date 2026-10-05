"""One strict model-loading path for evaluation, inference, and the app."""

from pathlib import Path
import warnings

import torch

from config import BEST_MODEL_PATH, LEGACY_MODEL_PATH, ATTRIBUTE_MODEL_PATH, DEVICE
from models.complete_model import CompleteMALViT
from models.attribute_wbc_classifier import AttributeWBCClassifier


def read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Supply the trained weights described in README.md "
            "or an exported checkpoint. Retraining is not required for existing trained weights."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a checkpoint dictionary: {path}")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"Invalid model state dictionary: {path}")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError(f"Checkpoint contains non-finite weights: {path}")
    return checkpoint, state


def load_model(checkpoint_path=None, device=DEVICE, attribute_checkpoint_path=None):
    """Load a concept checkpoint, or compose the two existing trained stages.

    Legacy register-feature classifier weights are discarded. Every image-to-
    attribute parameter and every replacement head parameter must match strictly.
    An incompatible/missing replacement never leaves a random WBC head in use.
    """
    if checkpoint_path is None:
        checkpoint_path = BEST_MODEL_PATH if BEST_MODEL_PATH.is_file() else LEGACY_MODEL_PATH
    checkpoint_path = Path(checkpoint_path)
    checkpoint, state = read_checkpoint(checkpoint_path)
    model = CompleteMALViT()
    is_legacy = any(key.startswith("wbc_classifier.classifier.") for key in state)
    if is_legacy:
        for key in ("attribute_names", "attribute_encoders", "cell_labels"):
            if key in checkpoint and checkpoint[key] != model.checkpoint_metadata()[key]:
                raise ValueError(f"Legacy image checkpoint {key} does not match the current model")
        backbone = {k: v for k, v in state.items() if not k.startswith("wbc_classifier.")}
        expected_backbone = {k for k in model.state_dict() if not k.startswith("wbc_classifier.")}
        if set(backbone) != expected_backbone:
            raise ValueError("Legacy checkpoint has missing or unexpected image-to-attribute weights")
        head_path = Path(attribute_checkpoint_path or ATTRIBUTE_MODEL_PATH)
        head_checkpoint, head_state = read_checkpoint(head_path)
        for key in ("attribute_names", "attribute_encoders", "cell_labels"):
            if key in head_checkpoint and head_checkpoint[key] != model.checkpoint_metadata()[key]:
                raise ValueError(f"Attribute checkpoint {key} does not match the current model")
        head = AttributeWBCClassifier()
        head.load_state_dict(head_state, strict=True)
        combined = dict(backbone)
        combined.update({f"wbc_classifier.{k}": v for k, v in head_state.items()})
        model.load_state_dict(combined, strict=True)
        model.checkpoint_source = {
            "mode": "composed", "image_model": str(checkpoint_path),
            "attribute_model": str(head_path),
        }
        warnings.warn(
            "Using the legacy image-to-attribute weights with the trained attribute-to-WBC "
            "head. The old image-feature WBC head is excluded. Evaluate this composed "
            "pipeline separately; historical WBC scores do not apply.",
            UserWarning, stacklevel=2,
        )
    else:
        for key, expected in model.checkpoint_metadata().items():
            if key in checkpoint and checkpoint[key] != expected:
                raise ValueError(f"Checkpoint {key} does not match the current model")
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is incompatible with the attribute bottleneck. "
                "Train with python train.py, or provide the legacy image model and its "
                "separate trained attribute head."
            ) from exc
        model.checkpoint_source = {"mode": "native", "model": str(checkpoint_path)}
        if checkpoint.get("checkpoint_kind") == "inference_export":
            model.checkpoint_source.update(
                mode="exported", source=checkpoint.get("source_checkpoints", {}),
            )
    return model.to(device).eval()
