"""Application reuse, export, and audit regressions; no dataset required."""

import json

import numpy as np
import pytest
import torch
from PIL import Image, UnidentifiedImageError

from data.encoders import ATTRIBUTE_NAMES, ATTRIBUTE_CLASS_COUNTS
from export_model import export_checkpoint
from models.complete_model import CompleteMALViT
from utils.checkpoint_manager import CheckpointManager
from utils.model_loading import load_model
from wbc_predictor import WBCPredictor


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def checkpoint(tmp_path):
    return CheckpointManager(tmp_path).save(CompleteMALViT(), filename="source.pth")


def test_predictor_accepts_file_pil_and_upload_bytes(checkpoint, tmp_path):
    image = Image.fromarray(np.random.default_rng(7).integers(0, 256, (120, 140, 3), dtype=np.uint8))
    path = tmp_path / "image.png"
    image.save(path)
    predictor = WBCPredictor(checkpoint, device="cpu")
    before = {name: tensor.clone() for name, tensor in predictor.model.state_dict().items()}
    results = [predictor.predict(path), predictor.predict(image), predictor.predict(path.read_bytes())]
    assert results[0] == results[1] == results[2]
    json.dumps(results[0])
    assert len(results[0]["attributes"]) == 11
    assert len(results[0]["wbc_probabilities"]) == 5
    assert sum(len(x) for x in results[0]["attribute_probabilities"].values()) == 31
    for name, tensor in predictor.model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_export_loads_without_original_checkpoint(checkpoint, tmp_path):
    original = load_model(checkpoint, device="cpu")
    exported = export_checkpoint(tmp_path / "export.pth", checkpoint)
    checkpoint.unlink()
    restored = load_model(exported, device="cpu")
    assert restored.checkpoint_source["mode"] == "exported"
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    inputs = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(original(inputs)["wbc_logits"], restored(inputs)["wbc_logits"],
                                   rtol=0, atol=0)
    saved = torch.load(exported, weights_only=True)
    assert "optimizer_state_dict" not in saved
    assert saved["attribute_names"] == ATTRIBUTE_NAMES


def test_export_never_overwrites_weights(checkpoint):
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        export_checkpoint(checkpoint, checkpoint)
    assert checkpoint.read_bytes() == before


def test_corrupt_image_is_not_classified(checkpoint):
    predictor = WBCPredictor(checkpoint, device="cpu")
    with pytest.raises(UnidentifiedImageError):
        predictor.predict(b"not an encoded image")
    with pytest.raises(TypeError):
        predictor.predict(None)


def test_non_finite_checkpoint_is_rejected(checkpoint):
    saved = torch.load(checkpoint, weights_only=True)
    saved["model_state_dict"]["position_embedding"][0, 0, 0] = float("nan")
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="non-finite"):
        load_model(checkpoint, device="cpu")


def test_attribute_metrics_follow_encoder_order():
    from utils.metrics import attribute_accuracy
    targets = torch.tensor([[i % ATTRIBUTE_CLASS_COUNTS[name] for i, name in enumerate(ATTRIBUTE_NAMES)]])
    predictions = {}
    for index in reversed(range(len(ATTRIBUTE_NAMES))):
        name = ATTRIBUTE_NAMES[index]
        predictions[name] = torch.nn.functional.one_hot(targets[:, index], ATTRIBUTE_CLASS_COUNTS[name]).float()
    assert attribute_accuracy(predictions, targets)["overall"] == 1.0


def test_explanations_leave_shared_parameter_gradients_untouched(monkeypatch):
    from utils.xai import vit_grad_cam as xai
    model = CompleteMALViT().eval()
    first = next(model.parameters())
    first.grad = torch.ones_like(first)
    monkeypatch.setattr(xai, "_MODEL", model)
    result = xai.generate_attribute_cam(Image.new("RGB", (224, 224), "pink"), "nucleus_shape")
    assert np.isfinite(result["heatmap"]).all()
    assert torch.equal(first.grad, torch.ones_like(first))
    assert all(p.grad is None for p in list(model.parameters())[1:])
