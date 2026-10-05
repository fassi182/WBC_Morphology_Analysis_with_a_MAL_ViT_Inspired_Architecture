"""Regression tests for the image -> attributes -> WBC contract."""

import importlib

import numpy as np
import pytest
import torch
from PIL import Image

from config import BEST_MODEL_NAME, LAST_CHECKPOINT_NAME
from data.encoders import ATTRIBUTE_NAMES, ATTRIBUTE_CLASS_COUNTS
from data.transforms import test_transform as image_transform
from models.complete_model import CompleteMALViT
from models.attribute_wbc_classifier import AttributeWBCClassifier, attributes_to_one_hot
from training.losses import compute_total_loss
from training.runner import TrainingRunner
from training.test import evaluate_test_set
from utils.checkpoint_manager import CheckpointManager
from utils.model_loading import load_model


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def batch(size=2):
    return {
        "images": torch.randn(size, 3, 224, 224),
        "labels": torch.arange(size) % 5,
        "attributes": torch.stack([
            torch.arange(size) % ATTRIBUTE_CLASS_COUNTS[name] for name in ATTRIBUTE_NAMES
        ], dim=1),
    }


def test_head_receives_only_normalized_attribute_predictions():
    model = CompleteMALViT().eval()
    inputs = []
    hook = model.wbc_classifier.register_forward_pre_hook(lambda _, args: inputs.append(args[0]))
    outputs = model(batch()["images"], return_attention=True)
    hook.remove()
    expected = torch.cat([
        outputs["attribute_predictions"][name].softmax(-1) for name in ATTRIBUTE_NAMES
    ], dim=1)
    torch.testing.assert_close(inputs[0], expected)
    assert inputs[0].shape == (2, 31)
    assert outputs["tokens"].shape == (2, 211, 192)
    assert len(outputs["attention"]) == 6
    assert outputs["wbc_logits"].shape == (2, 5)
    for distribution in inputs[0].split(list(ATTRIBUTE_CLASS_COUNTS.values()), dim=1):
        torch.testing.assert_close(distribution.sum(1), torch.ones(2))


def test_fixed_attributes_make_wbc_independent_of_image(monkeypatch):
    model = CompleteMALViT().eval()
    predictions = {name: torch.randn(2, ATTRIBUTE_CLASS_COUNTS[name]) for name in ATTRIBUTE_NAMES}
    monkeypatch.setattr(model, "extract_attributes", lambda images, **_: {
        "attribute_predictions": predictions,
        "patch_features": images.mean(),  # Deliberately changes between images.
    })
    with torch.no_grad():
        a = model(torch.zeros(2, 3, 224, 224))
        b = model(torch.ones(2, 3, 224, 224))
    torch.testing.assert_close(a["wbc_logits"], b["wbc_logits"])


def test_wbc_loss_backpropagates_through_all_attributes():
    model = CompleteMALViT().eval()
    outputs = model(batch()["images"])
    for logits in outputs["attribute_predictions"].values():
        logits.retain_grad()
    torch.nn.functional.cross_entropy(outputs["wbc_logits"], torch.tensor([0, 1])).backward()
    for logits in outputs["attribute_predictions"].values():
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum() > 0
    assert model.patch_embedding.projection.weight.grad.abs().sum() > 0


def test_categorical_intervention_uses_the_same_head():
    model = CompleteMALViT().eval()
    standalone = AttributeWBCClassifier().eval()
    standalone.load_state_dict(model.wbc_classifier.state_dict())
    attributes = batch()["attributes"]
    with torch.no_grad():
        expected = standalone(attributes)
        actual = model.classify_approved_attributes(attributes)["wbc_logits"]
    torch.testing.assert_close(actual, expected)
    assert attributes_to_one_hot(attributes).sum().item() == 22


@pytest.mark.parametrize("attributes", [torch.zeros(2, 11), torch.zeros(2, 10, dtype=torch.long),
                                         torch.full((2, 11), -1, dtype=torch.long)])
def test_invalid_categorical_attributes_are_rejected(attributes):
    with pytest.raises(ValueError):
        attributes_to_one_hot(attributes)


def test_checkpoint_roundtrip_and_label_validation(tmp_path):
    model = CompleteMALViT().eval()
    path = CheckpointManager(tmp_path).save(model, epoch=1)
    restored = load_model(path, device="cpu")
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name])
    raw_path = tmp_path / "raw.pth"
    torch.save(model.state_dict(), raw_path)
    load_model(raw_path, device="cpu")
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["attribute_names"] = list(reversed(ATTRIBUTE_NAMES))
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="attribute_names"):
        load_model(path, device="cpu")


def legacy_files(tmp_path):
    model = CompleteMALViT()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("wbc_classifier.")}
    state["wbc_classifier.classifier.0.weight"] = torch.randn(192)
    legacy = tmp_path / "legacy.pth"
    head_path = tmp_path / "head.pth"
    torch.save({"model_state_dict": state}, legacy)
    head = AttributeWBCClassifier()
    torch.save({"model_state_dict": head.state_dict()}, head_path)
    return legacy, head_path, model, head


def test_legacy_composition_replaces_the_old_head(tmp_path):
    legacy, head_path, original, head = legacy_files(tmp_path)
    with pytest.warns(UserWarning, match="old image-feature WBC head is excluded"):
        composed = load_model(legacy, device="cpu", attribute_checkpoint_path=head_path)
    for key, value in head.state_dict().items():
        torch.testing.assert_close(composed.wbc_classifier.state_dict()[key], value)
    torch.testing.assert_close(composed.position_embedding, original.position_embedding)
    assert composed.checkpoint_source["mode"] == "composed"


def test_legacy_composition_requires_a_complete_trained_head(tmp_path):
    legacy, head_path, _, _ = legacy_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_model(legacy, device="cpu", attribute_checkpoint_path=tmp_path / "missing.pth")
    torch.save({"network.0.weight": torch.zeros(1)}, head_path)
    with pytest.raises(RuntimeError):
        load_model(legacy, device="cpu", attribute_checkpoint_path=head_path)


def test_legacy_head_label_order_must_match(tmp_path):
    legacy, head_path, _, _ = legacy_files(tmp_path)
    checkpoint = torch.load(head_path, weights_only=True)
    checkpoint["attribute_names"] = list(reversed(ATTRIBUTE_NAMES))
    torch.save(checkpoint, head_path)
    with pytest.raises(ValueError, match="Attribute checkpoint attribute_names"):
        load_model(legacy, device="cpu", attribute_checkpoint_path=head_path)


@pytest.mark.parametrize("key", ["attribute_names", "attribute_encoders", "cell_labels"])
def test_legacy_image_label_metadata_must_match(tmp_path, key):
    legacy, head_path, _, _ = legacy_files(tmp_path)
    checkpoint = torch.load(legacy, weights_only=True)
    checkpoint[key] = ["wrong_order"] if key == "attribute_names" else {"wrong_label": 0}
    torch.save(checkpoint, legacy)
    with pytest.raises(ValueError, match=f"Legacy image checkpoint {key}"):
        load_model(legacy, device="cpu", attribute_checkpoint_path=head_path)


def test_default_loader_prefers_new_checkpoint_and_does_not_hide_corruption(tmp_path, monkeypatch):
    from utils import model_loading
    current = tmp_path / "current.pth"
    legacy, head_path, _, _ = legacy_files(tmp_path)
    monkeypatch.setattr(model_loading, "BEST_MODEL_PATH", current)
    monkeypatch.setattr(model_loading, "LEGACY_MODEL_PATH", legacy)
    monkeypatch.setattr(model_loading, "ATTRIBUTE_MODEL_PATH", head_path)
    with pytest.warns(UserWarning):
        assert load_model(device="cpu").checkpoint_source["mode"] == "composed"
    torch.save({"model_state_dict": CompleteMALViT().state_dict()}, current)
    assert load_model(device="cpu").checkpoint_source["mode"] == "native"
    torch.save({"model_state_dict": {"bad": torch.zeros(1)}}, current)
    with pytest.raises(RuntimeError, match="incompatible"):
        load_model(device="cpu")


def test_train_evaluate_and_reload_use_same_contract(tmp_path):
    model = CompleteMALViT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batches = [batch(2), batch(1)]
    runner = TrainingRunner(model, batches, batches, optimizer, compute_total_loss, "cpu",
                            checkpoint_manager=CheckpointManager(tmp_path))
    history = runner.fit(1)
    assert len(history) == 1
    assert (tmp_path / LAST_CHECKPOINT_NAME).is_file()
    restored = load_model(tmp_path / BEST_MODEL_NAME, device="cpu")
    results = evaluate_test_set(restored, batches, compute_total_loss, "cpu")
    assert len(results["predictions"]) == 3
    assert set(results["attribute_metrics"]) == set(ATTRIBUTE_NAMES)
    assert np.isfinite(results["total_loss"])


def test_inference_and_xai_share_preprocessing_and_predictions(monkeypatch):
    from inference import predict_details
    from utils.xai import vit_grad_cam as xai
    model = CompleteMALViT().eval()
    monkeypatch.setattr(xai, "_MODEL", model)
    image = Image.fromarray(np.random.default_rng(0).integers(0, 256, (240, 300, 3), dtype=np.uint8))
    torch.testing.assert_close(xai.preprocess_image(image)[0], image_transform(image), rtol=0, atol=0)
    prediction = predict_details(image, model)
    cam = xai.generate_attribute_cam(image, "cell_size")
    assert cam["predicted_class"] == prediction["attributes"]["cell_size"]
    assert cam["heatmap"].shape == (240, 300)
    assert np.isfinite(cam["heatmap"]).all()
    assert len(prediction["attribute_probabilities"]) == 11


def test_attribute_evaluator_accepts_current_batches(monkeypatch):
    import evaluate_attributes
    from evaluate_attributes import collect_predictions
    monkeypatch.setattr(evaluate_attributes, "DEVICE", "cuda")
    model = CompleteMALViT().train()
    true, predicted = collect_predictions(model, [batch()])
    assert not model.training
    assert set(true) == set(ATTRIBUTE_NAMES)
    assert all(len(values) == 2 for values in predicted.values())


@pytest.mark.parametrize("module", ["train", "test", "inference", "evaluate_attributes",
                                    "training.train_one_epoch", "training.validate"])
def test_entry_points_import_without_loading_weights(module):
    importlib.import_module(module)
