"""Exercise the Streamlit upload and analysis flow without a browser or dataset."""

from io import BytesIO
from pathlib import Path
import sys

from PIL import Image
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import torch

from models.complete_model import CompleteMALViT
from utils.checkpoint_manager import CheckpointManager


APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def app_environment(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(APP)])
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()
    torch.set_num_threads(previous)


def test_gui_waits_for_upload():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any("Upload a WBC" in message.value for message in app.info)


def test_gui_rejects_invalid_image(monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: BytesIO(b"invalid"))
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    assert any("could not be read" in message.value for message in app.error)


def test_gui_uses_selected_checkpoint_and_displays_all_explanations(tmp_path, monkeypatch):
    checkpoint = CheckpointManager(tmp_path).save(CompleteMALViT(), filename="gui.pth")
    monkeypatch.setattr(sys, "argv", [str(APP), "--checkpoint", str(checkpoint)])
    uploaded = BytesIO()
    Image.new("RGB", (128, 128), "pink").save(uploaded, format="PNG")
    uploaded.seek(0)
    uploaded.name = "cell.png"
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: uploaded)
    monkeypatch.setattr(st, "button", lambda *a, **kw: True)
    app = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not app.exception
    assert not app.error
    assert any("Analysis completed successfully" in message.value for message in app.success)
    assert any("WBC Type from the 11 Attributes" == header.value for header in app.header)
    assert len(app.get("imgs")) == 34  # Uploaded image plus three maps for each attribute.
    assert len(app.json) == 1
