"""
vit_grad_cam.py

Grad-CAM style explainability for MAL-ViT.

Generates one CAM for each morphology attribute.

Pipeline
--------
Image
    ↓
MAL-ViT
    ↓
Transformer Tokens
    ↓
Target Attribute Logit
    ↓
Gradients
    ↓
Patch Importance
    ↓
14 × 14 CAM
    ↓
224 × 224 Heatmap

No changes are required to the trained model architecture.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


import sys
from pathlib import Path
# Adds the project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

from data.transforms import test_transform

from data.encoders import (
    ATTRIBUTE_NAMES,
)

from models.complete_model import ExplainableWBCModel


# ==========================================================
# Configuration
# ==========================================================
ATTRIBUTE_COUNT = len(
    ATTRIBUTE_NAMES
)

IMAGE_SIZE = 224
PATCH_SIZE = 16

NUM_PATCHES_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = NUM_PATCHES_SIDE * NUM_PATCHES_SIDE


# ==========================================================
# Grad-CAM Class
# ==========================================================

class ViTGradCAM:

    def __init__(self, model):

        self.model = model

        self.activations = None
        self.gradients = None

        # --------------------------------------------------
        # Target Transformer Block
        # --------------------------------------------------
        #
        # We use the final Transformer encoder block.
        #
        # Its output contains:
        #
        # 11 attribute tokens
        # +
        # 196 image patch tokens
        #
        # Shape:
        # (B, 207, 192)
        #
        # --------------------------------------------------

        self.target_layer = (
            self.model
            .mal_vit
            .transformer
            .blocks[-1]
        )

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    # ======================================================
    # Forward Hook
    # ======================================================

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

        # We need gradients for this intermediate
        # tensor during backward.
        #
        # PyTorch normally does not retain gradients
        # for non-leaf tensors unless requested.

        if output.requires_grad:

            output.retain_grad()

    # ======================================================
    # Generate CAM
    # ======================================================

    def generate(
        self,
        image,
        attribute_name,
        target_class=None,
    ):

        self.model.zero_grad(set_to_none=True)

        self.activations = None

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------

        outputs = self.model(image)

        logits = outputs[
            "attribute_predictions"
        ][attribute_name]

        # --------------------------------------------------
        # Select Target Class
        # --------------------------------------------------

        if target_class is None:

            target_class = logits.argmax(
                dim=1
            ).item()

        target = logits[
            0,
            target_class
        ]

        # --------------------------------------------------
        # Backward
        # --------------------------------------------------

        target.backward()

        # --------------------------------------------------
        # Check Activations
        # --------------------------------------------------

        if self.activations is None:

            raise RuntimeError(
                "Transformer activations were not captured."
            )

        if self.activations.grad is None:

            raise RuntimeError(
                "Transformer gradients were not captured."
            )

        activations = self.activations[
            0
        ]

        gradients = self.activations.grad[
            0
        ]

        # --------------------------------------------------
        # Remove Attribute Tokens
        # --------------------------------------------------
        #
        # First 11 tokens:
        #
        # attribute tokens
        #
        # Remaining 196:
        #
        # image patch tokens
        #
        # --------------------------------------------------

        patch_activations = activations[
            ATTRIBUTE_COUNT:
        ]

        patch_gradients = gradients[
            ATTRIBUTE_COUNT:
        ]

        # --------------------------------------------------
        # Grad-CAM Weight
        # --------------------------------------------------
        #
        # Average gradient over embedding dimension.
        #
        # Shape:
        #
        # (196, 192)
        #       ↓ mean
        # (196,)
        #
        # Each patch receives an importance weight.
        #
        # --------------------------------------------------

        weights = patch_gradients.mean(
            dim=1
        )

        # --------------------------------------------------
        # Weighted Patch Activation
        # --------------------------------------------------

        cam = (
            patch_activations
            * weights.unsqueeze(-1)
        ).sum(dim=-1)

        # --------------------------------------------------
        # ReLU
        # --------------------------------------------------

        cam = F.relu(cam)

        # --------------------------------------------------
        # Convert 196 patches → 14 × 14
        # --------------------------------------------------

        cam = cam.reshape(
            NUM_PATCHES_SIDE,
            NUM_PATCHES_SIDE,
        )

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        # --------------------------------------------------
        # Resize to Image Size
        # --------------------------------------------------

        cam = cv2.resize(
            cam,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        # --------------------------------------------------
        # Final Normalization
        # --------------------------------------------------

        cam = np.clip(
            cam,
            0,
            1,
        )

        return cam, target_class

    # ======================================================
    # Cleanup
    # ======================================================

    def remove_hooks(self):

        self.forward_handle.remove()


# ==========================================================
# Number of Attribute Tokens
# ==========================================================



# ==========================================================
# Load Model
# ==========================================================

def load_model():

    model = (
        ExplainableWBCModel()
        .to(DEVICE)
    )

    checkpoint = torch.load(
        CHECKPOINT_DIR / BEST_MODEL_NAME,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model


# ==========================================================
# Prepare Image
# ==========================================================

def prepare_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    original = np.array(
        image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )
    )

    tensor = test_transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    return tensor, original


# ==========================================================
# Create Heatmap
# ==========================================================

def create_heatmap(
    original_image,
    cam,
    alpha=0.45,
):

    heatmap = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = (
        original_image.astype(
            np.float32
        )
        * (1 - alpha)
        +
        heatmap.astype(
            np.float32
        )
        * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    return overlay


# ==========================================================
# Generate All Attribute CAMs
# ==========================================================

def generate_all_attribute_cams(
    image_path
):

    model = load_model()

    image_tensor, original_image = (
        prepare_image(image_path)
    )

    grad_cam = ViTGradCAM(
        model
    )

    results = {}

    try:

        for attribute_name in ATTRIBUTE_NAMES:

            cam, predicted_class = (
                grad_cam.generate(
                    image_tensor,
                    attribute_name,
                )
            )

            overlay = create_heatmap(
                original_image,
                cam,
            )

            results[
                attribute_name
            ] = {
                "cam": cam,
                "overlay": overlay,
                "class_index": predicted_class,
            }

    finally:

        grad_cam.remove_hooks()

    return results


# ==========================================================
# Quick Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MAL-ViT Attribute Grad-CAM Test")
    print("=" * 70)

    image_path = input(
        "\nEnter image path: "
    ).strip()

    image_path = Path(
        image_path
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    results = generate_all_attribute_cams(
        image_path
    )

    print("\nGenerated Grad-CAMs:")

    for attribute_name in results:

        print(
            f"  ✓ {attribute_name}"
        )

    print(
        "\nTotal CAMs:",
        len(results),
    )

    print("\nDone.")