"""
utils/xai/vit_grad_cam.py

MAL-ViT Attribute-Level Explainability
--------------------------------------

Generates attribute-specific spatial attribution maps for MAL-ViT.

The model contains:

    11 attribute tokens
    4 register tokens
    196 image patch tokens
    6 transformer blocks
    6 attention heads

Token layout:

    0  - 10   : attribute tokens
    11 - 14   : register tokens
    15 - 210  : image patch tokens

For each morphology attribute, this module:

    1. Loads the trained MAL-ViT checkpoint.
    2. Runs the input image through the model.
    3. Selects the predicted class for the attribute.
    4. Computes the gradient of that class logit.
    5. Extracts attribute-token -> image-patch attention.
    6. Produces an attention-only spatial map.
    7. Produces a gradient-only spatial map.
    8. Combines attention and gradient information.
    9. Produces a combined Grad-CAM-style map.
   10. Saves all three visualizations.

This is a ViT-specific attribution method rather than
conventional CNN Grad-CAM.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from config import (
    DEVICE,
    BEST_MODEL_PATH,
    IMAGE_SIZE,
    NUM_PATCHES,
    NUM_ATTRIBUTE_TOKENS,
    NUM_REGISTER_TOKENS,
    ATTRIBUTE_NAMES,
)

from utils.model_loading import load_model as load_pipeline_model
from data.transforms import test_transform
from inference import predict_details, load_rgb_image

from data.encoders import (
    decode_wbc,
    decode_attribute,
)


# ============================================================
# CONSTANTS
# ============================================================

PATCH_GRID_SIZE = int(
    NUM_PATCHES ** 0.5
)

ATTRIBUTE_TOKEN_START = 0

REGISTER_TOKEN_START = (
    NUM_ATTRIBUTE_TOKENS
)

PATCH_TOKEN_START = (
    NUM_ATTRIBUTE_TOKENS
    + NUM_REGISTER_TOKENS
)


# ============================================================
# MODEL LOADING
# ============================================================

_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_pipeline_model(device=DEVICE)
    return _MODEL


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """Use exactly the normalization and resize used for validation/inference."""
    return test_transform(image.convert("RGB")).unsqueeze(0)


# ============================================================
# HEATMAP NORMALIZATION
# ============================================================

def normalize_heatmap(
    heatmap,
):
    """
    Normalize heatmap to [0, 1].
    """

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    heatmap = np.maximum(
        heatmap,
        0.0,
    )

    minimum = heatmap.min()

    maximum = heatmap.max()

    if maximum - minimum < 1e-8:

        return np.zeros_like(
            heatmap
        )

    heatmap = (
        heatmap - minimum
    ) / (
        maximum - minimum
    )

    return heatmap


# ============================================================
# PATCH MAP
# ============================================================

def patches_to_map(
    patch_scores,
):
    """
    Convert 196 patch scores into a 14 x 14 map.

    Parameters
    ----------
    patch_scores : torch.Tensor or numpy.ndarray

    Returns
    -------
    numpy.ndarray
        Shape:
            (14, 14)
    """

    if isinstance(
        patch_scores,
        torch.Tensor,
    ):

        patch_scores = (
            patch_scores.detach()
            .cpu()
            .numpy()
        )

    patch_scores = np.asarray(
        patch_scores,
        dtype=np.float32,
    )

    expected_patches = (
        PATCH_GRID_SIZE
        * PATCH_GRID_SIZE
    )

    if patch_scores.size != expected_patches:

        raise ValueError(
            f"Expected "
            f"{expected_patches} patch scores, "
            f"but received "
            f"{patch_scores.size}."
        )

    patch_scores = patch_scores.reshape(
        PATCH_GRID_SIZE,
        PATCH_GRID_SIZE,
    )

    return normalize_heatmap(
        patch_scores
    )


# ============================================================
# HEATMAP RESIZING
# ============================================================

def resize_heatmap(
    heatmap,
    size,
):
    """
    Resize heatmap to image dimensions.

    Parameters
    ----------
    heatmap : numpy.ndarray
        2D heatmap.

    size : tuple
        (width, height)

    Returns
    -------
    numpy.ndarray
    """

    heatmap_tensor = torch.from_numpy(
        np.asarray(
            heatmap,
            dtype=np.float32,
        )
    ).float()

    heatmap_tensor = (
        heatmap_tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )

    heatmap_tensor = F.interpolate(
        heatmap_tensor,
        size=(
            size[1],
            size[0],
        ),
        mode="bilinear",
        align_corners=False,
    )

    heatmap = (
        heatmap_tensor
        .squeeze()
        .numpy()
    )

    return normalize_heatmap(
        heatmap
    )


# ============================================================
# SIMPLE HEATMAP COLORIZATION
# ============================================================

def colorize_heatmap(
    heatmap,
):
    """
    Convert normalized heatmap into RGB colors.

    Uses a simple blue-to-red visualization.

    Returns
    -------
    numpy.ndarray
        uint8 RGB image.
    """

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    red = (
        255.0
        * heatmap
    )

    blue = (
        255.0
        * (1.0 - heatmap)
    )

    green = (
        255.0
        * (
            1.0
            - np.abs(
                heatmap - 0.5
            )
            * 2.0
        )
    )

    rgb = np.stack(
        [
            red,
            green,
            blue,
        ],
        axis=-1,
    )

    return rgb.astype(
        np.uint8
    )


# ============================================================
# OVERLAY
# ============================================================

def create_overlay(
    image,
    heatmap,
    alpha=0.45,
):
    """
    Create heatmap overlay on original image.

    Parameters
    ----------
    image : PIL.Image.Image

    heatmap : numpy.ndarray

    alpha : float
        Heatmap blending factor.

    Returns
    -------
    PIL.Image.Image
    """

    image = image.convert(
        "RGB"
    )

    original = np.asarray(
        image,
        dtype=np.float32,
    )

    heatmap_rgb = colorize_heatmap(
        heatmap
    ).astype(
        np.float32
    )

    overlay = (
        (1.0 - alpha)
        * original
        +
        alpha
        * heatmap_rgb
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        overlay
    )


# ============================================================
# ATTRIBUTE ATTRIBUTION
# ============================================================

def generate_attribute_cam(
    image,
    attribute_name,
    model=None,
):
    """
    Generate attribute-specific XAI maps.

    Three attribution maps are produced:

        1. Attention-only
        2. Gradient-only
        3. Attention × Gradient

    Parameters
    ----------
    image : PIL.Image.Image

    attribute_name : str

        One of ATTRIBUTE_NAMES.

    Returns
    -------
    dict
        Contains prediction, logits, heatmaps,
        attention maps, gradient maps and overlays.
    """

    if attribute_name not in ATTRIBUTE_NAMES:

        raise ValueError(
            f"Unknown attribute: "
            f"{attribute_name}. "
            f"Expected one of "
            f"{ATTRIBUTE_NAMES}"
        )

    model = load_model() if model is None else model

    device = next(model.parameters()).device

    input_tensor = preprocess_image(
        image
    ).to(device)

    # --------------------------------------------------------
    # Enable gradient
    # --------------------------------------------------------

    input_tensor.requires_grad_()

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    outputs = model.extract_attributes(
        input_tensor,
        return_attention=True,
    )

    # --------------------------------------------------------
    # Attribute logits
    # --------------------------------------------------------

    attribute_logits = outputs[
        "attribute_predictions"
    ][
        attribute_name
    ]

    # --------------------------------------------------------
    # Predicted class
    # --------------------------------------------------------

    predicted_index = int(
        torch.argmax(
            attribute_logits,
            dim=1,
        ).item()
    )

    predicted_logit = (
        attribute_logits[
            0,
            predicted_index,
        ]
    )

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    input_gradient, = torch.autograd.grad(predicted_logit, input_tensor)

    # --------------------------------------------------------
    # Transformer attention
    # --------------------------------------------------------

    attentions = outputs.get(
        "attention"
    )

    if attentions is None:

        raise RuntimeError(
            "Model did not return "
            "attention matrices."
        )

    if len(attentions) == 0:

        raise RuntimeError(
            "No transformer attention "
            "matrices were returned."
        )

    # --------------------------------------------------------
    # Attribute token index
    # --------------------------------------------------------

    attribute_index = (
        ATTRIBUTE_NAMES.index(
            attribute_name
        )
    )

    attribute_token_index = (
        ATTRIBUTE_TOKEN_START
        + attribute_index
    )

    # --------------------------------------------------------
    # Aggregate attention across blocks
    # --------------------------------------------------------

    attention_maps = []

    for attention in attentions:

        # Expected:
        #
        # (B, H, N, N)

        if attention.ndim != 4:

            raise RuntimeError(
                "Unexpected attention shape: "
                f"{tuple(attention.shape)}"
            )

        current = attention[
            0
        ]

        # ----------------------------------------------------
        # Average attention heads
        # ----------------------------------------------------

        current = current.mean(
            dim=0
        )

        # ----------------------------------------------------
        # Attribute token -> patch tokens
        # ----------------------------------------------------

        current = current[
            attribute_token_index,
            PATCH_TOKEN_START:
            PATCH_TOKEN_START
            + NUM_PATCHES,
        ]

        attention_maps.append(
            current
        )

    # --------------------------------------------------------
    # Average transformer blocks
    # --------------------------------------------------------

    attention_map = torch.stack(
        attention_maps,
        dim=0,
    ).mean(
        dim=0
    )

    attention_map = torch.relu(
        attention_map
    )

    # --------------------------------------------------------
    # Attention spatial map
    # --------------------------------------------------------

    attention_spatial = patches_to_map(
        attention_map
    )

    # --------------------------------------------------------
    # Gradient information
    # --------------------------------------------------------


    if input_gradient is None:

        raise RuntimeError(
            "Input gradient was not generated."
        )

    # --------------------------------------------------------
    # Gradient magnitude
    # --------------------------------------------------------

    gradient_strength = (
        input_gradient
        .abs()
        .mean(
            dim=1,
        )
    )

    # --------------------------------------------------------
    # Resize gradient to patch grid
    # --------------------------------------------------------

    gradient_strength = F.interpolate(
        gradient_strength.unsqueeze(1),
        size=(
            PATCH_GRID_SIZE,
            PATCH_GRID_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
    )

    gradient_strength = (
        gradient_strength
        .squeeze()
        .detach()
        .cpu()
    )

    gradient_spatial = normalize_heatmap(
        gradient_strength.numpy()
    )

    # --------------------------------------------------------
    # Combine attention and gradient
    # --------------------------------------------------------

    combined = (
        attention_spatial
        *
        gradient_spatial
    )

    combined = normalize_heatmap(
        combined
    )

    # --------------------------------------------------------
    # Resize all maps to original image
    # --------------------------------------------------------

    original_size = image.size

    attention_heatmap = resize_heatmap(
        attention_spatial,
        original_size,
    )

    gradient_heatmap = resize_heatmap(
        gradient_spatial,
        original_size,
    )

    combined_heatmap = resize_heatmap(
        combined,
        original_size,
    )

    # --------------------------------------------------------
    # Create overlays
    # --------------------------------------------------------

    attention_overlay = create_overlay(
        image,
        attention_heatmap,
        alpha=0.45,
    )

    gradient_overlay = create_overlay(
        image,
        gradient_heatmap,
        alpha=0.45,
    )

    combined_overlay = create_overlay(
        image,
        combined_heatmap,
        alpha=0.45,
    )

    # --------------------------------------------------------
    # Decode prediction
    # --------------------------------------------------------

    predicted_class = decode_attribute(
        attribute_name,
        predicted_index,
    )

    # --------------------------------------------------------
    # Logits
    # --------------------------------------------------------

    logits = (
        attribute_logits[
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return {

        "attribute":
            attribute_name,

        "predicted_class_index":
            predicted_index,

        "predicted_class":
            predicted_class,

        "logits":
            logits,

        # Combined map
        "heatmap":
            combined_heatmap,

        "overlay":
            combined_overlay,

        # Attention
        "attention_map":
            attention_spatial,

        "attention_heatmap":
            attention_heatmap,

        "attention_overlay":
            attention_overlay,

        # Gradient
        "gradient_map":
            gradient_spatial,

        "gradient_heatmap":
            gradient_heatmap,

        "gradient_overlay":
            gradient_overlay,

        # Combined
        "combined_heatmap":
            combined_heatmap,

        "combined_overlay":
            combined_overlay,
    }


# ============================================================
# ALL ATTRIBUTE CAMS
# ============================================================

def generate_all_attribute_cams(image, model=None):
    image = load_rgb_image(image)
    model = load_model() if model is None else model
    results = {}
    for attribute_name in ATTRIBUTE_NAMES:
        print(f"Generating attribute explanation: {attribute_name}")
        results[attribute_name] = generate_attribute_cam(image, attribute_name, model=model)
    return results


# ============================================================
# COMPLETE PREDICTION + EXPLANATION
# ============================================================

def generate_explanations(image, model=None):
    image = load_rgb_image(image)
    model = load_model() if model is None else model
    results = predict_details(image, model=model)
    results["cams"] = generate_all_attribute_cams(image, model=model)
    return results


# ============================================================
# QUICK TEST
# ============================================================

def main():

    print("=" * 70)

    print(
        "MAL-ViT ATTRIBUTE XAI TEST"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\nConfiguration")

    print(
        f"Device              : {DEVICE}"
    )

    print(
        f"Image size          : {IMAGE_SIZE}"
    )

    print(
        f"Patch tokens        : {NUM_PATCHES}"
    )

    print(
        f"Patch grid          : "
        f"{PATCH_GRID_SIZE} x "
        f"{PATCH_GRID_SIZE}"
    )

    print(
        f"Attribute tokens    : "
        f"{NUM_ATTRIBUTE_TOKENS}"
    )

    print(
        f"Register tokens     : "
        f"{NUM_REGISTER_TOKENS}"
    )

    print(
        f"Patch token start   : "
        f"{PATCH_TOKEN_START}"
    )

    print(
        f"Attributes          : "
        f"{len(ATTRIBUTE_NAMES)}"
    )

    # --------------------------------------------------------
    # Find sample image
    # --------------------------------------------------------

    image_folder = Path(
        "images"
    )

    if not image_folder.exists():

        raise FileNotFoundError(
            "images/ folder not found."
        )

    image_files = sorted(
        [
            p
            for p in image_folder.iterdir()
            if p.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png",
            ]
        ]
    )

    if not image_files:

        raise FileNotFoundError(
            "No sample images found "
            "inside images/."
        )

    image_path = image_files[0]

    print(
        f"\nSample image: "
        f"{image_path}"
    )

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Generate explanations
    # --------------------------------------------------------

    print(
        "\nGenerating attribute explanations..."
    )

    results = generate_all_attribute_cams(
        image
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "ATTRIBUTE PREDICTIONS"
    )

    print(
        "-" * 70
    )

    for attribute_name, result in results.items():

        print(
            f"{attribute_name:35s}: "
            f"{result['predicted_class']}"
        )

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    output_folder = (
        Path("outputs")
        / "xai"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save visualizations
    # --------------------------------------------------------

    print(
        "\nSaving XAI visualizations..."
    )

    for attribute_name, result in results.items():

        safe_name = (
            attribute_name
            .replace(
                " ",
                "_",
            )
        )

        # ----------------------------------------------------
        # Attention-only
        # ----------------------------------------------------

        attention_path = (
            output_folder
            / f"{safe_name}_attention.png"
        )

        result[
            "attention_overlay"
        ].save(
            attention_path
        )

        print(
            f"Saved: {attention_path}"
        )

        # ----------------------------------------------------
        # Gradient-only
        # ----------------------------------------------------

        gradient_path = (
            output_folder
            / f"{safe_name}_gradient.png"
        )

        result[
            "gradient_overlay"
        ].save(
            gradient_path
        )

        print(
            f"Saved: {gradient_path}"
        )

        # ----------------------------------------------------
        # Combined
        # ----------------------------------------------------

        combined_path = (
            output_folder
            / f"{safe_name}_gradcam.png"
        )

        result[
            "combined_overlay"
        ].save(
            combined_path
        )

        print(
            f"Saved: {combined_path}"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert len(
        results
    ) == len(
        ATTRIBUTE_NAMES
    )

    for attribute_name in ATTRIBUTE_NAMES:

        assert (
            attribute_name
            in results
        )

        result = results[
            attribute_name
        ]

        # ----------------------------------------------------
        # Combined heatmap
        # ----------------------------------------------------

        assert (
            result["heatmap"].ndim
            == 2
        )

        assert (
            result["heatmap"].shape
            == (
                image.height,
                image.width,
            )
        )

        assert (
            0.0
            <= result["heatmap"].min()
            <= 1.0
        )

        assert (
            0.0
            <= result["heatmap"].max()
            <= 1.0
        )

        # ----------------------------------------------------
        # Attention heatmap
        # ----------------------------------------------------

        assert (
            result[
                "attention_heatmap"
            ].ndim
            == 2
        )

        assert (
            result[
                "attention_heatmap"
            ].shape
            == (
                image.height,
                image.width,
            )
        )

        assert (
            0.0
            <= result[
                "attention_heatmap"
            ].min()
            <= 1.0
        )

        assert (
            0.0
            <= result[
                "attention_heatmap"
            ].max()
            <= 1.0
        )

        # ----------------------------------------------------
        # Gradient heatmap
        # ----------------------------------------------------

        assert (
            result[
                "gradient_heatmap"
            ].ndim
            == 2
        )

        assert (
            result[
                "gradient_heatmap"
            ].shape
            == (
                image.height,
                image.width,
            )
        )

        assert (
            0.0
            <= result[
                "gradient_heatmap"
            ].min()
            <= 1.0
        )

        assert (
            0.0
            <= result[
                "gradient_heatmap"
            ].max()
            <= 1.0
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ATTRIBUTE XAI VALIDATION: PASSED"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()