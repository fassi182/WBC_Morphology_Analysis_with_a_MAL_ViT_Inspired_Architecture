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
Transformer Attention (attribute token → patch tokens)
    ↓
Target Attribute Logit
    ↓
Gradients w.r.t. Attention Weights
    ↓
Patch Importance
    ↓
14 × 14 CAM
    ↓
224 × 224 Heatmap

No changes are required to the trained model architecture.

--------------------------------------------------------------------
WHY THE OLD VERSION PRODUCED THE SAME MAP FOR EVERY ATTRIBUTE
--------------------------------------------------------------------

The previous implementation hooked the *output* of
`transformer.blocks[-1]` (the last encoder block) and did classic
Grad-CAM on it:

    weight  = mean_over_channels( d(logit) / d(block_output) )
    cam     = relu( sum_over_channels( block_output * weight ) )

That output tensor has shape (B, 207, 192): 11 attribute-token rows
followed by 196 patch-token rows.

Look at what happens *after* that block in `VisionTransformer.forward`:

    for block in self.blocks:
        x = block(x)
    x = self.norm(x)          # <-- final LayerNorm

`nn.LayerNorm` normalizes each token independently over the embedding
dimension — it never mixes information *across* tokens. And in
`MALViT.forward`, only the first 11 rows of the normalized output are
ever used:

    attribute_features = tokens[:, :NUM_ATTRIBUTE_TOKENS, :]
    attribute_predictions = self.attribute_heads(attribute_features)

The 196 patch-token rows of `blocks[-1]`'s output are therefore
**never consumed by anything downstream of that block**. There is
simply no path in the computational graph from those patch-token
outputs to any attribute logit, so:

    d(logit) / d(blocks[-1].output[patch_rows])  ==  0.0   (exactly)

for *every* attribute, every class, every image. This was verified
directly on this repo's model:

    grad1 patch abs max/mean: 0.0 0.0
    grad2 patch abs max/mean: 0.0 0.0

That's why every attribute produced an (almost) identical, blob-like,
un-differentiated heatmap — the "CAM" was really just noise from the
zero/near-zero floor after normalization, not a real signal.

--------------------------------------------------------------------
THE FIX
--------------------------------------------------------------------

MAL-ViT doesn't have a convolutional feature map to run classic
Grad-CAM on — it has attention. The thing that actually tells you
"where did attribute token i look when producing its prediction" is
the **self-attention matrix**: specifically, row `i` (the attribute
token's query) against the 196 patch-token columns (keys), in the
block(s) where that attention is computed.

Each attribute token has its *own* learned query vector, so its
attention row over the patches is naturally different from every
other attribute token's row — this is exactly the per-attribute
localization signal we want.

To make it class/attribute-*discriminative* (not just "where does
this token attend on average", which attention alone doesn't
capture), we weight the attention weights by their gradient w.r.t.
the target attribute logit — the standard "Gradient × Attention"
idea used in transformer explainability methods:

    attention  = softmax(QK^T / sqrt(d))      # captured pre-dropout
    grad       = d(logit) / d(attention)
    relevance  = relu(attention * grad)       # per head
    cam        = mean over heads, over the last K blocks

This is captured with a forward hook on `attention.attention_dropout`
in each target block — its *input* is exactly the post-softmax,
pre-dropout attention matrix, so no changes to the model's source
files are needed, and gradients can be retained on it because it is
still a live intermediate tensor in the graph.
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

# Number of final transformer blocks whose attention maps are
# combined to build each attribute's CAM. Using more than one
# smooths out single-layer noise while still keeping the map
# attribute-specific (each layer still gets its own gradient
# w.r.t. that attribute's logit).
NUM_CAM_LAYERS = 4


# ==========================================================
# Grad-CAM Class
# ==========================================================

class ViTGradCAM:

    def __init__(self, model, num_layers=NUM_CAM_LAYERS):

        self.model = model

        # --------------------------------------------------
        # Target Blocks
        # --------------------------------------------------
        #
        # Instead of the *output* of the last encoder block
        # (which, as explained above, has zero gradient on the
        # patch-token rows), we hook the *attention weights*
        # inside the last `num_layers` encoder blocks.
        #
        # Each block's self-attention tensor has shape:
        #
        # (B, heads, 207, 207)
        #
        # Row i, columns [11:] give us: "how much did attribute
        # token i attend to each of the 196 image patches".
        #
        # --------------------------------------------------

        all_blocks = self.model.mal_vit.transformer.blocks

        num_layers = min(
            num_layers,
            len(all_blocks),
        )

        self.target_blocks = list(
            all_blocks[-num_layers:]
        )

        self.attention_maps = [None] * len(self.target_blocks)

        self.forward_handles = []

        for idx, block in enumerate(self.target_blocks):

            handle = (
                block
                .attention
                .attention_dropout
                .register_forward_hook(
                    self._make_forward_hook(idx)
                )
            )

            self.forward_handles.append(handle)

    # ======================================================
    # Forward Hook Factory
    # ======================================================
    #
    # `attention_dropout` is applied directly to the softmax
    # attention weights, before the weighted sum with V. Its
    # *input* (not output) is therefore exactly the attention
    # matrix we want, still connected to the autograd graph.
    #
    # ======================================================

    def _make_forward_hook(self, idx):

        def hook(module, inputs, output):

            attention_weights = inputs[0]

            if attention_weights.requires_grad:

                attention_weights.retain_grad()

            self.attention_maps[idx] = attention_weights

        return hook

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

        self.attention_maps = [None] * len(self.target_blocks)

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
        # Check Attention Maps
        # --------------------------------------------------

        for attention_weights in self.attention_maps:

            if attention_weights is None:

                raise RuntimeError(
                    "Attention weights were not captured."
                )

            if attention_weights.grad is None:

                raise RuntimeError(
                    "Attention gradients were not captured."
                )

        # --------------------------------------------------
        # Attribute Token Row Index
        # --------------------------------------------------
        #
        # Every attribute has its own dedicated token, at a
        # fixed position among the first ATTRIBUTE_COUNT
        # tokens. We look at *that* token's attention row.
        #
        # --------------------------------------------------

        token_index = ATTRIBUTE_NAMES.index(
            attribute_name
        )

        # --------------------------------------------------
        # Gradient-Weighted Attention, Per Layer
        # --------------------------------------------------
        #
        # For each of the last `num_layers` blocks:
        #
        # attn  : (heads, 207, 207)  -> post-softmax weights
        # grad  : (heads, 207, 207)  -> d(logit) / d(attn)
        #
        # We take row `token_index` (the attribute token's
        # query), restricted to patch-token columns, and
        # combine attention with its gradient -- this is what
        # makes the map specific to *this* attribute's logit,
        # not just "where this token generically looks".
        #
        # --------------------------------------------------

        layer_cams = []

        for attention_weights in self.attention_maps:

            attn = attention_weights[0]

            grad = attention_weights.grad[0]

            attn_row = attn[
                :,
                token_index,
                ATTRIBUTE_COUNT:,
            ]

            grad_row = grad[
                :,
                token_index,
                ATTRIBUTE_COUNT:,
            ]

            relevance = F.relu(
                attn_row * grad_row
            )

            # Average over attention heads
            layer_cam = relevance.mean(dim=0)

            layer_cams.append(layer_cam)

        # --------------------------------------------------
        # Combine Layers
        # --------------------------------------------------

        cam = torch.stack(
            layer_cams,
            dim=0,
        ).mean(dim=0)

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

        for handle in self.forward_handles:

            handle.remove()


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

            # A fresh forward pass is required for every
            # attribute: backward() consumes the graph, and
            # each attribute needs its own gradients w.r.t.
            # the attention weights.

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
'''

{
  "scripts_executed": [
    "f:/projects/WBC Morphology Analysis using MAL-ViT/check_attention_sink.py",
    "f:/projects/WBC Morphology Analysis using MAL-ViT/evaluate_attributes.py"
  ],
  "attention_sink_diagnostic": {
    "tested_images": "6 images from 'images'",
    "most_frequently_winning_patches": [
      {
        "patch": 106,
        "grid_position": "row=7, col=8",
        "wins": "14/66 times (21.2%)"
      },
      {
        "patch": 90,
        "grid_position": "row=6, col=6",
        "wins": "11/66 times (16.7%)"
      },
      {
        "patch": 117,
        "grid_position": "row=8, col=5",
        "wins": "8/66 times (12.1%)"
      },
      {
        "patch": 131,
        "grid_position": "row=9, col=5",
        "wins": "8/66 times (12.1%)"
      },
      {
        "patch": 195,
        "grid_position": "row=13, col=13",
        "wins": "8/66 times (12.1%)"
      }
    ],
    "winning_patches_note": "No single patch dominates disproportionately – winning patches look content-dependent, not sink-like.",
    "highest_average_incoming_attention": [
      {
        "patch": 93,
        "grid_position": "row=6, col=9",
        "avg_attention": 0.03119,
        "z_score": "+5.3"
      },
      {
        "patch": 131,
        "grid_position": "row=9, col=5",
        "avg_attention": 0.03029,
        "z_score": "+5.2"
      },
      {
        "patch": 119,
        "grid_position": "row=8, col=7",
        "avg_attention": 0.02745,
        "z_score": "+4.6"
      },
      {
        "patch": 116,
        "grid_position": "row=8, col=4",
        "avg_attention": 0.02517,
        "z_score": "+4.1"
      },
      {
        "patch": 134,
        "grid_position": "row=9, col=8",
        "avg_attention": 0.02406,
        "z_score": "+3.9"
      }
    ],
    "highest_average_hidden_state_norm": [
      {
        "patch": 136,
        "grid_position": "row=9, col=10",
        "avg_norm": 53.781,
        "z_score": "+2.5"
      },
      {
        "patch": 6,
        "grid_position": "row=0, col=6",
        "avg_norm": 53.421,
        "z_score": "+2.3"
      },
      {
        "patch": 195,
        "grid_position": "row=13, col=13",
        "avg_norm": 53.192,
        "z_score": "+2.2"
      },
      {
        "patch": 36,
        "grid_position": "row=2, col=8",
        "avg_norm": 53.107,
        "z_score": "+2.1"
      },
      {
        "patch": 151,
        "grid_position": "row=10, col=11",
        "avg_norm": 53.094,
        "z_score": "+2.1"
      }
    ]
  },
  "per_attribute_evaluation": {
    "cell_size": {
      "model_accuracy": "76.19%",
      "majority_class_guess": "54.99% (always predicting 'big')",
      "gap_over_baseline": "+21.20 points",
      "classification_report": {
        "big": { "precision": 0.720, "recall": 0.928, "f1-score": 0.811, "support": 1704 },
        "small": { "precision": 0.865, "recall": 0.558, "f1-score": 0.679, "support": 1395 }
      },
      "confusion_matrix": [
        [1582, 122],
        [616, 779]
      ]
    },
    "cell_shape": {
      "model_accuracy": "80.83%",
      "majority_class_guess": "77.41% (always predicting 'round')",
      "gap_over_baseline": "+3.42 points",
      "warning": "model is barely beating the majority-class guess. This head likely has NOT learned a real signal.",
      "classification_report": {
        "irregular": { "precision": 0.636, "recall": 0.354, "f1-score": 0.455, "support": 700 },
        "round": { "precision": 0.833, "recall": 0.941, "f1-score": 0.884, "support": 2399 }
      },
      "confusion_matrix": [
        [248, 452],
        [142, 2257]
      ]
    },
    "nucleus_shape": {
      "model_accuracy": "65.38%",
      "majority_class_guess": "29.27% (always predicting 'segmented-bilobed')",
      "gap_over_baseline": "+36.11 points",
      "classification_report": {
        "irregular": { "precision": 0.471, "recall": 0.500, "f1-score": 0.485, "support": 264 },
        "segmented-bilobed": { "precision": 0.667, "recall": 0.699, "f1-score": 0.683, "support": 907 },
        "segmented-multilobed": { "precision": 0.429, "recall": 0.007, "f1-score": 0.014, "support": 418 },
        "unsegmented-band": { "precision": 0.629, "recall": 0.867, "f1-score": 0.729, "support": 769 },
        "unsegmented-indented": { "precision": 0.720, "recall": 0.789, "f1-score": 0.753, "support": 436 },
        "unsegmented-round": { "precision": 0.759, "recall": 0.807, "f1-score": 0.782, "support": 305 }
      }
    },
    "nuclear_cytoplasmic_ratio": {
      "model_accuracy": "97.55%",
      "majority_class_guess": "87.90% (always predicting 'low')",
      "gap_over_baseline": "+9.65 points",
      "classification_report": {
        "high": { "precision": 0.960, "recall": 0.832, "f1-score": 0.891, "support": 375 },
        "low": { "precision": 0.977, "recall": 0.995, "f1-score": 0.986, "support": 2724 }
      },
      "confusion_matrix": [
        [312, 63],
        [13, 2711]
      ]
    },
    "chromatin_density": {
      "model_accuracy": "95.29%",
      "majority_class_guess": "90.74% (always predicting 'densely')",
      "gap_over_baseline": "+4.55 points",
      "warning": "model is barely beating the majority-class guess. This head likely has NOT learned a real signal.",
      "classification_report": {
        "densely": { "precision": 0.983, "recall": 0.964, "f1-score": 0.974, "support": 2812 },
        "loosely": { "precision": 0.707, "recall": 0.840, "f1-score": 0.768, "support": 287 }
      },
      "confusion_matrix": [
        [2712, 100],
        [46, 241]
      ]
    },
    "cytoplasm_vacuole": {
      "model_accuracy": "93.80%",
      "majority_class_guess": "92.09% (always predicting 'no')",
      "gap_over_baseline": "+1.71 points",
      "warning": "model is barely beating the majority-class guess. This head likely has NOT learned a real signal.",
      "classification_report": {
        "no": { "precision": 0.979, "recall": 0.953, "f1-score": 0.966, "support": 2854 },
        "yes": { "precision": 0.582, "recall": 0.767, "f1-score": 0.662, "support": 245 }
      },
      "confusion_matrix": [
        [2719, 135],
        [57, 188]
      ]
    },
    "cytoplasm_texture": {
      "model_accuracy": "94.64%",
      "majority_class_guess": "79.35% (always predicting 'clear')",
      "gap_over_baseline": "+15.30 points",
      "classification_report": {
        "clear": { "precision": 0.976, "recall": 0.956, "f1-score": 0.966, "support": 2459 },
        "frosted": { "precision": 0.843, "recall": 0.909, "f1-score": 0.875, "support": 640 }
      },
      "confusion_matrix": [
        [2351, 108],
        [58, 582]
      ]
    },
    "cytoplasm_colour": {
      "model_accuracy": "90.77%",
      "majority_class_guess": "74.80% (always predicting 'light blue')",
      "gap_over_baseline": "+15.97 points",
      "classification_report": {
        "blue": { "precision": 0.815, "recall": 0.601, "f1-score": 0.692, "support": 441 },
        "light blue": { "precision": 0.981, "recall": 0.987, "f1-score": 0.984, "support": 2318 },
        "purple blue": { "precision": 0.587, "recall": 0.762, "f1-score": 0.663, "support": 340 }
      },
      "confusion_matrix": [
        [265, 18, 158],
        [5, 2289, 24],
        [55, 26, 259]
      ]
    },
    "granule_colour": {
      "model_accuracy": "97.68%",
      "majority_class_guess": "31.91% (always predicting 'pink')",
      "gap_over_baseline": "+65.76 points",
      "classification_report": {
        "nil": { "precision": 0.984, "recall": 0.978, "f1-score": 0.981, "support": 809 },
        "pink": { "precision": 0.969, "recall": 0.995, "f1-score": 0.982, "support": 989 },
        "purple": { "precision": 0.927, "recall": 0.920, "f1-score": 0.923, "support": 374 },
        "red": { "precision": 1.000, "recall": 0.980, "f1-score": 0.990, "support": 927 }
      }
    },
    "granule_type": {
      "model_accuracy": "97.93%",
      "majority_class_guess": "32.78% (always predicting 'small')",
      "gap_over_baseline": "+65.15 points",
      "classification_report": {
        "coarse": { "precision": 0.922, "recall": 0.957, "f1-score": 0.939, "support": 347 },
        "nil": { "precision": 0.990, "recall": 0.978, "f1-score": 0.984, "support": 810 },
        "round": { "precision": 0.999, "recall": 0.975, "f1-score": 0.987, "support": 926 },
        "small": { "precision": 0.974, "recall": 0.992, "f1-score": 0.983, "support": 1016 }
      }
    },
    "granularity": {
      "model_accuracy": "98.94%",
      "majority_class_guess": "73.89% (always predicting 'yes')",
      "gap_over_baseline": "+25.04 points",
      "classification_report": {
        "no": { "precision": 0.987, "recall": 0.972, "f1-score": 0.979, "support": 809 },
        "yes": { "precision": 0.990, "recall": 0.996, "f1-score": 0.993, "support": 2290 }
      },
      "confusion_matrix": [
        [786, 23],
        [10, 2280]
      ]
    }
  },
  "summary_sorted_by_gap": [
    { "attribute": "cytoplasm_vacuole", "accuracy": "93.80%", "baseline": "92.09%", "gap": "+1.71", "status": "likely shortcut / undertrained" },
    { "attribute": "cell_shape", "accuracy": "80.83%", "baseline": "77.41%", "gap": "+3.42", "status": "likely shortcut / undertrained" },
    { "attribute": "chromatin_density", "accuracy": "95.29%", "baseline": "90.74%", "gap": "+4.55", "status": "likely shortcut / undertrained" },
    { "attribute": "nuclear_cytoplasmic_ratio", "accuracy": "97.55%", "baseline": "87.90%", "gap": "+9.65", "status": "normal" },
    { "attribute": "cytoplasm_texture", "accuracy": "94.64%", "baseline": "79.35%", "gap": "+15.30", "status": "normal" },
    { "attribute": "cytoplasm_colour", "accuracy": "90.77%", "baseline": "74.80%", "gap": "+15.97", "status": "normal" },
    { "attribute": "cell_size", "accuracy": "76.19%", "baseline": "54.99%", "gap": "+21.20", "status": "normal" },
    { "attribute": "granularity", "accuracy": "98.94%", "baseline": "73.89%", "gap": "+25.04", "status": "normal" },
    { "attribute": "nucleus_shape", "accuracy": "65.38%", "baseline": "29.27%", "gap": "+36.11", "status": "normal" },
    { "attribute": "granule_type", "accuracy": "97.93%", "baseline": "32.78%", "gap": "+65.15", "status": "normal" },
    { "attribute": "granule_colour", "accuracy": "97.68%", "baseline": "31.91%", "gap": "+65.76", "status": "normal" }
  ]
}'''