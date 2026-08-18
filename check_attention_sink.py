"""
check_attention_sink.py

Diagnostic #1: Attention-Sink Detection.

WHY THIS SCRIPT EXISTS
-----------------------
In the Cell Size / Nucleus Shape example, two DIFFERENT attribute
tokens produced an almost pixel-identical hotspot in the same corner
of the image. Two independent attribute heads landing on the exact
same tiny region is a classic symptom of an "attention sink": one
particular patch token that many query tokens attend to heavily
regardless of image content, simply because it's a convenient place
for the network to dump attention mass during training. (See Darcet
et al., "Vision Transformers Need Registers", 2023.)

If that's what's happening, the *same patch index* (not just the
same rough location, but literally the same one of the 196 patches)
should light up as the top-attended patch:

- across many DIFFERENT images
- for many DIFFERENT attribute tokens

...regardless of what's actually in the image at that location. A
real, content-driven signal would move around depending on where the
cell / nucleus / cytoplasm actually is in each image.

WHAT THIS SCRIPT DOES
----------------------
1. Loads the model and the last transformer block's raw attention
   weights (not gradient-weighted — we want to see raw attention
   behavior here, independent of any specific logit).
2. For every image in a folder and every attribute token, finds
   which patch index receives the most attention from that token.
3. Reports:
   - the single most frequently "winning" patch index, and what
     fraction of (image, attribute) pairs it wins
   - the average incoming attention each patch receives, aggregated
     over all queries and all images (a sink patch will have an
     outlier-high value here)
   - the L2 norm of each patch token's hidden representation,
     averaged over images (sink tokens are also known to have
     anomalously high activation norms)

HOW TO READ THE OUTPUT
------------------------
- If ONE patch index wins across most images/attributes -> attention
  sink confirmed. The localization in your Grad-CAMs for the
  attributes that route through that patch is not trustworthy;
  fixing this generally requires retraining with "register tokens"
  (extra learnable tokens with no output constraint) or simply
  excluding known sink patches from the CAM.
- If winning patches vary sensibly with image content -> the
  identical hotspot in your example was likely a coincidence /
  genuinely shared cue for that specific image, not a structural
  bug. Re-run on a handful of images to double check before
  concluding it's a sink.

USAGE
-----
    python check_attention_sink.py --image_dir images
    python check_attention_sink.py --image_dir path/to/more/images
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import (
    DEVICE,
    CHECKPOINT_DIR,
    BEST_MODEL_NAME,
)

from data.transforms import test_transform
from data.encoders import ATTRIBUTE_NAMES
from models.complete_model import ExplainableWBCModel


ATTRIBUTE_COUNT = len(ATTRIBUTE_NAMES)
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES_SIDE = IMAGE_SIZE // PATCH_SIZE


# ==========================================================
# Load Model
# ==========================================================

def load_model():

    model = ExplainableWBCModel().to(DEVICE)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR / BEST_MODEL_NAME,
            map_location=DEVICE,
        )
    )

    model.eval()

    return model


# ==========================================================
# Hook Last Block's Attention + Output Norms
# ==========================================================

class SinkProbe:

    def __init__(self, model):

        self.model = model

        self.attention_weights = None
        self.block_output = None

        target_block = model.mal_vit.transformer.blocks[-1]

        self.attn_handle = (
            target_block
            .attention
            .attention_dropout
            .register_forward_hook(self._attn_hook)
        )

        self.out_handle = (
            target_block
            .register_forward_hook(self._out_hook)
        )

    def _attn_hook(self, module, inputs, output):
        # inputs[0]: (B, heads, 207, 207), post-softmax, pre-dropout
        self.attention_weights = inputs[0].detach()

    def _out_hook(self, module, inputs, output):
        # (B, 207, 192)
        self.block_output = output.detach()

    def remove(self):
        self.attn_handle.remove()
        self.out_handle.remove()


def prepare_image(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = test_transform(image)

    tensor = tensor.unsqueeze(0).to(DEVICE)

    return tensor


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_dir",
        type=str,
        default="images",
        help="Folder of images to test across.",
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)

    image_paths = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {image_dir}"
        )

    print("=" * 70)
    print("Attention-Sink Diagnostic")
    print("=" * 70)
    print(f"\nTesting {len(image_paths)} images from '{image_dir}'\n")

    model = load_model()
    probe = SinkProbe(model)

    winning_patches = []  # (image_name, attribute_name, patch_idx)

    # running sums for average incoming attention / norm per patch
    incoming_attention_sum = np.zeros(NUM_PATCHES_SIDE * NUM_PATCHES_SIDE)
    patch_norm_sum = np.zeros(NUM_PATCHES_SIDE * NUM_PATCHES_SIDE)
    n_images = 0

    with torch.no_grad():

        for image_path in image_paths:

            tensor = prepare_image(image_path)

            model(tensor)  # triggers hooks

            attn = probe.attention_weights[0]  # (heads, 207, 207)
            attn = attn.mean(dim=0)            # avg over heads -> (207, 207)

            patch_cols = attn[:, ATTRIBUTE_COUNT:]  # (207, 196)

            # For each attribute token's row, which patch wins?
            for idx, name in enumerate(ATTRIBUTE_NAMES):

                row = patch_cols[idx]  # (196,)

                winner = row.argmax().item()

                winning_patches.append(
                    (image_path.name, name, winner)
                )

            # Incoming attention per patch, averaged over ALL query
            # rows (not just attribute tokens) -> classic sink metric
            incoming = patch_cols.mean(dim=0).cpu().numpy()  # (196,)
            incoming_attention_sum += incoming

            # Hidden-state norm per patch token (sinks tend to have
            # outlier-high norm)
            block_out = probe.block_output[0]  # (207, 192)
            patch_out = block_out[ATTRIBUTE_COUNT:]  # (196, 192)
            norms = patch_out.norm(dim=-1).cpu().numpy()  # (196,)
            patch_norm_sum += norms

            n_images += 1

    probe.remove()

    # ------------------------------------------------------
    # Report: Winning Patch Frequency
    # ------------------------------------------------------

    winner_counts = Counter(
        winner for _, _, winner in winning_patches
    )

    total_pairs = len(winning_patches)

    print("-" * 70)
    print("Most frequently 'winning' patch index across all "
          "(image, attribute) pairs:")
    print("-" * 70)

    for patch_idx, count in winner_counts.most_common(5):

        row = patch_idx // NUM_PATCHES_SIDE
        col = patch_idx % NUM_PATCHES_SIDE

        pct = 100 * count / total_pairs

        print(
            f"  patch #{patch_idx:3d}  (grid row={row}, col={col})  "
            f"-> won {count}/{total_pairs} times ({pct:.1f}%)"
        )

    top_patch, top_count = winner_counts.most_common(1)[0]
    top_pct = 100 * top_count / total_pairs

    print()

    if top_pct > 25:
        print(f"⚠️  WARNING: a single patch (#{top_patch}) accounts for "
              f"{top_pct:.1f}% of all top-attended results across "
              f"different images and attributes.")
        print("    This strongly suggests an attention-sink token, "
              "not content-driven localization.")
    else:
        print("No single patch dominates disproportionately — "
              "winning patches look content-dependent, not sink-like.")

    # ------------------------------------------------------
    # Report: Average Incoming Attention Per Patch
    # ------------------------------------------------------

    avg_incoming = incoming_attention_sum / n_images

    top5_idx = np.argsort(avg_incoming)[::-1][:5]

    print()
    print("-" * 70)
    print("Patches with highest AVERAGE incoming attention "
          "(across all queries, all images):")
    print("-" * 70)

    mean_val = avg_incoming.mean()
    std_val = avg_incoming.std()

    for idx in top5_idx:
        row = idx // NUM_PATCHES_SIDE
        col = idx % NUM_PATCHES_SIDE
        z = (avg_incoming[idx] - mean_val) / (std_val + 1e-8)
        print(
            f"  patch #{idx:3d} (row={row}, col={col})  "
            f"avg attention = {avg_incoming[idx]:.5f}  (z-score {z:+.1f})"
        )

    # ------------------------------------------------------
    # Report: Average Patch Token Norm
    # ------------------------------------------------------

    avg_norm = patch_norm_sum / n_images

    top5_norm_idx = np.argsort(avg_norm)[::-1][:5]

    print()
    print("-" * 70)
    print("Patches with highest AVERAGE hidden-state norm "
          "(classic sink-token signature):")
    print("-" * 70)

    mean_norm = avg_norm.mean()
    std_norm = avg_norm.std()

    for idx in top5_norm_idx:
        row = idx // NUM_PATCHES_SIDE
        col = idx % NUM_PATCHES_SIDE
        z = (avg_norm[idx] - mean_norm) / (std_norm + 1e-8)
        print(
            f"  patch #{idx:3d} (row={row}, col={col})  "
            f"avg norm = {avg_norm[idx]:.3f}  (z-score {z:+.1f})"
        )

    print()
    print("If the SAME patch index shows up as an outlier in all three "
          "reports above, that patch is almost certainly a learned "
          "attention sink rather than a real morphological cue.")


if __name__ == "__main__":
    main()