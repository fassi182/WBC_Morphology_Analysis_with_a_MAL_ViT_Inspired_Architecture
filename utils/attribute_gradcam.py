"""
attribute_gradcam.py

Attribute-specific Grad-CAM for MAL-ViT.

Generates a spatial attribution map for each of the
11 morphology attributes by tracing the selected
attribute prediction back to the ViT patch tokens.

Pipeline:

Image
  ↓
MAL-ViT
  ↓
Attribute Token
  ↓
Attribute Class Logit
  ↓
Gradients
  ↓
196 Patch Tokens
  ↓
14 × 14 Attribution Map
  ↓
224 × 224 Heatmap
"""

import torch
import torch.nn.functional as F
import numpy as np


class AttributeGradCAM:

    def __init__(self, model, device):

        self.model = model
        self.device = device

        self.activations = None
        self.gradients = None

        # --------------------------------------------------
        # Hook the final Transformer output
        # --------------------------------------------------

        self.forward_handle = (
            self.model.mal_vit.transformer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            self.model.mal_vit.transformer.register_full_backward_hook(
                self._backward_hook
            )
        )

    # ======================================================
    # Hooks
    # ======================================================

    def _forward_hook(self, module, inputs, output):

        self.activations = output

    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = grad_output[0]

    # ======================================================
    # Generate CAM
    # ======================================================

    def generate(
        self,
        image,
        attribute_name,
        class_index=None,
    ):

        self.model.zero_grad(set_to_none=True)

        image = image.to(self.device)

        # --------------------------------------------------
        # Forward pass
        # --------------------------------------------------

        outputs = self.model(image)

        logits = outputs[
            "attribute_predictions"
        ][attribute_name]

        # --------------------------------------------------
        # Select predicted class
        # --------------------------------------------------

        if class_index is None:

            class_index = logits.argmax(
                dim=1
            ).item()

        target = logits[
            0,
            class_index
        ]

        # --------------------------------------------------
        # Backward pass
        # --------------------------------------------------

        target.backward(
            retain_graph=True
        )

        # --------------------------------------------------
        # Transformer output
        #
        # Shape:
        # (B, 207, 192)
        #
        # 11 attribute tokens
        # 196 patch tokens
        # --------------------------------------------------

        activations = self.activations[
            0
        ]

        gradients = self.gradients[
            0
        ]

        # --------------------------------------------------
        # Remove attribute tokens
        # --------------------------------------------------

        patch_activations = activations[
            11:
        ]

        patch_gradients = gradients[
            11:
        ]

        # --------------------------------------------------
        # Grad-CAM weights
        # --------------------------------------------------

        weights = patch_gradients.mean(
            dim=1
        )

        # --------------------------------------------------
        # Weighted patch activation
        # --------------------------------------------------

        cam = (
            patch_activations
            * weights.unsqueeze(1)
        ).sum(dim=1)

        # --------------------------------------------------
        # ReLU
        # --------------------------------------------------

        cam = F.relu(cam)

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        # --------------------------------------------------
        # Convert 196 patches → 14 × 14
        # --------------------------------------------------

        cam = cam.reshape(
            1,
            1,
            14,
            14,
        )

        # --------------------------------------------------
        # Upsample → 224 × 224
        # --------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0
        ].detach().cpu().numpy()

        return cam, class_index

    # ======================================================
    # Cleanup
    # ======================================================

    def remove_hooks(self):

        self.forward_handle.remove()

        self.backward_handle.remove()