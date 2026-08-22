"""
app.py

Streamlit UI for MAL-ViT WBC Morphology Analysis.

Pipeline
--------
Upload Image
    ↓
MAL-ViT
    ↓
WBC Prediction
    ↓
11 Morphology Predictions
    ↓
Attribute-Level XAI
    ├── Grad-CAM-style attribution
    ├── Attention map
    └── Gradient map
"""

from pathlib import Path

import streamlit as st
import numpy as np

from PIL import Image

from data.encoders import ATTRIBUTE_NAMES

from utils.xai.vit_grad_cam import (
    generate_explanations,
    colorize_heatmap,
    resize_heatmap,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MAL-ViT WBC Analysis",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #777777;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f5f7fa;
        border: 1px solid #dddddd;
        margin-bottom: 20px;
    }

    .attribute-title {
        font-size: 20px;
        font-weight: 600;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">MAL-ViT WBC Morphology Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "White Blood Cell Classification, Morphology Prediction "
    "and Attribute-Level Explainability"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About MAL-ViT")

    st.write(
        """
        This system uses MAL-ViT to analyze white blood cell
        morphology through multiple attribute-specific tokens.
        """
    )

    st.divider()

    st.write("### Model")

    st.write("• Vision Transformer")
    st.write("• 11 morphology attributes")
    st.write("• 4 register tokens")
    st.write("• 196 image patch tokens")
    st.write("• 6 transformer blocks")
    st.write("• 6 attention heads")

    st.divider()

    st.write("### Explainability")

    st.write("• Attribute Grad-CAM")
    st.write("• Attribute Attention")
    st.write("• Gradient Attribution")


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.header("Upload WBC Image")

uploaded_file = st.file_uploader(
    "Choose a WBC image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


# ============================================================
# STOP IF NO IMAGE
# ============================================================

if uploaded_file is None:

    st.info(
        "Upload a WBC microscopy image to begin the analysis."
    )

    st.stop()


# ============================================================
# LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception:

    st.error(
        "The uploaded file could not be read as an image."
    )

    st.stop()


# ============================================================
# DISPLAY INPUT IMAGE
# ============================================================

st.header("Input Image")

image_column, info_column = st.columns(
    [1, 1]
)

with image_column:

    st.image(
        image,
        caption="Uploaded WBC Image",
        use_container_width=True,
    )

with info_column:

    st.write("### Image Information")

    st.write(
        f"**Filename:** {uploaded_file.name}"
    )

    st.write(
        f"**Image size:** {image.width} × {image.height}"
    )

    st.write(
        f"**Format:** {image.format or 'RGB image'}"
    )


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze = st.button(
    "🔬 Analyze WBC",
    type="primary",
    use_container_width=True,
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    progress = st.progress(
        0
    )

    status = st.empty()

    # --------------------------------------------------------
    # Generate predictions + XAI
    # --------------------------------------------------------

    try:

        status.write(
            "Loading MAL-ViT and generating predictions..."
        )

        progress.progress(
            10
        )

        results = generate_explanations(
            image
        )

        progress.progress(
            100
        )

        status.success(
            "Analysis completed successfully."
        )

    except Exception as error:

        progress.empty()

        status.empty()

        st.error(
            "Analysis failed."
        )

        st.write(
            "Please check the PowerShell terminal running "
            "Streamlit for the detailed error."
        )

        st.stop()


    # ========================================================
    # WBC PREDICTION
    # ========================================================

    st.divider()

    st.header("WBC Classification")

    wbc_column, index_column = st.columns(
        [2, 1]
    )

    with wbc_column:

        st.markdown(
            '<div class="prediction-box">',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Predicted WBC Type"
        )

        st.success(
            results["wbc"].replace(
                "_",
                " ",
            ).title()
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with index_column:

        st.metric(
            "Class Index",
            results["wbc_index"],
        )


    # ========================================================
    # MORPHOLOGY PREDICTIONS
    # ========================================================

    st.divider()

    st.header(
        "Morphology Predictions"
    )

    attributes = results[
        "attributes"
    ]

    # --------------------------------------------------------
    # Display attributes in two columns
    # --------------------------------------------------------

    attribute_items = list(
        attributes.items()
    )

    left_column, right_column = st.columns(
        2
    )

    for index, (
        attribute_name,
        predicted_value,
    ) in enumerate(
        attribute_items
    ):

        if index % 2 == 0:

            column = left_column

        else:

            column = right_column

        with column:

            st.markdown(
                '<div class="prediction-box">',
                unsafe_allow_html=True,
            )

            st.write(
                f"**{attribute_name.replace('_', ' ').title()}**"
            )

            st.success(
                str(
                    predicted_value
                ).replace(
                    "_",
                    " ",
                ).title()
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


    # ========================================================
    # XAI SECTION
    # ========================================================

    st.divider()

    st.header(
        "Attribute-Level Explainability"
    )

    st.write(
        """
        Each morphology attribute has its own explanation.
        The Grad-CAM-style map combines the attribute-token
        attention with gradient information from the selected
        attribute prediction.
        """
    )


    # ========================================================
    # ATTRIBUTE XAI
    # ========================================================

    cams = results[
        "cams"
    ]


    for attribute_name in ATTRIBUTE_NAMES:

        if attribute_name not in cams:

            continue

        result = cams[
            attribute_name
        ]

        predicted_class = result[
            "predicted_class"
        ]

        # ----------------------------------------------------
        # Attribute heading
        # ----------------------------------------------------

        st.subheader(
            attribute_name
            .replace(
                "_",
                " ",
            )
            .title()
        )

        st.write(
            f"Predicted value: **"
            f"{str(predicted_class).replace('_', ' ').title()}"
            f"**"
        )


        # ====================================================
        # PREPARE XAI MAPS
        # ====================================================

        gradcam_image = result[
            "overlay"
        ]

        attention_map = result[
            "attention_map"
        ]

        gradient_map = result[
            "gradient_map"
        ]


        # ----------------------------------------------------
        # Resize attention map
        # ----------------------------------------------------

        attention_resized = resize_heatmap(
            attention_map,
            image.size,
        )

        attention_rgb = colorize_heatmap(
            attention_resized
        )

        attention_image = Image.fromarray(
            attention_rgb
        )


        # ----------------------------------------------------
        # Resize gradient map
        # ----------------------------------------------------

        gradient_resized = resize_heatmap(
            gradient_map,
            image.size,
        )

        gradient_rgb = colorize_heatmap(
            gradient_resized
        )

        gradient_image = Image.fromarray(
            gradient_rgb
        )


        # ====================================================
        # DISPLAY XAI
        # ====================================================

        col1, col2, col3 = st.columns(
            3
        )

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        with col1:

            st.image(
                gradcam_image,
                caption="Grad-CAM",
                use_container_width=True,
            )


        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        with col2:

            st.image(
                attention_image,
                caption="Attribute Attention",
                use_container_width=True,
            )


        # ----------------------------------------------------
        # Gradient
        # ----------------------------------------------------

        with col3:

            st.image(
                gradient_image,
                caption="Input Gradient",
                use_container_width=True,
            )


        # ----------------------------------------------------
        # Logits
        # ----------------------------------------------------

        with st.expander(
            f"View {attribute_name} logits"
        ):

            logits = result[
                "logits"
            ]

            st.write(
                logits
            )


        st.divider()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="text-align:center; color:#888888; padding:20px;">
        MAL-ViT • WBC Morphology Analysis • Attribute-Level XAI
    </div>
    """,
    unsafe_allow_html=True,
)