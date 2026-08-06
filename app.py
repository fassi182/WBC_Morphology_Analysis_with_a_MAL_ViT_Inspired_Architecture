"""
Streamlit App


Run:
streamlit run app.py
"""

from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from inference import predict


# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="WBC Morphology Analysis with a MAL-ViT Inspired Architecture",
    page_icon="🩸",
    layout="wide",
)

st.title("🩸 WBC Morphology Analysis with a MAL-ViT Inspired Architecture")
st.write(
    "Predict White Blood Cell type and 11 morphology attributes using MAL-ViT."
)

st.divider()

# ==========================================================
# Image Source
# ==========================================================

option = st.radio(
    "Choose Image Source",
    (
        "📂 Upload Image",
        "📷 Camera",
        "🖼 Sample Images",
    ),
)

image = None

# ----------------------------------------------------------
# Upload
# ----------------------------------------------------------

if option == "📂 Upload Image":

    uploaded = st.file_uploader(
        "Upload a WBC Image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")


# ----------------------------------------------------------
# Camera
# ----------------------------------------------------------

elif option == "📷 Camera":

    captured = st.camera_input(
        "Take a Picture"
    )

    if captured is not None:
        image = Image.open(captured).convert("RGB")


# ----------------------------------------------------------
# Sample Images
# ----------------------------------------------------------

else:

    image_folder = Path("images")

    if not image_folder.exists():

        st.error(
            "images/ folder not found."
        )

    else:

        image_files = sorted(
            [
                p.name
                for p in image_folder.iterdir()
                if p.suffix.lower()
                in [".jpg", ".jpeg", ".png"]
            ]
        )

        if len(image_files) == 0:

            st.warning(
                "No sample images found."
            )

        else:

            selected = st.selectbox(
                "Select Sample Image",
                image_files,
            )

            image = Image.open(
                image_folder / selected
            ).convert("RGB")


# ==========================================================
# Prediction
# ==========================================================

if image is not None:

    col1, col2 = st.columns([1, 1])

    with col1:

        st.subheader("Input Image")

        st.image(
            image,
            use_container_width=True,
        )

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False,
    ) as temp:

        image.save(temp.name)

        cell, attributes = predict(temp.name)

    with col2:

        st.subheader("Prediction")

        st.success(
            f"### 🩸 {cell.upper()}"
        )

        st.markdown("---")

        st.subheader(
            "Morphology Attributes"
        )

        for key, value in attributes.items():

            st.write(
                f"**{key.replace('_',' ').title()}** : {value}"
            )