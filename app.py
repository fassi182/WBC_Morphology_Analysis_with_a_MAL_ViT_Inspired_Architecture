"""
Streamlit App
-------------

Explainable WBC Morphology Analysis using MAL-ViT.

Features
--------
1. Upload WBC image
2. Capture image using camera
3. Select sample image from images/
4. Predict WBC type
5. Predict 11 morphology attributes
6. Generate Grad-CAM for every morphology attribute
7. View individual attribute explanations
8. View all 11 Grad-CAM explanations

Run:
    streamlit run app.py
"""

from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from inference import predict
from utils.xai.vit_grad_cam import generate_all_attribute_cams


# ==========================================================
# Page Config
# ==========================================================

st.set_page_config(
    page_title="WBC Morphology Analysis with MAL-ViT",
    page_icon="🩸",
    layout="wide",
)


# ==========================================================
# Title
# ==========================================================

st.title(
    "🩸 WBC Morphology Analysis "
    "with a MAL-ViT Inspired Architecture"
)

st.write(
    "Predict White Blood Cell type and 11 morphology "
    "attributes with attribute-level Grad-CAM explanations."
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
    horizontal=True,
)

image = None
image_name = None


# ==========================================================
# Upload Image
# ==========================================================

if option == "📂 Upload Image":

    uploaded = st.file_uploader(
        "Upload a WBC Image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is not None:

        image = Image.open(
            uploaded
        ).convert("RGB")

        image_name = uploaded.name


# ==========================================================
# Camera
# ==========================================================

elif option == "📷 Camera":

    captured = st.camera_input(
        "Take a Picture"
    )

    if captured is not None:

        image = Image.open(
            captured
        ).convert("RGB")

        image_name = "camera_image.jpg"


# ==========================================================
# Sample Images
# ==========================================================

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

            image_name = selected


# ==========================================================
# Prediction
# ==========================================================

if image is not None:

    st.divider()

    # ------------------------------------------------------
    # Display Input Image
    # ------------------------------------------------------

    col1, col2 = st.columns(
        [1, 1]
    )

    with col1:

        st.subheader(
            "🔬 Input Image"
        )

        st.image(
            image,
            use_container_width=True,
        )

        if image_name:
            st.caption(
                f"Image: {image_name}"
            )

    # ------------------------------------------------------
    # Save temporary image
    # ------------------------------------------------------

    with tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False,
    ) as temp:

        image.save(
            temp.name
        )

        temp_path = temp.name

    # ------------------------------------------------------
    # Run Prediction
    # ------------------------------------------------------

    with st.spinner(
        "Analyzing WBC morphology..."
    ):

        cell, attributes = predict(
            temp_path
        )

    # ------------------------------------------------------
    # Prediction Results
    # ------------------------------------------------------

    with col2:

        st.subheader(
            "🩸 WBC Prediction"
        )

        st.success(
            f"### {cell.upper()}"
        )

        st.markdown("---")

        st.subheader(
            "🔬 Morphology Attributes"
        )

        for key, value in attributes.items():

            st.write(
                f"**{key.replace('_', ' ').title()}** : "
                f"{value}"
            )


    # ======================================================
    # Grad-CAM
    # ======================================================

    st.divider()

    st.header(
        "🔥 Attribute-Level Explainability"
    )

    st.write(
        "Each Grad-CAM highlights image regions that "
        "contribute to the prediction of a specific "
        "morphology attribute."
    )


    # ------------------------------------------------------
    # Generate Grad-CAMs
    # ------------------------------------------------------

    with st.spinner(
        "Generating 11 attribute Grad-CAM explanations..."
    ):

        cam_results = (
            generate_all_attribute_cams(
                temp_path
            )
        )


    st.success(
        f"Generated {len(cam_results)} attribute explanations."
    )


    # ======================================================
    # Individual Attribute Explanation
    # ======================================================

    st.subheader(
        "🔎 Explore Individual Attribute"
    )

    attribute_names = list(
        cam_results.keys()
    )

    selected_attribute = st.selectbox(
        "Select morphology attribute",
        attribute_names,
    )


    selected_result = cam_results[
        selected_attribute
    ]


    # ------------------------------------------------------
    # Selected Attribute Information
    # ------------------------------------------------------

    selected_prediction = attributes[
        selected_attribute
    ]

    st.markdown(
        f"""
        ### {selected_attribute.replace('_', ' ').title()}

        **Predicted value:** `{selected_prediction}`
        """
    )


    # ------------------------------------------------------
    # Selected CAM
    # ------------------------------------------------------

    cam_col1, cam_col2 = st.columns(
        [1, 1]
    )


    with cam_col1:

        st.image(
            image,
            caption="Original Image",
            use_container_width=True,
        )


    with cam_col2:

        st.image(
            selected_result["overlay"],
            caption=(
                f"Grad-CAM — "
                f"{selected_attribute.replace('_', ' ').title()}"
            ),
            use_container_width=True,
        )


    # ======================================================
    # All Attribute Grad-CAMs
    # ======================================================

    st.divider()

    st.subheader(
        "🧠 All 11 Attribute Explanations"
    )

    st.write(
        "Each panel shows which image regions are "
        "most influential for the corresponding "
        "morphology prediction."
    )


    # ------------------------------------------------------
    # Display CAMs in rows of 3
    # ------------------------------------------------------

    for start in range(
        0,
        len(attribute_names),
        3,
    ):

        row_attributes = attribute_names[
            start:start + 3
        ]

        columns = st.columns(
            len(row_attributes)
        )

        for column, attribute_name in zip(
            columns,
            row_attributes,
        ):

            result = cam_results[
                attribute_name
            ]

            prediction = attributes[
                attribute_name
            ]

            with column:

                st.markdown(
                    f"**{attribute_name.replace('_', ' ').title()}**"
                )

                st.image(
                    result["overlay"],
                    use_container_width=True,
                )

                st.caption(
                    f"Prediction: {prediction}"
                )


    # ======================================================
    # Interpretation
    # ======================================================

    st.divider()

    st.subheader(
        "💡 Explanation"
    )

    st.write(
        "The highlighted regions represent image patches "
        "that contributed most strongly to the selected "
        "attribute prediction. Different attributes can "
        "focus on different morphological regions of the "
        "same WBC image."
    )

    st.info(
        "Grad-CAM provides a visual explanation of the "
        "model's decision. It should be interpreted as "
        "model attention/importance rather than a clinical "
        "diagnosis."
    )