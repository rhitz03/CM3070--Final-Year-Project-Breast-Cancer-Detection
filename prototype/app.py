# ============================================================
# Author: S Rhithika
# Project:
# Deep Learning Breast Cancer Detection Using
# Transfer Learning and CNN-Based Mammogram Classification
#
## ============================================================
# Streamlit Research Prototype
# ------------------------------------------------------------
# Provides image upload and demo-case selection, basic input
# checks, predictions from the available CNN models, and
# Grad-CAM and occlusion visualizations.
# ============================================================
# ============================================================

import time
import numpy as np
import streamlit as st
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from utils.model_loader import load_models
from utils.predictor import (
    predict_all_models,
    generate_prediction_summary,
    format_results_table,
    PREPROCESSORS,
    DISPLAY_NAMES
)
from utils.gradcam import (
    find_last_conv_layer,
    generate_gradcam,
    resize_gradcam_heatmap,
    overlay_heatmap_on_image
)
from utils.roi import load_demo_manifest, get_roi_mask_path, prepare_aligned_roi, compute_iou
from utils.occlusion import compute_occlusion_heatmap
from utils.preprocessing import preprocess_for_display

DEMO_MANIFEST = load_demo_manifest()

DEMO_CASE_FILES = {
    "case_1.jpg": "demo_images/case_1.jpg",
    "case_2.jpg": "models/case_2.jpg",
    "case_3.jpg": "models/case_3.jpg",
    "case_4.jpg": "models/case_4.jpg",
    "case_5.jpg": "models/case_5.jpg",
    "case_6.jpg": "models/case_6.jpg",
}

# ============================================================
# Input Validation
# ============================================================
# Best-effort checks, not a guarantee the image is a real mammogram --
# this is a research demonstrator, so validation is a safety net against
# obviously wrong input (a photo, an icon, a corrupt file), not a medical
# image quality control system.

MIN_DIMENSION = 100  # pixels, below this a crop is almost certainly not usable
# Mean per-pixel |R-G|+|G-B|+|R-B|, out of 255 per channel. A true
# greyscale image (R==G==B at every pixel) scores 0 regardless of overall
# brightness/contrast. FIX vs. the first version of this check: comparing
# the three channels' WHOLE-IMAGE AVERAGES missed ordinary colour photos
# whose averages happen to land close together (skin tones, mixed
# backgrounds) even though individual pixels are clearly colourful.
# Per-pixel difference catches that; whole-image averages don't.
PER_PIXEL_COLOUR_THRESHOLD = 10
MAX_ASPECT_RATIO = 3.0  # width:height or height:width beyond this isn't a lesion crop

# Fixed thumbnail width for the input preview, regardless of the uploaded
# image's actual resolution -- a large photo would otherwise stretch to
# fill the column (the old use_container_width=True), dwarfing the rest
# of the page.
PREVIEW_WIDTH = 320

# These checks catch obviously unsuitable uploads; they do not verify that an image is a mammogram.
def validate_image(image):
    """Returns (is_valid, warnings). is_valid=False blocks inference;
    a warning alone does not."""
    warnings = []

    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        return False, [f"Image is too small ({width}x{height}px, minimum {MIN_DIMENSION}px)."]

    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False, [
            f"Image has an extreme aspect ratio ({width}x{height}px). Lesion crops "
            "are roughly square; this doesn't look like one."
        ]

    rgb = np.array(image.convert("RGB")).astype(np.float32)
    per_pixel_colour = (
        np.abs(rgb[..., 0] - rgb[..., 1])
        + np.abs(rgb[..., 1] - rgb[..., 2])
        + np.abs(rgb[..., 0] - rgb[..., 2])
    )
    colour_spread = float(np.mean(per_pixel_colour))
    if colour_spread > PER_PIXEL_COLOUR_THRESHOLD:
        warnings.append(
            "This image has noticeable colour variation. Mammograms are "
            "greyscale, so this may not be a mammographic image -- results may "
            "not be meaningful."
        )

    return True, warnings


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# Disclaimer
# ============================================================

st.warning(
    "**Research demonstrator, not a clinical tool.** This app shows how the "
    "trained models and Grad-CAM explanations behave on an image. It has not "
    "been validated for, and must not be used for, real diagnostic or "
    "clinical decisions."
)

# ============================================================
# Load Models
# ============================================================
# Keep the loaded models in memory across Streamlit reruns.
@st.cache_resource
def get_models():
    return load_models()

MODELS = get_models()

# ============================================================
# Load Grad-CAM Target Layers
# ============================================================
# Grad-CAM is generated for the four models actually carried through the
# report's interpretability analysis (Section 7): the baseline plus each
# architecture's PARTIAL FINE-TUNING winner (ARCHITECTURE_WINNERS in the
# notebook), not all five/seven trained configs.
#
# FIX vs. the current app.py: this list previously said "VGG16 Frozen",
# which was never one of the models Section 7 actually analysed -- the
# selected model per architecture is the Partial FT variant. Also adds
# EfficientNetB0 Partial, one of the three officially selected winners
# (Table 5.8), previously missing from the prototype entirely.

GRADCAM_MODEL_NAMES = [
    "Baseline CNN",
    "VGG16 Partial",
    "ResNet50 Partial",
    "EfficientNetB0 Partial",
]

@st.cache_resource
def get_gradcam_target_layers():
    return {
        model_name: find_last_conv_layer(MODELS[model_name])
        for model_name in GRADCAM_MODEL_NAMES
        if model_name in MODELS
    }

GRADCAM_TARGET_LAYERS = get_gradcam_target_layers()

# ============================================================
# Banner
# ============================================================

header_left, header_right = st.columns([1, 5])

with header_left:
    st.image("assets/logo.png", width=170)

with header_right:
    st.title("Deep Learning Breast Cancer Detection")
    st.markdown("### Comparative Deep Learning for Mammographic Lesion Classification")
    st.caption("Research Prototype • Transfer Learning Model Evaluation")
    st.markdown(
        """
🗂 **CBIS-DDSM Dataset** &nbsp;&nbsp;&nbsp;&nbsp;
🧠 **7 CNN Models** &nbsp;&nbsp;&nbsp;&nbsp;
⚙ **TensorFlow / Keras** &nbsp;&nbsp;&nbsp;&nbsp;
🎯 **Binary Classification**
""",
        unsafe_allow_html=True
    )

st.divider()

# ============================================================
# Main Layout
# ============================================================

left_col, right_col = st.columns([1.15, 1])

with left_col:

    st.subheader("🖼 Lesion Image Input")
    st.caption(
        "Expects a cropped lesion image, not a full mammogram. The models were "
        "trained on CBIS-DDSM's cropped mass/calcification crops, not whole scans."
    )

    

    demo_choice = st.selectbox(
        "Try a demo case with known ground truth (optional)",
        ["- Upload my own image -"] + list(DEMO_CASE_FILES.keys())
    )

    image_placeholder = st.empty()
    preprocessed_placeholder = st.empty()

    image = None
    demo_case_selected = None
    image_warnings = []
    image_is_valid = True

    if demo_choice != "- Upload my own image -":
        demo_case_selected = demo_choice
        image = Image.open(DEMO_CASE_FILES[demo_choice])
        image_placeholder.image(image, width=PREVIEW_WIDTH)

        true_label = "Malignant" if DEMO_MANIFEST[demo_choice]["label"] == 1 else "Benign"
        st.caption(f"📄 {demo_choice} (demo case)   |   Ground truth: {true_label}")

    else:
        uploaded_file = st.file_uploader(
            "",
            type=["png", "jpg", "jpeg"],
            help="Drag and drop a mammogram image here."
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                image.load()  # force full decode now -- catches truncated/corrupt data immediately, not lazily later
            except (UnidentifiedImageError, OSError):
                image = None
                image_is_valid = False
                st.error(
                    f"Could not read \"{uploaded_file.name}\" as an image. "
                    "It may be corrupted or in an unsupported format."
                )
            else:
                image_is_valid, image_warnings = validate_image(image)

                image_placeholder.image(image, width=PREVIEW_WIDTH)
                st.caption(
                    f"📄 {uploaded_file.name}   |   "
                    f"📐 {image.size[0]} × {image.size[1]} pixels"
                )

                if not image_is_valid:
                    for message in image_warnings:
                        st.error(message)
                else:
                    for message in image_warnings:
                        st.warning(message)

    if image is not None and image_is_valid:
        preprocessed_placeholder.image(
            preprocess_for_display(image),
            width=PREVIEW_WIDTH,
            caption="What the model sees: resized to 224×224, CLAHE contrast-enhanced.",
        )
    
    
    run_button = st.button(
        "Run Comparative Analysis",
        use_container_width=True,
        type="primary",
        disabled=(image is not None and not image_is_valid),
    )

with right_col:

    st.subheader("📊 Prediction Summary")
    prediction_placeholder = st.empty()
    agreement_placeholder = st.empty()
    disagreement_placeholder = st.empty()

    st.divider()

    st.subheader("🔥 Explainability Analysis (Grad-CAM)")
    st.caption(
        "Shown for the four models carried through the project's "
        "interpretability analysis: Baseline CNN and each architecture's "
        "Partial Fine-Tuning winner (VGG16, ResNet50, EfficientNetB0)."
    )
    st.caption(
        "⚠️ Grad-CAM shows which pixels influenced the model's own prediction, "
        "not whether that prediction is clinically correct. A confident, "
        "well-localised heatmap can still accompany a wrong prediction."
    )
    gradcam_placeholder = st.empty()

    with gradcam_placeholder.container():
        st.info("Upload an image and run the analysis to see the Grad-CAM heatmaps here.")

# ============================================================
# Run Comparative Analysis
# ============================================================

if run_button and image is not None and image_is_valid:

    inference_start = time.perf_counter()
    results = predict_all_models(image, MODELS)
    inference_latency_ms = (time.perf_counter() - inference_start) * 1000

    summary = generate_prediction_summary(results)
    table = format_results_table(results)

    prediction_placeholder.metric("Majority Prediction", summary["majority_prediction"])

    if summary["majority_prediction"] == "Benign":
        prediction_placeholder.success("### 🟢 Benign")
    else:
        prediction_placeholder.error("### 🔴 Malignant")

    agreement_placeholder.info(
        f"### 🤝 Model Agreement\n\n"
        f"{summary['agreement']} models predicted **{summary['majority_prediction']}**"
    )

    if summary["disagreements"]:
        # Binary classification -- every disagreeing model predicted the
        # SAME other class (there's only one alternative), so that's
        # stated once, not repeated per model.
        dissenting_prediction = results[summary["disagreements"][0]]["prediction"]
        dissenting_names = [DISPLAY_NAMES.get(name, name) for name in summary["disagreements"]]
        disagreement_placeholder.warning(
            f"Disagreed with the majority ({summary['majority_prediction']}), "
            f"predicted {dissenting_prediction} instead: " + ", ".join(dissenting_names)
        )
    else:
        disagreement_placeholder.success("All CNN models agree.")

    st.divider()
    st.markdown("---")
    st.subheader("📈 Transfer Learning Model Evaluation")
    st.caption("Prediction results generated using all trained CNN architectures.")

    uncalibrated = [
        DISPLAY_NAMES.get(name, name) for name, r in results.items() if not r["calibrated"]
    ]
    if uncalibrated:
        st.caption(
            "ℹ️ Scores are calibrated where a calibrator exists (see report Section 5.2.8); "
            f"shown as raw, uncalibrated model output for: {', '.join(uncalibrated)}."
        )
    else:
        st.caption("ℹ️ All scores below are calibrated (see report Section 5.2.8).")

    comparison_df = pd.DataFrame(
        table, columns=["CNN Architecture", "Predicted Class", "Model Score (%)", "Calibrated?", "Decision Threshold"]
    )
    st.dataframe(comparison_df, height=215, use_container_width=True, hide_index=True)
    st.caption(
        "Decision Threshold: each model's own cutoff on raw probability "
        "(Youden's J, picked on validation, report Section 3.7.1), not a flat 0.5."
    )

    st.caption(f"⏱ Inference across all {len(results)} models: {inference_latency_ms:.0f} ms")

    # ========================================================
    # Grad-CAM Explainability (three selected models)
    # ========================================================

    gradcam_start = time.perf_counter()

    with gradcam_placeholder.container():

        gradcam_cols = st.columns(len(GRADCAM_MODEL_NAMES))

        # Only demo cases have a known expert ROI to compare against --
        # this stays None for any freely uploaded image.
        roi_binary = None

        if demo_case_selected is not None:
            try:
                original_path = DEMO_MANIFEST[demo_case_selected]["original_image_path"]
                roi_path = get_roi_mask_path(original_path)

                with Image.open(DEMO_CASE_FILES[demo_case_selected]) as demo_img:
                    crop_size = demo_img.size

                roi_binary = prepare_aligned_roi(roi_path, crop_size=crop_size, target_size=(224, 224))

            except (FileNotFoundError, ValueError) as e:
                st.warning(f"Could not load expert ROI for this demo case: {e}")

        for col, model_name in zip(gradcam_cols, GRADCAM_MODEL_NAMES):

            if model_name not in MODELS:
                continue

            processed_image = PREPROCESSORS[model_name](image)

            # This model's own fixed threshold (from results), not 0.5.
            heatmap, probability, predicted_class = generate_gradcam(
                MODELS[model_name],
                tf.convert_to_tensor(processed_image),
                GRADCAM_TARGET_LAYERS[model_name],
                model_name,
                threshold=results[model_name]["threshold"],
            )

            heatmap = resize_gradcam_heatmap(heatmap, target_size=(224, 224))
            display_image = image.convert("RGB").resize((224, 224))
            overlay_image = overlay_heatmap_on_image(display_image, heatmap)

            with col:
                st.image(overlay_image, use_container_width=True)

                predicted_label = "Malignant" if predicted_class == 1 else "Benign"
                caption_text = (
                    f"**{DISPLAY_NAMES.get(model_name, model_name)}**  \n"
                    f"Predicted: {predicted_label} ({probability:.1%})"
                )

                if roi_binary is not None:
                    iou = compute_iou(heatmap, roi_binary)
                    caption_text += f"  \nGrad-CAM vs expert ROI: IoU = {iou:.3f}"

                st.caption(caption_text)

                # ResNet50 (Partial FT) blank malignant heatmaps are a
                # documented finding (report Table 4.10, Table 5.9), not a
                # bug -- shown here so it reads as a known result, not an
                # app error.
                if np.max(heatmap) < 1e-6:
                    st.info(
                        "This model's Grad-CAM map is blank for this prediction. "
                        "This is a documented limitation (see report Section 5.4.2), "
                        "not an error in this app."
                    )

    gradcam_latency_ms = (time.perf_counter() - gradcam_start) * 1000
    st.caption(f"⏱ Grad-CAM for {len(GRADCAM_MODEL_NAMES)} models: {gradcam_latency_ms:.0f} ms")

    # ========================================================
    # Occlusion Sensitivity (Cross-Check for Baseline CNN)
    # ========================================================

    st.divider()
    st.subheader("🧩 Occlusion Sensitivity (Cross-Check)")
    st.caption(
        "A second, gradient-free explainability method, shown for "
        "Baseline CNN only, to check it agrees with Grad-CAM."
    )

    processed_image_baseline = PREPROCESSORS["Baseline CNN"](image)
    occlusion_heatmap, _ = compute_occlusion_heatmap(MODELS["Baseline CNN"], processed_image_baseline)

    display_image = image.convert("RGB").resize((224, 224))
    occlusion_overlay = overlay_heatmap_on_image(display_image, occlusion_heatmap)

    st.image(occlusion_overlay, width=300)
    st.caption("Baseline CNN - Occlusion Sensitivity")




































































