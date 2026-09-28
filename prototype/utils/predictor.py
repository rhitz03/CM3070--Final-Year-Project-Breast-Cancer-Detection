# # ============================================================
# # Author: S Rhithika
# # Project:
#============================================================
# Model Prediction
# ------------------------------------------------------------
# Preprocesses an image for each loaded model, obtains its raw
# malignant probability, and applies that model's threshold.
# Available calibrators adjust the displayed confidence.
# ============================================================





from utils.preprocessing import (
    preprocess_baseline,
    preprocess_vgg,
    preprocess_resnet,
    preprocess_efficientnet
)
from utils.calibration import calibrate_probability
from utils.thresholds import get_threshold
from utils.frozen_backbones import get_frozen_backbone, FROZEN_MODEL_NAMES

BENIGN = "Benign"
MALIGNANT = "Malignant"

PREPROCESSORS = {
    "Baseline CNN": preprocess_baseline,
    "VGG16 Frozen": preprocess_vgg,
    "VGG16 Partial": preprocess_vgg,
    "ResNet50 Frozen": preprocess_resnet,
    "ResNet50 Partial": preprocess_resnet,
    "EfficientNetB0 Frozen": preprocess_efficientnet,
    "EfficientNetB0 Partial": preprocess_efficientnet,
}


def predict_all_models(image, models):
    results = {}

    for model_name, model in models.items():
        processed_image = PREPROCESSORS[model_name](image)

        if model_name in FROZEN_MODEL_NAMES:
            # This checkpoint is head-only (see frozen_backbones.py)
            # extract pooled ImageNet features first, same as training.
            # Frozen checkpoints contain only the classifier head, so convert the image to backbone features first.
            backbone = get_frozen_backbone(model_name)
            model_input = backbone.predict(processed_image, verbose=0)
        else:
            model_input = processed_image

        raw_probability = float(
            model.predict(model_input, verbose=0)[0][0]
        )

        # The raw probability determines the class; calibration only adjusts the displayed confidence.
        # This model's own fixed threshold, not a shared 0.5.
        threshold = get_threshold(model_name)

        # Use this model's validation-selected threshold to turn its raw malignant probability into a label.
        if raw_probability >= threshold:
            prediction = MALIGNANT
        else:
            prediction = BENIGN

        # Calibration adjusts the probability shown as confidence; it does not change the label decision above.
        calibrated_probability, was_calibrated = calibrate_probability(
            model_name, raw_probability
        )

        confidence = (
            calibrated_probability if raw_probability >= threshold
            else 1 - calibrated_probability
        )

        results[model_name] = {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "calibrated": was_calibrated,
            "threshold": threshold,
        }

    return results


def generate_prediction_summary(results):
    prediction_counts = {}
    disagreements = []

    for result in results.values():
        prediction = result["prediction"]
        prediction_counts[prediction] = prediction_counts.get(prediction, 0) + 1

    majority_prediction = max(prediction_counts, key=prediction_counts.get)

    for model_name, result in results.items():
        if result["prediction"] != majority_prediction:
            disagreements.append(model_name)

    return {
        "majority_prediction": majority_prediction,
        "agreement": f"{prediction_counts[majority_prediction]} / {len(results)}",
        "disagreements": disagreements,
    }


DISPLAY_NAMES = {
    "Baseline CNN": "Baseline CNN (Scratch)",
    "VGG16 Frozen": "VGG16 (Frozen)",
    "VGG16 Partial": "VGG16 (Partial Fine-Tuning)",
    "ResNet50 Frozen": "ResNet50 (Frozen)",
    "ResNet50 Partial": "ResNet50 (Partial Fine-Tuning)",
    "EfficientNetB0 Frozen": "EfficientNetB0 (Frozen)",
    "EfficientNetB0 Partial": "EfficientNetB0 (Partial Fine-Tuning)",
}


def format_results_table(results):
    table = []
    for model_name, result in results.items():
        table.append([
            DISPLAY_NAMES.get(model_name, model_name),
            result["prediction"],
            f'{result["confidence"]:.1f}%',
            "Yes" if result["calibrated"] else "No",
            f'{result["threshold"]:.3f}',
        ])
    return table