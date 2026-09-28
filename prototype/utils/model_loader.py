# ============================================================
# Model Loader
#============================================================
# 
# Loads the configured CNN checkpoints from the models folder.
# Missing checkpoints are skipped; the app can use the models
# that were successfully loaded.
# ============================================================

from pathlib import Path
from tensorflow.keras.models import load_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_FILES = {
    "Baseline CNN": "baseline_cnn__trial1.keras",
    "VGG16 Frozen": "vgg16_frozen__trial1.keras",
    "VGG16 Partial": "vgg16_partial__trial1.keras",
    "ResNet50 Frozen": "resnet50_frozen__trial3.keras",
    "ResNet50 Partial": "resnet50_partial__trial3.keras",
    "EfficientNetB0 Frozen": "effnet_frozen__trial9.keras",
    "EfficientNetB0 Partial": "effnet_partial__trial4.keras",
}


def load_models():
    loaded_models = {}
    print("\nLoading trained models...\n")

    for model_name, filename in MODEL_FILES.items():
        model_path = MODELS_DIR / filename

        if model_path.exists():
            loaded_models[model_name] = load_model(model_path)
            print(f"OK {model_name}")
        else:
            print(f"! {model_name} not found (skipped) -- expected {model_path}")

    print(f"\nSuccessfully loaded {len(loaded_models)} model(s).\n")
    return loaded_models