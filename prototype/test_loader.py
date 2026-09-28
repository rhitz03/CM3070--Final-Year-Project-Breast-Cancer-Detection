# ============================================================
# Test Script
# ------------------------------------------------------------
# Tests whether the application can locate and load all
# available trained models.
# ============================================================

from utils.model_loader import load_models


# Load every available model
models = load_models()


# Print loaded model names
print(models.keys())