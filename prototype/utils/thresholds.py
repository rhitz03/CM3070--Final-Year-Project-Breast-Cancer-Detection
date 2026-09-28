# ============================================================
# Per-Model Decision Thresholds
# ------------------------------------------------------------
# Loads each model's classification cutoff from thresholds.pkl.
# If the file or a model's entry is missing, get_threshold()
# falls back to 0.5.
# ============================================================

import pickle
from pathlib import Path

THRESHOLDS_PATH = Path(__file__).resolve().parent.parent / "models" / "thresholds.pkl"

_thresholds = None


def load_thresholds():
    global _thresholds
    if _thresholds is not None:
        return _thresholds

    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH, "rb") as f:
            _thresholds = pickle.load(f)
        print(f"Loaded per-model thresholds for: {list(_thresholds.keys())}")
    else:
        print(f"! No thresholds.pkl found at {THRESHOLDS_PATH} -- falling back to 0.5 for "
              f"every model. Run export_prototype_settings.py in the notebook to fix this.")
        _thresholds = {}

    return _thresholds


def get_threshold(model_name):
    """This model's own checkpoint's Youden's-J threshold (validation-set,
    single seed -- see fix_thresholds_correct.py), or 0.5 if none was exported."""
    thresholds = load_thresholds()
    return thresholds.get(model_name, 0.5)