# ============================================================
# Author: S Rhithika
# Project:
# Deep Learning Breast Cancer Detection Using
# Transfer Learning and CNN-Based Mammogram Classification
#
# Confidence calibration for the Streamlit prototype.
# Loads the correctors saved by the notebook and uses them to fix up
# the confidence numbers shown to the user.
# ============================================================

import pickle
from pathlib import Path

# Where the calibrators.pkl file lives -- the models folder, next to
# your .keras files.
CALIBRATORS_PATH = Path(__file__).resolve().parent.parent / "models" / "calibrators.pkl"

# Keeps the loaded file in memory after the first load, so it isn't
# re-read from disk every single time a prediction is made.
_calibrators = None


def load_calibrators():
    """
    Load the calibrators.pkl file once. If the file is missing for
    some reason, the app still works -- it just shows raw (uncorrected)
    confidence instead of crashing.
    """
    global _calibrators

    if _calibrators is not None:
        return _calibrators

    if CALIBRATORS_PATH.exists():
        with open(CALIBRATORS_PATH, "rb") as f:
            _calibrators = pickle.load(f)
        print(f"✓ Loaded calibrators for: {list(_calibrators.keys())}")
    else:
        print(f"!: No calibrators.pkl found at {CALIBRATORS_PATH} — showing raw confidence.")
        _calibrators = {}

    return _calibrators


def calibrate_probability(model_name, raw_probability):
    """
    Takes the model's raw output (e.g. 0.87) and returns a corrected
    version, using that model's calibrator. If no calibrator exists
    for this model, just returns the raw number unchanged.

    Returns two things: the (possibly corrected) probability, and
    True/False for whether correction was actually applied.
    """
    calibrators = load_calibrators()
    calibrator = calibrators.get(model_name)

    if calibrator is None:
        return raw_probability, False

    calibrated = float(calibrator.predict([raw_probability])[0])
    calibrated = min(max(calibrated, 0.0), 1.0)

    return calibrated, True