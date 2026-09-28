# =============================================================================
# Section 5.6: Prototype verification against n=50 pipeline predictions
# =============================================================================
# Objective 7 requires the prototype's predictions to match the pipeline's
# on a sample of test images. This runs the PROTOTYPE'S OWN CODE (not a
# re-implementation of it, so nothing here can silently drift from what the
# app actually does) on 50 real test images and checks each prediction
# against POOLED -- the same seed-averaged numbers the report's own result
# tables are built from (Section 6.2).
#
# Note on what "match" means here: POOLED averages 3 trained seeds per
# model, while the prototype loads and runs a SINGLE checkpoint file. Exact
# agreement on every image is not guaranteed even with fully correct
# prototype code -- a few disagreements near the decision threshold are
# expected. The match RATE is the meaningful pass/fail signal, not 100%.
# If the match rate is unexpectedly low (e.g. well under 90%), the most
# likely cause is that the prototype's models/ folder is pointing at a
# different checkpoint than the one POOLED was built from (see
# model_loader.py's MODEL_FILES), or that models/thresholds.pkl was built
# from the pooled (3-seed) threshold instead of this single checkpoint's
# own Youden's-J threshold -- check both before assuming the code above is
# broken.
#
# Images come from pooled_export.pkl's embedded bytes (see
# export_pooled_for_verification.py), not from image_paths -- those paths
# are Colab paths (/content/dataset/...) and do not exist on the machine
# this script actually runs on.
#
# How to run: on the machine with the prototype folder (not Colab), with
# the prototype's own Python environment (needs the prototype's
# dependencies -- streamlit is imported by app.py but not needed here,
# only utils/*). Requires POOLED_EXPORT_PATH (below) pointing at a copy of
# pooled_export.pkl exported from the notebook (see
# export_pooled_for_verification.py) -- no other local files needed, since
# the sampled images are embedded in that pickle.

import os
import io
import sys
import pickle
import time
import numpy as np
import pandas as pd
from PIL import Image

# --- Configuration: adjust these two paths for wherever this actually runs ---
PROTOTYPE_DIR = "."          # folder containing app.py, utils/
POOLED_EXPORT_PATH = "pooled_export.pkl"  # see export_pooled_for_verification.py
N_CHECK = 50
RANDOM_SEED = 0

sys.path.insert(0, PROTOTYPE_DIR)
os.chdir(PROTOTYPE_DIR)  # utils/model_loader.py resolves MODELS_DIR relative to cwd's parent

from utils.model_loader import load_models, MODEL_FILES
from utils.predictor import PREPROCESSORS
from utils.thresholds import get_threshold
from utils.frozen_backbones import get_frozen_backbone, FROZEN_MODEL_NAMES

with open(POOLED_EXPORT_PATH, "rb") as f:
    pooled_export = pickle.load(f)
# pooled_export: {prototype_model_name: {"image_paths": [...] (reference only),
#                                         "image_bytes": [...], "y_true": [...],
#                                         "y_prob": [...], "y_pred": [...]}}

print(f"Loading prototype models (this loads all {len(MODEL_FILES)} .keras files)...")
models = load_models()

rng = np.random.default_rng(RANDOM_SEED)
rows = []

for model_name, model in models.items():
    if model_name not in pooled_export:
        print(f"skipping {model_name}: not in pooled_export")
        continue

    pooled = pooled_export[model_name]
    n_available = len(pooled["image_bytes"])
    check_indices = rng.choice(n_available, size=min(N_CHECK, n_available), replace=False)

    n_match = 0
    latencies_ms = []

    for idx in check_indices:
        # image_paths is kept in pooled_export.pkl for reference/debugging
        # only -- it's a Colab path and is never opened here.
        image_path = pooled["image_paths"][idx]
        pipeline_pred = int(pooled["y_pred"][idx])

        image = Image.open(io.BytesIO(pooled["image_bytes"][idx]))
        processed = PREPROCESSORS[model_name](image)

        start = time.perf_counter()

        # Frozen configs need their pooled ImageNet feature vector, not the
        # raw image, fed to model.predict() -- mirrors
        # utils.predictor.predict_all_models exactly, same as app.py.
        if model_name in FROZEN_MODEL_NAMES:
            backbone = get_frozen_backbone(model_name)
            model_input = backbone.predict(processed, verbose=0)
        else:
            model_input = processed

        raw_probability = float(model.predict(model_input, verbose=0)[0][0])
        threshold = get_threshold(model_name)
        prototype_pred = int(raw_probability >= threshold)
        latency_ms = (time.perf_counter() - start) * 1000

        latencies_ms.append(latency_ms)
        is_match = prototype_pred == pipeline_pred
        n_match += int(is_match)

        rows.append({
            "model": model_name, "image_path": image_path,
            "pipeline_pred": pipeline_pred, "prototype_pred": prototype_pred,
            "match": is_match, "latency_ms": round(latency_ms, 1),
        })

    n_checked = len(check_indices)
    print(f"{model_name}: {n_match}/{n_checked} matched pipeline prediction "
          f"({n_match / n_checked:.1%}), mean inference latency "
          f"{np.mean(latencies_ms):.1f} ms")

results_df = pd.DataFrame(rows)
results_df.to_csv("prototype_verification_results.csv", index=False)
print("\nSaved full per-image results to prototype_verification_results.csv")

summary = results_df.groupby("model").agg(
    n=("match", "size"),
    n_match=("match", "sum"),
    match_rate=("match", "mean"),
    mean_latency_ms=("latency_ms", "mean"),
).reset_index()
print("\nSummary:")
print(summary.to_string(index=False))
