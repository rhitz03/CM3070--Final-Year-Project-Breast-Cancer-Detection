# ============================================================
# Frozen ImageNet Backbones
# ------------------------------------------------------------
# Frozen-model checkpoints contain trained classifier heads
# that expect pooled ImageNet features. This module rebuilds
# and caches the matching frozen feature extractors.
# ============================================================
import threading
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0

_BASE_CLASS = {
    "VGG16 Frozen": VGG16,
    "ResNet50 Frozen": ResNet50,
    "EfficientNetB0 Frozen": EfficientNetB0,
}

FROZEN_MODEL_NAMES = set(_BASE_CLASS.keys())

_cache = {}
# Streamlit can run more than one script execution concurrently (e.g. two
# reruns in flight at once). Without a lock, two threads can both see an
# empty _cache and both try to build/download the same backbone at the
# same time -- on Windows this is a file-write race on the same ImageNet
# weights file and raises PermissionError. This lock makes only one
# thread build each backbone; the rest wait, then reuse the cached result.
_lock = threading.Lock()


def get_frozen_backbone(model_name):
    """Cached (built once, reused) frozen ImageNet backbone with
    pooling='avg' for this Frozen model name. Thread-safe."""
    if model_name in _cache:
        return _cache[model_name]
    with _lock:
        if model_name not in _cache:  # re-check: another thread may have built it while we waited
            base_class = _BASE_CLASS[model_name]
            # Rebuild the frozen ImageNet feature extractor because the saved checkpoint contains only its trained head.
            base = base_class(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))
            base.trainable = False
            _cache[model_name] = base
    return _cache[model_name]