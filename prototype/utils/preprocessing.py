# # ============================================================
# # Author: S Rhithika
# # Project:
# # Deep Learning Breast Cancer Detection Using
# # Transfer Learning and CNN-Based Mammogram Classification
# 
# ============================================================
# Image Preprocessing
# ------------------------------------------------------------
# Resizes input images to 224×224, applies CLAHE using the
# prototype configuration (or defaults), then applies the
# preprocessing expected by each model architecture.
# ============================================================

import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from PIL import Image

IMAGE_SIZE = (224, 224)

# Resize with BILINEAR to match tf.image.resize's default method, used by
# every process_image*() function in the notebook (Sections 3.2.2, 4.2, 4.3,
# 4.4). PIL's own default resize filter is BICUBIC, not bilinear -- a silent
# mismatch here shifted pixel values slightly before CLAHE ever runs, which
# CLAHE's local-tile histogram equalisation can then amplify.

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "models" / "prototype_config.json"
_DEFAULT_CLAHE_CLIP_LIMIT = 2.0
_DEFAULT_CLAHE_TILE_GRID_SIZE = (8, 8)


def _load_clahe_settings():
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
        return cfg["clahe_clip_limit"], tuple(cfg["clahe_tile_grid_size"])
    print(f"! No prototype_config.json at {_CONFIG_PATH} -- using default CLAHE settings. "
          f"Run export_prototype_settings.py in the notebook to pin the real values.")
    return _DEFAULT_CLAHE_CLIP_LIMIT, _DEFAULT_CLAHE_TILE_GRID_SIZE


CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE = _load_clahe_settings()

# All architectures use the same resize and CLAHE steps before model-specific preprocessing.
def apply_clahe(image_float32_hwc):
    """Same operation as the notebook's apply_clahe: local contrast
    enhancement on a resized, RGB (R==G==B) image, values in [0, 255].
    image_float32_hwc: (224, 224, 3) numpy float32 array."""
    image_uint8 = np.clip(image_float32_hwc, 0, 255).astype(np.uint8)
    gray = image_uint8[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    enhanced = clahe.apply(gray)
    return np.stack([enhanced, enhanced, enhanced], axis=-1).astype(np.float32)


def preprocess_for_display(image):
    """Resize + CLAHE only, no architecture-specific normalisation -- shown
    to the user as 'what the model sees'. Stops here rather than also
    applying preprocess_baseline/vgg/resnet/efficientnet because those
    produce mean-centred, sometimes-negative values (VGG/ResNet) that
    aren't a displayable image anymore."""
    resized = image.convert("RGB").resize(IMAGE_SIZE, resample=Image.BILINEAR)
    array = np.array(resized).astype(np.float32)
    enhanced = apply_clahe(array)
    return Image.fromarray(enhanced.astype(np.uint8))


# The baseline model handles 0–255 to 0–1 scaling inside its own first layer.
def preprocess_baseline(image):
    # baseline_cnn__trial1.keras (build_baseline_cnn) has its own internal
    # Rescaling(1/255) as its first layer, trained on process_image's raw
    # 0-255 CLAHE output -- no separate preprocess_input
    image = image.convert("RGB").resize(IMAGE_SIZE, resample=Image.BILINEAR)
    image = np.array(image).astype(np.float32)
    image = apply_clahe(image)
    return np.expand_dims(image, axis=0)


def preprocess_vgg(image):
    image = image.convert("RGB").resize(IMAGE_SIZE, resample=Image.BILINEAR)
    image = np.array(image).astype(np.float32)
    image = apply_clahe(image)
    image = vgg_preprocess(image)
    return np.expand_dims(image, axis=0)


def preprocess_resnet(image):
    image = image.convert("RGB").resize(IMAGE_SIZE, resample=Image.BILINEAR)
    image = np.array(image).astype(np.float32)
    image = apply_clahe(image)
    image = resnet_preprocess(image)
    return np.expand_dims(image, axis=0)


def preprocess_efficientnet(image):
    # EfficientNet's preprocess_input is effectively a no-op (the model has
    # its own rescaling/normalisation built into its first layers and
    # expects raw 0-255 float input) -- kept for consistency with the other
    # two pipelines, same as the notebook's own prepare_gradcam_image.
    image = image.convert("RGB").resize(IMAGE_SIZE, resample=Image.BILINEAR)
    image = np.array(image).astype(np.float32)
    image = apply_clahe(image)
    image = efficientnet_preprocess(image)
    return np.expand_dims(image, axis=0)