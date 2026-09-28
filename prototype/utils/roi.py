# ============================================================
# Author: S Rhithika
# Project:
# Deep Learning Breast Cancer Detection Using
# Transfer Learning and CNN-Based Mammogram Classification
#
# Live Grad-CAM vs expert ROI overlap, for the six bundled demo cases.
# Ported from the training notebook (Sections 6.6/6.7) -- this only
# works for the demo cases, since it needs the expert-annotated mask
# file that only exists for images already in the CBIS-DDSM dataset.
# ============================================================

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from PIL import Image

# demo_reference.xls maps each demo case's filename to its original
# path in your dataset, and its true label.
DEMO_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "models" / "demo_reference.xls"


def load_demo_manifest():
    # Despite the .xls extension, this file is actually plain CSV --
    # checked directly, so it's read as CSV here rather than as Excel.
    manifest = pd.read_csv(DEMO_REFERENCE_PATH)
    return manifest.set_index("filename").to_dict(orient="index")


def get_roi_mask_path(image_path):
    """
    Each case's folder in your dataset has exactly two JPEGs: the
    lesion crop, and the expert ROI mask. The mask is reliably the
    larger of the two files.
    """
    folder_path = os.path.dirname(image_path)
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

    if len(jpg_files) != 2:
        raise ValueError(f"Expected two JPEG files in {folder_path}, found {len(jpg_files)}.")

    image_sizes = []
    for file_name in jpg_files:
        with Image.open(os.path.join(folder_path, file_name)) as img:
            image_sizes.append((img.size[0] * img.size[1], file_name))
    
    # For these demo folders, the ROI mask is identified as the larger of the two JPEG files.
    roi_file = max(image_sizes, key=lambda item: item[0])[1]
    return os.path.join(folder_path, roi_file)

# Place the cropped ROI outline at the center of a canvas matching the lesion crop's dimensions.
def prepare_aligned_roi(roi_path, crop_size, target_size=(224, 224)):
    """
    The ROI mask and the lesion crop don't share a coordinate system
    by default. This places the mask's lesion outline centred inside
    a blank canvas the same size as the crop, so it lines up properly
    with the Grad-CAM heatmap.
    """
    crop_width, crop_height = crop_size

    roi_array = np.array(Image.open(roi_path).convert("L"))
    non_black = roi_array > 0

    if not np.any(non_black):
        raise ValueError(f"No ROI pixels found in: {roi_path}")

    rows, columns = np.where(non_black)
    y_min, y_max = rows.min(), rows.max() + 1
    x_min, x_max = columns.min(), columns.max() + 1

    roi_bbox = (roi_array[y_min:y_max, x_min:x_max] > 0).astype(np.float32)
    roi_height, roi_width = roi_bbox.shape

    if roi_width > crop_width or roi_height > crop_height:
        raise ValueError("ROI bounding box is larger than the corresponding lesion crop.")

    aligned_roi = np.zeros((crop_height, crop_width), dtype=np.float32)
    x_offset = (crop_width - roi_width) // 2
    y_offset = (crop_height - roi_height) // 2
    aligned_roi[y_offset:y_offset + roi_height, x_offset:x_offset + roi_width] = roi_bbox

    aligned_roi = tf.image.resize(aligned_roi[..., np.newaxis], target_size, method="nearest")
    return aligned_roi[..., 0].numpy()


def compute_iou(heatmap, roi_binary, threshold=0.5):
    """
    The actual overlap score: what fraction of "heatmap says important
    here" and "expert says the lesion is here" overlap, out of their
    combined area. Same definition as Table 5.7 in your report.
    """
    heatmap_binary = (heatmap >= threshold).astype(np.float32)
    intersection = np.logical_and(heatmap_binary, roi_binary).sum()
    union = np.logical_or(heatmap_binary, roi_binary).sum()
    return 0.0 if union == 0 else intersection / union