# ============================================================
# Author: S Rhithika
# Project:
# Deep Learning Breast Cancer Detection Using
# Transfer Learning and CNN-Based Mammogram Classification
#
# Occlusion sensitivity -- a second, gradient-free explainability
# method used as a cross-check against Grad-CAM. Instead of using
# gradients, small patches of the image are covered up one at a
# time, and the drop in the model's predicted probability is
# measured. A bigger drop means that patch mattered more.
# ============================================================

import numpy as np


def compute_occlusion_heatmap(model, processed_image, patch_size=32, stride=32):
    # Covers small squares of the image one at a time and checks
    # how much the model's prediction changes. Bigger change = that
    # square mattered more. Returns a 224x224 heatmap of that.

    image_size = processed_image.shape[1]  # 224

    # Model's original prediction, before covering anything up
    baseline_probability = float(model.predict(processed_image, verbose=0)[0][0])

    # Colour used to cover a patch: this image's own average pixel value
    occlusion_value = float(np.mean(processed_image))

    heatmap = np.zeros((image_size, image_size), dtype=np.float32)

    # Slide the patch across the image
    for top in range(0, image_size - patch_size + 1, stride):
        for left in range(0, image_size - patch_size + 1, stride):

            # Cover this patch on a copy of the image
            occluded_image = processed_image.copy()
            occluded_image[0, top:top + patch_size, left:left + patch_size, :] = occlusion_value

            # Re-run the model with the patch covered
            occluded_probability = float(model.predict(occluded_image, verbose=0)[0][0])

            # How much the prediction changed because of covering it
            # Store the size of the prediction change caused by hiding this patch, regardless of direction.
            probability_drop = abs(baseline_probability - occluded_probability)

            heatmap[top:top + patch_size, left:left + patch_size] = probability_drop

    # Scale to 0-1 so it can be coloured like the Grad-CAM heatmaps
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap, baseline_probability