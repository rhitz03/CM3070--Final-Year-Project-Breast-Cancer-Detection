# # ============================================================
# # Author: S Rhithika
# # Project:
# # Deep Learning Breast Cancer Detection Using
# # Transfer Learning and CNN-Based Mammogram Classification
# 
# ============================================================
# Grad-CAM Explainability
# ------------------------------------------------------------
# Builds a heatmap from the predicted class's gradient at a
# selected convolutional layer, then provides helpers to resize
# and overlay the heatmap for display.
# ============================================================





import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from PIL import Image


def find_last_conv_layer(model):
    """Final feature layer for Grad-CAM: conv5_block3_out for ResNet50,
    else the last Conv2D layer found (nested model or direct)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            if "resnet" in layer.name.lower():
                for nested_layer in layer.layers:
                    if nested_layer.name == "conv5_block3_out":
                        return nested_layer
            for nested_layer in reversed(layer.layers):
                if isinstance(nested_layer, Conv2D):
                    return nested_layer
        if isinstance(layer, Conv2D):
            return layer
    raise ValueError(f"No suitable Grad-CAM target layer found in model: {model.name}")


def _logit_layer(final_dense):
    """final_dense's own weights on a plain Dense(1, activation=None), i.e.
    the same layer minus its fused sigmoid. A proper Keras layer, not a raw
    tf.matmul, since a bare tf.matmul on a symbolic KerasTensor errors out
    while building a functional-API graph."""
    layer = tf.keras.layers.Dense(1, activation=None, name=f"{final_dense.name}_logit")
    layer.build((None, final_dense.kernel.shape[0]))
    layer.set_weights(final_dense.get_weights())
    return layer


def generate_gradcam(model, image, target_layer, model_name, threshold):
    """
    Generate a Grad-CAM heatmap for a single image.

    Parameters
    ----------
    model : tf.keras.Model, trained classification model.
    image : tf.Tensor, preprocessed image, shape (1, 224, 224, 3).
    target_layer : tf.keras.layers.Layer, from find_last_conv_layer.
    model_name : str, must be "Baseline CNN" for the from-scratch model,
        anything else is treated as a nested-pretrained-backbone model.
    threshold : float, this model's own fixed decision threshold (not 0.5).

    Returns
    -------
    heatmap : np.ndarray, normalised Grad-CAM heatmap.
    prediction_probability : float, sigmoid output (malignant probability).
    predicted_class : int, 0 = benign, 1 = malignant.
    """
    final_dense = model.layers[-1]
    logit_layer = _logit_layer(final_dense)

    if model_name == "Baseline CNN":
        gradcam_input = tf.keras.Input(shape=(224, 224, 3))
        x = gradcam_input
        target_activation = None
        for layer in model.layers[:-1]:
            x = layer(x, training=False)
            if layer.name == target_layer.name:
                target_activation = x
        if target_activation is None:
            raise ValueError(f"Target layer '{target_layer.name}' was not found in the Baseline CNN.")
        logit = logit_layer(x)
        grad_model = tf.keras.Model(inputs=gradcam_input, outputs=[target_activation, logit])
    else:
        nested_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                nested_layer_names = [nested_layer.name for nested_layer in layer.layers]
                if target_layer.name in nested_layer_names:
                    nested_model = layer
                    break
        if nested_model is None:
            raise ValueError(f"Could not find the pretrained model containing '{target_layer.name}'.")
        nested_grad_model = tf.keras.Model(
            inputs=nested_model.input, outputs=[target_layer.output, nested_model.output]
        )
        gradcam_input = tf.keras.Input(shape=(224, 224, 3))
        conv_outputs, x = nested_grad_model(gradcam_input, training=False)
        for layer in model.layers[:-1]:
            if layer is nested_model:
                continue
            x = layer(x, training=False)
        logit = logit_layer(x)
        grad_model = tf.keras.Model(inputs=gradcam_input, outputs=[conv_outputs, logit])

    with tf.GradientTape() as tape:
        conv_outputs, logit = grad_model(image, training=False)
        logit = logit[:, 0]
        prediction_probability = tf.sigmoid(logit)  # for reporting/thresholding, not differentiated
        predicted_class = tf.cast(prediction_probability >= threshold, tf.int32)
        # Evidence for the PREDICTED class, in logit space: +logit if
        # predicted malignant, -logit if predicted benign. No sigmoid
        # saturation, unlike differentiating the probability directly.
        # Differentiate the predicted class's logit to avoid sigmoid saturation in the gradients.
        # Choose the logit direction for the predicted class so Grad-CAM highlights evidence supporting that prediction.
        class_score = tf.where(predicted_class == 1, logit, -logit)

    gradients = tape.gradient(class_score, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]
    weighted_activations = conv_outputs * pooled_gradients
    heatmap = tf.reduce_sum(weighted_activations, axis=-1)
    # Remove negative Grad-CAM values; if all values are negative, the resulting heatmap is blank.
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
    heatmap = heatmap.numpy()
    prediction_probability = float(prediction_probability[0].numpy())
    predicted_class = int(predicted_class[0].numpy())
    return heatmap, prediction_probability, predicted_class


def resize_gradcam_heatmap(heatmap, target_size=(224, 224)):
    """Resize a Grad-CAM heatmap to the display size. Unchanged from the
    current file -- no bug here."""
    heatmap_tensor = tf.convert_to_tensor(heatmap[..., np.newaxis], dtype=tf.float32)
    resized_heatmap = tf.image.resize(heatmap_tensor, target_size)
    return resized_heatmap[..., 0].numpy()


def overlay_heatmap_on_image(original_image, heatmap, alpha=0.4):
    """Blend a normalised Grad-CAM heatmap onto the original image.
    Unchanged from the current file -- no bug here."""
    heatmap_rgb = np.zeros((*heatmap.shape, 3), dtype=np.float32)
    heatmap_rgb[..., 0] = np.clip(1.5 * heatmap, 0, 1)
    heatmap_rgb[..., 1] = np.clip(1.5 * heatmap - 0.5, 0, 1)
    heatmap_rgb[..., 2] = np.clip(1.5 * heatmap - 1.0, 0, 1)
    heatmap_rgb = (heatmap_rgb * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_rgb).convert("RGB")
    return Image.blend(original_image.convert("RGB"), heatmap_image, alpha)