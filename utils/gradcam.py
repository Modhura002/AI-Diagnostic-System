import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

brain_model = load_model(
    "models/brain_model.keras",
    safe_mode=False,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

def load_image(image_path):

    img = image.load_img(
        image_path,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    original = cv2.imread(image_path)

    return img_array, original

def compute_heatmap(img_array):

    base_model = brain_model.get_layer("resnet50")

    last_conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("conv5_block3_out").output
    )

    with tf.GradientTape() as tape:

        conv_output = last_conv_model(
            img_array,
            training=False
        )

        tape.watch(conv_output)

        x = brain_model.get_layer(
            "global_average_pooling2d_1"
        )(conv_output)

        x = brain_model.get_layer(
            "dense_2"
        )(x)

        x = brain_model.get_layer(
            "dropout_1"
        )(x, training=False)

        predictions = brain_model.get_layer(
            "dense_3"
        )(x)

        predicted_class = tf.argmax(predictions[0])

        class_score = predictions[:, predicted_class]

    grads = tape.gradient(
        class_score,
        conv_output
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(heatmap,0)

    heatmap /= (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()

def overlay_heatmap(heatmap, original, image_path):

    # Resize heatmap to match image
    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    # Convert to uint8
    heatmap_display = np.uint8(255 * heatmap)

    # Apply JET color map
    heatmap_color = cv2.applyColorMap(
        heatmap_display,
        cv2.COLORMAP_JET
    )

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.5,
        0
    )

    os.makedirs(
    "static/results",
    exist_ok=True
)

    # Save GradCAM image
    filename = os.path.basename(image_path)

    save_path = os.path.join(
        "static",
        "results",
        filename
    )

    cv2.imwrite(
        save_path,
        superimposed_img
    )

    return save_path

def generate_gradcam(image_path):

    img_array, original = load_image(
        image_path
    )

    heatmap = compute_heatmap(
        img_array
    )

    gradcam_path = overlay_heatmap(
        heatmap,
        original,
        image_path
    )

    return gradcam_path