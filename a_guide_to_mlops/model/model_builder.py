
import tensorflow as tf
from typing import Tuple

def get_model(image_shape: Tuple[int, int, int], conv_size: int, dense_size: int, output_classes: int) -> tf.keras.Model:
    """Creates a simple CNN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(conv_size, (3, 3), activation="relu", input_shape=image_shape),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_size, activation="relu"),
        tf.keras.layers.Dense(output_classes)
    ])
    return model
            