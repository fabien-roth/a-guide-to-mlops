
import numpy as np
import tensorflow as tf
from PIL.Image import Image
from typing import Tuple, Dict

def preprocess(x: Image, grayscale: bool, image_size: Tuple[int, int]) -> np.ndarray:
    """Convert PIL image to tensor for model inference."""
    x = x.convert('L' if grayscale else 'RGB')
    x = x.resize(image_size)
    x = np.array(x) / 255.0  # Normalize
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    return x

def postprocess(predictions: np.ndarray, labels: Dict[int, str]) -> Dict:
    """Convert model predictions to a readable format."""
    return {
        "prediction": labels[tf.argmax(predictions, axis=-1).numpy()[0]],
        "probabilities": {
            labels[i]: prob
            for i, prob in enumerate(tf.nn.softmax(predictions).numpy()[0].tolist())
        },
    }
            