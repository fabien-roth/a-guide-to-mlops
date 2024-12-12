import json
import sys
from pathlib import Path
from typing import Tuple
import numpy as np
import tensorflow as tf
import yaml
import bentoml
from PIL.Image import Image
from a_guide_to_mlops.utils.seed import set_seed
import os
import time
import psutil
import tensorflow.lite as tflite
from a_guide_to_mlops.utils.config import EVALUATION_BASELINE_DIR, EVALUATION_PTQ_DIR, EVALUATION_QAT_DIR

def get_output_dir(model_folder: Path) -> Path:
    """
    Dynamically determine the evaluation output directory based on the model folder.
    """
    if "baseline" in model_folder.parts:
        return EVALUATION_BASELINE_DIR
    elif "ptq" in model_folder.parts:
        return EVALUATION_PTQ_DIR / model_folder.parts[-1]
    elif "qat" in model_folder.parts:
        return EVALUATION_QAT_DIR / model_folder.parts[-1]
    else:
        raise ValueError(f"Unknown model type in path: {model_folder}")

def apply_qat(model: tf.keras.Model):

    os.environ["TF_KERAS"] = "1"

    try:
        import tensorflow_model_optimization as tfmot
        print("✅ TensorFlow Model Optimization imported successfully!")
    except ImportError as e:
        raise ImportError(
            "⚠️ TensorFlow Model Optimization toolkit is not installed or incompatible. "
            "Please install with `pip install tensorflow-model-optimization`."
        ) from e

    # Apply QAT
    quantize_model = tfmot.quantization.keras.quantize_model
    qat_model = quantize_model(model)
    print("🔥 QAT model prepared successfully.")
    return qat_model

import bentoml
import tensorflow as tf
import json
from pathlib import Path


def quantize_model(model, model_folder, quantization_type="DYNAMIC", is_qat=False, ds_train=None):
    from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR

    prepared_dataset_folder = PREPARED_DATA_DIR
    labels_file_path = prepared_dataset_folder / "labels.json"
    print(f"Loading labels from {labels_file_path}", flush=True)

    if not labels_file_path.exists():
        print(f"Error: Labels file does not exist at {labels_file_path}", flush=True)
        exit(1)

    with open(labels_file_path) as f:
        labels = json.load(f)

    # Create a TFLiteConverter from the Keras model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization_type.upper() == "INT8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if is_qat:
            print("⚡ Using QAT logic for INT8 quantization.")
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            if ds_train:
                def representative_dataset():
                    for images, _ in ds_train.take(200):
                        for image in images:
                            yield [np.expand_dims(image.numpy(), axis=0)]

                converter.representative_dataset = representative_dataset
        else:
            print("⚡ Using PTQ logic for INT8 quantization.")
            if ds_train is None:
                raise ValueError("A training dataset is required for PTQ INT8 quantization.")

            def representative_dataset():
                for images, _ in ds_train.take(100):
                    for image in images:
                        yield [np.expand_dims(image.numpy(), axis=0)]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    elif quantization_type.upper() == "FLOAT16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("⚡ Using FLOAT16 quantization.")

    elif quantization_type.upper() == "DYNAMIC":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("⚡ Using DYNAMIC quantization.")

    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")

    try:
        # Convert the model
        quantized_model = converter.convert()

    except Exception as e:
        raise RuntimeError(f"Quantization failed for {quantization_type}: {e}")

    quantized_model_path = model_folder / f"quantized_model.tflite"
    with open(quantized_model_path, "wb") as f:
        f.write(quantized_model)

    print(f"✅ Quantized model ({quantization_type}) saved at: {quantized_model_path}")
    return quantized_model_path


def get_memory_consumption():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

def evaluate_quantized_model(quantized_model_path, ds_test, labels, quantization_type="INT8"):

    interpreter = tflite.Interpreter(model_path=str(quantized_model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]

    true_labels = []
    predictions = []
    latencies = []

    for images, label_idxs in ds_test:
        for image, true_label in zip(images, label_idxs):
            true_labels.append(true_label.numpy())

            # Preprocess image
            input_data = np.expand_dims(image.numpy(), axis=0).astype(input_dtype)
            if quantization_type.upper() == "INT8":
                input_data = (input_data * 127.5 - 1).astype(np.int8)  # Normalize for INT8

            start_time = time.time()
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            latencies.append((time.time() - start_time) * 1000)  # Convert to ms

            # Get predictions
            output_data = interpreter.get_tensor(output_details[0]["index"])
            predictions.append(np.argmax(output_data))

    overall_accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)

    class_accuracies = {}

    for idx, label in enumerate(labels):
        true_positives = sum(1 for p, t in zip(predictions, true_labels) if p == t == idx)
        total_true = sum(1 for t in true_labels if t == idx)
        class_accuracies[label] = true_positives / total_true if total_true > 0 else 0.0

    average_latency_ms = np.mean(latencies)

    print ("⚙️ model metrics calculation done succeffuly.")

    return overall_accuracy, class_accuracies, average_latency_ms


def evaluate_base_model(model_path_or_tag, ds_test, labels):

    # Check if the input is a path to a `.keras` model or a BentoML tag
    
    #if model_path_or_tag.endswith(".keras"):
    print(f"⚙️ Loading model from Keras file: {model_path_or_tag}", flush=True)
    model = tf.keras.models.load_model(model_path_or_tag)
    '''
    else:
    
        print(f"⚙️ Loading model from BentoML: {model_path_or_tag}", flush=True)
        model_name = os.path.basename(model_path_or_tag).split(".")[0]  # Remove extension
        model_tag = f"{model_name}:latest"
        model = bentoml.keras.load_model(str(model_tag))
    '''
    
    true_labels = []
    predictions = []
    latencies = []

    # Iterate through test dataset
    for images, label_idxs in ds_test:
        for image, true_label in zip(images, label_idxs):
            true_labels.append(true_label.numpy())

            # Preprocess image for model (assuming float32 input)
            input_data = np.expand_dims(image.numpy(), axis=0).astype(np.float32)

            # Measure latency
            start_time = time.time()
            output_data = model(input_data)
            latencies.append((time.time() - start_time) * 1000)  # Convert to ms

            # Get predictions
            predictions.append(np.argmax(output_data.numpy()))

    # Calculate overall accuracy
    overall_accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(true_labels)

    # Calculate per-class accuracies
    class_accuracies = {}
    for idx, label in enumerate(labels):
        true_positives = sum(1 for p, t in zip(predictions, true_labels) if p == t == idx)
        total_true = sum(1 for t in true_labels if t == idx)
        class_accuracies[label] = true_positives / total_true if total_true > 0 else 0.0

    # Calculate average latency
    average_latency_ms = np.mean(latencies)

    print("⚙️ Model metrics calculation completed successfully.")
    return overall_accuracy, class_accuracies, average_latency_ms

