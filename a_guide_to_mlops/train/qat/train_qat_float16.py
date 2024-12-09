import sys
from pathlib import Path
import json
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf2onnx
import bentoml
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

import numpy as np
import os

# Ajouter le répertoire principal du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model
from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, QAT_MODEL_FLOAT16_DIR  # Import paths from config

def main():
    # Command line argument parsing
    if len(sys.argv) == 3:
        prepared_dataset_folder = Path(sys.argv[1])
        model_folder = Path(sys.argv[2])
    else:
        print("Usage: python train_qat_float16.py <prepared-dataset-folder> <model-folder>")
        print("No arguments provided, using default paths from configuration.", flush=True)
        prepared_dataset_folder = PREPARED_DATA_DIR
        model_folder = QAT_MODEL_FLOAT16_DIR

    # Ensure the model folder exists
    model_folder.mkdir(parents=True, exist_ok=True)

    # Debug print statements for directories
    print(f"Prepared Dataset Folder Path: {prepared_dataset_folder}", flush=True)
    print(f"Model Folder Path: {model_folder}", flush=True)

    # Load parameters
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

    # Model parameters
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    # Set random seed for reproducibility
    set_seed(train_params["seed"])

    # Load training and validation datasets
    try:
        print("Loading datasets...", flush=True)
        ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
        ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))
        print("Datasets loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading datasets: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

    # Load labels
    labels_file_path = prepared_dataset_folder / "labels.json"
    print(f"Loading labels from {labels_file_path}", flush=True)

    if not labels_file_path.exists():
        print(f"Error: Labels file does not exist at {labels_file_path}", flush=True)
        exit(1)

    with open(labels_file_path) as f:
        labels = json.load(f)

    # Define model and apply QAT
    print("Defining model and applying QAT...", flush=True)
    model = get_model(image_shape, train_params["conv_size"], train_params["dense_size"], train_params["output_classes"])
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
    quantize_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # Compile the quantized model
    print("Compiling quantized model...", flush=True)
    quantize_model.compile(
        optimizer=tf.keras.optimizers.Adam(train_params["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    quantize_model.summary()

    # Train the quantized model
    print("Starting QAT training...", flush=True)
    try:
        history = quantize_model.fit(ds_train, epochs=train_params["epochs"], validation_data=ds_test)
    except Exception as e:
        print(f"Error during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

    # Save training history
    history_path = model_folder / "history.npy"
    try:
        with open(history_path, "wb") as f:
            np.save(f, history.history)
        print(f"Training history saved at {history_path}", flush=True)
    except Exception as e:
        print(f"Error saving training history: {e}", flush=True)
        exit(1)

    # Save the model in `.keras` format for backup
    keras_backup_path = model_folder / "celestial_bodies_classifier_model_qat_float16.keras"
    try:
        quantize_model.save(keras_backup_path)
        print(f"Model saved in Keras format at {keras_backup_path}", flush=True)
    except Exception as e:
        print(f"Error saving model in Keras format: {e}", flush=True)
        exit(1)

    # Convert the trained Keras model to ONNX
    try:
        print("Converting model to ONNX...", flush=True)
        spec = (tf.TensorSpec((None, *image_shape), tf.float32, name="input"),)
        onnx_output_path = model_folder / "celestial_bodies_classifier_model_qat_float16.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(quantize_model, input_signature=spec, opset=13, output_path=str(onnx_output_path))
        print(f"Model converted to ONNX and saved at {onnx_output_path}", flush=True)
    except Exception as e:
        print(f"Failed to convert model to ONNX. Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # Apply dynamic quantization to the ONNX model
    quantized_model_path = model_folder / "celestial_bodies_classifier_model_qat_float16_quantized.onnx"
    try:
        print("Applying dynamic quantization to ONNX model...", flush=True)
        quantize_dynamic(
            model_input=str(onnx_output_path),
            model_output=str(quantized_model_path),
            op_types_to_quantize=['Conv', 'MatMul'],
            weight_type=QuantType.QInt8
        )
        print(f"Quantized ONNX model saved at {quantized_model_path}", flush=True)
    except Exception as e:
        print(f"Failed to quantize the ONNX model. Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # Save the quantized ONNX model with BentoML
    bentoml_model_name = "celestial_bodies_classifier_qat_float16"
    try:
        print("Saving ONNX model with BentoML...", flush=True)
        quantized_model = onnx.load(str(quantized_model_path))
        bentoml.onnx.save_model(
            bentoml_model_name,
            quantized_model,
            signatures={"run": {"batchable": True}},
            labels={"model": "quantized", "framework": "onnx"},
            metadata={"description": "Quantized ONNX model of celestial bodies classifier"}
        )
        print("Quantized ONNX model successfully saved to BentoML store.", flush=True)

        # Export the BentoML model for deployment
        bentoml.models.export_model(
            f"{bentoml_model_name}:latest",
            str(model_folder / "celestial_bodies_classifier_model.bentomodel")
        )
        print(f"BentoML model exported at {model_folder}", flush=True)
    except Exception as e:
        print(f"Failed to save model with BentoML. Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
