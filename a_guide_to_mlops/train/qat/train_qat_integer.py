import sys
from pathlib import Path
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf2onnx
import bentoml
import json
import numpy as np
import onnx
import os

# Ajouter le répertoire principal du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model
from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, QAT_MODEL_INTEGER_DIR

def main():
    # Check command-line arguments for dataset and model output paths
    if len(sys.argv) == 3:
        prepared_dataset_folder = Path(sys.argv[1])
        model_folder = Path(sys.argv[2])
    else:
        print("Usage: python train_qat_integer.py <prepared-dataset-folder> <model-folder>")
        print("No arguments provided, falling back to default paths from configuration.", flush=True)

        # Fallback to config paths if no arguments are provided
        prepared_dataset_folder = PREPARED_DATA_DIR
        model_folder = QAT_MODEL_INTEGER_DIR

    # Load parameters from the configuration file
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
        exit(1)

    # Load labels
    labels = None
    try:
        with open(prepared_dataset_folder / "labels.json") as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"Error: labels.json not found at {prepared_dataset_folder}", flush=True)
        exit(1)

    # Define model
    print("Defining model...", flush=True)
    base_model = get_model(image_shape, train_params["conv_size"], train_params["dense_size"], train_params["output_classes"])

    # Annotate and apply QAT
    print("Annotating model for QAT and applying quantization...", flush=True)
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(base_model)
    quantize_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # Compile the quantized model
    print("Compiling quantized model...", flush=True)
    quantize_model.compile(
        optimizer=tf.keras.optimizers.Adam(train_params["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # Train the quantized model
    print("Starting QAT model training...", flush=True)
    history = quantize_model.fit(ds_train, epochs=train_params["epochs"], validation_data=ds_test)

    # Save the quantized model in `.keras` format for backup
    model_folder.mkdir(parents=True, exist_ok=True)
    keras_backup_path = model_folder / "celestial_bodies_classifier_qat_integer.keras"
    print("Saving model in Keras format...", flush=True)
    try:
        quantize_model.save(keras_backup_path)
        print(f"Model saved in Keras format at {keras_backup_path}", flush=True)
    except Exception as e:
        print(f"Error saving model in Keras format: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Convert the quantized model to ONNX format
    print("Converting quantized model to ONNX...", flush=True)
    try:
        spec = (tf.TensorSpec((None, *image_shape), tf.float32, name="input"),)
        onnx_output_path = model_folder / "celestial_bodies_classifier_qat_integer.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(quantize_model, input_signature=spec, opset=13, output_path=str(onnx_output_path))
        print(f"Model converted to ONNX and saved at {onnx_output_path}", flush=True)
    except Exception as e:
        print(f"Failed to convert model to ONNX: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    # Save the ONNX model with BentoML using the unified name
    print("Saving the ONNX model with BentoML...", flush=True)
    try:
        onnx_model = onnx.load(str(onnx_output_path))
        bentoml_model_name = "celestial_bodies_classifier_model"  # Unified name for consistency across all variants
        bentoml.onnx.save_model(
            bentoml_model_name,
            onnx_model,
            labels={"model": "quantized", "framework": "onnx"},
            metadata={"description": "Quantized ONNX model of celestial bodies classifier"}
        )
        # Export the BentoML model for deployment
        export_path = model_folder / "celestial_bodies_classifier_model.bentomodel"
        bentoml.models.export_model(
            f"{bentoml_model_name}:latest",
            str(export_path)
        )
        print(f"BentoML model exported successfully to {export_path}", flush=True)
    except Exception as e:
        print(f"Failed to save ONNX model with BentoML: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Save model training history for evaluation purposes
    history_path = model_folder / "history.npy"
    try:
        with open(history_path, "wb") as f:
            np.save(f, history.history)
        print(f"Model training history saved at {history_path}", flush=True)
    except Exception as e:
        print(f"Error saving training history: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}", flush=True)
        import traceback
        traceback.print_exc()
