# Import statements
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
from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, QAT_MODEL_DYNAMIC_DIR
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model

def main():
    print("Script started...", flush=True)

    # Load parameters
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

    # Set paths using config.py
    prepared_dataset_folder = PREPARED_DATA_DIR
    model_folder = QAT_MODEL_DYNAMIC_DIR
    model_folder.mkdir(parents=True, exist_ok=True)

    # Debug print statements for directories
    print(f"Prepared Dataset Folder Path: {prepared_dataset_folder}", flush=True)
    print(f"Model Folder Path: {model_folder}", flush=True)

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
    try:
        labels_file_path = prepared_dataset_folder / "labels.json"
        with open(labels_file_path) as f:
            labels = json.load(f)
        print(f"Labels loaded from {labels_file_path}", flush=True)
    except Exception as e:
        print(f"Error loading labels file: {e}", flush=True)
        exit(1)

    # Define and annotate the model for quantization
    try:
        print("Defining model and applying QAT...", flush=True)
        model = get_model(image_shape, train_params["conv_size"], train_params["dense_size"], train_params["output_classes"])
        annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
        quantize_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    except Exception as e:
        print(f"Error defining/annotating model: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

    # Compile the quantized model
    try:
        quantize_model.compile(
            optimizer=tf.keras.optimizers.Adam(train_params["lr"]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        quantize_model.summary()
    except Exception as e:
        print(f"Error compiling model: {e}", flush=True)
        exit(1)

    # Train the quantized model
    try:
        print("Starting QAT training...", flush=True)
        history = quantize_model.fit(ds_train, epochs=train_params["epochs"], validation_data=ds_test)
    except Exception as e:
        print(f"Error during training: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

    # Save the training history as `.npy` file
    history_path = model_folder / "history.npy"
    try:
        with open(history_path, "wb") as f:
            np.save(f, history.history)
        print(f"Training history saved at {history_path}", flush=True)
    except Exception as e:
        print(f"Error saving training history: {e}", flush=True)
        exit(1)

    # Save the model in `.keras` format for backup
    keras_backup_path = model_folder / "celestial_bodies_classifier_model_qat_dynamic.keras"
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
        onnx_output_path = model_folder / "celestial_bodies_classifier_model_qat_dynamic.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(quantize_model, input_signature=spec, opset=13, output_path=str(onnx_output_path))
        print(f"Model converted to ONNX and saved at {onnx_output_path}", flush=True)
    except Exception as e:
        print("Failed to convert model to ONNX. Error:", str(e), flush=True)
        return

    # Quantize the ONNX model using ONNX Runtime
    try:
        print("Applying dynamic quantization to ONNX model...", flush=True)
        quantized_model_path = model_folder / "celestial_bodies_classifier_model_qat_dynamic_quantized.onnx"
        quantize_dynamic(model_input=str(onnx_output_path),
                         model_output=str(quantized_model_path),
                         op_types_to_quantize=['Conv', 'MatMul'],
                         weight_type=QuantType.QInt8)
        print(f"Quantized ONNX model saved at {quantized_model_path}", flush=True)
    except Exception as e:
        print("Failed to quantize the ONNX model. Error:", str(e), flush=True)
        import traceback
        traceback.print_exc()
        return

    # Load and save the quantized ONNX model with BentoML
    try:
        print("Saving ONNX model with BentoML...", flush=True)
        quantized_model = onnx.load(str(quantized_model_path))
        bentoml_model_name = "celestial_bodies_classifier_qat_dynamic"  # Unique name for tracking the specific variant
        bentoml.onnx.save_model(
            bentoml_model_name,
            quantized_model,
            signatures={"run": {"batchable": True}},
            labels={"model": "quantized", "framework": "onnx"},
            metadata={"description": "Quantized ONNX model of celestial bodies classifier"}
        )
        print("Quantized ONNX model successfully saved to BentoML store.", flush=True)
    except Exception as e:
        print("Failed to save model with BentoML. Error:", str(e), flush=True)
        import traceback
        traceback.print_exc()

    # Export the BentoML model to the specified folder for deployment
    try:
        print("Exporting the BentoML model...", flush=True)
        bentoml.models.export_model(
            f"{bentoml_model_name}:latest",
            str(model_folder / "celestial_bodies_classifier_model.bentomodel")
        )
        print(f"BentoML model exported at {model_folder}", flush=True)
    except Exception as e:
        print("Failed to export the BentoML model. Error:", str(e), flush=True)

if __name__ == "__main__":
    main()
