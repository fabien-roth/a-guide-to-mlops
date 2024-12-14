import os
import sys
from pathlib import Path
import time
import tensorflow as tf
import bentoml
import json
import numpy as np

from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, PTQ_MODEL_DYNAMIC_DIR
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.preprocessing import preprocess, postprocess
from a_guide_to_mlops.utils.quantization_func import get_output_dir, quantize_model
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model

def main():
    print("Script started...", flush=True)

    # Load parameters from the configuration file
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

    # Set paths using paths from config.py
    prepared_dataset_folder = PREPARED_DATA_DIR
    model_folder = PTQ_MODEL_DYNAMIC_DIR
    model_folder.mkdir(parents=True, exist_ok=True)

    # Debug print statements
    print(f"Prepared Dataset Folder Path: {prepared_dataset_folder}", flush=True)
    print(f"Model Folder Path: {model_folder}", flush=True)

    # Verify if 'train' folder exists and contains the necessary files
    train_path = prepared_dataset_folder / "train"
    if not train_path.exists():
        print(f"Error: {train_path} does not exist.", flush=True)
        exit(1)

    if not (train_path / "dataset_spec.pb").exists():
        print(f"Error: dataset_spec.pb not found at {train_path}.", flush=True)
        exit(1)

    # Set up model parameters
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    # Set random seed for reproducibility
    set_seed(train_params["seed"])

    # Load training and validation datasets
    try:
        print("Loading datasets...", flush=True)
        ds_train = tf.data.Dataset.load(str(train_path))
        ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))
        print("Datasets loaded successfully.", flush=True)
    except Exception as e:
        print(f"Error loading datasets: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)

    # Load labels
    labels = None
    labels_file_path = prepared_dataset_folder / "labels.json"
    print(f"Loading labels from {labels_file_path}", flush=True)

    if not labels_file_path.exists():
        print(f"Error: Labels file does not exist at {labels_file_path}", flush=True)
        exit(1)

    with open(labels_file_path) as f:
        labels = json.load(f)

    # Define model
    print("Defining model...", flush=True)
    model = get_model(image_shape, train_params["conv_size"], train_params["dense_size"], train_params["output_classes"])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(train_params["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.summary()

    # Train the model
    print("Starting training...", flush=True)
    start_time = time.time()
    history = model.fit(ds_train, epochs=train_params["epochs"], validation_data=ds_test)
    training_time = time.time() - start_time
    
    # Apply Dynamic Range Quantization
    print("🔧 Applying PTQ Dynamic Range Quantization...", flush=True)
    
    quantisation_method_choice = "PTQ"
    quantization_type = "DYNAMIC"
    is_qat=False
    quantized_model_path = quantize_model(
            model,
            model_folder,
            quantization_type=quantization_type,
            is_qat=None,
            ds_train=ds_train if quantisation_method_choice.upper() == "PTQ" or is_qat else None
        )
    
    quantized_model_size = os.path.getsize(quantized_model_path) / (1024 * 1024)

    metrics = {
        "model_size_mb": quantized_model_size,
        "training_time_sec": training_time,
    }

    print("Saving the model using BentoML...", flush=True)
    # Load the quantized model binary
    with open(quantized_model_path, "rb") as f:
        quantized_model_data = f.read()
    
    # Unique name for tracking the specific variant
    bento_model_name = "quantized_model"  # Unique name for tracking the specific variant

    with bentoml.models.create(
        name=bento_model_name,
        module="tensorflow-lite",
        api_version="v1",
        options={"model_binary": quantized_model_data},
        labels={"quantization": quantization_type, "method": quantisation_method_choice},
        metadata={
            "description": "Quantized TFLite model",
            "training_time_sec": training_time,
            "model_size_mb": quantized_model_size,
        },
        custom_objects={
            "preprocess": lambda x: (x / 255.0),  # Example preprocessing logic
            "postprocess": lambda x: labels[np.argmax(x)],  # Example postprocessing logic
        },
    ) as bento_model:
        print(f"✅ Quantized model saved to BentoML: {bento_model.tag}")

    # Export the Bento model to a .bentomodel file for deployment
    print("Exporting the Bento model...", flush=True)
    bentoml.models.export_model(
        f"{bento_model.tag}",
        str(model_folder / "celestial_bodies_classifier_model.bentomodel"),
    )
    print(f"✅ Bento model exported to: {model_folder / 'celestial_bodies_classifier_model.bentomodel'}")

    # Save model training history for evaluation purposes
    history_path = model_folder / "history.npy"
    np.save(history_path, history.history)

    output_dir = get_output_dir(model_folder)
    shared_metrics_file = Path(f"{output_dir}/metrics.json")
    label = f"PTQ_DYNAMIC_{int(time.time())}"

    # Load existing metrics if file exists
    if shared_metrics_file.exists():
        with open(shared_metrics_file, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    all_metrics[label] = metrics

    # Save updated metrics back to the file
    shared_metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(shared_metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"✅ Bqseline - Metrics appended to shared file: {shared_metrics_file}")

    print(f"\nModel and training history saved at {model_folder.absolute()}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred: {e}", flush=True)
        import traceback
        traceback.print_exc()
