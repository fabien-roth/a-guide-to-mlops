import os
import sys
import time
import tensorflow as tf
import bentoml
import json
from pathlib import Path
import numpy as np

from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, BASELINE_MODEL_DIR
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.quantization_func import get_output_dir

def main():
    # Initial debug prints
    print("Script started...", flush=True)
    print(f"Command Line Arguments: {sys.argv}", flush=True)

    # Set paths using the paths from config.py
    prepared_dataset_folder = PREPARED_DATA_DIR  # This should correctly point to data/prepared
    model_folder = BASELINE_MODEL_DIR
    model_folder.mkdir(parents=True, exist_ok=True)

    # Debug print statements
    print(f"Prepared Dataset Folder Path: {prepared_dataset_folder}", flush=True)
    print(f"Model Folder Path: {model_folder}", flush=True)

    # Load parameters from the configuration file
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

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

    # Load datasets
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
    labels_file_path = prepared_dataset_folder / "labels.json"
    print(f"Loading labels from {labels_file_path}", flush=True)

    if not labels_file_path.exists():
        print(f"Error: Labels file does not exist at {labels_file_path}", flush=True)
        exit(1)

    with open(labels_file_path) as f:
        labels = json.load(f)

    # Define model and compile it
    print("Defining model...", flush=True)
    model = get_model(image_shape, train_params["conv_size"], train_params["dense_size"], train_params["output_classes"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(train_params["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.summary()

    # Train the model
    print("⚙️ Starting training...", flush=True)
    start_time = time.time()
    history = model.fit(ds_train, epochs=train_params["epochs"], validation_data=ds_test)
    training_time = time.time() - start_time
    print("✅ Model trained successfully")

    # Save the model in Keras' native format
    print("⚙️ Saving the model in Keras' native format (.keras)...", flush=True)
    keras_model_path = model_folder / "celestial_bodies_classifier_model.keras"
    model.save(keras_model_path)
    print(f"✅ Model saved in Keras format at {keras_model_path}")

    # Save the model using BentoML with a unique name
    print("⚙️ Saving the model using BentoML...", flush=True)
    bentoml_model_name = "celestial_bodies_classifier_baseline"
    bentoml.keras.save_model(
        bentoml_model_name,
        model,
        include_optimizer=True,
        custom_objects={
            "preprocess": lambda x: (x / 255.0),
            "postprocess": lambda x: labels[tf.argmax(x)],
        }
    )
    print(f"✅ Model saved with BentoML as {bentoml_model_name}")

    # Export the BentoML model for deployment
    print("⚙️ Exporting the BentoML model...", flush=True)
    bentoml.models.export_model(
        f"{bentoml_model_name}:latest",
        str(model_folder / "celestial_bodies_classifier_model.bentomodel")
    )
    print("✅ Model exported successfully")
    # Save the model history for evaluation purposes
    np.save(model_folder / "history.npy", history.history)

    model_size = sum(os.path.getsize(f) for f in os.scandir(model_folder) if f.is_file()) / (1024 * 1024)
    metrics = {
        "model_size_mb": model_size,
        "training_time_sec": training_time,
    }
    
    output_dir = get_output_dir(model_folder)
    shared_metrics_file = Path(f"{output_dir}/metrics.json")
    label = f"baseline_{int(time.time())}"

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
