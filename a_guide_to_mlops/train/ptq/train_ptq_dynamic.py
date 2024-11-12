import sys
from pathlib import Path
import tensorflow as tf
import bentoml
import json

from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, PTQ_MODEL_DIR
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.preprocessing import preprocess, postprocess
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model

def main():
    print("Script started...", flush=True)

    # Load parameters
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

    # Model paths from the config
    prepared_dataset_folder = PREPARED_DATA_DIR
    model_folder = PTQ_MODEL_DIR
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

    # Model parameters
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
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # Define model
    print("Defining model...", flush=True)
    model = get_model(image_shape, train_params["conv_size"], train_params["dense_size"], train_params["output_classes"])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(train_params["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.summary()

    # Train model
    print("Starting training...", flush=True)
    history = model.fit(ds_train, epochs=train_params["epochs"], validation_data=ds_test)

    # Apply Dynamic Range Quantization
    print("Applying Dynamic Range Quantization...", flush=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    # Save the quantized TFLite model
    with open(model_folder / "celestial_bodies_classifier_model_ptq_dynamic.tflite", "wb") as f:
        f.write(quantized_model)

    print(f"\nModel and training history saved at {model_folder.absolute()}", flush=True)

if __name__ == "__main__":
    main()
