import sys
import tensorflow as tf
import bentoml
import json
from pathlib import Path

from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, PTQ_MODEL_DIR
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.preprocessing import preprocess, postprocess
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model

def main():
    print("Script started...", flush=True)

    # Load parameters from configuration
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

    # Paths from config.py
    prepared_dataset_folder = PREPARED_DATA_DIR
    model_folder = PTQ_MODEL_DIR
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

    # Apply Float16 Quantization
    print("Applying Float16 Quantization...", flush=True)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()

    # Save the quantized TFLite model
    with open(model_folder / "celestial_bodies_classifier_model_ptq_float16.tflite", "wb") as f:
        f.write(quantized_model)

    print(f"\nModel and training history saved at {model_folder.absolute()}", flush=True)

if __name__ == "__main__":
    main()
