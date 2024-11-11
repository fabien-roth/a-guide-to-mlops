import sys
from pathlib import Path
import tensorflow as tf
import bentoml
import json
from model.model_builder import get_model
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.seed import set_seed


def representative_dataset_gen(ds_train):
    """Generate representative data for calibration during full integer quantization."""
    for input_value, _ in ds_train.take(100):
        # Remove batch dimension if it exists and ensure correct input shape
        yield [tf.cast(input_value, tf.float32)]


def main():
    if len(sys.argv) != 3:
        print("Usage: python train_ptq_integer.py <prepared-dataset-folder> <model-folder>")
        exit(1)

    # Load parameters
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

    prepared_dataset_folder = Path(sys.argv[1])
    model_folder = Path(sys.argv[2])

    # Model parameters
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    # Set random seed for reproducibility
    set_seed(train_params["seed"])

    # Load training and validation datasets
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    # Load labels
    labels = None
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # Define model
    model = get_model(image_shape, train_params["conv_size"], train_params["dense_size"], train_params["output_classes"])

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(train_params["lr"]),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # Train model
    model.fit(ds_train, epochs=train_params["epochs"], validation_data=ds_test)

    # Apply Full Integer Quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = lambda: representative_dataset_gen(ds_train)
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8

    try:
        quantized_model = converter.convert()

        # Save the quantized TFLite model
        model_folder.mkdir(parents=True, exist_ok=True)
        with open(model_folder / "celestial_bodies_classifier_model_ptq_integer.tflite", "wb") as f:
            f.write(quantized_model)

        print(f"\nModel and training history saved at {model_folder.absolute()}")

    except Exception as e:
        print(f"Failed to quantize the model. Error: {str(e)}")


if __name__ == "__main__":
    main()
