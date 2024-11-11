import sys
from pathlib import Path
import json
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import tf2onnx
import bentoml
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.seed import set_seed
from model.model_builder import get_model

def main():
    if len(sys.argv) != 3:
        print("Usage: python train_qat_dynamic.py <prepared-dataset-folder> <model-folder>")
        exit(1)

    # Load parameters
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

    prepared_dataset_folder = Path(sys.argv[1])
    model_folder = Path(sys.argv[2])

    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    set_seed(train_params["seed"])

    # Load training and validation datasets
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    # Load labels
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # Define and annotate the model for quantization
    model = get_model(image_shape, train_params["conv_size"], train_params["dense_size"], train_params["output_classes"])
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
    quantize_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    # Compile the quantized model
    quantize_model.compile(
        optimizer=tf.keras.optimizers.Adam(train_params["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # Train the quantized model
    quantize_model.fit(ds_train, epochs=train_params["epochs"], validation_data=ds_test)

    # Save the model in `.keras` format
    keras_backup_path = model_folder / "celestial_bodies_classifier_model_qat_dynamic.keras"
    quantize_model.save(keras_backup_path)

    # Convert the trained Keras model to ONNX
    try:
        spec = (tf.TensorSpec((None, *image_shape), tf.float32, name="input"),)
        onnx_output_path = model_folder / "celestial_bodies_classifier_model_qat_dynamic.onnx"
        model_proto, _ = tf2onnx.convert.from_keras(quantize_model, input_signature=spec, opset=13, output_path=str(onnx_output_path))
        print(f"Model converted to ONNX and saved at {onnx_output_path}")
    except Exception as e:
        print("Failed to convert model to ONNX. Error:", str(e))
        return

    # Quantize the ONNX model using ONNX Runtime
    try:
        quantized_model_path = model_folder / "celestial_bodies_classifier_model_qat_dynamic_quantized.onnx"
        quantize_dynamic(
            model_input=str(onnx_output_path),
            model_output=str(quantized_model_path),
            op_types_to_quantize=['Conv', 'MatMul'],
            weight_type=QuantType.QInt8
        )
        print(f"Quantized model saved at {quantized_model_path}")
    except Exception as e:
        print("Failed to quantize the ONNX model. Error:", str(e))
        return

    # Load and save the quantized ONNX model with BentoML
    try:
        quantized_model = onnx.load(str(quantized_model_path))
        bentoml.onnx.save_model(
            "celestial_bodies_classifier_model_qat_dynamic",
            quantized_model,
            signatures={"run": {"batchable": True}},
            labels={"model": "quantized", "framework": "onnx"},
            metadata={"description": "Quantized ONNX model of celestial bodies classifier"}
        )
        print("Quantized ONNX model successfully saved to BentoML store.")
    except Exception as e:
        print("Failed to save model with BentoML. Error:", str(e))

if __name__ == "__main__":
    main()
