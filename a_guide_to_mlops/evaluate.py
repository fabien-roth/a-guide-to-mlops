import json
import sys
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import bentoml
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


def get_training_plot(model_history: dict) -> plt.Figure:
    """Plot the training and validation loss"""
    epochs = range(1, len(model_history["loss"]) + 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, model_history["loss"], label="Training loss")
    plt.plot(epochs, model_history["val_loss"], label="Validation loss")
    plt.xticks(epochs)
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    return fig


def get_pred_preview_plot(
    model: tf.keras.Model, ds_test: tf.data.Dataset, labels: List[str]
) -> plt.Figure:
    """Plot a preview of the predictions"""
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds_test.take(1):
        preds = model.predict(images)
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            img = (images[i].numpy() * 255).astype("uint8")
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
                img = np.stack((img,) * 3, axis=-1)
            true_label = labels[label_idxs[i].numpy()]
            pred_label = labels[np.argmax(preds[i])]
            img = np.pad(img, pad_width=((1, 1), (1, 1), (0, 0)))
            if true_label != pred_label:
                img[0, :, 0] = 255
                img[-1, :, 0] = 255
                img[:, 0, 0] = 255
                img[:, -1, 0] = 255
            else:
                img[0, :, 1] = 255
                img[-1, :, 1] = 255
                img[:, 0, 1] = 255
                img[:, -1, 1] = 255
            plt.imshow(img)
            plt.title(f"True: {true_label}\nPred: {pred_label}")
            plt.axis("off")
    return fig


def get_confusion_matrix_plot(
    model: tf.keras.Model, ds_test: tf.data.Dataset, labels: List[str]
) -> plt.Figure:
    """Plot the confusion matrix"""
    preds = model.predict(ds_test)
    conf_matrix = tf.math.confusion_matrix(
        labels=tf.concat([y for _, y in ds_test], axis=0),
        predictions=tf.argmax(preds, axis=1),
        num_classes=len(labels),
    )

    fig = plt.figure(figsize=(6, 6), tight_layout=True)
    normalized_conf_matrix = conf_matrix / tf.reduce_sum(conf_matrix, axis=1, keepdims=True)
    plt.imshow(normalized_conf_matrix, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = conf_matrix[i, j].numpy()
            color = "white" if value > tf.reduce_max(conf_matrix).numpy() / 2 else "black"
            plt.text(
                j, i, f"{value}", ha="center", va="center", color=color
            )
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")

    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython -m a_guide_to_mlops.evaluate <model-folder> <prepared-dataset-folder>\n")
        exit(1)

    # Load arguments
    model_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])

    print(f"Model folder: {model_folder}")
    print(f"Prepared dataset folder: {prepared_dataset_folder}")

    # Determine output directory dynamically
    output_dir = get_output_dir(model_folder)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model_file_path = model_folder / "celestial_bodies_classifier_model.bentomodel"
    if not model_file_path.exists():
        print(f"Error: Model file '{model_file_path}' does not exist.")
        exit(1)

    labels_file = prepared_dataset_folder / "labels.json"
    if not labels_file.exists():
        print(f"Error: Labels file '{labels_file}' not found.")
        exit(1)

    with open(labels_file) as f:
        labels = json.load(f)

    test_dataset_path = prepared_dataset_folder / "test"
    if not test_dataset_path.exists():
        print(f"Error: Test dataset not found at '{test_dataset_path}'.")
        exit(1)

    ds_test = tf.data.Dataset.load(str(test_dataset_path))

    print(f"Loading model from '{model_file_path}' using BentoML...")
    try:
        model = bentoml.keras.load_model("celestial_bodies_classifier_baseline")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Evaluate the model
    val_loss, val_acc = model.evaluate(ds_test)
    print(f"Validation loss: {val_loss:.2f}")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({"val_loss": val_loss, "val_acc": val_acc}, f)

    # Save confusion matrix plot
    get_confusion_matrix_plot(model, ds_test, labels).savefig(plots_dir / "confusion_matrix.png")
    # Save prediction preview plot
    get_pred_preview_plot(model, ds_test, labels).savefig(plots_dir / "pred_preview.png")
    # Save training history plot
    history_file = model_folder / "history.npy"
    model_history = np.load(history_file, allow_pickle=True).item()
    get_training_plot(model_history).savefig(plots_dir / "training_history.png")

    print(f"Evaluation completed. Metrics saved to '{metrics_file}'.")


if __name__ == "__main__":
    main()
