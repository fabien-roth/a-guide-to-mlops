import json
import sys
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf
import bentoml
import onnxruntime as rt
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from a_guide_to_mlops.utils.config import (
    EVALUATION_DIR,
    PTQ_MODEL_DIR,
    QAT_MODEL_DIR,
    BASELINE_MODEL_DIR,
)

# Function to generate the training and validation loss plot
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

# Function to generate a preview of predictions
def get_pred_preview_plot(model, ds_test: tf.data.Dataset, labels: List[str]) -> plt.Figure:
    """Plot a preview of the predictions"""
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds_test.take(1):
        preds = model.run(None, {model.get_inputs()[0].name: images.numpy()})[0]
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            img = (images[i].numpy() * 255).astype("uint8")
            # Convert image to RGB if grayscale
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
                img = np.stack((img,) * 3, axis=-1)
            true_label = labels[label_idxs[i].numpy()]
            pred_label = labels[np.argmax(preds[i])]
            # Add red border if the prediction is wrong, else add green border
            img = np.pad(img, pad_width=((1, 1), (1, 1), (0, 0)))
            if true_label != pred_label:
                img[0, :, 0] = 255  # Top border
                img[-1, :, 0] = 255  # Bottom border
                img[:, 0, 0] = 255  # Left border
                img[:, -1, 0] = 255  # Right border
            else:
                img[0, :, 1] = 255  # Green border for correct prediction
                img[-1, :, 1] = 255
                img[:, 0, 1] = 255
                img[:, -1, 1] = 255

            plt.imshow(img)
            plt.title(f"True: {true_label}\nPred: {pred_label}")
            plt.axis("off")

    return fig

# Function to generate a confusion matrix plot
def get_confusion_matrix_plot(model, ds_test: tf.data.Dataset, labels: List[str]) -> plt.Figure:
    """Plot the confusion matrix"""
    all_true_labels = []
    all_predictions = []

    # Loop over dataset to get predictions and true labels
    for images, true_labels in ds_test:
        preds = model.run(None, {model.get_inputs()[0].name: images.numpy()})[0]
        all_true_labels.extend(true_labels.numpy())
        all_predictions.extend(np.argmax(preds, axis=1))

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predictions, labels=range(len(labels)))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    return fig

# Main function for evaluation
def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython evaluate.py <model-folder> <prepared-dataset-folder>\n")
        exit(1)

    model_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])

    # Determine the evaluation folder using config paths dynamically
    model_type = model_folder.name  # Extract the model type, e.g., 'baseline', 'ptq/dynamic'
    
    # Determine evaluation folder based on model type
    if model_type == 'baseline':
        evaluation_folder = EVALUATION_DIR / 'baseline'
    elif 'ptq' in model_folder.parts:
        evaluation_folder = EVALUATION_DIR / 'ptq' / model_type.split('/')[-1]
    elif 'qat' in model_folder.parts:
        evaluation_folder = EVALUATION_DIR / 'qat' / model_type.split('/')[-1]
    else:
        evaluation_folder = EVALUATION_DIR / model_type
    
    plots_folder = evaluation_folder / "plots"

    # Create evaluation output directories
    plots_folder.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))
    labels = None
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # Load model using the consistent name from BentoML store
    bentoml_model_name = "celestial_bodies_classifier_model"
    print(f"Loading model '{bentoml_model_name}' from BentoML...", flush=True)

    try:
        # Load the ONNX model
        model = bentoml.onnx.load_model(f"{bentoml_model_name}:latest")
        print(f"Model '{bentoml_model_name}' successfully loaded from BentoML store.", flush=True)
    except bentoml.exceptions.NotFound:
        print(f"Error: Model '{bentoml_model_name}' not found in BentoML store.", flush=True)
        exit(1)

    # Load model training history for evaluation purposes
    model_history = np.load(model_folder / "history.npy", allow_pickle=True).item()

    # Set up the ONNX InferenceSession for evaluation
    session = model  # The loaded model is an ONNX Runtime InferenceSession
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Prepare variables to calculate metrics manually
    all_true_labels = []
    all_predictions = []

    # Loop through the test dataset
    print("Evaluating model on test dataset...", flush=True)
    for images, true_labels in ds_test:
        # Convert images to numpy arrays for ONNX
        images_np = images.numpy()
        
        # Run inference with ONNX
        preds = session.run([output_name], {input_name: images_np})[0]

        # Collect true labels and predictions for metrics calculation
        all_true_labels.extend(true_labels.numpy())
        all_predictions.extend(preds)

    # Calculate metrics manually
    val_loss = log_loss(all_true_labels, all_predictions, labels=range(len(labels)))
    val_acc = accuracy_score(all_true_labels, np.argmax(all_predictions, axis=1))

    print(f"Validation loss: {val_loss:.2f}")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")

    # Save metrics to JSON file
    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"val_loss": val_loss, "val_acc": val_acc}, f)

    # Save training history plot
    fig = get_training_plot(model_history)
    fig.savefig(plots_folder / "training_history.png")

    # Save predictions preview plot
    fig = get_pred_preview_plot(session, ds_test, labels)
    fig.savefig(plots_folder / "pred_preview.png")

    # Save confusion matrix plot
    fig = get_confusion_matrix_plot(session, ds_test, labels)
    fig.savefig(plots_folder / "confusion_matrix.png")

    print(f"\nEvaluation metrics and plot files saved at {evaluation_folder.absolute()}")

if __name__ == "__main__":
    main()
