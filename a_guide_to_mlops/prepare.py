import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt  # Added this import
import tensorflow as tf
import yaml

from a_guide_to_mlops.utils.seed import set_seed


def get_preview_plot(ds: tf.data.Dataset, labels: List[str]) -> plt.Figure:
    """Plot a preview of the prepared dataset"""
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds.take(1):
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
            plt.title(labels[label_idxs[i].numpy()])
            plt.axis("off")

    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 prepare.py <raw-dataset-folder> <prepared-dataset-folder>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]

    raw_dataset_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])
    seed = prepare_params["seed"]
    split = prepare_params["split"]
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]

    # Set seed for reproducibility
    set_seed(seed)

    # Read data
    ds_train, ds_test = tf.keras.utils.image_dataset_from_directory(
        raw_dataset_folder,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale" if grayscale else "rgb",
        batch_size=32,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=split,
        subset="both",
    )
    labels = ds_train.class_names

    if not prepared_dataset_folder.exists():
        prepared_dataset_folder.mkdir(parents=True)

    # Save the preview plot
    preview_plot = get_preview_plot(ds_train, labels)
    preview_plot.savefig(prepared_dataset_folder / "preview.png")

    # Normalize the data
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
    ds_test = ds_test.map(lambda x, y: (normalization_layer(x), y))

    # Save the prepared dataset
    with open(prepared_dataset_folder / "labels.json", "w") as f:
        json.dump(labels, f)
    train_path = prepared_dataset_folder / "train"
    test_path = prepared_dataset_folder / "test"
    tf.data.Dataset.save(ds_train, str(train_path))
    tf.data.Dataset.save(ds_test, str(test_path))

    # Debug: Confirm that the datasets can be loaded right after saving
    print("Attempting to load saved datasets to verify...")
    try:
        ds_train_loaded = tf.data.Dataset.load(str(train_path))
        ds_test_loaded = tf.data.Dataset.load(str(test_path))
        print(f"Successfully loaded datasets from {train_path} and {test_path}")
    except Exception as e:
        print(f"Error loading datasets: {e}")

    print(f"\nDataset saved at {prepared_dataset_folder.absolute()}")


if __name__ == "__main__":
    main()
