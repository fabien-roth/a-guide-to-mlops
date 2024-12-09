from __future__ import annotations
from bentoml.validators import ContentType
from typing import Annotated
from PIL import Image
from pydantic import Field
import bentoml
import json
import time
import os
import numpy as np
from tensorflow.keras.models import Sequential
from pathlib import Path
from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR


@bentoml.service(name="celestial_bodies_classifier_canary")
class CelestialBodiesClassifierCanaryService:
    def __init__(self) -> None:
        """
        Initialize the service. Models will be dynamically loaded based on the request.
        """
        self.loaded_models = {}
        self.labels = self.load_labels()

    def load_labels(self):
        """
        Load the labels mapping from the training phase, considering the container environment.
        """
        labels_path = Path(PREPARED_DATA_DIR) / "labels.json"
        print(f"DEBUG: Attempting to load labels from {labels_path}")

        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with open(labels_path, "r") as f:
            labels = json.load(f)

        print(f"DEBUG: Loaded labels: {labels}")
        return labels

    def load_model(self, model_type: str):
        """
        Load and cache the model dynamically from BentoML's model store.
        """
        if model_type in self.loaded_models:
            # Return cached model
            print(f"DEBUG: Using cached model for type '{model_type}'")
            return self.loaded_models[model_type]

        model_tag = f"celestial_bodies_classifier_{model_type}"
        try:
            # Fetch the model from the BentoML model store
            model = bentoml.keras.load_model(model_tag)
            print(f"DEBUG: Loaded model '{model_tag}' from BentoML model store.")
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_tag}': {e}")

        # Define preprocessing and postprocessing
        preprocess = lambda x: np.expand_dims(np.array(x.resize((32, 32)).convert("L"), dtype=np.float32) / 255.0, axis=-1)
        postprocess = lambda x: self.labels[np.argmax(x)]

        # Cache the model
        self.loaded_models[model_type] = {
            "model": model,
            "preprocess": preprocess,
            "postprocess": postprocess,
        }
        return self.loaded_models[model_type]

    @bentoml.api()
    def predict(
        self,
        image: Annotated[Image.Image, ContentType("image/jpeg")] = Field(description="Planet image to analyze"),
        model_type: Annotated[str, ContentType("application/json")] = Field(default=None, description="Model type to use"),
    ) -> Annotated[str, ContentType("application/json")]:
        """
        Predict celestial body classification based on the input image and selected model type.
        """
        # Use the model_type from the request or fallback to the environment variable or default to 'baseline'
        model_type = model_type or os.getenv("MODEL_TYPE", "baseline")

        # Validate model type and load the model
        try:
            model_data = self.load_model(model_type)
        except ValueError as e:
            return json.dumps({"error": str(e)})

        preprocess = model_data["preprocess"]
        postprocess = model_data["postprocess"]

        # Preprocess the image and make predictions
        try:
            start_time = time.time()
            image_array = preprocess(image)
            predictions = model_data["model"].predict(np.expand_dims(image_array, axis=0))

            # Convert predictions into probabilities and extract the prediction
            probabilities = {label: float(prob) for label, prob in zip(self.labels, predictions[0])}
            prediction = postprocess(predictions)

            # Log latency for monitoring
            latency = time.time() - start_time
            print(f"[{model_type}] Prediction latency: {latency:.4f}s")

            return json.dumps({
                "model_type": model_type,
                "prediction": prediction,
                "probabilities": probabilities,
                "latency": latency
            })

        except Exception as e:
            return json.dumps({"error": f"An error occurred during prediction: {str(e)}"})
