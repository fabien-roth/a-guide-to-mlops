from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict

import bentoml
from fastapi import FastAPI
from PIL import Image

# 1. Define constants
MODEL_VERSION = "v0.0.1"
DATA_FOLDER_PATH = Path("extra-data/extra_data")

# 2. Initialize FastAPI app
app = FastAPI()
# Load the BentoML model
bento_model = bentoml.keras.get("celestial_bodies_classifier_model")
preprocess = bento_model.custom_objects["preprocess"]
postprocess = bento_model.custom_objects["postprocess"]
model = bento_model.load_model()


# 3. Define API endpoints
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/setup")
async def setup(_: Dict):
    # Currently, we don't need to do anything with the setup
    # This is why we ignore the input data
    return {"model_version": MODEL_VERSION}


@app.post("/webhook")
async def webhook():
    # Currently, we don't need to do anything with the webhook
    # It is used to start training the model from the Label Studio UI
    # Refer to:
    # https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/api.py#L118
    return {"status": "Unknown event"}


@app.post("/predict")
async def predict(data: Dict):
    # For request and response types, refer to:
    # https://labelstud.io/guide/ml_create#3-Implement-prediction-logic
    task = data["tasks"][0]
    # This is a simplification to get the filename from the image id.
    # In a real-world scenario, you would store the image on S3
    # and fetch it here.
    filename = "".join(task["data"]["image"].split("-")[1:])
    image_path = DATA_FOLDER_PATH / filename
    image = Image.open(image_path)

    preprocessed_image = preprocess(image)
    prediction = model.predict(preprocessed_image)
    result = postprocess(prediction)
    prediction = result["prediction"]
    score = result["probabilities"][prediction]
    print(score, type(score))

    return {
        "results": [
            {
                "model_version": MODEL_VERSION,
                "score": score,
                "result": [
                    {
                        "value": {"choices": [prediction]},
                        "id": str(uuid.uuid4()),
                        "from_name": "choice",
                        "to_name": "image",
                        "type": "choices",
                    }
                ],
            }
        ],
    }