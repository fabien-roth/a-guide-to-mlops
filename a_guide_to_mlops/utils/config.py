from pathlib import Path
import os

# Determine if the code is running inside a BentoML container
IS_BENTO_CONTAINER = os.getenv("BENTOML_CONTAINER") is not None

# Define the root of the project as the base directory
if IS_BENTO_CONTAINER:
    # Paths inside the container
    PROJECT_ROOT = Path("/home/bentoml/bento")
else:
    # Local paths during development
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define data paths
if IS_BENTO_CONTAINER:
    DATA_DIR = PROJECT_ROOT / "src" / "data"
    PREPARED_DATA_DIR = DATA_DIR / "prepared"
else:
    DATA_DIR = PROJECT_ROOT / "data"
    PREPARED_DATA_DIR = DATA_DIR / "prepared"

# Define common directories within the project
MODEL_DIR = PROJECT_ROOT / "model"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UTILS_DIR = PROJECT_ROOT / "utils"
TRAIN_DIR = PROJECT_ROOT / "a_guide_to_mlops" / "train"

# Specific folders for different stages
BASELINE_MODEL_DIR = MODEL_DIR / "baseline"
PTQ_MODEL_DIR = MODEL_DIR / "ptq"
PTQ_MODEL_DYNAMIC_DIR = PTQ_MODEL_DIR / "dynamic"
PTQ_MODEL_FLOAT16_DIR = PTQ_MODEL_DIR / "float16"
PTQ_MODEL_INTEGER_DIR = PTQ_MODEL_DIR / "integer"
QAT_MODEL_DIR = MODEL_DIR / "qat"
QAT_MODEL_DYNAMIC_DIR = QAT_MODEL_DIR / "dynamic"
QAT_MODEL_FLOAT16_DIR = QAT_MODEL_DIR / "float16"
QAT_MODEL_INTEGER_DIR = QAT_MODEL_DIR / "integer"

# Model configuration constants
EXTRACTED_MODEL_SUBDIR = "extracted_model"
MODEL_FILENAME = "celestial_bodies_classifier_model.bentomodel"

# Model name
BENTOML_MODEL_NAME = MODEL_FILENAME

# Model paths (for different quantization types)
BASELINE_MODEL_PATH = BASELINE_MODEL_DIR / BENTOML_MODEL_NAME
PTQ_DYNAMIC_MODEL_PATH = PTQ_MODEL_DYNAMIC_DIR / BENTOML_MODEL_NAME
PTQ_FLOAT16_MODEL_PATH = PTQ_MODEL_FLOAT16_DIR / BENTOML_MODEL_NAME
PTQ_INTEGER_MODEL_PATH = PTQ_MODEL_INTEGER_DIR / BENTOML_MODEL_NAME
QAT_DYNAMIC_MODEL_PATH = QAT_MODEL_DYNAMIC_DIR / BENTOML_MODEL_NAME
QAT_FLOAT16_MODEL_PATH = QAT_MODEL_FLOAT16_DIR / BENTOML_MODEL_NAME
QAT_INTEGER_MODEL_PATH = QAT_MODEL_INTEGER_DIR / BENTOML_MODEL_NAME

# Unified model paths mapping for dynamic loading
MODEL_PATHS = {
    "baseline": BASELINE_MODEL_PATH,
    "ptq_dynamic": PTQ_DYNAMIC_MODEL_PATH,
    "ptq_float16": PTQ_FLOAT16_MODEL_PATH,
    "ptq_integer": PTQ_INTEGER_MODEL_PATH,
    "qat_dynamic": QAT_DYNAMIC_MODEL_PATH,
    "qat_float16": QAT_FLOAT16_MODEL_PATH,
    "qat_integer": QAT_INTEGER_MODEL_PATH,
}

# Evaluation paths
EVALUATION_BASELINE_DIR = EVALUATION_DIR / "baseline"
EVALUATION_PTQ_DIR = EVALUATION_DIR / "ptq"
EVALUATION_QAT_DIR = EVALUATION_DIR / "qat"

if __name__ == "__main__":
    # Example debug output to check paths
    print("Running in:", "Container" if IS_BENTO_CONTAINER else "Local Environment")
    print("Data Directory:", DATA_DIR)
    print("Prepared Data Directory:", PREPARED_DATA_DIR)
    print("Model Paths:", MODEL_PATHS)
    print(f"Does labels.json exist? {PREPARED_DATA_DIR / 'labels.json'} -> {(PREPARED_DATA_DIR / 'labels.json').exists()}")
    print("Resolved paths in MODEL_PATHS:")
    for model_type, path in MODEL_PATHS.items():
        print(f"{model_type}: {path} -> Exists: {path.exists()}")