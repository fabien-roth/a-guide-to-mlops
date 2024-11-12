from pathlib import Path
import os

# Define the root of the project as the base directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define common directories within the project
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UTILS_DIR = PROJECT_ROOT / "utils"
TRAIN_DIR = PROJECT_ROOT / "a_guide_to_mlops" / "train"

# Specific folders for different stages
BASELINE_MODEL_DIR = MODEL_DIR / "baseline"
PTQ_MODEL_DIR = MODEL_DIR / "ptq"
QAT_MODEL_DIR = MODEL_DIR / "qat"

# Environment variables (useful for overriding defaults, especially in different environments)
BASE_DIR = Path(os.getenv("BASE_DIR", PROJECT_ROOT))

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
#PREPARED_DATA_DIR = DATA_DIR / "prepared"
PREPARED_DATA_DIR = PROJECT_ROOT / "data" / "prepared"


# Model paths (for different quantization types)
BASELINE_MODEL_PATH = BASELINE_MODEL_DIR / "celestial_bodies_classifier_baseline.tflite"
PTQ_DYNAMIC_MODEL_PATH = PTQ_MODEL_DIR / "celestial_bodies_classifier_model_ptq_dynamic.tflite"
PTQ_FLOAT16_MODEL_PATH = PTQ_MODEL_DIR / "celestial_bodies_classifier_model_ptq_float16.tflite"
PTQ_INTEGER_MODEL_PATH = PTQ_MODEL_DIR / "celestial_bodies_classifier_model_ptq_integer.tflite"
QAT_DYNAMIC_MODEL_PATH = QAT_MODEL_DIR / "celestial_bodies_classifier_model_qat_dynamic.tflite"
QAT_INTEGER_MODEL_PATH = QAT_MODEL_DIR / "celestial_bodies_classifier_model_qat_integer.tflite"

# Evaluation paths
EVALUATION_BASELINE_DIR = EVALUATION_DIR / "baseline"
EVALUATION_PTQ_DIR = EVALUATION_DIR / "ptq"
EVALUATION_QAT_DIR = EVALUATION_DIR / "qat"

if __name__ == "__main__":
    # Example debug output to check paths
    print("Project Root:", PROJECT_ROOT)
    print("Baseline Model Path:", BASELINE_MODEL_PATH)
    print("Prepared Data Directory:", PREPARED_DATA_DIR)
