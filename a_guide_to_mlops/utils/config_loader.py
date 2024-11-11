
import yaml

def load_config(file_path="params.yaml"):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
            