import json
import shutil
from pathlib import Path

# Constants
EXTRA_DATA_FOLDER_PATH = Path("extra-data/extra_data")
NEW_DATA_FOLDER_PATH = Path("data/raw")

# Read annotations and copy images to annotated folders
with open("extra-data/annotations.json") as f:
    annotations = json.load(f)

for annotation in annotations:
    # Here we perform the same manipulation as `src/serve_label_studio.py`
    # to retrieve the correct filename
    filename = "".join(annotation["image"].split("-")[1:])
    choice = annotation["choice"]

    source_path = EXTRA_DATA_FOLDER_PATH / filename
    # Note: Here we use the choice as the folder name, as
    #       this is how we organised the data
    dest_path = NEW_DATA_FOLDER_PATH / choice / filename

    print(f"Copying {source_path} -> {dest_path}")
    shutil.copy(source_path, dest_path)