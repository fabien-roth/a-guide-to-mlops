#!/bin/bash

# Define the correct IP and port of the remote service
SERVICE_URL="http://34.65.152.219:3000/predict"

# Full path to the local test image file
IMAGE_PATH="/Users/fabien/Library/CloudStorage/GoogleDrive-roth.fabien@gmail.com/My Drive/MSC_Datascience/02 - Courses/TSM-MachLeData/a-guide-to-mlops/RWFydGhfMTQ4.jpg"

# Check if the image file exists before proceeding
if [ ! -f "$IMAGE_PATH" ]; then
  echo "Error: Image file not found at $IMAGE_PATH"
  exit 1
fi

# Array of valid model types
model_types=("baseline" "ptq_integer" "ptq_float16" "ptq_dynamic")

# Test valid cases
for model in "${model_types[@]}"; do
  echo "Testing valid case for model_type: $model"
  curl -X 'POST' \
    "$SERVICE_URL" \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F "image=@${IMAGE_PATH};type=image/jpeg" \
    -F "model_type=$model"
  echo -e "\n-----------------------------"
done

# Test invalid cases
echo "Testing missing model_type"
curl -X 'POST' \
  "$SERVICE_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F "image=@${IMAGE_PATH};type=image/jpeg"
echo -e "\n-----------------------------"

echo "Testing invalid model_type"
curl -X 'POST' \
  "$SERVICE_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F "image=@${IMAGE_PATH};type=image/jpeg" \
  -F 'model_type=invalid_model'
echo -e "\n-----------------------------"

echo "Testing missing image"
curl -X 'POST' \
  "$SERVICE_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'model_type=baseline'
echo -e "\n-----------------------------"

echo "Testing invalid image"
curl -X 'POST' \
  "$SERVICE_URL" \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@/nonexistent.jpg;type=image/jpeg' \
  -F 'model_type=baseline'
echo -e "\n-----------------------------"

# Additional cases to trigger specific metrics
echo "Simulating CPU-intensive task"
python3 -c "for _ in range(10**6): pass"
echo -e "\n-----------------------------"

echo "Simulating memory load"
python3 -c "x = [0] * (10**7); del x"
echo -e "\n-----------------------------"

echo "Simulating combined CPU and Memory load"
python3 -c "x = [0] * (10**6); [None for _ in range(10**6)]; del x"
echo -e "\n-----------------------------"