#!/usr/bin/env bash
#
# Extended test script for baseline and canary services.
# Generates more requests (multiple rounds) to ensure plenty of metrics data.
# Sends various valid/invalid requests to both endpoints.
#
# Adjust IMAGE_PATH before running.
#
# Run:
#   chmod +x test_both_containers.txt
#   ./test_both_containers.txt

set -euo pipefail
set -x

#####################################
# CONFIGURE THESE VARIABLES FIRST
#####################################

# Baseline and canary service URLs
BASELINE_URL="http://34.65.152.219:3000/predict"
CANARY_URL="http://34.65.100.195:3000/predict"

# Path to your test image
IMAGE_PATH="/Users/fabien/Library/CloudStorage/GoogleDrive-roth.fabien@gmail.com/My Drive/MSC_Datascience/02 - Courses/TSM-MachLeData/a-guide-to-mlops/RWFydGhfMTQ4.jpg"

# Validate image
if [ ! -f "$IMAGE_PATH" ]; then
  echo "Error: Image file not found at $IMAGE_PATH"
  exit 1
fi

model_types=("baseline" "ptq_integer" "ptq_float16" "ptq_dynamic")

# Function to print a header
print_section() {
  echo -e "\n============================="
  echo -e "$1"
  echo -e "=============================\n"
}

# We'll send multiple rounds of requests
ROUNDS=5

send_requests() {
  local SERVICE_URL="$1"
  local SERVICE_NAME="$2"

  print_section "Testing Valid model_types on $SERVICE_NAME ($ROUNDS rounds)"
  for (( round=1; round<=$ROUNDS; round++ )); do
    for model in "${model_types[@]}"; do
      echo "[$SERVICE_NAME - Round $round] Valid model_type: $model"
      curl -X POST "$SERVICE_URL" \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F "image=@${IMAGE_PATH};type=image/jpeg" \
        -F "model_type=$model"
      echo "Curl exit code: $?"
      echo -e "-----------------------------\n"
    done
  done

  print_section "Testing missing model_type on $SERVICE_NAME ($ROUNDS times)"
  for (( i=1; i<=$ROUNDS; i++ )); do
    echo "[$SERVICE_NAME - Missing model_type - Attempt $i]"
    curl -X POST "$SERVICE_URL" \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F "image=@${IMAGE_PATH};type=image/jpeg"
    echo "Curl exit code: $?"
    echo -e "-----------------------------\n"
  done

  print_section "Testing invalid model_type on $SERVICE_NAME ($ROUNDS times)"
  for (( i=1; i<=$ROUNDS; i++ )); do
    echo "[$SERVICE_NAME - Invalid model_type=invalid_model - Attempt $i]"
    curl -X POST "$SERVICE_URL" \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F "image=@${IMAGE_PATH};type=image/jpeg" \
      -F "model_type=invalid_model"
    echo "Curl exit code: $?"
    echo -e "-----------------------------\n"
  done

  print_section "Testing missing image on $SERVICE_NAME ($ROUNDS times)"
  for (( i=1; i<=$ROUNDS; i++ )); do
    echo "[$SERVICE_NAME - Missing image - Attempt $i]"
    curl -X POST "$SERVICE_URL" \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'model_type=baseline'
    echo "Curl exit code: $?"
    echo -e "-----------------------------\n"
  done

  print_section "Testing invalid image on $SERVICE_NAME ($ROUNDS times)"
  for (( i=1; i<=$ROUNDS; i++ )); do
    echo "[$SERVICE_NAME - Invalid image path - Attempt $i]"
    curl -X POST "$SERVICE_URL" \
      -H 'accept: application/json' \
      -H 'Content-Type: multipart/form-data' \
      -F 'image=@/nonexistent.jpg;type=image/jpeg' \
      -F 'model_type=baseline'
    echo "Curl exit code: $?"
    echo -e "-----------------------------\n"
  done
}

# Run tests on both baseline and canary
send_requests "$BASELINE_URL" "Baseline Service"
send_requests "$CANARY_URL" "Canary Service"

print_section "All tests completed"
echo "Check your Prometheus/Grafana dashboards now to see the newly collected data."
