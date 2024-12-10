#!/bin/bash

set -e  # Exit on error

REPOSITORY_OWNER=fabien-roth
REPOSITORY_NAME=a-guide-to-mlops

# Set the runner token (expires after 1 hour)
set_token() {
    REG_TOKEN=$(curl -s -X POST -H "Accept: application/vnd.github.v3+json" -H "Authorization: token ${GITHUB_RUNNER_PAT}" https://api.github.com/repos/${REPOSITORY_OWNER}/${REPOSITORY_NAME}/actions/runners/registration-token | jq -r .token)
}

# Configure the runner
set_token
./config.sh --unattended \
    --url https://github.com/${REPOSITORY_OWNER}/${REPOSITORY_NAME} \
    --replace --labels ${GITHUB_RUNNER_LABEL} --token ${REG_TOKEN}

# Cleanup the runner
cleanup() {
    echo "Removing runner..."
    set_token
    ./config.sh remove --unattended --token ${REG_TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# Start the runner
./run.sh > run.log 2>&1 & wait $!