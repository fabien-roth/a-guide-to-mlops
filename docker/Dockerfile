# Base image
FROM ubuntu:22.04

ENV RUNNER_VERSION=2.319.1

LABEL RunnerVersion=${RUNNER_VERSION}
LABEL org.opencontainers.image.source="https://github.com/fabien-roth/a-guide-to-mlops"

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y build-essential lsb-release python3 python3-pip \
    curl jq vim gpg wget git unzip tar gettext-base && \
    apt-get clean

# Add a non-root user
RUN useradd -m runner

# Set up the working directory for GitHub Actions Runner
WORKDIR /home/actions-runner

# Install GitHub Actions Runner
RUN curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz && \
    rm actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Install extra dependencies for the runner
RUN ./bin/installdependencies.sh

# Add BentoML environment setup
WORKDIR /home/bentoml/bento
COPY . /home/bentoml/bento
RUN pip install --upgrade pip && \
    pip install -r /home/bentoml/bento/env/python/requirements.lock.txt

# Ensure BentoML can run the service
CMD ["bentoml", "serve", "/home/bentoml/bento"]

# Ensure startup script for GitHub Actions Runner is preserved
COPY startup.sh /home/actions-runner/startup.sh
RUN chmod +x /home/actions-runner/startup.sh

# Switch to the non-root user for security
USER runner

# Entrypoint for the container
ENTRYPOINT ["./startup.sh"]