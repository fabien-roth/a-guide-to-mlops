FROM ubuntu:22.04

ENV RUNNER_VERSION=2.319.1

LABEL RunnerVersion=${RUNNER_VERSION}
LABEL org.opencontainers.image.source="https://github.com/fabien-roth/a-guide-to-mlops"

# Install dependencies
RUN apt-get update -y && \
    apt-get install -y build-essential lsb-release python3 python3-pip \
    curl jq vim gpg wget git unzip tar gettext-base

# Add a non-root user
RUN useradd -m runner

WORKDIR /home/actions-runner

# Install GitHub Actions Runner
RUN curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
RUN tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Install extra dependencies for the runner
RUN ./bin/installdependencies.sh

COPY startup.sh .

RUN chmod +x startup.sh

USER runner

ENTRYPOINT ["./startup.sh"]