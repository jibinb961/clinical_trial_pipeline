FROM python:3.12-slim

# Install Poetry
RUN apt-get update && apt-get install -y curl git build-essential
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

# Set working directory
WORKDIR /app

# Install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy full source code
COPY . .

# Set environment variables for Prefect Cloud
#ENV PREFECT_API_URL="https://api.prefect.cloud/api/accounts/6c1c0ae5-5bb6-4bc8-ac9e-c62e0156e715/workspaces/5d172eba-a9ce-4180-87e8-eba9b6b5e425"
#ENV PREFECT_UI_URL="https://app.prefect.cloud"
ENV PREFECT_LOGGING_LEVEL="INFO"

# Entrypoint to register and run the deployment agent
CMD bash -c "\
    prefect cloud login --key $PREFECT_API_KEY --workspace $PREFECT_WORKSPACE && \
    prefect deployment build src/pipeline/flow.py:clinical_trials_pipeline \
      -n cloud-deploy \
      -q default \
      --infra process \
      --output deployment.yaml \
      --skip-upload && \
    prefect deployment apply deployment.yaml && \
    prefect agent start -q default"