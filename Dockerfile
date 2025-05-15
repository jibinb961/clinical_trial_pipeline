FROM python:3.12-slim

# Install Poetry and system dependencies
RUN apt-get update && apt-get install -y curl git build-essential
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

# Set working directory
WORKDIR /app

# Install Python dependencies using Poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy source code
COPY . .

# Prefect logging level
ENV PREFECT_LOGGING_LEVEL="INFO"

# Start a dummy HTTP server in the background to satisfy Cloud Run's TCP probe,
# then start the Prefect agent with queue `default`
CMD ["sh", "-c", "python3 -m http.server 8080 & prefect cloud login --key $PREFECT_API_KEY --workspace $PREFECT_WORKSPACE && prefect deployment build src/pipeline/flow.py:clinical_trials_pipeline -n cloud-deploy -q default --infra process --output deployment.yaml --skip-upload && prefect deployment apply deployment.yaml && prefect agent start -q default"]
