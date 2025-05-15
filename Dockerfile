FROM python:3.12-slim

# Install Poetry
RUN apt-get update && apt-get install -y curl git build-essential
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Copy dependency files and install
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy source code
COPY . .

# Set logging level
ENV PREFECT_LOGGING_LEVEL="INFO"

# Agent-only entrypoint for Cloud Run
CMD ["sh", "-c", "prefect config set PREFECT_API_KEY=$PREFECT_API_KEY && prefect agent start --pool default-agent-pool"]