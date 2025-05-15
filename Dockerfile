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

# Run Prefect worker (new CLI, replaces agent)
CMD ["prefect", "worker", "start", "--pool", "default-agent-pool"]
