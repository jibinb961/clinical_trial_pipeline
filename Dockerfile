FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl git build-essential

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-interaction --no-ansi --no-root

COPY . .

ENV PREFECT_LOGGING_LEVEL="INFO"

# Just keep it idle. Prefect will auto-invoke flows in Cloud Run jobs.
CMD ["echo", "Container ready for Prefect Cloud Run job."]