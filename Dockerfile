FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies required for python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.6.1

# Copy poetry configuration
COPY pyproject.toml poetry.lock* ./

# Configure poetry to not use a virtual environment in the container
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw data/processed data/figures release

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the application
CMD ["python", "-m", "src.pipeline.flow"] 