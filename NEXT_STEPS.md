# Next Steps for Clinical Trials Pipeline

This document provides instructions on how to set up and run the clinical trials data pipeline after cloning the repository.

## 1. Set Up Environment Variables

First, create a `.env` file with your API keys and configuration:

```bash
cp .env.example .env
```

Edit the `.env` file to add your API keys:

```
# Clinical trials extraction parameters
DISEASE=Familial Hypercholesterolemia
YEAR_START=2008
YEAR_END=2023

# API key for drug information enrichment 
GEMINI_API_KEY=your_actual_gemini_key_here

# Optional configuration
LOG_LEVEL=INFO
```

At minimum, you should provide a `GEMINI_API_KEY` as it's the fallback enrichment source when ChEMBL doesn't have data.

## 2. Install Dependencies with Poetry

Install Poetry if you don't have it already:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then install the project dependencies:

```bash
poetry install
```

## 3. Running the Pipeline

### Option 1: Direct Python Execution

```bash
poetry run python -m src.pipeline.flow
```

### Option 2: Using Prefect UI

Start the Prefect server:

```bash
poetry run prefect server start
```

In a new terminal, create and apply the deployment:

```bash
poetry run prefect deployment build src.pipeline.flow:clinical_trials_pipeline -n default -q default
poetry run prefect deployment apply clinical_trials_pipeline-deployment.yaml
```

Start a Prefect agent to execute the flow:

```bash
poetry run prefect agent start -q default
```

Then run the flow from the Prefect UI (http://localhost:4200).

## 4. Using Docker

Build and run with Docker Compose:

```bash
docker-compose up --build
```

This will start both the Prefect server and the clinical trials pipeline container.

Access the Prefect UI at http://localhost:4200 to monitor the flow run.

## 5. Running the Dashboard

After the pipeline has completed, you can run the Streamlit dashboard:

```bash
poetry run streamlit run app/streamlit_app.py
```

The dashboard will be available at http://localhost:8501.

## 6. CI/CD Setup

To enable CI/CD with GitHub Actions:

1. Push the repository to GitHub:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/clinical-trials-pipeline.git
git push -u origin main
```

2. Add repository secrets in GitHub:
   - Go to your repository → Settings → Secrets and variables → Actions
   - Add secrets for `GEMINI_API_KEY`