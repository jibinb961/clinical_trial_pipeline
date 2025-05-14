# Clinical Trial Pipeline: Technical Explanation

## Overview

This document provides a deep, code-level explanation of the clinical trial pipeline, focusing on the enrichment and analysis logic, async flows, and the overall workflow. The pipeline is designed to efficiently extract, enrich, and analyze clinical trial data, with robust handling of drug enrichment using ChEMBL and Gemini (LLM), and strong guarantees on output consistency and reproducibility.

---

## Pipeline Workflow

The pipeline is orchestrated using [Prefect](https://www.prefect.io/) and consists of the following main stages:

1. **ETL (Extract, Transform, Load)**
2. **Drug Enrichment (Two-Stage: ChEMBL, then Gemini)**
3. **Application of Enrichment to Trials**
4. **Analysis and Reporting**
5. **Release and Artifact Generation**

Each stage is implemented as a Prefect task or async function, with careful management of concurrency, caching, and error handling. The modular design allows for easy extension, robust error recovery, and reproducibility of results.

---

## 1. ETL (Extract, Transform, Load)

### What Happens and Why
- **Extraction**: The pipeline fetches clinical trial data from the ClinicalTrials.gov API for a specified disease and year range. This ensures the dataset is up-to-date and relevant to the research question.
- **Transformation**: Raw JSON data is normalized into a structured DataFrame, making downstream analysis and enrichment possible.

### Code Implementation
- **Extraction** is handled by `extract_all_clinical_trials` in `etl.py`. It:
  - Builds API queries for the disease and date range.
  - Handles pagination using `nextPageToken`.
  - Applies rate limiting and retries (using `tenacity` and `fetch_with_retry` from `utils.py`).
  - Filters studies to include only those with drug interventions, industry sponsors, and interventional study type.
  - Saves both raw and filtered data to disk for reproducibility.
- **Transformation** is performed by `transform_clinical_trials`, which:
  - Extracts and normalizes fields like intervention names, outcomes, sponsors, and dates.
  - Handles missing or malformed data gracefully.
  - Returns a Pandas DataFrame ready for enrichment and analysis.

### Edge Cases and Design Decisions
- **Error Handling**: Network errors, malformed JSON, and API rate limits are all handled with retries and logging. Malformed responses are saved for debugging.
- **Reproducibility**: All raw and filtered data is timestamped and saved, so any run can be reproduced exactly.
- **Extensibility**: The ETL logic is modular, so new filters or data sources can be added easily.

### Example
```python
studies = await extract_all_clinical_trials(disease="FH", year_start=2010, year_end=2024)
df = transform_clinical_trials(studies)
```

---

## 2. Drug Enrichment (Two-Stage Pipeline)

### What Happens and Why
- The goal is to annotate each drug intervention with its **modality** (e.g., small-molecule, antibody) and **target** (e.g., gene, protein, pathway), using trusted sources and minimizing LLM calls.
- The two-stage approach ensures:
  - **Efficiency**: ChEMBL is used first for all drugs, as it is fast and free.
  - **Completeness**: Only unresolved drugs are sent to Gemini (LLM), and only in a single batch, reducing cost and latency.
  - **Consistency**: Placebos and edge cases are handled strictly, and all enrichment is cached.

### Code Implementation
- **Preprocessing** (`preprocess_drug_name` in `enrich.py`):
  - Handles placebos, strips dosages, splits combos (e.g., `DrugA + DrugB`), and removes annotations.
  - Ensures that variants (e.g., with/without dosage) are normalized for lookup and caching.
- **ChEMBL Query** (`query_chembl_client`):
  - Checks the local SQLite cache first (`get_cached_drug`).
  - If not cached, queries ChEMBL for modality and target.
  - Maps ChEMBL molecule types to standardized modalities.
  - Caches results for future runs (`cache_drug`).
- **Gemini (LLM) Query** (`batch_query_gemini`):
  - All unresolved drugs (not placebos, not resolved by ChEMBL) are sent in a single batch to Gemini.
  - The prompt is carefully crafted to:
    - Return lists for combos, use gene symbols/protein names for targets, and label placebos/devices/procedures correctly.
    - Only return "Unknown" if truly no information is available.
  - Results are parsed and cached, with source attribution set to "Gemini" if attempted.
- **Main Enrichment Function** (`enrich_drugs`):
  - Orchestrates the two-stage process, manages unresolved drugs, and ensures output consistency.
  - Handles combos, placebos, and edge cases.

### Edge Cases and Design Decisions
- **Placebo Handling**: Placebos are always labeled as "placebo" for all fields and never sent to ChEMBL or Gemini.
- **Unknowns**: If both ChEMBL and Gemini fail, the source is still set to "Gemini" (never "Unknown").
- **Caching**: All enrichment results are cached in a local SQLite DB, reducing redundant API/LLM calls and ensuring reproducibility.
- **Batching**: Gemini is always called in a single batch for all unresolved drugs, minimizing LLM usage and cost.

### Example
```python
drug_names = extract_intervention_names(studies)
drug_info = await enrich_drugs(drug_names)
```

---

## 3. Application of Enrichment to Trials

### What Happens and Why
- The enriched drug info is mapped back to each trial's interventions, so that every trial row in the DataFrame has its interventions annotated with modality, target, and source.
- This enables downstream analysis by modality/target and ensures full traceability of enrichment sources.

### Code Implementation
- `apply_enrichment_to_trials` in `enrich.py`:
  - For each trial, looks up each intervention in the `drug_info` dictionary.
  - Adds columns: `modalities`, `targets`, `enrichment_sources` (all as lists, flattened for combos).
  - Saves the enriched DataFrame as Parquet and generates a detailed enrichment report CSV for auditing.

### Edge Cases and Design Decisions
- **Missing Drugs**: If a drug is not found in `drug_info`, it is labeled as "Unknown" for all fields.
- **Combos**: For combination drugs, lists of modalities/targets/sources are kept in order.
- **Auditability**: The enrichment report CSV provides a flat, row-wise view for easy inspection.

### Example
```python
enriched_df = apply_enrichment_to_trials(trials_df, drug_info)
```

---

## 4. Analysis and Reporting

### What Happens and Why
- The pipeline generates summary statistics, top modalities/targets, trends, and visualizations to provide actionable insights from the enriched clinical trial data.
- Markdown summaries and demo scripts are generated for communication and reproducibility.

### Code Implementation
- `analysis.py` contains:
  - `generate_summary_statistics`: Computes total trials, completed/ongoing counts, enrollment/duration stats, phase counts.
  - `generate_modality_counts`, `generate_target_counts`: Analyze top modalities/targets.
  - `generate_yearly_modality_data`, `generate_sponsor_activity_over_time`: Trends over time.
  - Plotting functions: Generate figures (matplotlib/plotly) for modalities, targets, sponsors, enrollment, etc.
  - `analyze_trials`: Orchestrates the analysis and returns both stats and markdown insights.
- **LLM-based outcome clustering**: This step is now removed from the main pipeline; raw outcomes are kept for later clustering if needed.

### Edge Cases and Design Decisions
- **Empty Data**: If the DataFrame is empty, the analysis functions return minimal output and log a warning.
- **Extensibility**: New analysis or visualization functions can be added easily.
- **Reproducibility**: All figures and markdown summaries are saved to the release directory.

### Example
```python
stats, insights = analyze_trials(enriched_df)
```

---

## 5. Async and Task Orchestration

### What Happens and Why
- The pipeline is orchestrated as a Prefect flow, with each stage as a Prefect task. This enables robust scheduling, monitoring, and error recovery.
- Async functions are used for extraction and enrichment to maximize efficiency and minimize latency.

### Code Implementation
- `flow.py` defines the main Prefect flow (`clinical_trials_pipeline`) and all tasks:
  - Extraction, transformation, drug extraction, enrichment, application, analysis, and release are all separate tasks.
  - Async tasks (extraction, enrichment) are awaited; sync tasks (transformation, analysis) are run in sequence.
- **Async Batching**: Drug enrichment batches Gemini calls for all unresolved drugs, and uses async DB/cache operations.
- **Error Handling**: Retries (with exponential backoff) are implemented for all network/API calls using `retry_async` from `utils.py`.
- **Logging**: All stages log progress, errors, and key events for traceability.

### Example
```python
from src.pipeline.flow import clinical_trials_pipeline
import asyncio
asyncio.run(clinical_trials_pipeline(...))
```

---

## 6. LLM (Gemini) Usage Minimization

### What Happens and Why
- To reduce cost and latency, the pipeline minimizes LLM (Gemini) usage by:
  - Always using ChEMBL first for all drugs.
  - Only sending unresolved drugs to Gemini, and only in a single batch call.
  - Never sending placebos to Gemini.
- The Gemini prompt is optimized for batch, structured, and minimal output.

### Code Implementation
- `enrich.py` and `gemini_utils.py`:
  - ChEMBL is always queried first; only unresolved drugs are batched for Gemini.
  - The Gemini prompt is designed to return lists for combos, use gene/protein names, and handle placebos/devices/procedures.
  - All Gemini results are cached for future runs.

### Example
```python
# Only unresolved drugs are sent to Gemini in a single batch
unresolved = [d for d in drug_names if not resolved_by_chembl(d)]
gemini_results = await batch_query_gemini(unresolved)
```

---

## 7. Output Consistency and Placebo Handling

### What Happens and Why
- The pipeline enforces strict output consistency:
  - Placebos are always labeled as "placebo" for all fields and never sent to ChEMBL or Gemini.
  - If ChEMBL returns "Unknown", Gemini is always used, and the source is set to "Gemini" if attempted.
  - No drug has "Unknown" as the source if Gemini was attempted.
  - Output is consistent for all drugs, including those with dosage/annotation variants.

### Code Implementation
- `enrich.py`:
  - Placebo detection and labeling in `preprocess_drug_name` and `enrich_drugs`.
  - Source attribution logic ensures "Gemini" is set if Gemini was attempted.
  - Output is always a dict with `modality`, `target`, and `source` for every drug.

### Example
```python
# Placebos are always handled strictly
drug_info = await enrich_drugs({"Placebo", "DrugA 10mg"})
assert drug_info["Placebo"]["modality"] == "placebo"
```

---

## 8. Configuration and Extensibility

### What Happens and Why
- All pipeline settings (API keys, paths, disease, year range, concurrency, etc.) are managed via a Pydantic config class, making the pipeline easy to configure and extend.
- Environment variables and `.env` files are supported for secrets and overrides.
- All data and artifact paths are auto-created and managed.

### Code Implementation
- `config.py`:
  - `Settings` class manages all configuration, using Pydantic for validation and environment variable support.
  - Paths for raw, processed, cache, figures, and release artifacts are auto-created.
- **Extensibility**: New settings can be added easily; the config is imported everywhere as `settings`.

### Example
```python
from src.pipeline.config import settings
print(settings.disease, settings.year_start, settings.api_keys.gemini)
```

---

## 9. Release and Artifacts

### What Happens and Why
- At the end of the pipeline, all results (enriched data, figures, markdown insights, demo scripts) are saved to a timestamped release directory for reproducibility and sharing.
- Artifacts include CSV/Parquet data, enrichment reports, figures, markdown summaries, and demo scripts.

### Code Implementation
- `flow.py`:
  - The `generate_release_files` task saves all outputs to the release directory.
  - Figures are copied, markdown summaries and demo scripts are generated, and all data is saved with timestamps.

### Example
```python
release_dir = generate_release_files(enriched_df, insights, timestamp)
```

---

## Example: End-to-End Workflow (Code)

```python
from src.pipeline.flow import clinical_trials_pipeline

# Run the full pipeline (async)
import asyncio
asyncio.run(clinical_trials_pipeline(
    disease="Familial Hypercholesterolemia",
    year_start=2010,
    year_end=2024,
    use_cached_raw_data=False
))
```

---

## Key Takeaways
- The pipeline is robust, efficient, and reproducible.
- LLM (Gemini) usage is minimized and always batched.
- Placebo and edge-case handling is strict and consistent.
- All enrichment is cached, and output is fully auditable.
- The workflow is fully async and orchestrated with Prefect for reliability and scalability. 