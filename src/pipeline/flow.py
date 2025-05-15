"""Prefect flow for the clinical trials pipeline.

This module defines the Prefect flow for orchestrating the clinical trials
data pipeline, including extraction, transformation, enrichment, and analysis.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import tempfile

import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ---- Health check server (required for Cloud Run TCP probe) ----
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_response(404)
            self.end_headers()

def start_health_server():
    server = HTTPServer(("0.0.0.0", 8080), HealthCheckHandler)
    print("Health check server running on port 8080")
    server.serve_forever()

threading.Thread(target=start_health_server, daemon=True).start()

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.task_runners import SequentialTaskRunner

from src.pipeline.analysis import analyze_trials
from src.pipeline.config import settings
from src.pipeline.enrich import apply_enrichment_to_trials, enrich_drugs
from src.pipeline.etl import (
    extract_all_clinical_trials,
    extract_intervention_names,
    load_and_transform_from_raw,
    transform_clinical_trials,
)
from src.pipeline.utils import get_timestamp, upload_to_gcs, download_from_gcs


@task(name="extract_clinical_trials")
async def extract_trials_task(
    disease: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> List[Dict]:
    """Extract clinical trials data from ClinicalTrials.gov.
    
    Args:
        disease: Disease condition to search for
        year_start: Start year for study search
        year_end: End year for study search
        
    Returns:
        List of clinical trial data
    """
    logger = get_run_logger()
    
    logger.info(f"Extracting clinical trials for {disease or settings.disease}")
    studies = await extract_all_clinical_trials(disease, year_start, year_end)
    
    logger.info(f"Extracted {len(studies)} clinical trials")
    return studies


@task(name="transform_clinical_trials")
def transform_trials_task(studies: List[Dict]) -> pd.DataFrame:
    """Transform raw clinical trial data into a structured DataFrame.
    
    Args:
        studies: List of clinical trial studies
        
    Returns:
        DataFrame with transformed clinical trial data
    """
    logger = get_run_logger()
    
    logger.info("Transforming clinical trial data")
    df = transform_clinical_trials(studies)
    
    logger.info(f"Transformed data with {len(df)} rows")
    return df


@task(name="extract_drug_names")
def extract_drugs_task(studies: List[Dict]) -> Set[str]:
    """Extract unique drug names from clinical trial data.
    
    Args:
        studies: List of clinical trial studies
        
    Returns:
        Set of unique drug names
    """
    logger = get_run_logger()
    
    logger.info("Extracting unique drug names")
    drug_names = extract_intervention_names(studies)
    
    logger.info(f"Extracted {len(drug_names)} unique drug names")
    return drug_names


@task(name="enrich_drug_data")
async def enrich_drugs_task(drug_names: Set[str]) -> Dict:
    """Enrich drug data with modality and target information.
    
    Args:
        drug_names: Set of drug names to enrich
        
    Returns:
        Dictionary mapping drug names to their enrichment information
    """
    logger = get_run_logger()
    
    logger.info(f"Enriching {len(drug_names)} drugs")
    drug_info = await enrich_drugs(drug_names)
    
    logger.info(f"Enriched {len(drug_info)} drugs")
    return drug_info


@task(name="apply_drug_enrichment")
def apply_enrichment_task(
    trials_df: pd.DataFrame, drug_info: Dict
) -> pd.DataFrame:
    """Apply drug enrichment data to trials DataFrame.
    
    Args:
        trials_df: DataFrame with clinical trial data
        drug_info: Dictionary mapping drug names to their enrichment information
        
    Returns:
        Enriched DataFrame
    """
    logger = get_run_logger()
    
    logger.info("Applying drug enrichment to trials data")
    enriched_df = apply_enrichment_to_trials(trials_df, drug_info)
    
    logger.info(f"Applied enrichment to {len(enriched_df)} trials")
    return enriched_df


@task(name="analyze_trials")
def analyze_trials_task(
    enriched_df: pd.DataFrame,
) -> Tuple[Dict, str]:
    """Analyze clinical trial data and generate insights.
    
    Args:
        enriched_df: Enriched DataFrame with clinical trial data
        
    Returns:
        Tuple of (summary stats dictionary, insights text)
    """
    logger = get_run_logger()
    
    logger.info("Analyzing clinical trial data")
    
    # Check if DataFrame is empty
    if enriched_df.empty:
        logger.warning("DataFrame is empty, generating minimal analysis output")
    
    stats, insights = analyze_trials(enriched_df)
    
    logger.info("Analysis completed")
    return stats, insights


@task(name="generate_release_files")
def generate_release_files(
    enriched_df: pd.DataFrame, 
    insights: str,
    timestamp: Optional[str] = None,
) -> Path:
    """Generate release files with analysis results.
    
    Args:
        enriched_df: Enriched DataFrame with clinical trial data
        insights: Markdown-formatted insights text
        timestamp: Timestamp string for file naming
        
    Returns:
        Path to the release directory
    """
    logger = get_run_logger()
    if timestamp is None:
        timestamp = get_timestamp()
    # GCS base path for this run
    gcs_base = f"runs/{timestamp}/release"
    # Write CSV to temp file, upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tmp_csv:
        enriched_df.to_csv(tmp_csv.name, index=False)
        upload_to_gcs(tmp_csv.name, f"{gcs_base}/clinical_trials_{timestamp}.csv")
    # Write README.md to temp file, upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".md", delete=True, mode="w+") as tmp_readme:
        tmp_readme.write(f"# Clinical Trials Analysis for {settings.disease}\n\n")
        tmp_readme.write(f"Analysis run on: {timestamp}\n\n")
        tmp_readme.write(insights)
        tmp_readme.flush()
        upload_to_gcs(tmp_readme.name, f"{gcs_base}/README.md")
    # Write demo.md to temp file, upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".md", delete=True, mode="w+") as tmp_demo:
        tmp_demo.write(f"# 5-Minute Demo Script: Clinical Trials Analysis for {settings.disease}\n\n")
        tmp_demo.write("## 1. Introduction (30 seconds)\n")
        tmp_demo.write("- Brief explanation of the dataset and analysis goals\n")
        tmp_demo.write("- Overview of the disease area studied\n\n")
        tmp_demo.write("## 2. Data Collection Process (1 minute)\n")
        tmp_demo.write("- ClinicalTrials.gov API extraction workflow\n")
        tmp_demo.write("- Drug enrichment process using multiple data sources\n\n")
        tmp_demo.write("## 3. Key Findings (2 minutes)\n")
        if enriched_df.empty:
            tmp_demo.write("- **No data found** for the specified criteria\n")
            tmp_demo.write("- Discuss potential reasons for lack of data\n")
            tmp_demo.write("- Suggest alternative search strategies\n\n")
        else:
            tmp_demo.write("- Present top modalities and targets\n")
            tmp_demo.write("- Show enrollment and duration patterns\n")
            tmp_demo.write("- Highlight any notable trends over time\n\n")
        tmp_demo.write("## 4. Interactive Visualization Demo (1 minute)\n")
        if enriched_df.empty:
            tmp_demo.write("- Discuss how visualizations would be presented if data were available\n")
            tmp_demo.write("- Show sample visualizations from previous analyses\n\n")
        else:
            tmp_demo.write("- Walk through Plotly interactive charts\n")
            tmp_demo.write("- Show how to filter and explore the data\n\n")
        tmp_demo.write("## 5. Conclusions & Next Steps (30 seconds)\n")
        tmp_demo.write("- Summarize key insights\n")
        tmp_demo.write("- Suggest potential follow-up analyses\n")
        tmp_demo.flush()
        upload_to_gcs(tmp_demo.name, f"{gcs_base}/demo.md")
    logger.info(f"Generated release files and uploaded to GCS under {gcs_base}")
    return Path(gcs_base)


@flow(
    name="clinical_trials_pipeline",
    task_runner=SequentialTaskRunner(),
    description="End-to-end pipeline for clinical trials data analysis",
)
async def clinical_trials_pipeline(
    disease: str = "Familial Hypercholesterolemia",
    year_start: int = 2009,
    year_end: int = 2024,
    max_studies: int = 1000,
    max_pages: int = 20,
    use_cached_raw_data: bool = False,
    timestamp: Optional[str] = None,
) -> Path:
    """Run the complete clinical trials data pipeline.
    
    Args:
        disease: Disease condition to search for
        year_start: Start year for study search
        year_end: End year for study search
        max_studies: Maximum number of studies to extract
        max_pages: Maximum number of pages to extract
        use_cached_raw_data: Whether to use cached raw data
        timestamp: Timestamp string for file naming
        
    Returns:
        Path to the release directory
    """
    logger = get_run_logger()

    # Download the shared drug cache from GCS at the start
    local_cache_path = str(settings.cache_db_path)
    gcs_cache_path = "cache/drug_cache.sqlite"
    download_from_gcs(gcs_cache_path, local_cache_path)

    # Override singleton settings with UI parameters if provided
    if disease is not None:
        settings.disease = disease
    if year_start is not None:
        settings.year_start = year_start
    if year_end is not None:
        settings.year_end = year_end
    if max_studies is not None:
        settings.max_studies = max_studies
    if max_pages is not None:
        settings.max_pages = max_pages

    disease = settings.disease
    year_start = settings.year_start
    year_end = settings.year_end
    
    if timestamp is None:
        timestamp = get_timestamp()
    
    logger.info(f"Starting clinical trials pipeline for {disease} ({year_start}-{year_end})")
    
    # Step 1: Extract data
    if use_cached_raw_data:
        logger.info("Using cached raw data")
        trials_df = await load_and_transform_from_raw(timestamp)
        # We also need the raw studies for drug extraction
        from src.pipeline.utils import get_raw_data_path, load_json
        raw_path = get_raw_data_path(timestamp) / "all_studies.json"
        raw_data = load_json(raw_path)
        studies = raw_data.get("studies", [])
    else:
        studies = await extract_trials_task(disease, year_start, year_end)
        trials_df = transform_trials_task(studies)
    
    # Step 2: Extract and enrich drug data
    drug_names = extract_drugs_task(studies)
    drug_info = await enrich_drugs_task(drug_names)
    
    # Step 3: Apply enrichment and analyze
    enriched_df = apply_enrichment_task(trials_df, drug_info)
    stats, insights = analyze_trials_task(enriched_df)
    
    # Step 4: Generate release files
    release_dir = generate_release_files(enriched_df, insights, timestamp)
    
    # Upload the updated drug cache to GCS at the end
    upload_to_gcs(local_cache_path, gcs_cache_path)

    logger.info(f"Pipeline completed successfully. Release files available at {release_dir}")
    return release_dir


if __name__ == "__main__":
    # This allows the flow to be run as a script
    asyncio.run(clinical_trials_pipeline()) 