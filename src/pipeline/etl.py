"""ETL module for extracting and transforming clinical trial data.

This module handles extraction of clinical trial data from the ClinicalTrials.gov API
and transformation of the raw data into structured formats.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
import json
import hashlib
import logging

from src.pipeline.config import settings
from src.pipeline.utils import (
    fetch_with_retry,
    get_raw_data_path,
    get_timestamp,
    load_json,
    log_execution_time,
    logger,
    save_json,
)


async def build_clinical_trials_query(
    disease: str, year_start: int, year_end: int
) -> Dict[str, Any]:
    """Build query parameters for ClinicalTrials.gov API.
    
    Args:
        disease: Disease condition to search for
        year_start: Start year for study search
        year_end: End year for study search
        
    Returns:
        Dictionary of query parameters
    """
    # Use a simple search term for maximum results
    search_term = f'AREA[ConditionSearch] "{disease}"'
    
    logger.info(f"Starting extraction for disease: {disease}, years: {year_start}-{year_end}")
    query_params = {
        "query.term": search_term,
        "pageSize": settings.ctgov.page_size,
        "format": "json",
        "countTotal": "true",
        "fields": "protocolSection,hasResults",
    }
    logger.info(f"API query params: {query_params}")
    
    logger.info(
        f"Built query for {disease} clinical trials between {year_start} and {year_end}"
    )
    return query_params


async def extract_clinical_trials_page(
    session: aiohttp.ClientSession,
    url: str,
    params: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Extract a single page of clinical trial data from the API.
    
    Args:
        session: aiohttp client session
        url: API URL
        params: Query parameters
        
    Returns:
        Tuple of (list of studies, next page URL or None)
    """
    # STEP 2: Extract a single page of data
    logger.debug(f"Fetching page with params: {params}")
    
    try:
        response_data = await fetch_with_retry(session, url, params)
        
        studies = response_data.get("studies", [])
        links = response_data.get("links", [])
        
        # Find the "next" link if it exists
        next_url = None
        for link in links:
            if link.get("rel") == "next":
                next_url = link.get("href")
                break
        
        return studies, next_url
        
    except aiohttp.ClientResponseError as e:
        # Per API docs: Handle different status codes appropriately
        if e.status == 429:  # Too Many Requests
            logger.warning(f"Rate limit exceeded. Retrying with backoff: {str(e)}")
            # This will be retried by the fetch_with_retry function
            raise
        elif e.status >= 500:  # Server errors
            logger.warning(f"Server error from ClinicalTrials.gov API: {str(e)}")
            # This will also be retried
            raise
        elif e.status >= 400:  # Client errors (except 429)
            logger.error(f"Client error when querying ClinicalTrials.gov API: {str(e)}")
            # For client errors, return empty results instead of retrying
            return [], None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"Network error when querying ClinicalTrials.gov API: {str(e)}")
        # Network errors will be retried
        raise
    except json.JSONDecodeError as e:
        # Handle malformed JSON
        logger.error(f"Malformed JSON response from ClinicalTrials.gov API: {str(e)}")
        # Save the raw response for debugging
        error_path = settings.paths.raw_data / "errors"
        error_path.mkdir(exist_ok=True)
        with open(error_path / f"malformed_json_{get_timestamp()}.txt", "w") as f:
            f.write(f"URL: {url}\nParams: {params}\nError: {str(e)}")
        return [], None


@log_execution_time
async def extract_all_clinical_trials(
    disease: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    save_raw: bool = True,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract all clinical trial data matching the criteria.
    
    Args:
        disease: Disease condition to search for (defaults to settings)
        year_start: Start year for study search (defaults to settings)
        year_end: End year for study search (defaults to settings)
        save_raw: Whether to save raw JSON responses
        logger: Optional logger object
        
    Returns:
        List of clinical trial studies
    """
    # Use Prefect logger if not provided
    if logger is None:
        try:
            from prefect import get_run_logger
            logger = get_run_logger()
        except ImportError:
            from src.pipeline.utils import logger as default_logger
            logger = default_logger
    
    # STEP 3: Extract all pages of data
    disease = disease or settings.disease
    year_start = year_start or settings.year_start
    year_end = year_end or settings.year_end
    
    logger.info(f"Extracting clinical trials for {disease} ({year_start}-{year_end})")
    
    query_params = await build_clinical_trials_query(disease, year_start, year_end)
    url = str(settings.ctgov.base_url)
    
    all_studies = []
    timestamp = get_timestamp()
    metadata_checked = False
    
    async with aiohttp.ClientSession() as session:
        # Check API metadata to detect schema changes
        metadata_info = await check_api_metadata(session)
        metadata_checked = True
        
        if metadata_info.get("error"):
            logger.warning("Failed to check API metadata, proceeding with extraction anyway")
        else:
            # Save metadata for reference
            raw_path = get_raw_data_path(timestamp)
            save_json(metadata_info, raw_path / "api_metadata.json")
        
        page_number = 1
        next_url = None
        
        while True:
            # Use the next URL if available, otherwise use the base URL with parameters
            current_url = next_url or url
            current_params = {} if next_url else query_params
            
            studies, next_url = await extract_clinical_trials_page(
                session, current_url, current_params
            )
            
            logger.info(f"Retrieved page {page_number} with {len(studies)} studies")
            all_studies.extend(studies)
            
            if save_raw:
                raw_path = get_raw_data_path(timestamp)
                save_json(
                    {"studies": studies, "page": page_number},
                    raw_path / f"page_{page_number}.json",
                )
            
            if not next_url:
                break
                
            page_number += 1
    
    logger.info(f"Extracted total of {len(all_studies)} clinical trials before filtering")
    # Save raw extraction
    raw_path = get_raw_data_path(timestamp)
    raw_path.mkdir(parents=True, exist_ok=True)
    raw_file = raw_path / f"raw_{timestamp}.json"
    save_json({"studies": all_studies}, raw_file)
    logger.info(f"Saved raw extracted studies to {raw_file}")

    # Post-process: filter by start_date, intervention, sponsor, and study type in Python
    def is_in_date_range(study):
        protocol_section = study.get("protocolSection", {})
        status_module = protocol_section.get("statusModule", {})
        start_date = status_module.get("startDateStruct", {}).get("date")
        if not start_date:
            return False
        try:
            year = int(start_date.split("-")[0])
            return year_start <= year <= year_end
        except Exception:
            return False

    def has_drug_intervention(study):
        protocol_section = study.get("protocolSection", {})
        arms_module = protocol_section.get("armsInterventionsModule", {})
        interventions = arms_module.get("interventions", [])
        for intervention in interventions:
            if intervention.get("type") == "DRUG":
                return True
        return False

    def has_industry_sponsor(study):
        protocol_section = study.get("protocolSection", {})
        sponsor_module = protocol_section.get("sponsorCollaboratorsModule", {})
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        return lead_sponsor.get("class") == "INDUSTRY"

    def is_interventional(study):
        protocol_section = study.get("protocolSection", {})
        design_module = protocol_section.get("designModule", {})
        return design_module.get("studyType") == "INTERVENTIONAL"

    filtered_studies = [
        s for s in all_studies
        if is_in_date_range(s)
        and has_drug_intervention(s)
        and has_industry_sponsor(s)
        and is_interventional(s)
    ]
    logger.info(f"Filtered to {len(filtered_studies)} studies in date range {year_start}-{year_end} with drug intervention, industry sponsor, and interventional type")
    # Save filtered studies
    filtered_file = raw_path / f"filtered_{timestamp}.json"
    save_json({"studies": filtered_studies}, filtered_file)
    logger.info(f"Saved filtered studies to {filtered_file}")

    # Save combined results (for backward compatibility)
    if save_raw:
        save_json(
            {
                "metadata": {
                    "disease": disease,
                    "year_start": year_start,
                    "year_end": year_end,
                    "timestamp": timestamp,
                    "total_studies": len(filtered_studies),
                    "api_metadata_checked": metadata_checked,
                },
                "studies": filtered_studies,
            },
            raw_path / "all_studies.json",
        )
        logger.info(f"Saved all_studies.json for backward compatibility")

    return filtered_studies


def extract_intervention_names(studies: List[Dict[str, Any]]) -> Set[str]:
    """Extract unique intervention names from clinical trial studies.
    
    Args:
        studies: List of clinical trial studies
        
    Returns:
        Set of unique intervention names
    """
    # STEP 4: Extract unique drug names for enrichment
    intervention_names = set()
    
    # Valid intervention types from API documentation
    valid_types = {
        "DRUG", "BIOLOGICAL", "DEVICE", "BEHAVIORAL", 
        "PROCEDURE", "RADIATION", "DIETARY_SUPPLEMENT"
    }
    
    for study in studies:
        protocol_section = study.get("protocolSection", {})
        arms_module = protocol_section.get("armsInterventionsModule", {})
        interventions = arms_module.get("interventions", [])
        
        for intervention in interventions:
            int_type = intervention.get("type")
            int_name = intervention.get("name")
            
            # Only extract interventions with valid types
            if int_type in valid_types and int_name:
                # For drug enrichment, we focus on drug and biological interventions
                if int_type in ["DRUG", "BIOLOGICAL"]:
                    intervention_names.add(int_name)
    
    logger.info(f"Extracted {len(intervention_names)} unique drug/biological interventions")
    return intervention_names


def parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string from ClinicalTrials.gov API, handling partial dates (YYYY-MM, YYYY)."""
    if not date_str:
        return None
    try:
        # Try full ISO format first
        return datetime.fromisoformat(date_str.split("T")[0])
    except (ValueError, TypeError):
        # Try YYYY-MM format
        try:
            if len(date_str) == 7 and '-' in date_str:
                return datetime.strptime(date_str, "%Y-%m")
            # Try YYYY format
            if len(date_str) == 4 and date_str.isdigit():
                return datetime.strptime(date_str, "%Y")
        except Exception:
            pass
        logger.warning(f"Failed to parse date: {date_str}")
        return None


@log_execution_time
def transform_clinical_trials(
    studies: List[Dict[str, Any]], timestamp: Optional[str] = None
) -> pd.DataFrame:
    """Transform raw clinical trial data into a structured DataFrame.
    
    Args:
        studies: List of clinical trial studies
        timestamp: Optional timestamp string (defaults to current date)
        
    Returns:
        DataFrame with transformed clinical trial data
    """
    # STEP 5: Transform raw data into structured format based on the API contract
    logger.info("Transforming clinical trial data")
    
    # Extract relevant fields from each study according to the API contract
    transformed_data = []
    
    for study in studies:
        protocol_section = study.get("protocolSection", {})
        identification_module = protocol_section.get("identificationModule", {})
        status_module = protocol_section.get("statusModule", {})
        design_module = protocol_section.get("designModule", {})
        sponsor_module = protocol_section.get("sponsorCollaboratorsModule", {})
        conditions_module = protocol_section.get("conditionsModule", {})
        arms_module = protocol_section.get("armsInterventionsModule", {})
        contacts_locations_module = protocol_section.get("contactsLocationsModule", {})
        
        # Required fields (API contract)
        nct_id = identification_module.get("nctId")
        brief_title = identification_module.get("briefTitle")
        study_type = design_module.get("studyType")
        
        # Skip studies without required fields
        if not all([nct_id, brief_title, study_type]):
            logger.warning(f"Skipping study with missing required fields: {nct_id}")
            continue
        
        # Basic study information
        study_data = {
            # Required fields
            "nct_id": nct_id,
            "brief_title": brief_title,
            "study_type": study_type,
            "overall_status": status_module.get("overallStatus"),
            "has_results": study.get("hasResults", False),
            
            # Optional fields
            "official_title": identification_module.get("officialTitle"),
            
            # Dates
            "start_date": status_module.get("startDateStruct", {}).get("date"),
            "primary_completion_date": status_module.get("primaryCompletionDateStruct", {}).get("date"),
            "completion_date": status_module.get("completionDateStruct", {}).get("date"),
            
            # Enrollment
            "enrollment_count": design_module.get("enrollmentInfo", {}).get("count"),
            "enrollment_type": design_module.get("enrollmentInfo", {}).get("type"),
            
            # Conditions
            "conditions": conditions_module.get("conditions", []),
            
            # Sponsor/Collaborators
            "lead_sponsor": sponsor_module.get("leadSponsor", {}).get("name"),
            "sponsor_class": sponsor_module.get("leadSponsor", {}).get("class"),
        }
        
        # Get the first location's geo point if available
        locations = contacts_locations_module.get("locations", [])
        if locations:
            geo_point = locations[0].get("geoPoint", {})
            if geo_point:
                study_data["geo_point_lat"] = geo_point.get("lat")
                study_data["geo_point_lon"] = geo_point.get("lon")
        
        # Extract phases - take the first phase if available
        phases = design_module.get("phases", [])
        study_data["phase"] = phases[0] if phases else None
        
        # Extract collaborators
        collaborators = sponsor_module.get("collaborators", [])
        study_data["collaborators"] = [collab.get("name") for collab in collaborators if collab.get("name")]
        
        # Extract interventions
        interventions = arms_module.get("interventions", [])
        study_data["interventions"] = []
        
        # Keep only drug interventions
        drug_intervention_types = ["DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"]
        for intervention in interventions:
            int_type = intervention.get("type")
            int_name = intervention.get("name")
            
            if int_type and int_name:
                study_data["interventions"].append({
                    "intervention_type": int_type,
                    "intervention_name": int_name
                })
        
        # Parse dates
        study_data["start_date_parsed"] = parse_date(study_data["start_date"])
        study_data["completion_date_parsed"] = parse_date(study_data["completion_date"])
        
        # Calculate duration in days if both dates are available
        if study_data["start_date_parsed"] and study_data["completion_date_parsed"]:
            study_data["duration_days"] = (
                study_data["completion_date_parsed"] - study_data["start_date_parsed"]
            ).days
        else:
            study_data["duration_days"] = None
        
        # Add pipeline timestamp
        study_data["data_pull_timestamp"] = datetime.now().isoformat()
        
        transformed_data.append(study_data)
    
    # Create DataFrame
    df = pd.DataFrame(transformed_data)
    
    # Clean up data types
    if "enrollment_count" in df.columns:
        df["enrollment_count"] = pd.to_numeric(df["enrollment_count"], errors="coerce")
    
    if "conditions" in df.columns and len(df) > 0:
        # Only convert to string if it's a list
        df["conditions_str"] = df["conditions"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
    
    if "collaborators" in df.columns and len(df) > 0:
        # Only convert to string if it's a list
        df["collaborators_str"] = df["collaborators"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
    
    # Extract drug names from interventions for enrichment
    if 'interventions' in df.columns:
        df['intervention_names'] = df['interventions'].apply(
            lambda lst: [d['intervention_name'] for d in lst if d.get('intervention_type') == 'DRUG'] if isinstance(lst, list) else []
        )
    
    # Save transformed data
    if timestamp is None:
        timestamp = get_timestamp()
    
    output_path = settings.paths.processed_data / f"trials_{timestamp}.parquet"
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Transformed {len(df)} studies into tabular data saved to {output_path}")
    
    return df


async def load_and_transform_from_raw(
    timestamp: Optional[str] = None,
) -> pd.DataFrame:
    """Load raw data from disk and transform it.
    
    Args:
        timestamp: Optional timestamp string (defaults to current date)
        
    Returns:
        DataFrame with transformed clinical trial data
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    raw_path = get_raw_data_path(timestamp) / "all_studies.json"
    logger.info(f"Loading raw data from {raw_path}")
    
    data = load_json(raw_path)
    studies = data.get("studies", [])
    
    return transform_clinical_trials(studies, timestamp)


async def check_api_metadata(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """Check API metadata to detect schema changes.
    
    Args:
        session: aiohttp client session
        
    Returns:
        Dictionary with API metadata and version info
    """
    logger.info("Checking ClinicalTrials.gov API metadata")
    
    try:
        # Fetch metadata
        metadata_url = str(settings.ctgov.metadata_url)
        metadata = await fetch_with_retry(session, metadata_url)
        
        # Fetch version information
        version_url = "https://clinicaltrials.gov/api/v2/version"
        version_info = await fetch_with_retry(session, version_url)
        
        # Calculate hash of metadata to detect changes
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        
        # Get timestamp from version info
        data_timestamp = version_info.get("dataTimestamp", "unknown")
        
        logger.info(f"API data timestamp: {data_timestamp}")
        logger.info(f"Metadata hash: {metadata_hash}")
        
        # Check if we've seen this metadata hash before or if it's the first time
        cache_dir = settings.paths.cache
        cache_dir.mkdir(exist_ok=True)
        
        hash_file = cache_dir / "metadata_hash.txt"
        if hash_file.exists():
            with open(hash_file, "r") as f:
                old_hash = f.read().strip()
                
            if old_hash != metadata_hash:
                logger.warning(
                    f"API metadata has changed! Old hash: {old_hash}, New hash: {metadata_hash}"
                )
                logger.warning("Schema changes might affect data extraction and processing")
        
        # Save the current hash
        with open(hash_file, "w") as f:
            f.write(metadata_hash)
        
        return {
            "metadata": metadata,
            "version": version_info,
            "hash": metadata_hash,
            "timestamp": data_timestamp,
        }
    
    except Exception as e:
        logger.error(f"Error checking API metadata: {e}")
        return {
            "error": str(e),
            "metadata": None,
            "version": None,
            "hash": None,
            "timestamp": None,
        } 