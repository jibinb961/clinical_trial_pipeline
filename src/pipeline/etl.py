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
import tempfile

from src.pipeline.config import settings
from src.pipeline.utils import (
    fetch_with_retry,
    get_raw_data_path,
    get_timestamp,
    load_json,
    log_execution_time,
    logger,
    save_json,
    upload_to_gcs,
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
    """Extract a single page of clinical trial data from the API (supports nextPageToken pagination)."""
    logger = params.get('logger', None)
    if logger is None:
        try:
            from prefect import get_run_logger
            logger = get_run_logger()
        except ImportError:
            from src.pipeline.utils import logger as default_logger
            logger = default_logger
    logger.debug(f"Fetching page with params: {params}")
    try:
        response_data = await fetch_with_retry(session, url, params)
        studies = response_data.get("studies", [])
        next_page_token = response_data.get("nextPageToken")
        return studies, next_page_token
    except aiohttp.ClientResponseError as e:
        if e.status == 429:
            logger.warning(f"Rate limit exceeded. Retrying with backoff: {str(e)}")
            raise
        elif e.status >= 500:
            logger.warning(f"Server error from ClinicalTrials.gov API: {str(e)}")
            raise
        elif e.status >= 400:
            logger.error(f"Client error when querying ClinicalTrials.gov API: {str(e)}")
            return [], None
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.error(f"Network error when querying ClinicalTrials.gov API: {str(e)}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Malformed JSON response from ClinicalTrials.gov API: {str(e)}")
        return [], None


@log_execution_time
async def extract_all_clinical_trials(
    disease: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    save_raw: bool = True,
    logger: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Extract all clinical trial data matching the criteria (supports nextPageToken pagination and max_studies/max_pages limits)."""
    if logger is None:
        try:
            from prefect import get_run_logger
            logger = get_run_logger()
        except ImportError:
            from src.pipeline.utils import logger as default_logger
            logger = default_logger
    disease = disease or settings.disease
    year_start = year_start or settings.year_start
    year_end = year_end or settings.year_end
    logger.info(f"Extracting clinical trials for {disease} ({year_start}-{year_end})")
    query_params = await build_clinical_trials_query(disease, year_start, year_end)
    url = str(settings.ctgov.base_url)
    all_studies = []
    timestamp = get_timestamp()
    metadata_checked = False
    max_studies = settings.max_studies
    max_pages = settings.max_pages
    async with aiohttp.ClientSession() as session:
        metadata_info = await check_api_metadata(session)
        metadata_checked = True
        if metadata_info.get("error"):
            logger.warning("Failed to check API metadata, proceeding with extraction anyway")
        page_number = 1
        next_page_token = None
        while True:
            if max_pages is not None and page_number > max_pages:
                logger.info(f"Reached max_pages limit: {max_pages}")
                break
            params = dict(query_params)  # Copy base params
            if next_page_token:
                params["pageToken"] = next_page_token
            studies, next_page_token = await extract_clinical_trials_page(session, url, params)
            logger.info(f"Retrieved page {page_number} with {len(studies)} studies, next_page_token: {next_page_token}")
            all_studies.extend(studies)
            if not next_page_token:
                break
            page_number += 1
    logger.info(f"Extracted total of {len(all_studies)} clinical trials before filtering")
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
    # Save filtered studies ONLY
    raw_path = get_raw_data_path(timestamp)
    raw_path.mkdir(parents=True, exist_ok=True)
    filtered_file = raw_path / f"filtered_{timestamp}.json"
    save_json({"studies": filtered_studies}, filtered_file)
    logger.info(f"Saved filtered studies to {filtered_file}")
    return filtered_studies


# This function returns a set of unique drug names across all trials, to be used for enrichment.
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
        "DRUG", "BIOLOGICAL", 
    "DIETARY_SUPPLEMENT"
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
        eligibility_module = protocol_section.get("eligibilityModule", {})
        outcomes_module = protocol_section.get("outcomesModule", {})
        
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
        # Set study_phase for downstream compatibility
        study_data["study_phase"] = study_data["phase"]
        
        # Extract collaborators
        collaborators = sponsor_module.get("collaborators", [])
        study_data["collaborators"] = [collab.get("name") for collab in collaborators if collab.get("name")]
        
        # Extract interventions and arm groups for treatment details
        interventions = arms_module.get("interventions", [])
        arm_groups = arms_module.get("armGroups", [])
        study_data["treatment_details"] = []
        # Build a mapping from arm group label to description
        arm_group_map = {ag.get("label"): ag for ag in arm_groups if ag.get("label")}
        for intervention in interventions:
            int_type = intervention.get("type")
            int_name = intervention.get("name")
            int_desc = intervention.get("description")
            arm_labels = intervention.get("armGroupLabels", [])
            if int_type and int_name:
                # For each arm group this intervention is used in
                if arm_labels:
                    for arm_label in arm_labels:
                        ag = arm_group_map.get(arm_label, {})
                        study_data["treatment_details"].append({
                            "intervention_name": int_name,
                            "intervention_type": int_type,
                            "intervention_description": int_desc,
                            "arm_group_label": arm_label,
                            "arm_group_description": ag.get("description"),
                        })
                else:
                    # No arm group, just intervention
                    study_data["treatment_details"].append({
                        "intervention_name": int_name,
                        "intervention_type": int_type,
                        "intervention_description": int_desc,
                        "arm_group_label": None,
                        "arm_group_description": None,
                    })
        
        # Parse dates
        start_date = status_module.get("startDateStruct", {}).get("date")
        completion_date = status_module.get("completionDateStruct", {}).get("date")
        primary_completion_date = status_module.get("primaryCompletionDateStruct", {}).get("date")
        start_date_parsed = parse_date(start_date)
        completion_date_parsed = parse_date(completion_date)
        primary_completion_date_parsed = parse_date(primary_completion_date)
        # Use completion_date if available, else primary_completion_date
        end_date_parsed = completion_date_parsed or primary_completion_date_parsed
        if start_date_parsed and end_date_parsed:
            study_data["duration_days"] = (end_date_parsed - start_date_parsed).days
        else:
            study_data["duration_days"] = None
        
        # --- Age extraction and preprocessing ---
        def parse_age(age_str):
            if not age_str or age_str in ("N/A", "None", ""):
                return None
            try:
                parts = age_str.strip().split()
                value = float(parts[0])
                unit = parts[1].lower() if len(parts) > 1 else "years"
                if unit.startswith("year"):
                    return value
                elif unit.startswith("month"):
                    return value / 12
                elif unit.startswith("week"):
                    return value / 52.1429
                elif unit.startswith("day"):
                    return value / 365.25
                else:
                    return value  # fallback
            except Exception:
                return None
        study_data["minimum_age"] = parse_age(eligibility_module.get("minimumAge"))
        study_data["maximum_age"] = parse_age(eligibility_module.get("maximumAge"))
        study_data["std_ages"] = eligibility_module.get("stdAges", []) or []
        # --- Outcome extraction ---
        def extract_measures(outcomes):
            if not outcomes or not isinstance(outcomes, list):
                return []
            return [o.get("measure", "").strip() for o in outcomes if o.get("measure")]
        study_data["primary_outcomes"] = extract_measures(outcomes_module.get("primaryOutcomes", []))
        study_data["secondary_outcomes"] = extract_measures(outcomes_module.get("secondaryOutcomes", []))
        
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
    if 'treatment_details' in df.columns:
        df['intervention_names'] = df['treatment_details'].apply(
            lambda lst: [d['intervention_name'] for d in lst if d.get('intervention_type') == 'DRUG'] if isinstance(lst, list) else []
        )
    
    # Save transformed data
    if timestamp is None:
        timestamp = get_timestamp()
    # Write DataFrame to temp parquet, upload to GCS
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp_parquet:
        df.to_parquet(tmp_parquet.name, index=False)
        upload_to_gcs(tmp_parquet.name, f"runs/{timestamp}/trials_{timestamp}.parquet")
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