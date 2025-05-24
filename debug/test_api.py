import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import aiohttp

from src.pipeline.config import settings
from src.pipeline.utils import fetch_with_retry, get_timestamp
from src.pipeline.etl import extract_intervention_names

# Fallback logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("standalone_extractor")

from difflib import get_close_matches

from src.pipeline.gemini_utils import query_gemini_for_drug_grounded_search

async def test_gemini_enrichment():
    test_drugs = [
        ("CKD-498 dose#1", "Alopecia Areata")
    ]

    for drug_name, disease in test_drugs:
        result = await query_gemini_for_drug_grounded_search(drug_name, disease)
        print(f" Result for {drug_name} and {disease}: {result}")



# -------------------------
# PREPROCESSING FUNCTION
# -------------------------
def preprocess_drug_name(drug_name):
    """
    Simple preprocessing:
    - Filter out known non-drugs (e.g., placebo, vehicle)
    - Split on '+' and ';' only
    """
    name = drug_name.lower().strip()

    NON_DRUG_TERMS = [
        "placebo", "vehicle", "normal saline", "formulation only",
        "background drug", "matching placebo", "reference", "food effect"
    ]

    # Filter out non-drugs
    if any(term in name for term in NON_DRUG_TERMS):
        return [], [], [], []

    # Split using '+' and ';'
    parts = re.split(r'\s*(\+|;)\s*', drug_name)
    cleaned = [part.strip() for part in parts if part.strip() and part not in ['+', ';']]

    return cleaned, [drug_name] * len(cleaned), [drug_name] * len(cleaned), [''] * len(cleaned)

# -------------------------
# FILTER FUNCTIONS
# -------------------------
def is_in_date_range(study: Dict[str, Any], year_start: int, year_end: int) -> bool:
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

def has_drug_intervention(study: Dict[str, Any]) -> bool:
    interventions = study.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions", [])
    return any(i.get("type") == "DRUG" for i in interventions)

def has_industry_sponsor(study: Dict[str, Any]) -> bool:
    lead_sponsor = study.get("protocolSection", {}).get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    return lead_sponsor.get("class") == "INDUSTRY"

def is_interventional(study: Dict[str, Any]) -> bool:
    return study.get("protocolSection", {}).get("designModule", {}).get("studyType") == "INTERVENTIONAL"

# -------------------------
# MAIN EXECUTION
# -------------------------
async def extract_studies_limited(disease: str, year_start: int, year_end: int, max_pages: int = 20) -> List[Dict[str, Any]]:
    search_term = f'AREA[ConditionSearch] "{disease}"'
    params = {
        "query.term": search_term,
        "pageSize": settings.ctgov.page_size,
        "format": "json",
        "countTotal": "true",
        "fields": "protocolSection,hasResults",
    }
    logger.info(f"Fetching studies for: {disease} ({year_start}-{year_end})")
    all_studies = []
    next_page_token = None
    url = str(settings.ctgov.base_url)

    async with aiohttp.ClientSession() as session:
        for page in range(1, max_pages + 1):
            if next_page_token:
                params["pageToken"] = next_page_token
            logger.info(f"Fetching page {page}...")
            response = await fetch_with_retry(session, url, params)
            studies = response.get("studies", [])
            all_studies.extend(studies)
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    logger.info(f"✅ Total studies retrieved (limited to {max_pages} pages): {len(all_studies)}")
    return all_studies

async def run_standalone(disease: str, year_start: int, year_end: int):
    all_studies = await extract_studies_limited(disease, year_start, year_end, max_pages=20)

    # Apply filters
    filtered = [
        s for s in all_studies
        if is_in_date_range(s, year_start, year_end)
        and has_drug_intervention(s)
        and has_industry_sponsor(s)
        and is_interventional(s)
    ]
    logger.info(f"✅ Filtered studies: {len(filtered)}")

    # Extract raw drugs
    raw_drug_names = extract_intervention_names(filtered)
    with open("raw_drug_names.txt", "w") as f:
        f.write("\n".join(raw_drug_names))
    logger.info(f"🧪 Raw extracted drugs: {len(raw_drug_names)}")

    # Preprocess drug names
    cleaned_drugs: Set[str] = set()
    for name in raw_drug_names:
        parts, *_ = preprocess_drug_name(name)
        for p in parts:
            cleaned_drugs.add(p)


    logger.info(f"🧼 Cleaned & unique drug names: {len(cleaned_drugs)}")

    # Generate timestamped file names
    timestamp = get_timestamp()
    json_path = Path(f"debug_filtered_trials_{disease.replace(' ', '_')}_{timestamp}.json")
    txt_path = Path(f"extracted_drugs_{disease.replace(' ', '_')}_{timestamp}.txt")

    # Write JSON
    with open(json_path, "w") as f:
        json.dump({"studies": filtered}, f, indent=2)
    logger.info(f"📁 Saved filtered studies to: {json_path.resolve()}")

    # Write cleaned TXT list
    with open(txt_path, "w") as f:
        f.write("\n".join(sorted(cleaned_drugs)))
    logger.info(f"📁 Saved extracted drug names to: {txt_path.resolve()}")


if __name__ == "__main__":
    #asyncio.run(run_standalone("Alopecia Areata", 2020, 2025))
    asyncio.run(test_gemini_enrichment())