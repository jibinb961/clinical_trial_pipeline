import sys
from src.pipeline.enrich import query_chembl_client, query_gemini_for_drug_info
import requests

# --- Config ---
DEFAULT_DRUG = "Bempedoic acid"
DEFAULT_DISEASE = "Diabetes"

# --- Drug enrichment test ---
def test_enrichment(drug_name):
    print(f"\nTesting enrichment for drug: {drug_name}")
    chembl_result = query_chembl_client(drug_name)
    if chembl_result:
        print(f"ChEMBL result: {chembl_result}")
    else:
        print("ChEMBL: No result, trying Gemini...")
        gemini_result = None
        try:
            import asyncio
            gemini_result = asyncio.run(query_gemini_for_drug_info(drug_name))
        except Exception as e:
            print(f"Error calling Gemini: {e}")
        print(f"Gemini result: {gemini_result}")

# --- ClinicalTrials.gov API test ---
def test_clinicaltrials_api(disease=DEFAULT_DISEASE):
    print(f"\nTesting ClinicalTrials.gov API for disease: {disease}")
    url = "https://clinicaltrials.gov/api/v2/studies"
    params = {
        "query.term": f'AREA[ConditionSearch] "{disease}"',
        "pageSize": 1,
        "format": "json",
        "fields": "protocolSection,hasResults",
    }
    try:
        resp = requests.get(url, params=params)
        print(f"Status code: {resp.status_code}")
        data = resp.json()
        studies = data.get("studies", [])
        if studies:
            study = studies[0]
            interventions = study.get("protocolSection", {}).get("armsInterventionsModule", {}).get("interventions", [])
            print(f"First study NCT ID: {study.get('protocolSection', {}).get('identificationModule', {}).get('nctId')}")
            print(f"Interventions: {interventions}")
        else:
            print("No studies found.")
    except Exception as e:
        print(f"Error querying ClinicalTrials.gov: {e}")

if __name__ == "__main__":
    drug = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DRUG
    test_enrichment(drug)
    test_clinicaltrials_api() 