import requests
import json
import pprint

# Constants
API_V2_URL = "https://clinicaltrials.gov/api/v2/studies"

def debug_api_response():
    """
    Debug function to fetch a sample study from the API and print its structure
    to understand where the date fields are located.
    """
    # Use the same parameters as the pipeline
    params = {
        "query.term": 'AREA[ConditionSearch] "Familial Hypercholesterolemia" AND AREA[InterventionSearch] "Drug" AND AREA[SponsorSearch] "Industry" AND AREA[StudyType] INTERVENTIONAL',
        "pageSize": 1,
        "format": "json",
        "fields": "protocolSection,hasResults",
        "filter.startDate": "2010/01/01 TO 2025/12/31"
    }
    
    print("Fetching a sample study from ClinicalTrials.gov API with pipeline parameters...")
    
    try:
        response = requests.get(API_V2_URL, params=params)
        
        print(f"Request URL: {response.url}")
        print(f"Status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return
        
        data = response.json()
        
        # Check if we got any studies
        studies = data.get("studies", [])
        if not studies:
            print("No studies found in the response")
            return
        
        # Get the first study
        study = studies[0]
        
        # Print the NCT ID for reference
        protocol_section = study.get("protocolSection", {})
        identification_module = protocol_section.get("identificationModule", {})
        nct_id = identification_module.get("nctId", "Unknown")
        
        print(f"\nExamining study with NCT ID: {nct_id}\n")
        
        # Look for status module to find date fields
        status_module = protocol_section.get("statusModule", {})
        
        print("Status Module Structure:")
        print("------------------------")
        pprint.pprint(status_module)
        
        # Print just the date fields for clarity
        print("\nDate Fields:")
        print("------------")
        date_fields = {
            "startDate": status_module.get("startDateStruct", {}).get("date"),
            "primaryCompletionDate": status_module.get("primaryCompletionDateStruct", {}).get("date"),
            "completionDate": status_module.get("completionDateStruct", {}).get("date"),
            "lastUpdateSubmitDate": status_module.get("lastUpdateSubmitDate"),
            "statusVerifiedDate": status_module.get("statusVerifiedDate"),
            # Also show the date types where available
            "startDateType": status_module.get("startDateStruct", {}).get("type"),
            "primaryCompletionDateType": status_module.get("primaryCompletionDateStruct", {}).get("type"),
            "completionDateType": status_module.get("completionDateStruct", {}).get("type")
        }
        pprint.pprint(date_fields)
        
        # Check if there are any date structures (could be nested)
        print("\nSearching for other date-related fields in the entire study:")
        print("------------------------------------------------------------")
        
        # Save full study structure to a file for reference
        with open("sample_study_structure.json", "w") as f:
            json.dump(study, f, indent=2)
        print("Full study structure saved to 'sample_study_structure.json'")
        
        # Let's look for any field containing "date" in the protocol section
        print("\nAll fields containing 'date' in their name:")
        print("-------------------------------------------")
        
        def find_date_fields(obj, path=""):
            """Recursively find fields with 'date' in their name"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if "date" in key.lower():
                        print(f"{new_path}: {value}")
                    find_date_fields(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}[{i}]"
                    find_date_fields(item, new_path)
        
        find_date_fields(protocol_section)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    debug_api_response() 