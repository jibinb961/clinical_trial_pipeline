"""Pytest configuration for clinical trials pipeline tests."""

import json
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("sys.path:", sys.path)

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.pipeline.config import settings
from src.pipeline.enrich import Base, DrugCache



@pytest.fixture
def sample_clinical_trial():
    """Sample clinical trial data for testing."""
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT12345678",
                "briefTitle": "Test Clinical Trial",
                "officialTitle": "A Test Clinical Trial for Testing",
            },
            "statusModule": {
                "overallStatus": "Completed",
                "startDateStruct": {"date": "2020-01-01"},
                "completionDateStruct": {"date": "2022-01-01"},
            },
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Test Pharma Inc.", "class": "INDUSTRY"}
            },
            "conditionsModule": {
                "conditions": ["Familial Hypercholesterolemia"]
            },
            "designModule": {
                "phases": ["Phase 2"],
                "enrollmentInfo": {"count": 100},
                "studyType": "INTERVENTIONAL",
            },
            "armsInterventionsModule": {
                "interventions": [
                    {
                        "type": "DRUG",
                        "name": "Test Drug A",
                    },
                    {
                        "type": "DRUG",
                        "name": "Test Drug B",
                    },
                ],
            },
        },
    }


@pytest.fixture
def sample_drug_info():
    """Sample drug enrichment information for testing."""
    return {
        "Test Drug A": {
            "name": "Test Drug A",
            "modality": "small-molecule",
            "target": "PCSK9",
            "source": "DrugBank",
        },
        "Test Drug B": {
            "name": "Test Drug B",
            "modality": "monoclonal antibody",
            "target": "PCSK9",
            "source": "ChEMBL",
        },
    }


@pytest.fixture
def sample_transformed_df():
    """Sample transformed DataFrame for testing."""
    data = {
        "nct_id": ["NCT12345678", "NCT87654321"],
        "brief_title": ["Test Clinical Trial 1", "Test Clinical Trial 2"],
        "official_title": ["A Test Clinical Trial for Testing 1", "A Test Clinical Trial for Testing 2"],
        "study_phase": ["Phase 2", "Phase 3"],
        "start_date": ["2020-01-01", "2019-01-01"],
        "completion_date": ["2022-01-01", "2023-01-01"],
        "enrollment": [100, 200],
        "status": ["Completed", "Completed"],
        "sponsor": ["Test Pharma Inc.", "Other Pharma Inc."],
        "conditions": ["Familial Hypercholesterolemia", "Familial Hypercholesterolemia"],
        "intervention_names": [["Test Drug A", "Test Drug B"], ["Test Drug C"]],
        "start_date_parsed": [pd.Timestamp("2020-01-01"), pd.Timestamp("2019-01-01")],
        "completion_date_parsed": [pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01")],
        "duration_days": [731, 1461],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_enriched_df(sample_transformed_df):
    """Sample enriched DataFrame for testing."""
    df = sample_transformed_df.copy()
    df["modalities"] = [["small-molecule", "monoclonal antibody"], ["small-molecule"]]
    df["targets"] = [["PCSK9", "PCSK9"], ["LDLR"]]
    return df


@pytest.fixture
def temp_sqlite_db(tmp_path):
    """Temporary SQLite database for testing drug cache."""
    db_path = tmp_path / "test_drug_cache.sqlite"
    settings.cache_db_path = str(db_path)
    
    # Create the database and tables
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    
    # Add sample data
    with Session(engine) as session:
        drug1 = DrugCache(
            name="Test Drug A",
            modality="small-molecule",
            target="PCSK9",
            source="DrugBank",
            timestamp="2023-01-01T00:00:00",
        )
        drug2 = DrugCache(
            name="Test Drug B",
            modality="monoclonal antibody",
            target="PCSK9",
            source="ChEMBL",
            timestamp="2023-01-01T00:00:00",
        )
        session.add_all([drug1, drug2])
        session.commit()
    
    return str(db_path) 

def test_sample_clinical_trial_fixture(sample_clinical_trial):
    assert isinstance(sample_clinical_trial, dict)
    assert "protocolSection" in sample_clinical_trial
    assert "identificationModule" in sample_clinical_trial["protocolSection"]
    assert "nctId" in sample_clinical_trial["protocolSection"]["identificationModule"] 