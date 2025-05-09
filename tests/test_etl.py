"""Tests for the ETL module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import os
import logging

import pandas as pd
import pytest

from src.pipeline.etl import (
    build_clinical_trials_query,
    extract_all_clinical_trials,
    extract_clinical_trials_page,
    extract_intervention_names,
    parse_date,
    transform_clinical_trials,
)


@pytest.mark.asyncio
async def test_build_clinical_trials_query():
    """Test building query parameters for ClinicalTrials.gov API."""
    # Test with specific parameters
    disease = "Familial Hypercholesterolemia"
    year_start = 2020
    year_end = 2023
    
    query_params = await build_clinical_trials_query(disease, year_start, year_end)
    
    # Check that key parameters are set correctly
    assert query_params["query.term"] == f'AREA[ConditionSearch] "{disease}"'
    assert query_params["pageSize"] == 100 or isinstance(query_params["pageSize"], int)
    assert query_params["format"] == "json"
    assert query_params["countTotal"] == "true"
    assert "fields" in query_params


@pytest.mark.asyncio
async def test_extract_clinical_trials_page(monkeypatch):
    """Test extracting a single page of clinical trials."""
    # Patch get_run_logger to return a standard logger
    monkeypatch.setattr('prefect.get_run_logger', lambda: logging.getLogger("test"))
    # Create mock session and response
    mock_session = MagicMock()
    mock_session.get = AsyncMock()
    # Sample response data
    sample_response = {
        "studies": [
            {"id": "study1"},
            {"id": "study2"},
        ],
        "nextPageToken": "token123",
    }
    # Set up the mock to return our sample data
    mock_get_response = AsyncMock()
    mock_get_response.__aenter__ = AsyncMock(return_value=mock_get_response)
    mock_get_response.json = AsyncMock(return_value=sample_response)
    mock_get_response.raise_for_status = AsyncMock()
    mock_session.get.return_value = mock_get_response
    # Test the function
    with patch("src.pipeline.etl.fetch_with_retry", return_value=sample_response):
        studies, next_token = await extract_clinical_trials_page(
            mock_session, "http://example.com/page1", {"page": 1}
        )
    # Check the results
    assert len(studies) == 2
    assert studies[0]["id"] == "study1"
    assert next_token == "token123"


def test_extract_intervention_names(sample_clinical_trial):
    """Test extracting intervention names from clinical trial studies."""
    studies = [sample_clinical_trial]
    
    # Test extraction
    drug_names = extract_intervention_names(studies)
    
    # Check results
    assert drug_names == {"Test Drug A", "Test Drug B"}


def test_parse_date():
    """Test parsing date strings."""
    # Test valid date
    date_str = "2023-01-15"
    parsed_date = parse_date(date_str)
    assert parsed_date == datetime(2023, 1, 15)
    
    # Test date with time
    date_time_str = "2023-01-15T12:30:45"
    parsed_date_time = parse_date(date_time_str)
    assert parsed_date_time == datetime(2023, 1, 15)
    
    # Test None input
    assert parse_date(None) is None
    
    # Test invalid date
    assert parse_date("not-a-date") is None


def test_transform_clinical_trials(sample_clinical_trial, tmp_path, monkeypatch):
    """Test transforming raw clinical trial data into a DataFrame."""
    studies = [sample_clinical_trial]
    # Patch processed_data path to a temp directory
    monkeypatch.setattr('src.pipeline.etl.settings.paths.processed_data', tmp_path)
    df = transform_clinical_trials(studies, "test_timestamp")
    # Check the DataFrame
    assert not df.empty
    assert "nct_id" in df.columns
    assert "brief_title" in df.columns
    assert "study_type" in df.columns
    assert "enrollment_count" in df.columns
    assert "intervention_names" in df.columns
    assert "duration_days" in df.columns
    assert "phase" in df.columns
    # Check the values
    assert df.loc[0, "nct_id"] == "NCT12345678"
    assert df.loc[0, "brief_title"] == "Test Clinical Trial"
    assert df.loc[0, "study_type"] == "INTERVENTIONAL"
    assert df.loc[0, "enrollment_count"] == 100
    assert df.loc[0, "intervention_names"] == ["Test Drug A", "Test Drug B"]
    assert df.loc[0, "duration_days"] == 731  # 2022-01-01 - 2020-01-01
    # Check that the file was created
    assert (tmp_path / "trials_test_timestamp.parquet").exists()


@pytest.mark.asyncio
async def test_extract_all_clinical_trials(monkeypatch, tmp_path):
    """Test extracting all clinical trials."""
    # Patch get_run_logger to return a standard logger
    monkeypatch.setattr('prefect.get_run_logger', lambda: logging.getLogger("test"))
    # Patch settings to use tmp_path for file outputs
    monkeypatch.setattr('src.pipeline.etl.settings.paths.raw_data', tmp_path / "raw")
    monkeypatch.setattr('src.pipeline.etl.settings.paths.processed_data', tmp_path / "processed")
    monkeypatch.setattr('src.pipeline.etl.settings.max_studies', 2)
    monkeypatch.setattr('src.pipeline.etl.settings.max_pages', 1)
    monkeypatch.setattr('src.pipeline.etl.settings.disease', 'TestDisease')
    monkeypatch.setattr('src.pipeline.etl.settings.year_start', 2020)
    monkeypatch.setattr('src.pipeline.etl.settings.year_end', 2021)
    # Mock API metadata check
    async def mock_check_api_metadata(session):
        return {"metadata": {}, "version": {}, "hash": "abc", "timestamp": "now"}
    monkeypatch.setattr('src.pipeline.etl.check_api_metadata', mock_check_api_metadata)
    # Mock fetch_with_retry to return fake studies
    async def mock_fetch_with_retry(session, url, params):
        return {
            "studies": [
                {"protocolSection": {
                    "statusModule": {"startDateStruct": {"date": "2020-01-01"}},
                    "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "DrugA"}]},
                    "sponsorCollaboratorsModule": {"leadSponsor": {"class": "INDUSTRY"}},
                    "designModule": {"studyType": "INTERVENTIONAL"},
                }}
            ],
            "nextPageToken": None
        }
    monkeypatch.setattr('src.pipeline.etl.fetch_with_retry', mock_fetch_with_retry)
    # Run the function
    studies = await extract_all_clinical_trials()
    assert isinstance(studies, list)
    assert len(studies) == 1
    assert studies[0]["protocolSection"]["armsInterventionsModule"]["interventions"][0]["name"] == "DrugA"
    # Check that files are saved
    raw_dir = list((tmp_path / "raw").iterdir())
    assert raw_dir 