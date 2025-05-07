"""Tests for the ETL module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
    assert query_params["query.cond"] == disease
    assert query_params["query.intr"] == "Drug"
    assert query_params["query.spons"] == "Industry"
    assert query_params["query.stype"] == "Intr"
    assert query_params["filter.start"] == "2020/01/01,2023/12/31"
    assert query_params["filter.species"] == "Human"


@pytest.mark.asyncio
async def test_extract_clinical_trials_page():
    """Test extracting a single page of clinical trials."""
    # Create mock session and response
    mock_session = MagicMock()
    mock_session.get = AsyncMock()
    
    # Sample response data
    sample_response = {
        "studies": [
            {"id": "study1"},
            {"id": "study2"},
        ],
        "links": [
            {"rel": "self", "href": "http://example.com/page1"},
            {"rel": "next", "href": "http://example.com/page2"},
        ],
    }
    
    # Set up the mock to return our sample data
    mock_get_response = AsyncMock()
    mock_get_response.__aenter__ = AsyncMock(return_value=mock_get_response)
    mock_get_response.json = AsyncMock(return_value=sample_response)
    mock_get_response.raise_for_status = AsyncMock()
    mock_session.get.return_value = mock_get_response
    
    # Test the function
    with patch("src.pipeline.etl.fetch_with_retry", return_value=sample_response):
        studies, next_url = await extract_clinical_trials_page(
            mock_session, "http://example.com/page1", {"page": 1}
        )
    
    # Check the results
    assert len(studies) == 2
    assert studies[0]["id"] == "study1"
    assert next_url == "http://example.com/page2"


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


def test_transform_clinical_trials(sample_clinical_trial, tmp_path):
    """Test transforming raw clinical trial data into a DataFrame."""
    studies = [sample_clinical_trial]
    
    # Mock paths for testing
    test_processed_dir = tmp_path / "processed"
    test_processed_dir.mkdir()
    
    with patch("src.pipeline.etl.settings.paths.processed_data", test_processed_dir):
        df = transform_clinical_trials(studies, "test_timestamp")
    
    # Check the DataFrame
    assert not df.empty
    assert "nct_id" in df.columns
    assert "brief_title" in df.columns
    assert "study_phase" in df.columns
    assert "enrollment" in df.columns
    assert "intervention_names" in df.columns
    assert "duration_days" in df.columns
    
    # Check the values
    assert df.loc[0, "nct_id"] == "NCT12345678"
    assert df.loc[0, "brief_title"] == "Test Clinical Trial"
    assert df.loc[0, "study_phase"] == "Phase 2"
    assert df.loc[0, "enrollment"] == 100
    assert df.loc[0, "intervention_names"] == ["Test Drug A", "Test Drug B"]
    assert df.loc[0, "duration_days"] == 731  # 2022-01-01 - 2020-01-01
    
    # Check that the file was created
    assert (test_processed_dir / "trials_test_timestamp.parquet").exists() 