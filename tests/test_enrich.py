"""Tests for the drug enrichment module."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.pipeline.enrich import (
    apply_enrichment_to_trials,
    cache_drug,
    enrich_drug,
    enrich_drugs,
    get_cached_drug,
    query_chembl,
    query_drugbank,
    query_openai,
    setup_drug_cache_db,
)


def test_get_cached_drug(temp_sqlite_db):
    """Test retrieving a drug from the cache."""
    # Test getting an existing drug
    drug_info = get_cached_drug("Test Drug A")
    assert drug_info is not None
    assert drug_info["name"] == "Test Drug A"
    assert drug_info["modality"] == "small-molecule"
    assert drug_info["target"] == "PCSK9"
    assert drug_info["source"] == "DrugBank"
    
    # Test getting a non-existent drug
    assert get_cached_drug("Non-existent Drug") is None


def test_cache_drug(temp_sqlite_db):
    """Test caching a drug in the database."""
    # Cache a new drug
    cache_drug(
        "New Test Drug",
        "small-molecule",
        "LDLR",
        "OpenAI",
    )
    
    # Verify it was cached
    drug_info = get_cached_drug("New Test Drug")
    assert drug_info is not None
    assert drug_info["name"] == "New Test Drug"
    assert drug_info["modality"] == "small-molecule"
    assert drug_info["target"] == "LDLR"
    assert drug_info["source"] == "OpenAI"
    
    # Update an existing drug
    cache_drug(
        "Test Drug A",
        "updated-modality",
        "updated-target",
        "updated-source",
    )
    
    # Verify it was updated
    updated_drug = get_cached_drug("Test Drug A")
    assert updated_drug["modality"] == "updated-modality"
    assert updated_drug["target"] == "updated-target"
    assert updated_drug["source"] == "updated-source"


@pytest.mark.asyncio
async def test_query_drugbank():
    """Test querying the DrugBank API."""
    # Mock session
    mock_session = MagicMock()
    mock_session.get = AsyncMock()
    
    # Sample successful response
    sample_response = {
        "drugs": [
            {
                "name": "Test Drug A",
                "type": "small-molecule",
                "targets": [
                    {"name": "PCSK9"},
                ],
            }
        ]
    }
    
    # Mock the API response
    with patch("src.pipeline.enrich.retry_async", new_callable=AsyncMock) as mock_retry:
        mock_retry.return_value = sample_response
        with patch("src.pipeline.enrich.settings.api_keys.drugbank", "fake_key"):
            result = await query_drugbank("Test Drug A", mock_session)
    
    # Check result
    assert result is not None
    assert result["name"] == "Test Drug A"
    assert result["modality"] == "small-molecule"
    assert result["target"] == "PCSK9"
    assert result["source"] == "DrugBank"
    
    # Test with no API key
    with patch("src.pipeline.enrich.settings.api_keys.drugbank", None):
        result = await query_drugbank("Test Drug A", mock_session)
    
    # Should return None if no API key is set
    assert result is None
    
    # Test with API error
    with patch("src.pipeline.enrich.retry_async", side_effect=Exception("API Error")):
        with patch("src.pipeline.enrich.settings.api_keys.drugbank", "fake_key"):
            result = await query_drugbank("Test Drug A", mock_session)
    
    # Should return None on error
    assert result is None


@pytest.mark.asyncio
async def test_query_chembl():
    """Test querying the ChEMBL API."""
    # Mock session
    mock_session = MagicMock()
    mock_session.get = AsyncMock()
    
    # Sample successful response
    molecule_response = {
        "molecules": [
            {
                "molecule_type": "Small molecule",
                "molecule_chembl_id": "CHEMBL123",
            }
        ]
    }
    
    target_response = {
        "mechanisms": [
            {
                "target_name": "PCSK9",
            }
        ]
    }
    
    # Mock the API responses
    with patch("src.pipeline.enrich.retry_async", new_callable=AsyncMock) as mock_retry:
        mock_retry.side_effect = [molecule_response, target_response]
        with patch("src.pipeline.enrich.settings.api_keys.chembl", "fake_key"):
            result = await query_chembl("Test Drug A", mock_session)
    
    # Check result
    assert result is not None
    assert result["name"] == "Test Drug A"
    assert result["modality"] == "Small molecule"
    assert result["target"] == "PCSK9"
    assert result["source"] == "ChEMBL"


@pytest.mark.asyncio
async def test_query_openai():
    """Test querying the OpenAI API."""
    # Sample successful response
    openai_response = MagicMock()
    openai_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"modality": "small-molecule", "target": "PCSK9"}'
            )
        )
    ]
    
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=openai_response)
    
    # Mock the OpenAI class
    with patch("src.pipeline.enrich.AsyncOpenAI", return_value=mock_client):
        with patch("src.pipeline.enrich.settings.api_keys.openai", "fake_key"):
            result = await query_openai("Test Drug A")
    
    # Check result
    assert result is not None
    assert result["name"] == "Test Drug A"
    assert result["modality"] == "small-molecule"
    assert result["target"] == "PCSK9"
    assert result["source"] == "OpenAI"
    
    # Test with no API key
    with patch("src.pipeline.enrich.settings.api_keys.openai", None):
        result = await query_openai("Test Drug A")
    
    # Should return None if no API key is set
    assert result is None
    
    # Test with invalid JSON response
    openai_response.choices[0].message.content = "not valid json"
    with patch("src.pipeline.enrich.AsyncOpenAI", return_value=mock_client):
        with patch("src.pipeline.enrich.settings.api_keys.openai", "fake_key"):
            result = await query_openai("Test Drug A")
    
    # Should return None on parsing error
    assert result is None


@pytest.mark.asyncio
async def test_enrich_drug(temp_sqlite_db):
    """Test enriching a single drug."""
    # Mock session
    mock_session = MagicMock()
    
    # Test with a drug already in cache
    result = await enrich_drug("Test Drug A", mock_session)
    assert result["name"] == "Test Drug A"
    assert result["modality"] == "small-molecule"
    assert result["target"] == "PCSK9"
    assert result["source"] == "DrugBank"
    
    # Test with a drug not in cache - using DrugBank
    with patch("src.pipeline.enrich.get_cached_drug", return_value=None):
        with patch(
            "src.pipeline.enrich.query_drugbank",
            new_callable=AsyncMock,
            return_value={
                "name": "New Drug", 
                "modality": "small-molecule", 
                "target": "LDLR", 
                "source": "DrugBank"
            },
        ):
            with patch("src.pipeline.enrich.cache_drug") as mock_cache:
                result = await enrich_drug("New Drug", mock_session)
                
                # Check result
                assert result["name"] == "New Drug"
                assert result["modality"] == "small-molecule"
                assert result["target"] == "LDLR"
                assert result["source"] == "DrugBank"
                
                # Check that it was cached
                mock_cache.assert_called_once_with(
                    "New Drug", "small-molecule", "LDLR", "DrugBank"
                )


def test_apply_enrichment_to_trials(sample_transformed_df, sample_drug_info):
    """Test applying drug enrichment to trials DataFrame."""
    # Apply enrichment
    enriched_df = apply_enrichment_to_trials(sample_transformed_df, sample_drug_info)
    
    # Check that new columns were added
    assert "modalities" in enriched_df.columns
    assert "targets" in enriched_df.columns
    
    # Check first row which contains Test Drug A and Test Drug B
    assert "small-molecule" in enriched_df.iloc[0]["modalities"]
    assert "monoclonal antibody" in enriched_df.iloc[0]["modalities"]
    assert "PCSK9" in enriched_df.iloc[0]["targets"] 