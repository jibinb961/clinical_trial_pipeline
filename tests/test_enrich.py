"""Tests for the drug enrichment module."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import os

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
async def test_enrich_drug(temp_sqlite_db):
    """Test enriching a single drug."""
    # Test with a drug already in cache
    result = await enrich_drug("Test Drug A")
    assert result["name"] == "Test Drug A"
    assert result["modality"] == "small-molecule"
    assert result["target"] == "PCSK9"
    assert result["source"] == "DrugBank"
    
    # Test with a drug not in cache - using DrugBank
    # (patching for this case is handled in other tests)


def test_apply_enrichment_to_trials(sample_transformed_df, sample_drug_info, tmp_path, monkeypatch):
    """Test applying drug enrichment to trials DataFrame."""
    # Patch processed_data path to a temp directory
    monkeypatch.setattr('src.pipeline.enrich.settings.paths.processed_data', tmp_path)
    # Apply enrichment
    enriched_df = apply_enrichment_to_trials(sample_transformed_df, sample_drug_info)
    
    # Check that new columns were added
    assert "modalities" in enriched_df.columns
    assert "targets" in enriched_df.columns
    
    # Check first row which contains Test Drug A and Test Drug B
    assert "small-molecule" in enriched_df.iloc[0]["modalities"]
    assert "monoclonal antibody" in enriched_df.iloc[0]["modalities"]
    assert "PCSK9" in enriched_df.iloc[0]["targets"]


def test_setup_drug_cache_db(tmp_path, monkeypatch):
    # Patch cache_db_path to use a temp file
    db_path = tmp_path / "test_cache.sqlite"
    monkeypatch.setattr('src.pipeline.enrich.settings.cache_db_path', str(db_path))
    from src.pipeline.enrich import setup_drug_cache_db
    setup_drug_cache_db()
    assert db_path.exists()


def test_query_chembl_client(monkeypatch):
    # Patch new_client to return mock molecule, mechanism, target
    from src.pipeline import enrich
    class MockMolecule:
        def filter(self, **kwargs):
            if 'pref_name__iexact' in kwargs:
                return [{
                    'molecule_chembl_id': 'CHEMBL1',
                    'molecule_type': 'Small molecule',
                }]
            return []
    class MockMechanism:
        def filter(self, **kwargs):
            return [{'target_chembl_id': 'T1'}]
    class MockTarget:
        def filter(self, **kwargs):
            return [{'pref_name': 'PCSK9'}]
    monkeypatch.setattr(enrich.new_client, 'molecule', MockMolecule())
    monkeypatch.setattr(enrich.new_client, 'mechanism', MockMechanism())
    monkeypatch.setattr(enrich.new_client, 'target', MockTarget())
    result = enrich.query_chembl_client('TestDrug')
    assert result['name'] == 'TestDrug'
    assert result['modality'] == 'small-molecule'
    assert result['target'] == 'PCSK9'
    assert result['source'] == 'ChEMBL'


import asyncio
@pytest.mark.asyncio
async def test_query_gemini(monkeypatch):
    from src.pipeline import enrich
    async def mock_query_gemini_for_drug_info(drug_name):
        return {'name': drug_name, 'modality': 'mock-modality', 'target': 'mock-target', 'source': 'Gemini'}
    monkeypatch.setattr(enrich, 'query_gemini_for_drug_info', mock_query_gemini_for_drug_info)
    result = await enrich.query_gemini('TestDrug')
    assert result['name'] == 'TestDrug'
    assert result['modality'] == 'mock-modality'
    assert result['target'] == 'mock-target'
    assert result['source'] == 'Gemini'


@pytest.mark.asyncio
async def test_enrich_drugs(monkeypatch):
    from src.pipeline import enrich
    # Patch setup_drug_cache_db to do nothing
    monkeypatch.setattr(enrich, 'setup_drug_cache_db', lambda: None)
    # Patch enrich_drug to return a simple dict
    async def mock_enrich_drug(drug_name):
        return {'name': drug_name, 'modality': 'm', 'target': 't', 'source': 's'}
    monkeypatch.setattr(enrich, 'enrich_drug', mock_enrich_drug)
    result = await enrich.enrich_drugs({'A', 'B'})
    assert set(result.keys()) == {'A', 'B'}
    assert result['A']['modality'] == 'm'
    assert result['B']['target'] == 't' 