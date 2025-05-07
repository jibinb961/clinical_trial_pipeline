"""Drug enrichment module for adding modality and target information.

This module handles the enrichment of drug intervention data with
modality and target information from ChEMBL and Google Gemini.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import pandas as pd
from sqlalchemy import Column, String, Table, create_engine, insert, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from tqdm.asyncio import tqdm
from chembl_webresource_client.new_client import new_client

from src.pipeline.config import settings
from src.pipeline.gemini_utils import query_gemini_for_drug_info
from src.pipeline.utils import log_execution_time, logger, retry_async

# Define SQLAlchemy models
Base = declarative_base()


class DrugCache(Base):
    """SQLite table for caching drug enrichment data."""
    
    __tablename__ = "drug_cache"
    
    name = Column(String, primary_key=True)
    modality = Column(String, nullable=True)
    target = Column(String, nullable=True)
    source = Column(String, nullable=True)
    timestamp = Column(String, nullable=True)


def setup_drug_cache_db() -> None:
    """Set up the SQLite database for drug caching."""
    # STEP 1: Set up SQLite DB for caching drug enrichment data
    engine = create_engine(f"sqlite:///{settings.cache_db_path}")
    Base.metadata.create_all(engine)
    logger.info(f"Drug cache database setup at {settings.cache_db_path}")


def get_cached_drug(drug_name: str) -> Optional[Dict[str, str]]:
    """Get cached drug information if available.
    
    Args:
        drug_name: Name of the drug to look up
        
    Returns:
        Dictionary with modality and target, or None if not in cache
    """
    # STEP 2: Check if drug is in cache
    engine = create_engine(f"sqlite:///{settings.cache_db_path}")
    
    with Session(engine) as session:
        result = session.execute(
            select(DrugCache).where(DrugCache.name == drug_name)
        ).first()
        
        if result:
            drug = result[0]
            logger.debug(f"Found cached drug: {drug_name} from {drug.source}")
            return {
                "name": drug.name,
                "modality": drug.modality,
                "target": drug.target,
                "source": drug.source,
            }
    
    return None


def cache_drug(
    drug_name: str, modality: str, target: str, source: str
) -> None:
    """Cache drug information in SQLite database.
    
    Args:
        drug_name: Name of the drug
        modality: Drug modality
        target: Drug target
        source: Source of the information
    """
    # STEP 3: Save drug info to cache
    engine = create_engine(f"sqlite:///{settings.cache_db_path}")
    
    with Session(engine) as session:
        drug = DrugCache(
            name=drug_name,
            modality=modality,
            target=target,
            source=source,
            timestamp=datetime.now().isoformat(),
        )
        
        # Upsert (insert or update)
        existing = session.execute(
            select(DrugCache).where(DrugCache.name == drug_name)
        ).first()
        
        if existing:
            existing[0].modality = modality
            existing[0].target = target
            existing[0].source = source
            existing[0].timestamp = datetime.now().isoformat()
        else:
            session.add(drug)
        
        session.commit()
    
    logger.debug(f"Cached drug: {drug_name} from {source}")


def query_chembl_client(drug_name: str) -> Optional[Dict[str, str]]:
    """Query ChEMBL API via Python client to get drug modality and target.
    
    Args:
        drug_name: Name of the drug to query
        
    Returns:
        Dictionary with modality and target or None if not found
    """
    # STEP 4: Query ChEMBL API using Python client
    logger.debug(f"Querying ChEMBL for {drug_name}")
    
    try:
        # Initialize ChEMBL client resources
        molecule = new_client.molecule
        mechanism = new_client.mechanism
        target = new_client.target
        
        # Search for the drug by name
        results = molecule.filter(pref_name__iexact=drug_name)
        if not results:
            # Try a more flexible search if exact match fails
            results = molecule.filter(molecule_synonyms__molecule_synonym__icontains=drug_name)
            
        if not results:
            logger.debug(f"Drug not found in ChEMBL: {drug_name}")
            return None
            
        # Get the first matching molecule
        molecule_data = results[0]
        chembl_id = molecule_data['molecule_chembl_id']
        
        # Determine modality from molecule type
        molecule_type = molecule_data.get('molecule_type', '')
        
        # Map ChEMBL molecule types to our standardized modalities
        modality_mapping = {
            'Small molecule': 'small-molecule',
            'Protein': 'protein',
            'Antibody': 'monoclonal antibody',
            'Oligonucleotide': 'siRNA',
            'Enzyme': 'protein',
            'Cell': 'cell therapy',
            'Gene': 'gene therapy',
            'Oligosaccharide': 'small-molecule',
        }
        
        modality = modality_mapping.get(molecule_type, 'Unknown')
        
        # Get mechanism of action data
        moa_results = mechanism.filter(molecule_chembl_id=chembl_id)
        
        if not moa_results:
            # Return with just the modality if no mechanism is found
            return {
                "name": drug_name,
                "modality": modality,
                "target": "Unknown",
                "source": "ChEMBL",
            }
            
        # Extract target information
        target_ids = [m.get('target_chembl_id') for m in moa_results if 'target_chembl_id' in m]
        
        if not target_ids:
            return {
                "name": drug_name,
                "modality": modality,
                "target": "Unknown",
                "source": "ChEMBL",
            }
            
        # Get target details
        target_names = []
        for target_id in target_ids:
            target_info = target.filter(target_chembl_id=target_id)
            if target_info:
                target_name = target_info[0].get('pref_name', '')
                if target_name:
                    target_names.append(target_name)
                    
        # Join multiple targets with commas
        target_str = ", ".join(target_names) if target_names else "Unknown"
        
        return {
            "name": drug_name,
            "modality": modality,
            "target": target_str,
            "source": "ChEMBL",
        }
        
    except Exception as e:
        logger.warning(f"Error querying ChEMBL for {drug_name}: {e}")
        return None


async def query_gemini(drug_name: str) -> Optional[Dict[str, str]]:
    """Query Google Gemini to get drug modality and target.
    
    Args:
        drug_name: Name of the drug to query
        
    Returns:
        Dictionary with modality and target or None if not found
    """
    # STEP 5: Query Google Gemini API as fallback
    logger.debug(f"Querying Google Gemini for {drug_name}")
    
    # Use our utility function from gemini_utils
    return await query_gemini_for_drug_info(drug_name)


async def enrich_drug(drug_name: str) -> Dict[str, str]:
    """Enrich a drug with modality and target information.
    
    Args:
        drug_name: Name of the drug to enrich
        
    Returns:
        Dictionary with name, modality, target, and source
    """
    # STEP 6: Enrich a single drug with modality and target
    # First, check the cache
    cached_drug = get_cached_drug(drug_name)
    if cached_drug:
        return cached_drug
    
    # Try each source in order
    drug_info = None
    
    # 1. Try ChEMBL client first
    drug_info = query_chembl_client(drug_name)
    if drug_info:
        cache_drug(
            drug_info["name"],
            drug_info["modality"],
            drug_info["target"],
            drug_info["source"],
        )
        return drug_info
    
    # 2. Fallback to Google Gemini
    drug_info = await query_gemini(drug_name)
    if drug_info:
        cache_drug(
            drug_info["name"],
            drug_info["modality"],
            drug_info["target"],
            drug_info["source"],
        )
        return drug_info
    
    # If all sources fail, return unknown values
    unknown_info = {
        "name": drug_name,
        "modality": "Unknown",
        "target": "Unknown",
        "source": "None",
    }
    cache_drug(
        unknown_info["name"],
        unknown_info["modality"],
        unknown_info["target"],
        unknown_info["source"],
    )
    return unknown_info


@log_execution_time
async def enrich_drugs(drug_names: Set[str]) -> Dict[str, Dict[str, str]]:
    """Enrich multiple drugs with modality and target information.
    
    Args:
        drug_names: Set of drug names to enrich
        
    Returns:
        Dictionary mapping drug names to their enrichment information
    """
    # STEP 7: Enrich multiple drugs in parallel
    setup_drug_cache_db()
    
    logger.info(f"Enriching {len(drug_names)} drugs with modality and target info")
    
    drug_info = {}
    semaphore = asyncio.Semaphore(settings.concurrency_limit)
    
    async def _enrich_with_semaphore(drug_name: str) -> Tuple[str, Dict[str, str]]:
        # Use semaphore to limit concurrent API requests
        async with semaphore:
            result = await enrich_drug(drug_name)
            return drug_name, result
    
    # Use tqdm to show progress bar
    tasks = [_enrich_with_semaphore(drug) for drug in drug_names]
    results = await tqdm.gather(*tasks, desc="Enriching drugs")
    
    for drug_name, info in results:
        drug_info[drug_name] = info
    
    logger.info(f"Completed enrichment of {len(drug_info)} drugs")
    return drug_info


def apply_enrichment_to_trials(
    trials_df: pd.DataFrame, drug_info: Dict[str, Dict[str, str]]
) -> pd.DataFrame:
    """Apply drug enrichment information to the trials DataFrame.
    Adds modalities, targets, and enrichment_sources columns.
    """
    # STEP 8: Apply enrichment data to trials DataFrame
    logger.info("Applying drug enrichment data to trials DataFrame")
    
    # Create a copy to avoid modifying the original
    enriched_df = trials_df.copy()
    
    # Add columns for modality, target, and source if needed
    if "modalities" not in enriched_df.columns:
        enriched_df["modalities"] = None
    if "targets" not in enriched_df.columns:
        enriched_df["targets"] = None
    if "enrichment_sources" not in enriched_df.columns:
        enriched_df["enrichment_sources"] = None
    
    # Function to extract modalities, targets, and sources for a list of interventions
    def extract_enrichment(interventions: list) -> tuple:
        if not interventions or not isinstance(interventions, list):
            return [], [], []
        modalities, targets, sources = [], [], []
        for drug in interventions:
            info = drug_info.get(drug, {"modality": "Unknown", "target": "Unknown", "source": "Unknown"})
            modalities.append(info.get("modality", "Unknown"))
            targets.append(info.get("target", "Unknown"))
            sources.append(info.get("source", "Unknown"))
        return modalities, targets, sources
    
    # Apply the extraction to each row
    for i, row in enriched_df.iterrows():
        interventions = row.get("intervention_names")
        if interventions:
            modalities, targets, sources = extract_enrichment(interventions)
            enriched_df.at[i, "modalities"] = modalities
            enriched_df.at[i, "targets"] = targets
            enriched_df.at[i, "enrichment_sources"] = sources
    
    # Get timestamp for filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Save the enriched DataFrame
    output_path = settings.paths.processed_data / f"trials_enriched_{timestamp}.parquet"
    enriched_df.to_parquet(output_path, index=False)
    logger.info(f"Saved enriched DataFrame to {output_path}")
    
    # Generate enrichment report CSV
    try:
        rows = []
        for _, row in enriched_df.iterrows():
            nct_id = row.get('nct_id', '')
            for drug, modality, target, source in zip(
                row.get('intervention_names', []),
                row.get('modalities', []),
                row.get('targets', []),
                row.get('enrichment_sources', [])
            ):
                rows.append({
                    "nct_id": nct_id,
                    "drug": drug,
                    "modality": modality,
                    "target": target,
                    "source": source
                })
        report_df = pd.DataFrame(rows)
        report_path = settings.paths.processed_data / f"enrichment_report_{timestamp}.csv"
        report_df.to_csv(report_path, index=False)
        logger.info(f"Saved enrichment report to {report_path}")
    except Exception as e:
        logger.error(f"Error generating enrichment report: {e}")
    
    return enriched_df 