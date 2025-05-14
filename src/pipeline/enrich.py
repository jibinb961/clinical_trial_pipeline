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
from src.pipeline.gemini_utils import query_gemini_for_drug_info, initialize_gemini, batch_query_gemini
from src.pipeline.utils import log_execution_time, logger, retry_async

import re

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
    engine = create_engine(f"sqlite:///{settings.cache_db_path}")
    logger.info(f"[CACHE] Lookup for drug: '{drug_name}' in cache DB: {settings.cache_db_path}")
    with Session(engine) as session:
        result = session.execute(
            select(DrugCache).where(DrugCache.name == drug_name)
        ).first()
        
        if result:
            drug = result[0]
            logger.info(f"[CACHE] HIT: '{drug_name}' found in cache (source: {drug.source})")
            return {
                "name": drug.name,
                "modality": drug.modality,
                "target": drug.target,
                "source": drug.source,
            }
    
    logger.info(f"[CACHE] MISS: '{drug_name}' not found in cache")
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
    engine = create_engine(f"sqlite:///{settings.cache_db_path}")
    logger.info(f"[CACHE] WRITE: '{drug_name}' -> modality: '{modality}', target: '{target}', source: '{source}' in cache DB: {settings.cache_db_path}")
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
    logger.info(f"Querying ChEMBL for {drug_name}")
    
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


# --- New: Preprocessing helpers ---
def preprocess_drug_name(drug_name):
    # Lowercase for placebo check
    if 'placebo' in drug_name.lower():
        return ['placebo'], ['placebo'], ['placebo'], ['placebo']
    # Remove dosage info (e.g., (90 mg), 10 mg, 5mg, etc.)
    name = re.sub(r'\([^)]*mg[^)]*\)', '', drug_name)
    name = re.sub(r'\b\d+\s*mg\b', '', name, flags=re.IGNORECASE)
    # Remove annotations like (Background Drug), (oral), etc.
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    # Split combinations on '+', ' and ', or '/'
    if '+' in name:
        parts = [n.strip() for n in name.split('+')]
    elif ' and ' in name:
        parts = [n.strip() for n in name.split(' and ')]
    elif '/' in name:
        parts = [n.strip() for n in name.split('/')]
    else:
        parts = [name]
    return parts, [drug_name]*len(parts), [name]*len(parts), ['']*len(parts)


# --- New: Main enrichment function ---
@log_execution_time
async def enrich_drugs(drug_names: Set[str]) -> Dict[str, Dict[str, Any]]:
    logger.info(f"[CACHE] Using cache DB at: {settings.cache_db_path}")
    setup_drug_cache_db()
    logger.info(f"Enriching {len(drug_names)} drugs with ChEMBL, then Gemini batch fallback")
    chembl_results = {}
    unresolved = set()
    placebo_set = set()
    combo_map = {}
    # --- NEW: Track all preprocessed drug names ---
    all_preprocessed_drugs = set()
    for orig_name in tqdm(drug_names, desc="Preprocessing and ChEMBL lookup"):
        parts, origs, normed, _ = preprocess_drug_name(orig_name)
        all_preprocessed_drugs.update(parts)
        if len(parts) > 1:
            combo_map[orig_name] = parts
        modalities, targets, sources = [], [], []
        for part in parts:
            # Placebo strict handling
            if 'placebo' in part.lower() or 'simulant' in part.lower():
                modalities.append('placebo')
                targets.append('placebo')
                sources.append('placebo')
                cache_drug(part, 'placebo', 'placebo', 'placebo')
                placebo_set.add(part)
            else:
                cached = get_cached_drug(part)
                if cached and cached['modality'] != 'Unknown' and cached['target'] != 'Unknown':
                    modalities.append(cached['modality'])
                    targets.append(cached['target'])
                    sources.append(cached['source'])
                else:
                    info = query_chembl_client(part)
                    if info and info['modality'] != 'Unknown' and info['target'] != 'Unknown':
                        modalities.append(info['modality'])
                        targets.append(info['target'])
                        sources.append('ChEMBL')
                        cache_drug(part, info['modality'], info['target'], 'ChEMBL')
                    else:
                        modalities.append('Unknown')
                        targets.append('Unknown')
                        sources.append('Unknown')
                        unresolved.add(part)
        # Store as list for combos, single value for single drugs
        if len(parts) > 1:
            chembl_results[orig_name] = {"modality": modalities, "target": targets, "source": sources}
        else:
            chembl_results[orig_name] = {"modality": modalities[0], "target": targets[0], "source": sources[0]}
    # Stage 2: Gemini batch (skip placebos)
    unresolved_for_gemini = [name for name in unresolved if 'placebo' not in name.lower() and 'simulant' not in name.lower()]
    gemini_results = await batch_query_gemini(unresolved_for_gemini)
    for name, info in gemini_results.items():
        # Always set source to Gemini, even if Unknown
        info['source'] = 'Gemini'
        cache_drug(name, info['modality'], info['target'], 'Gemini')
    # Merge Gemini results into chembl_results
    for orig_name in drug_names:
        parts, _, _, _ = preprocess_drug_name(orig_name)
        if 'placebo' in orig_name.lower() or 'simulant' in orig_name.lower():
            chembl_results[orig_name] = {"modality": "placebo", "target": "placebo", "source": "placebo"}
        elif len(parts) > 1:
            # For combos, update unresolved parts with Gemini results
            modalities, targets, sources = [], [], []
            for part in parts:
                if part in gemini_results:
                    modalities.append(gemini_results[part]['modality'])
                    targets.append(gemini_results[part]['target'])
                    sources.append('Gemini')
                else:
                    entry = chembl_results.get(orig_name, {"modality": "Unknown", "target": "Unknown", "source": "Unknown"})
                    if isinstance(entry['modality'], list):
                        idx = parts.index(part)
                        modalities.append(entry['modality'][idx])
                        targets.append(entry['target'][idx])
                        sources.append(entry['source'][idx])
                    else:
                        modalities.append(entry['modality'])
                        targets.append(entry['target'])
                        sources.append(entry['source'])
            chembl_results[orig_name] = {"modality": modalities, "target": targets, "source": sources}
        elif orig_name in gemini_results:
            chembl_results[orig_name] = gemini_results[orig_name]
        elif orig_name in chembl_results:
            if (chembl_results[orig_name]['modality'] == 'Unknown' or chembl_results[orig_name]['target'] == 'Unknown'):
                if orig_name in unresolved_for_gemini:
                    chembl_results[orig_name]['source'] = 'Gemini'
                elif chembl_results[orig_name]['source'] == 'Unknown':
                    chembl_results[orig_name]['source'] = 'ChEMBL'
        else:
            chembl_results[orig_name] = {"modality": "Unknown", "target": "Unknown", "source": "Gemini"}
    logger.info(f"Completed enrichment of {len(chembl_results)} drugs")
    return chembl_results


def apply_enrichment_to_trials(trials_df: pd.DataFrame, drug_info: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    logger.info("Applying drug enrichment data to trials DataFrame")
    enriched_df = trials_df.copy()
    if "modalities" not in enriched_df.columns:
        enriched_df["modalities"] = None
    if "targets" not in enriched_df.columns:
        enriched_df["targets"] = None
    if "enrichment_sources" not in enriched_df.columns:
        enriched_df["enrichment_sources"] = None
    def extract_enrichment(interventions: list) -> tuple:
        if not interventions or not isinstance(interventions, list):
            return [], [], []
        modalities, targets, sources = [], [], []
        for drug in interventions:
            info = drug_info.get(drug, {"modality": "Unknown", "target": "Unknown", "source": "Unknown"})
            mod = info.get("modality", "Unknown")
            tar = info.get("target", "Unknown")
            src = info.get("source", "Unknown")
            # Always wrap in list if not already a list
            if not isinstance(mod, list):
                mod = [mod]
            if not isinstance(tar, list):
                tar = [tar]
            if not isinstance(src, list):
                src = [src]
            modalities.append(mod)
            targets.append(tar)
            sources.append(src)
        return modalities, targets, sources
    for i, row in enriched_df.iterrows():
        interventions = row.get("intervention_names")
        if interventions:
            modalities, targets, sources = extract_enrichment(interventions)
            enriched_df.at[i, "modalities"] = modalities
            enriched_df.at[i, "targets"] = targets
            enriched_df.at[i, "enrichment_sources"] = sources
    # Flatten list-of-lists in modalities, targets, enrichment_sources
    def flatten_list_of_lists(lst):
        if not isinstance(lst, list):
            return [lst]
        flat = []
        for item in lst:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    for col in ["modalities", "targets", "enrichment_sources"]:
        if col in enriched_df.columns:
            enriched_df[col] = enriched_df[col].apply(flatten_list_of_lists)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
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
        report_df = report_df.drop_duplicates(subset=["nct_id", "drug", "modality", "target", "source"])
        report_path = settings.paths.processed_data / f"enrichment_report_{timestamp}.csv"
        report_df.to_csv(report_path, index=False)
        logger.info(f"Saved enrichment report to {report_path}")
    except Exception as e:
        logger.error(f"Error generating enrichment report: {e}")
    return enriched_df 