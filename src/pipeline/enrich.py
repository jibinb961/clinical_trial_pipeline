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
from sqlalchemy.orm import Session, sessionmaker
from tqdm.asyncio import tqdm
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.settings import Settings

from src.pipeline.config import settings
from src.pipeline.gemini_utils import query_gemini_for_drug_grounded_search
from src.pipeline.utils import log_execution_time, logger, retry_async, upload_to_gcs
from prefect import get_run_logger

import re
import tempfile

# Define SQLAlchemy models
Base = declarative_base()


class DrugCache(Base):
    """SQLite table for caching drug enrichment data."""
    
    __tablename__ = "drug_cache"
    
    name = Column(String, primary_key=True)
    modality = Column(String, nullable=True)
    target = Column(String, nullable=True)
    source = Column(String, nullable=True)
    uri = Column(String, nullable=True)  # NEW: URI for grounded search
    timestamp = Column(String, nullable=True)


# --- SQLAlchemy global engine/session ---
engine = create_engine(f"sqlite:///{settings.cache_db_path}")
SessionLocal = sessionmaker(bind=engine)


def setup_drug_cache_db() -> None:
    """Set up the SQLite database for drug caching."""
    Base.metadata.create_all(engine)
    logger.info(f"Drug cache database setup at {settings.cache_db_path}")


def get_cached_drug(drug_name: str) -> Optional[Dict[str, str]]:
    """Get cached drug information if available.
    
    Args:
        drug_name: Name of the drug to look up
        
    Returns:
        Dictionary with modality and target, or None if not in cache
    """
    prefect_logger = get_run_logger()
    prefect_logger.info(f"[CACHE] Lookup for drug: '{drug_name}' in cache DB: {settings.cache_db_path}")
    with SessionLocal() as session:
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
                "uri": drug.uri,
            }
    
    logger.info(f"[CACHE] MISS: '{drug_name}' not found in cache")
    return None


def cache_drug(
    drug_name: str, modality: Any, target: Any, source: str, uri: Optional[str] = None
) -> None:
    """Cache drug information in SQLite database.

    Args:
        drug_name: Name of the drug
        modality: Drug modality (can be string or list)
        target: Drug target (can be string or list)
        source: Source of the information
        uri: URI for grounded search
    """
    # ✅ Convert lists to comma-separated strings
    if isinstance(modality, list):
        modality = ", ".join(modality)
    if isinstance(target, list):
        target = ", ".join(target)

    logger.info(f"[CACHE] WRITE: '{drug_name}' -> modality: '{modality}', target: '{target}', source: '{source}', uri: '{uri}' in cache DB: {settings.cache_db_path}")
    with SessionLocal() as session:
        drug = DrugCache(
            name=drug_name,
            modality=modality,
            target=target,
            source=source,
            uri=uri,
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
            existing[0].uri = uri
            existing[0].timestamp = datetime.now().isoformat()
        else:
            session.add(drug)

        session.commit()
    
    prefect_logger = get_run_logger()
    prefect_logger.debug(f"Cached drug: {drug_name} from {source}")


def query_chembl_client(drug_name: str) -> Optional[Dict[str, str]]:
    """Query ChEMBL API via Python client to get drug modality and target.
    
    Args:
        drug_name: Name of the drug to query
        
    Returns:
        Dictionary with modality and target or None if not found
    """
    logger.info(f"Querying ChEMBL for {drug_name}")
    try:
        # Disable ChEMBL's own local caching
        Settings.Instance().CACHING = True
        molecule = new_client.molecule
        mechanism = new_client.mechanism
        target = new_client.target
        # Try exact match
        results = molecule.filter(pref_name__iexact=drug_name).only(['molecule_chembl_id', 'molecule_type', 'pref_name'])
        if results:
            logger.info(f"ChEMBL exact match for {drug_name}")
        else:
            # Try synonym match
            results = molecule.filter(molecule_synonyms__molecule_synonym__icontains=drug_name).only(['molecule_chembl_id', 'molecule_type', 'pref_name'])
            if results:
                logger.info(f"ChEMBL synonym match for {drug_name}")
            else:
                # Try partial match
                results = molecule.filter(pref_name__icontains=drug_name).only(['molecule_chembl_id', 'molecule_type', 'pref_name'])
                if results:
                    logger.info(f"ChEMBL partial match for {drug_name}")
        if not results:
            logger.debug(f"Drug not found in ChEMBL: {drug_name}")
            return None
        molecule_data = results[0]
        chembl_id = molecule_data['molecule_chembl_id']
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
        # Get all mechanisms
        moa_results = mechanism.filter(molecule_chembl_id=chembl_id)
        target_ids = [m.get('target_chembl_id') for m in moa_results if 'target_chembl_id' in m]
        target_names = []
        for target_id in target_ids:
            target_info = target.filter(target_chembl_id=target_id).only(['pref_name'])
            if target_info:
                target_name = target_info[0].get('pref_name', '')
                if target_name:
                    target_names.append(target_name)
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


# --- New: Preprocessing helpers ---
def preprocess_drug_name(drug_name):
    """
    Simple preprocessing for enrichment:
    - Filter out known non-drugs (placebo, vehicle, etc.)
    - Split on '+' and ';' only
    - Return same structure as original function: 4 lists
    """
    name = drug_name.lower().strip()

    NON_DRUG_TERMS = [
        "placebo", "vehicle", "normal saline", "background drug",
        "matching placebo", "formulation only", "reference", "food effect"
    ]

    # Filter out non-drug terms
    if any(term in name for term in NON_DRUG_TERMS):
        return [], [], [], []

    # Split using only '+' and ';'
    parts = re.split(r'\s*(\+|;)\s*', drug_name)
    cleaned = [part.strip() for part in parts if part.strip() and part not in ['+', ';']]

    return cleaned, [drug_name] * len(cleaned), [drug_name] * len(cleaned), [''] * len(cleaned)


# --- New: Main enrichment function ---
@log_execution_time
async def enrich_drugs(drug_names: Set[str]) -> Dict[str, Dict[str, Any]]:
    prefect_logger = get_run_logger()
    prefect_logger.info(f"[CACHE] Using cache DB at: {settings.cache_db_path}")
    setup_drug_cache_db()
    prefect_logger.info(f"Enriching {len(drug_names)} drugs with ChEMBL, then Gemini grounded search fallback")
    chembl_results = {}
    unresolved = set()
    placebo_set = set()
    combo_map = {}
    all_preprocessed_drugs = set()
    for orig_name in tqdm(drug_names, desc="Preprocessing and ChEMBL lookup"):
        parts, origs, normed, _ = preprocess_drug_name(orig_name)
        all_preprocessed_drugs.update(parts)
        if len(parts) > 1:
            combo_map[orig_name] = parts
        modalities, targets, sources, uris = [], [], [], []
        for part in parts:
            if 'placebo' in part.lower() or 'simulant' in part.lower():
                modalities.append('placebo')
                targets.append('placebo')
                sources.append('placebo')
                uris.append(None)
                cache_drug(part, 'placebo', 'placebo', 'placebo', None)
                placebo_set.add(part)
            else:
                cached = get_cached_drug(part)
                if cached and cached['modality'] != 'Unknown' and cached['target'] != 'Unknown':
                    modalities.append(cached['modality'])
                    targets.append(cached['target'])
                    sources.append(cached['source'])
                    uris.append(cached.get('uri'))
                else:
                    info = query_chembl_client(part)
                    if info and info['modality'] != 'Unknown' and info['target'] != 'Unknown':
                        modalities.append(info['modality'])
                        targets.append(info['target'])
                        sources.append('ChEMBL')
                        uris.append(None)
                        cache_drug(part, info['modality'], info['target'], 'ChEMBL', None)
                    else:
                        modalities.append('Unknown')
                        targets.append('Unknown')
                        sources.append('Unknown')
                        uris.append(None)
                        unresolved.add(part)
        if len(parts) > 1:
            chembl_results[orig_name] = {"modality": modalities, "target": targets, "source": sources, "uri": uris}
        else:
            if not modalities or not targets or not sources or not uris:
                prefect_logger.info(
                    f"[ENRICH] Skipping enrichment for '{orig_name}' — "
                    f" Since it is non drug or placebo"
                )
            chembl_results[orig_name] = {
                "modality": modalities[0] if modalities else "Unknown",
                "target": targets[0] if targets else "Unknown",
                "source": sources[0] if sources else "Unknown",
                "uri": uris[0] if uris else None
            }
    # --- Gemini grounded search fallback ---
    unresolved_for_gemini = [name for name in unresolved if 'placebo' not in name.lower() and 'simulant' not in name.lower()]
    # --- Gemini API rate limiting ---
    import time
    from collections import deque
    gemini_minute_window = deque()
    gemini_100s_window = deque()
    gemini_daily_count = 0
    gemini_daily_limit = 100
    gemini_results = {}
    async def rate_limiter():
        now = time.time()
        # Remove old timestamps
        while gemini_minute_window and now - gemini_minute_window[0] > 60:
            gemini_minute_window.popleft()
        while gemini_100s_window and now - gemini_100s_window[0] > 100:
            gemini_100s_window.popleft()
        # Check limits
        if len(gemini_minute_window) >= 15 or len(gemini_100s_window) >= 10 or gemini_daily_count >= gemini_daily_limit:
            return False
        return True
    async def wait_for_slot():
        while not await rate_limiter():
            prefect_logger.info("[Gemini] Rate limit hit, waiting...")
            await asyncio.sleep(5)
    for name in tqdm(unresolved_for_gemini, desc="Gemini grounded search", mininterval=1):
        await wait_for_slot()
        retries = 0
        while retries < 5:
            try:
                info = await query_gemini_for_drug_grounded_search(name, settings.disease)
                now = time.time()
                gemini_minute_window.append(now)
                gemini_100s_window.append(now)
                gemini_daily_count += 1
                if info:
                    gemini_results[name] = info
                    cache_drug(name, info['modality'], info['target'], 'Gemini', info.get('uri'))
                else:
                    gemini_results[name] = {"modality": "Unknown", "target": "Unknown", "source": "Gemini", "uri": None}
                    cache_drug(name, "Unknown", "Unknown", "Gemini", None)
                break
            except Exception as e:
                prefect_logger.info(f"[Gemini Retry {retries}] Error for {name}: {e}")
                await asyncio.sleep(2 ** retries + 0.5)
                retries += 1
    # --- Merge Gemini results into chembl_results ---
    for orig_name in drug_names:
        parts, _, _, _ = preprocess_drug_name(orig_name)
        if 'placebo' in orig_name.lower() or 'simulant' in orig_name.lower():
            chembl_results[orig_name] = {"modality": "placebo", "target": "placebo", "source": "placebo", "uri": None}
        elif len(parts) > 1:
            modalities, targets, sources, uris = [], [], [], []
            for part in parts:
                if part in gemini_results:
                    modalities.append(gemini_results[part]['modality'])
                    targets.append(gemini_results[part]['target'])
                    sources.append('Gemini')
                    uris.append(gemini_results[part].get('uri'))
                else:
                    entry = chembl_results.get(orig_name, {"modality": "Unknown", "target": "Unknown", "source": "Unknown", "uri": None})
                    idx = parts.index(part)
                    if isinstance(entry['modality'], list):
                        modalities.append(entry['modality'][idx])
                        targets.append(entry['target'][idx])
                        sources.append(entry['source'][idx])
                        uris.append(entry['uri'][idx] if isinstance(entry['uri'], list) else entry['uri'])
                    else:
                        modalities.append(entry['modality'])
                        targets.append(entry['target'])
                        sources.append(entry['source'])
                        uris.append(entry['uri'])
            chembl_results[orig_name] = {"modality": modalities, "target": targets, "source": sources, "uri": uris}
        elif orig_name in gemini_results:
            chembl_results[orig_name] = gemini_results[orig_name]
        elif orig_name in chembl_results:
            entry = chembl_results[orig_name]
            if (entry['modality'] == 'Unknown' or entry['target'] == 'Unknown'):
                if orig_name in unresolved_for_gemini:
                    entry['source'] = 'Gemini'
                elif entry['source'] == 'Unknown':
                    entry['source'] = 'ChEMBL'
        else:
            chembl_results[orig_name] = {"modality": "Unknown", "target": "Unknown", "source": "Gemini", "uri": None}
    prefect_logger.info(f"Completed enrichment of {len(chembl_results)} drugs")
    return chembl_results


def apply_enrichment_to_trials(trials_df: pd.DataFrame, drug_info: Dict[str, Dict[str, str]], timestamp: Optional[str] = None) -> pd.DataFrame:
    prefect_logger = get_run_logger()
    prefect_logger.info("Applying drug enrichment data to trials DataFrame")
    enriched_df = trials_df.copy()
    # Ensure all columns exist
    for col in ["modalities", "targets", "enrichment_sources", "enrichment_uris", "intervention_cleaned"]:
        if col not in enriched_df.columns:
            enriched_df[col] = None

    def clean_and_extract_drugs(intervention_names):
        """Return cleaned drug names (excluding placebos/non-drugs) from intervention_names list."""
        if not intervention_names or not isinstance(intervention_names, list):
            return []
        cleaned = []
        for name in intervention_names:
            parts, _, _, _ = preprocess_drug_name(name)
            for part in parts:
                if part and 'placebo' not in part.lower() and 'simulant' not in part.lower():
                    cleaned.append(part)
        return cleaned

    def get_enrichment_lists(cleaned_drugs):
        """Return modalities, targets, sources, uris for a list of cleaned drugs."""
        modalities, targets, sources, uris = [], [], [], []
        for drug in cleaned_drugs:
            info = drug_info.get(drug, {"modality": "Unknown", "target": "Unknown", "source": "Unknown", "uri": None})
            modalities.append(info.get("modality", "Unknown"))
            targets.append(info.get("target", "Unknown"))
            sources.append(info.get("source", "Unknown"))
            uris.append(info.get("uri", None))
        return modalities, targets, sources, uris

    # Main enrichment loop
    for i, row in enriched_df.iterrows():
        intervention_names = row.get("intervention_names")
        cleaned_drugs = clean_and_extract_drugs(intervention_names)
        enriched_df.at[i, "intervention_cleaned"] = cleaned_drugs
        modalities, targets, sources, uris = get_enrichment_lists(cleaned_drugs)
        enriched_df.at[i, "modalities"] = modalities
        enriched_df.at[i, "targets"] = targets
        enriched_df.at[i, "enrichment_sources"] = sources
        enriched_df.at[i, "enrichment_uris"] = uris

    # Export logic (flatten lists for export)
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
    for col in ["modalities", "targets", "enrichment_sources", "enrichment_uris", "intervention_cleaned"]:
        if col in enriched_df.columns:
            enriched_df[col] = enriched_df[col].apply(flatten_list_of_lists)
    if timestamp is None:
        from src.pipeline.utils import get_timestamp
        timestamp = get_timestamp()
    output_path = settings.paths.processed_data / f"trials_enriched_{timestamp}.parquet"
    enriched_df.to_parquet(output_path, index=False)
    csv_path = settings.paths.processed_data / f"trials_enriched_{timestamp}.csv"
    enriched_df.to_csv(csv_path, index=False)
    prefect_logger.info(f"Saved enriched DataFrame to {output_path} and {csv_path}")
    # Generate enrichment report CSV
    try:
        rows = []
        for _, row in enriched_df.iterrows():
            nct_id = row.get('nct_id', '')
            cleaned_drugs = row.get('intervention_cleaned', [])
            modalities = row.get('modalities', [])
            targets = row.get('targets', [])
            sources = row.get('enrichment_sources', [])
            uris = row.get('enrichment_uris', [])
            for drug, modality, target, source, uri in zip(cleaned_drugs, modalities, targets, sources, uris):
                rows.append({
                    "nct_id": nct_id,
                    "drug": drug,
                    "modality": modality,
                    "target": target,
                    "source": source,
                    "uri": uri
                })
        report_df = pd.DataFrame(rows)
        for col in ["modality", "target", "source"]:
            if col in report_df.columns:
                report_df[col] = report_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x) if x is not None else "")
        report_df = report_df.drop_duplicates(subset=["nct_id", "drug", "modality", "target", "source", "uri"])
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tmp_csv:
            report_df.to_csv(tmp_csv.name, index=False)
            upload_to_gcs(tmp_csv.name, f"runs/{timestamp}/enrichment_report_{timestamp}.csv")
        prefect_logger.info(f"Saved enrichment report to GCS runs/{timestamp}/enrichment_report_{timestamp}.csv")
    except Exception as e:
        prefect_logger.error(f"Error generating enrichment report: {e}")
    return enriched_df 