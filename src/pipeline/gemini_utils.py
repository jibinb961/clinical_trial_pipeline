"""Google Gemini API integration for drug enrichment.

This module provides functions to leverage Google's Gemini AI
for enriching drug information with modality and target data.
"""

import json
import logging
import re
import asyncio
from typing import Dict, Optional, List
from random import uniform

from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

from src.pipeline.config import settings
from src.pipeline.utils import logger

    
async def query_gemini_for_drug_grounded_search(drug_name: str) -> Optional[Dict[str, str]]:
    """
    Use Google Gemini with grounded search to get drug modality and target.
    
    Args:
        drug_name: Name of the drug to query

    Returns:
        Dictionary with modality, target, source, and grounding URI
    """
    if not settings.api_keys.gemini:
        logger.warning("Gemini API key not set, skipping Gemini query")
        return None

    logger.debug(f"[Gemini Grounded] Querying for drug: {drug_name}")
    
    try:
        # Configure client
        client = genai.Client(api_key=settings.api_keys.gemini)

        # Define the Google Search tool
        google_search_tool = Tool(google_search=GoogleSearch())

        # Build the prompt
        prompt = f"""
        You are a pharmacology expert. Your task is to extract the *modality* and *target* of a drug given its name.

        Return a single JSON object with these two keys:

        "modality": Choose one of the following strings (case-sensitive):
        - "small-molecule"
        - "monoclonal antibody"
        - "antibody-drug conjugate"
        - "protein"
        - "peptide"
        - "siRNA"
        - "gene therapy"
        - "cell therapy"
        - "CAR-T"
        - "vaccine"
        - "Unknown" (only if you cannot infer the modality)

        "target": Return the HGNC gene symbol or pathway the drug acts on. Examples: "PCSK9", "EGFR", "JAK/STAT".
        If unknown, return "Unknown".

        **Important**: Think step by step internally, but output only the final JSON — no explanations, no code fences.

        Example:
        {{
        "modality": "small-molecule",
        "target": "PCSK9"
        }}

        DRUG = "{drug_name}"
        """

        # Retry logic with exponential backoff
        retries = 0
        while retries < 5:
            try:
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=prompt,
                    config=GenerateContentConfig(
                        tools=[google_search_tool],
                        response_modalities=["TEXT"],
                        max_output_tokens=1000
                    )
                )

                if not response or not response.candidates:
                    logger.warning(f"No candidate response for {drug_name}")
                    return None

                content = response.candidates[0].content.parts[0].text.strip()
                metadata = response.candidates[0].grounding_metadata

                # Extract URI if available
                uri = None
                if metadata and getattr(metadata, "grounding_chunks", None):
                    uri = metadata.grounding_chunks[0].web.uri

                # Parse JSON
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    # Try cleaning up and extracting JSON from raw text
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_str = content[start:end]
                    parsed = json.loads(json_str)

                return {
                    "name": drug_name,
                    "modality": parsed.get("modality", "Unknown"),
                    "target": parsed.get("target", "Unknown"),
                    "source": "Gemini",
                    "uri": uri or ""
                }

            except Exception as e:
                logger.warning(f"[Gemini Retry {retries}] Error querying {drug_name}: {e}")
                await asyncio.sleep(2 ** retries + uniform(0.1, 0.5))
                retries += 1

    except Exception as outer_e:
        logger.error(f"Fatal error calling Gemini for {drug_name}: {outer_e}")
        return None

    return None

# --- Utilities ---
def format_outcome_for_llm(outcome):
    parts = []
    if outcome.get('measure'):
        parts.append(outcome['measure'])
    if outcome.get('description'):
        parts.append(f"Description: {outcome['description']}")
    if outcome.get('timeFrame'):
        parts.append(f"Timeframe: {outcome['timeFrame']}")
    return " | ".join(parts)

def generate_pipeline_insights(prompt: str) -> str:
    if not settings.api_keys.gemini:
        return "Error: Gemini API key not configured."
    try:
        client = genai.Client(api_key=settings.api_keys.gemini)
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=GenerateContentConfig(max_output_tokens=2048)
        )
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        return f"Error generating pipeline insights: {str(e)}"

def cluster_outcomes_with_gemini(outcome_list, outcome_type="primary"):
    if not settings.api_keys.gemini:
        return {}
    client = genai.Client(api_key=settings.api_keys.gemini)

    formatted = [format_outcome_for_llm(o) for o in outcome_list]
    prompt = f"""
            You are an expert in clinical trial data curation. Group the following {outcome_type} outcomes:

            {chr(10).join([f"{i+1}. {s}" for i, s in enumerate(formatted)])}

            Output JSON:
            [
            {{
                "canonical": "...",
                "summary": "...",
                "originals": ["...", ...]
            }},
            ...
            ]
            """
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=GenerateContentConfig(max_output_tokens=2048)
        )
        content = response.candidates[0].content.parts[0].text.strip()
        start_idx = content.find('[')
        end_idx = content.rfind(']') + 1
        groups = json.loads(content[start_idx:end_idx])
        mapping = {}
        for group in groups:
            for orig in group.get("originals", []):
                mapping[orig.strip()] = {
                    "canonical": group.get("canonical", "").strip(),
                    "summary": group.get("summary", "").strip()
                }
        return mapping
    except Exception as e:
        logger.warning(f"Gemini clustering error: {e}")
        return {}

async def batch_query_gemini(drug_names):
    if not settings.api_keys.gemini or not drug_names:
        return {name: {"modality": "Unknown", "target": "Unknown", "source": "Gemini"} for name in drug_names}
    
    client = genai.Client(api_key=settings.api_keys.gemini)
    prompt = f"""
            You are a pharmacology and drug development expert. Provide modality and target for each drug:

            Drugs:
            {json.dumps(drug_names, indent=2)}

            Return JSON:
            {{
            "drugs": [
                {{
                "name": "...",
                "modality": "...",
                "target": "..."
                }},
                ...
            ]
            }}
            """
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=prompt,
            config=GenerateContentConfig(max_output_tokens=8000)
        )
        content = response.candidates[0].content.parts[0].text.strip()
        json_str = content[content.find('{'):content.rfind('}')+1]
        data = json.loads(json_str)
        return {
            d["name"]: {
                "modality": d.get("modality", "Unknown"),
                "target": d.get("target", "Unknown"),
                "source": "Gemini"
            } for d in data.get("drugs", [])
        }
    except Exception as e:
        logger.error(f"Batch Gemini query failed: {e}")
        return {name: {"modality": "Unknown", "target": "Unknown", "source": "Gemini"} for name in drug_names}

def categorize_outcomes_with_gemini(outcomes: List[str], outcome_type: str = "outcome", batch_size: int = 50) -> List[Dict]:
    if not settings.api_keys.gemini:
        return []
    client = genai.Client(api_key=settings.api_keys.gemini)

    all_results = []
    for i in range(0, len(outcomes), batch_size):
        batch = outcomes[i:i+batch_size]
        prompt = f"""
                Categorize each {outcome_type} outcome into:
                - Safety/Tolerability
                - Pharmacokinetics (PK)
                - Biomarkers
                - Efficacy: motor
                - Efficacy: cognitive
                - Efficacy: behavioral
                - Efficacy: QoL

                Respond with JSON:
                [
                {{"outcome": "...", "category": "..."}},
                ...
                ]

                Outcomes:
                {chr(10).join([f"- {o}" for o in batch])}
                """
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=GenerateContentConfig(max_output_tokens=3000)
            )
            content = response.candidates[0].content.parts[0].text.strip()
            json_str = content[content.find('['):content.rfind(']')+1]
            all_results.extend(json.loads(json_str))
        except Exception as e:
            logger.warning(f"Failed to categorize outcomes: {e}")
    return all_results

def categorize_primary_and_secondary_outcomes_with_gemini(primary_outcomes: list, secondary_outcomes: list, batch_size: int = 50) -> list:
    primary_results = categorize_outcomes_with_gemini(primary_outcomes, "primary", batch_size) if primary_outcomes else []
    for r in primary_results:
        r["type"] = "primary"
    secondary_results = categorize_outcomes_with_gemini(secondary_outcomes, "secondary", batch_size) if secondary_outcomes else []
    for r in secondary_results:
        r["type"] = "secondary"
    return primary_results + secondary_results