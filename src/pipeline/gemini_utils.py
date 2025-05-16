"""Google Gemini API integration for drug enrichment.

This module provides functions to leverage Google's Gemini AI
for enriching drug information with modality and target data.
"""

import json
import logging
from typing import Dict, Optional
import re

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

from src.pipeline.config import settings
from src.pipeline.utils import logger

# Initialize Gemini API
def initialize_gemini() -> bool:
    """Initialize the Google Gemini API with the API key.
    
    Returns:
        bool: True if successfully initialized, False otherwise
    """
    try:
        if not settings.api_keys.gemini:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            return False
            
        genai.configure(api_key=settings.api_keys.gemini)
        return True
    except Exception as e:
        logger.error(f"Error initializing Gemini API: {e}")
        return False

async def query_gemini_for_drug_info(drug_name: str) -> Optional[Dict[str, str]]:
    """Query Google Gemini to get drug modality and target information.
    
    Args:
        drug_name: Name of the drug to query
        
    Returns:
        Dictionary with modality and target or None if not found
    """
    if not settings.api_keys.gemini:
        logger.warning("Gemini API key not set, skipping Gemini query")
        return None
    
    logger.debug(f"Querying Gemini for {drug_name}")
    
    # Initialize Gemini
    if not initialize_gemini():
        return None
    
    try:
        # Prepare the prompt for drug information extraction
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
        
        # Set safety settings appropriate for medical/research context
        safety_settings = [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            }
        ]
        
        # Set generation config for more deterministic output
        generation_config = {
            "temperature": 0.2,  # Lower temperature for more factual responses
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 5000,  # Shorter output for JSON only
        }
        
        # Create model
        model = genai.GenerativeModel(
            settings.gemini_model, 
            safety_settings=safety_settings
        )
        
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        if not hasattr(response, 'text'):
            logger.warning(f"Unexpected response format from Gemini API for {drug_name}")
            return None
            
        # Parse JSON from response
        content = response.text.strip()
        try:
            # Remove any non-JSON text that might be in the response
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
                
            # Try to find a valid JSON object
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Validate expected format
                modality = data.get("modality", "Unknown")
                target = data.get("target", "Unknown")
                
                return {
                    "name": drug_name,
                    "modality": modality,
                    "target": target,
                    "source": "Gemini",
                }
            else:
                logger.warning(f"No valid JSON object found in Gemini response for {drug_name}")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from Gemini response for {drug_name}: {e}")
            logger.debug(f"Gemini response: {content}")
            return None
            
    except GoogleAPIError as e:
        logger.warning(f"Google API error for {drug_name}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error querying Gemini for {drug_name}: {e}")
        return None
        
    return None 

def format_outcome_for_llm(outcome):
    """Format an outcome dict for LLM input."""
    parts = []
    if outcome.get('measure'):
        parts.append(outcome['measure'])
    if outcome.get('description'):
        parts.append(f"Description: {outcome['description']}")
    if outcome.get('timeFrame'):
        parts.append(f"Timeframe: {outcome['timeFrame']}")
    return " | ".join(parts)


def cluster_outcomes_with_gemini(outcome_list, outcome_type="primary"):
    """
    Use Gemini to cluster and canonicalize outcome measures (primary or secondary).
    Args:
        outcome_list: List of dicts with keys 'measure', 'description', 'timeFrame'.
        outcome_type: 'primary' or 'secondary' (for prompt clarity)
    Returns:
        Dict mapping original outcome string to {'canonical': ..., 'summary': ...}
    """
    if not settings.api_keys.gemini:
        logger.warning("Gemini API key not set, skipping Gemini query")
        return {}
    if not initialize_gemini():
        return {}
    # Prepare the prompt
    formatted = [format_outcome_for_llm(o) for o in outcome_list]
    numbered = [f"{i+1}. {s}" for i, s in enumerate(formatted)]
    joined = "\n".join(numbered)
    prompt = f"""
You are an expert in clinical trial data curation. Given the following list of {outcome_type} outcome measures (with descriptions and timeframes), group together outcomes that are semantically equivalent or highly similar, even if phrased differently. For each group, provide:
- A canonical label (short, human-readable)
- A brief summary (1-2 sentences)
- The list of original outcome strings in that group

List of outcomes:
{joined}

Output as JSON with this structure:
[
  {{
    "canonical": "...",
    "summary": "...",
    "originals": ["...", ...]
  }},
  ...
]
"""
    # Set generation config
    generation_config = {
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    # Create model
    model = genai.GenerativeModel(
        settings.gemini_model,
        safety_settings=[
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
         tools=["google_search_retrieval"]
    )
    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        if not hasattr(response, 'text'):
            logger.warning("Unexpected response format from Gemini API for outcome clustering")
            return {}
        content = response.text.strip()
        # Try to extract JSON
        try:
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                groups = json.loads(json_str)
                # Build mapping from each original to canonical/summary
                mapping = {}
                for group in groups:
                    canonical = group.get('canonical', '').strip()
                    summary = group.get('summary', '').strip()
                    for orig in group.get('originals', []):
                        mapping[orig.strip()] = {"canonical": canonical, "summary": summary}
                return mapping
            else:
                logger.warning("No valid JSON array found in Gemini response for outcome clustering")
                return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from Gemini response for outcome clustering: {e}")
            logger.debug(f"Gemini response: {content}")
            return {}
    except GoogleAPIError as e:
        logger.warning(f"Google API error for outcome clustering: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Unexpected error querying Gemini for outcome clustering: {e}")
        return {}
    return {} 

def generate_pipeline_insights(prompt: str) -> str:
    """
    Generate a pipeline-level insights report using Gemini LLM given a prompt string.
    Args:
        prompt: The prompt string to send to Gemini
    Returns:
        The LLM-generated report as a string
    """
    if not settings.api_keys.gemini:
        return "Error: Gemini API key not configured. Please set the GEMINI_API_KEY environment variable."
    try:
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating pipeline insights: {str(e)}" 

async def batch_query_gemini(drug_names):
    if not drug_names:
        return {}
    if not initialize_gemini():
        return {name: {"modality": "Unknown", "target": "Unknown", "source": "Gemini"} for name in drug_names}
    prompt = f"""
You are a pharmacology and drug development expert. For each drug name below, provide its modality and target.

**Instructions:**
- Ignore dosage, formulation, and administration route information (e.g., mg, BID, capsules, tablets, oral, injection, etc.) and focus on the core drug name.
- For drug names with slashes ("/"), plus signs ("+"), or "and", treat these as combinations and return a list of modalities and a list of targets for each component, in the same order as the drugs in the name.
- If a drug is an internal code or investigational compound, attempt to infer its modality and target from any available public information or typical drug class. Only return 'Unknown' if absolutely no information is available.
- For persistent unknowns, suggest a likely class or mechanism if possible, even if generic (e.g., 'investigational small-molecule', 'unknown target').
- For each drug, return a JSON object with:
    - "name": the drug name
    - "modality": the type (e.g., small-molecule, monoclonal antibody, siRNA, peptide, gene therapy, cell therapy, vaccine, placebo, device, procedure, other)
    - "target": Prefer gene symbols, protein names, or well-known pathway names. If only a generic target is known (e.g., 'dopamine receptor', 'immune system', 'bacterial cell wall'), use that rather than 'Unknown'.
- For placebos, always return 'placebo' for both modality and target.
- If the drug is a device, procedure, or not a drug, return 'device', 'procedure', or 'other' as the modality and a generic target if possible.
- Only return 'Unknown' if there is truly no information available after considering all generic/functional targets.
- Respond in this JSON format:
{{
  "drugs": [
    {{
    ...
  ]
}}

Drugs:
{json.dumps(drug_names, indent=2)}
"""
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(prompt, generation_config={"temperature": 0.2, "max_output_tokens": 10000})
    content = response.text.strip()
    try:
        start = content.find('{')
        end = content.rfind('}')+1
        data = json.loads(content[start:end])
        return {d['name']: {"modality": d['modality'], "target": d['target'], "source": "Gemini"} for d in data['drugs']}
    except Exception as e:
        logger.error(f"Failed to parse Gemini batch response: {e}")
        return {name: {"modality": "Unknown", "target": "Unknown", "source": "Gemini"} for name in drug_names} 

def categorize_outcomes_with_gemini(outcomes: list, outcome_type: str = "outcome", batch_size: int = 50) -> list:
    """
    Use Gemini to categorize outcome measures into predefined endpoint categories, with batching for large lists.
    Args:
        outcomes: List of outcome strings (flattened, normalized)
        outcome_type: 'primary' or 'secondary' (for prompt clarity)
        batch_size: Number of outcomes per Gemini call
    Returns:
        List of dicts: {"outcome": ..., "category": ...}
    """
    if not settings.api_keys.gemini:
        logger.warning("Gemini API key not set, skipping Gemini query")
        return []
    if not initialize_gemini():
        return []
    if not outcomes:
        return []
    results = []
    for i in range(0, len(outcomes), batch_size):
        batch = outcomes[i:i+batch_size]
        categories = [
            "Safety/Tolerability",
            "Pharmacokinetics (PK)",
            "Biomarkers",
            "Efficacy: motor",
            "Efficacy: cognitive",
            "Efficacy: behavioral",
            "Efficacy: QoL"
        ]
        prompt = f"""
You are a clinical trial expert. Categorize each {outcome_type} outcome below into one of the following categories:
- Safety/Tolerability
- Pharmacokinetics (PK)
- Biomarkers
- Efficacy: motor
- Efficacy: cognitive
- Efficacy: behavioral
- Efficacy: QoL

Return ONLY a valid JSON array like this:
[
  {{"outcome": "Outcome text", "category": "Efficacy: motor"}},
  ...
]

Here are the outcomes:
""" + "\n".join([f"- {o}" for o in batch]) + "\n\nOnly return valid JSON. Do not include explanations, markdown, or extra text."
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(prompt, generation_config={"temperature": 0.2, "max_output_tokens": 13000})
        content = response.text.strip()
        if not content:
            logger.error("Gemini response was empty. No content to parse.")
            continue
        logger.debug(f"Gemini raw response content: {repr(content)}")
        # Improved JSON extraction logic
        json_str = None
        code_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
            else:
                json_str = content  # fallback
        try:
            batch_results = json.loads(json_str)
            if isinstance(batch_results, list):
                logger.info(f"Gemini attempted JSON: {len(json_str)} characters")
                results.extend(batch_results)
            else:
                logger.error(f"Gemini batch did not return a list: {batch_results}")
        except Exception as e:
            logger.error(f"Failed to parse Gemini categorization response: {e}")
            logger.debug(f"Gemini attempted JSON: {json_str}")
            continue
    return results


def categorize_primary_and_secondary_outcomes_with_gemini(primary_outcomes: list, secondary_outcomes: list, batch_size: int = 50) -> list:
    """
    Categorize primary and secondary outcomes using Gemini, making separate API calls for each, and combine the results.
    Returns a list of dicts: {"outcome": ..., "category": ..., "type": "primary"/"secondary"}
    """
    primary_results = categorize_outcomes_with_gemini(primary_outcomes, outcome_type="primary", batch_size=batch_size) if primary_outcomes else []
    for r in primary_results:
        r["type"] = "primary"
    secondary_results = categorize_outcomes_with_gemini(secondary_outcomes, outcome_type="secondary", batch_size=batch_size) if secondary_outcomes else []
    for r in secondary_results:
        r["type"] = "secondary"
    return primary_results + secondary_results