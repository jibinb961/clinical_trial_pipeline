"""Google Gemini API integration for drug enrichment.

This module provides functions to leverage Google's Gemini AI
for enriching drug information with modality and target data.
"""

import json
import logging
from typing import Dict, Optional

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
            "max_output_tokens": 256,  # Shorter output for JSON only
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