# Drug Enrichment Process Guide

This document explains how the drug enrichment process works in our clinical trials pipeline.

## Overview

The enrichment process adds modality and target information to drug interventions found in clinical trials. This helps in analyzing trends in drug development and understanding the biological mechanisms being targeted.

## Process Flow

1. **Extract Drug Names**: First, the pipeline extracts unique drug names from clinical trial interventions.

2. **Cache Check**: For each drug, the system first checks if information is already available in the SQLite cache database.

3. **ChEMBL Query**:
   - If not in cache, the system queries the ChEMBL database using the `chembl_webresource_client` Python library.
   - It searches for the drug by exact name, with a fallback to a broader search.
   - For matched drugs, it extracts:
     - **Modality**: Determined from the molecule type (e.g., small molecule, protein, antibody)
     - **Target**: Obtained from the mechanism of action records, looking up target information

4. **Google Gemini Fallback**:
   - If ChEMBL doesn't have information about the drug, the system falls back to Google's Gemini AI model.
   - The model is prompted to provide modality and target information in a structured JSON format.
   - The prompt specifies valid modality types and requests the primary protein or pathway target.

5. **Caching Results**:
   - All results are cached in an SQLite database for future use, regardless of source.
   - This improves performance for repeated queries and reduces API calls.

6. **Application to Trials Data**:
   - The enriched drug information is applied to the clinical trials dataset.
   - New columns for modalities and targets are added to the DataFrame.

## Modality Classification

The system maps ChEMBL molecule types to standardized modality categories:

| ChEMBL Type     | Standardized Modality           |
|-----------------|----------------------------------|
| Small molecule  | small-molecule                   |
| Protein         | protein                          |
| Antibody        | monoclonal antibody (mAb)        |
| Oligonucleotide | siRNA                            |
| Enzyme          | protein                          |
| Cell            | cell therapy                     |
| Gene            | gene therapy                     |
| Oligosaccharide | small-molecule                   |

## Usage Example

Here's how to use the enrichment module directly:

```python
from src.pipeline.enrich import enrich_drugs

# Example set of drug names
drug_names = {"Evolocumab", "Alirocumab", "Ezetimibe", "Rosuvastatin"}

# Enrich drugs
import asyncio
drug_info = asyncio.run(enrich_drugs(drug_names))

# Print results
for drug, info in drug_info.items():
    print(f"Drug: {drug}")
    print(f"  Modality: {info['modality']}")
    print(f"  Target: {info['target']}")
    print(f"  Source: {info['source']}")
    print()
```

## Requirements

- ChEMBL Web Resource Client: `pip install chembl_webresource_client`
- Google Generative AI Python Library: `pip install google-generativeai==0.5.2`
- Gemini API Key (only needed for the fallback mechanism)

## Tips

- The ChEMBL client doesn't require an API key and has access to a vast database of drug information.
- The Gemini model works best as a fallback for newer or experimental drugs not yet in ChEMBL.
- The SQLite cache significantly speeds up repeated runs of the pipeline.
- For large datasets, the process uses asyncio for parallel enrichment with a semaphore to control concurrency. 