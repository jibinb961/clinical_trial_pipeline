# **Summary of Feedback from Alex – Action Plan & Clarification**

**Date:** May 12, 2025

This document outlines the feedback received from Alex on the Huntington’s Disease pipeline insights and details the interpretation, required actions, and clarifications.

---

## ✅ High-Priority Action Items

### 1. **Add Visualizations for Endpoints**

- **What Alex Means:** Create visual charts (bar/pie/Sankey) that show how many trials focus on:
    - Safety/Tolerability
    - Pharmacokinetics (PK)
    - Biomarkers
    - Efficacy (motor, cognitive, behavioral, QoL)
- **Action:** Categorize each primary/secondary outcome into one of these endpoint groups and visualize their frequencies.

def categorize_outcomes_with_gemini(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use Gemini LLM to categorize each primary and secondary outcome measure into endpoint categories.
    Returns a DataFrame with columns: nct_id, measure_text, measure_type, category
    """
    import json
    from llm_module import gemini_batch_json
    # Step 1: Extract all outcome measures with nct_id and type
    records = []
    for idx, row in df.iterrows():
        nct_id = row.get('nct_id', None)
        # Primary outcomes
        if 'primary_outcomes' in row and row['primary_outcomes']:
            for o in row['primary_outcomes']:
                if isinstance(o, dict) and o.get('measure'):
                    measure_text = o['measure']
                elif isinstance(o, str):
                    measure_text = o
                else:
                    continue
                records.append({
                    'nct_id': nct_id,
                    'measure_text': measure_text,
                    'measure_type': 'primary'
                })
        # Secondary outcomes
        if 'secondary_outcomes' in row and row['secondary_outcomes']:
            for o in row['secondary_outcomes']:
                if isinstance(o, dict) and o.get('measure'):
                    measure_text = o['measure']
                elif isinstance(o, str):
                    measure_text = o
                else:
                    continue
                records.append({
                    'nct_id': nct_id,
                    'measure_text': measure_text,
                    'measure_type': 'secondary'
                })
    if not records:
        return pd.DataFrame(columns=['nct_id', 'measure_text', 'measure_type', 'category'])
    # Step 2: Prepare Gemini prompt
    prompt = f"""
You are a clinical trial data expert. Your task is to categorize each primary and secondary outcome measure from clinical trials into one of the following standardized endpoint categories:

- Safety/Tolerability  
- Pharmacokinetics (PK)  
- Biomarkers  
- Efficacy: motor  
- Efficacy: cognitive  
- Efficacy: behavioral  
- Efficacy: QoL  

Instructions:
1. Read each outcome measure carefully.
2. Use only one of the categories listed above. If the category is unclear, infer the best fit based on terminology.
3. Do **not** include notes or explanations — return only the structured data.
4. Keep `nct_id` and `measure_type` unchanged.
5. Return the result as a JSON array, structured like this:

[
  {{
    "nct_id": "NCT0123456",
    "measure_text": "Incidence of treatment-emergent adverse events",
    "measure_type": "primary",
    "category": "Safety/Tolerability"
  }},
  ...
]
Here is the list of outcome measures to categorize:
{json.dumps(records, indent=2)}
"""
    # Step 3: Call Gemini LLM
    try:
        gemini_response = gemini_batch_json(prompt)
        # Should return a JSON array as string
        categorized = json.loads(gemini_response)
        # Validate structure
        df_cat = pd.DataFrame(categorized)[['nct_id', 'measure_text', 'measure_type', 'category']]
        return df_cat
    except Exception as e:
        logger.error(f"Gemini categorization failed: {e}")
        # Return empty DataFrame on failure
        return pd.DataFrame(columns=['nct_id', 'measure_text', 'measure_type', 'category']) 
---

### 2. **List All Used Endpoints in the Quantitative Summary**

- **What Alex Means:** Provide a structured table (not just LLM summary) grouping each outcome measure under endpoint categories.
- **Clarification:**
    - **Endpoints** = conceptual categories (Safety, PK, etc.)
    - **Outcome Measures** = trial-reported descriptions (e.g., “Change in MoCA”)
- **Action:** Generate 2 tables (one each for Primary and Secondary outcomes) showing:

| Category | Outcome Measures |
| --- | --- |
| Safety/Tolerability | AE, SAE, Lab values |
| PK | AUC, Cmax |
| Biomarkers | Huntingtin CSF levels |
| Efficacy (Motor) | UHDRS-TMS |
| ... | ... |

---

## 🔧 Additional Suggestions and Fixes

### 3. **Fix “Placebo” Listed as Target**

- **Issue:** Placebo is a control, not a biological target.
- **Action:** Remove “placebo” from the list of targets. Keep it only as a comparator in trial design descriptions.

---

### 4. **Expand Visualizations for Target, Modality, and Sponsor**

- **What Alex Means:** Current plots may show just a few top items.
- **Action:** Extend plots to include either:
    - All entries
    - Or Top 40 (with scrollable/interactive charts if needed)

---

### 5. **Create Modality-by-Phase Distribution Chart**

- **What Alex Wants:** A bar or stacked bar chart showing:
    - X-axis = Trial Phase (Phase 1, Phase 2, etc.)
    - Y-axis = Count of Trials
    - Colors = Modalities (e.g., small-molecule, siRNA)

---

### 6. **Extract Treatment Details (Duration, Dose, Mode, Frequency)**

- **What Alex Means:** Pull these from intervention details for each trial:
    - Treatment Length (e.g., 12 weeks)
    - Dose (e.g., 300mg)
    - Mode (e.g., oral)
    - Frequency (e.g., twice daily)
- **Action:** Create a structured table from intervention descriptions.

---

### 7. **Investigate Incorrect Targets like “DMD exon 51/53”**

- **Issue:** These belong to Duchenne Muscular Dystrophy, not Huntington’s Disease.
- **Action:** Review enrichment logic or source filtering to remove such targets from unrelated diseases.

---

### 8. **Clarify Quantitative Summary Generation**

- **Alex Asked:** Was the summary manual or auto-generated?
- **Clarification:**
    - “Quantitative Summary (Manual)” = extracted via pipeline (no LLM)
    - “Analysis” = LLM-generated from structured values (e.g., using Gemini or GPT)