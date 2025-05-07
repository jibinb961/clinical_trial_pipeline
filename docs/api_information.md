**Creating API documentation**


* **Source system** ClinicalTrials.gov **JSON API v2** (production base URL: `https://clinicaltrials.gov/api/v2`).
* **Consumers** Your ETL flow (`pipeline/etl.py`) and any internal services that query the processed dataset.
* **Format** All requests and responses use JSON UTF‑8.
* **Auth** Public; no token required. (Your enrichment services *do* need keys—keep them in `.env`.)
* **Rate‑limit** \~ 10 req/s soft limit; back off with exponential retry (`aiohttp` + `tenacity`).

---

## 2  |  Endpoints & Requests

| # | Endpoint              | Verb    | Purpose                                                               |
| - | --------------------- | ------- | --------------------------------------------------------------------- |
| ① | `/studies`            | **GET** | **Search**: pull paginated study summaries for a disease term.        |
| ② | `/studies/{nctId}`    | **GET** | **Details**: fetch the full canonical record for one study.           |
| ③ | `/studies/metadata`   | **GET** | Schema introspection (versioning, new fields). Cached weekly.         |
| ④ | `/stats/field/values` | **GET** | Lookup valid enum values (e.g., `OverallStatus`). Rare use—manual QA. |

### 2.1  Common query parameters (search & stats)

| Param          | Type       | Default    | Description                                                                                                                    |
| -------------- | ---------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `query.term`   | string     | —          | Boolean search expression. Example: `AREA[ConditionSearch] "Familial Hypercholesterolemia" AND AREA[StudyType] INTERVENTIONAL` |
| `pageSize`     | integer    | `100`      | 1‑1000. Keep ≤ 500 to stay under 1 MB responses.                                                                               |
| `pageToken`    | string     | —          | Opaque cursor returned in `nextPageToken`.                                                                                     |
| `fields`       | comma‑list | *all*      | Restrict payload to the columns listed in **§ 3**.                                                                             |
| `format`       | enum       | `json`     | `json` \| `csv`. JSON only for the pipeline.                                                                                   |
| `markupFormat` | enum       | `markdown` | `markdown` \| `legacy` — use `markdown`.                                                                                       |

### 2.2  Request examples

```http
GET /api/v2/studies?query.term=AREA[ConditionSearch]%20"familial%20hypercholesterolemia"
                  &fields=nctId,briefTitle,studyType,phases,overallStatus,startDateStruct,date,
                          completionDateStruct,date,enrollmentInfo,count,
                          conditions,interventions,type,name,
                          sponsorCollaboratorsModule.leadSponsor.name,
                          sponsorCollaboratorsModule.leadSponsor.class
                  &pageSize=500
                  &format=json
```

```http
GET /api/v2/studies/NCT00582660?markupFormat=markdown
```

---

## 3  |  Response Contract (minified)

Below is the **flat view** your transformation code should emit.
The *JSON‑path* on the right shows where each value comes from in the v2 payload.

| Field name (output)       | Type                    | Required | Source JSON‑path                                                  |
| ------------------------- | ----------------------- | -------- | ----------------------------------------------------------------- |
| `nct_id`                  | string                  | ✔        | `protocolSection.identificationModule.nctId`                      |
| `brief_title`             | string                  | ✔        | `protocolSection.identificationModule.briefTitle`                 |
| `official_title`          | string                  |          | `protocolSection.identificationModule.officialTitle`              |
| `study_type`              | enum `StudyType`        | ✔        | `protocolSection.designModule.studyType`                          |
| `phase`                   | enum `Phase`            |          | `protocolSection.designModule.phases[0]`                          |
| `overall_status`          | enum `Status`           | ✔        | `protocolSection.statusModule.overallStatus`                      |
| `start_date`              | date                    | ✔        | `protocolSection.statusModule.startDateStruct.date`               |
| `primary_completion_date` | date                    |          | `protocolSection.statusModule.primaryCompletionDateStruct.date`   |
| `completion_date`         | date                    |          | `protocolSection.statusModule.completionDateStruct.date`          |
| `enrollment_count`        | int                     |          | `protocolSection.designModule.enrollmentInfo.count`               |
| `enrollment_type`         | enum `EnrollmentType`   |          | `protocolSection.designModule.enrollmentInfo.type`                |
| `conditions`              | string\[]               | ✔        | `protocolSection.conditionsModule.conditions[]`                   |
| `interventions`           | object\[]               | ✔        | `protocolSection.armsInterventionsModule.interventions[]`         |
| ↳ `intervention_type`     | enum `InterventionType` |          | `.type`                                                           |
| ↳ `intervention_name`     | string                  |          | `.name`                                                           |
| `lead_sponsor`            | string                  |          | `protocolSection.sponsorCollaboratorsModule.leadSponsor.name`     |
| `sponsor_class`           | enum `AgencyClass`      |          | `protocolSection.sponsorCollaboratorsModule.leadSponsor.class`    |
| `collaborators`           | string\[]               |          | `protocolSection.sponsorCollaboratorsModule.collaborators[].name` |
| `geo_point`               | object                  |          | `protocolSection.contactsLocationsModule.locations[0].geoPoint`   |
| `has_results`             | boolean                 | ✔        | `hasResults`                                                      |
| **ENRICHED FIELDS**       |                         |          |                                                                   |
| `drug_modality`           | enum (custom)           | •        | inferred via DrugBank/LLM                                         |
| `drug_target`             | string                  | •        | inferred via DrugBank/LLM                                         |
| `data_pull_timestamp`     | datetime                | ✔        | added by pipeline                                                 |

> **Required** (✔) means the pipeline should drop / retry the record if absent.
> **Enriched** fields (•) are filled *after* source ingestion.

---

## 4  |  Enum subsets you need to support

Only the values your analysis cares about are listed; the full lists are huge.

### 4.1  `StudyType`

`INTERVENTIONAL` • `OBSERVATIONAL`

### 4.2  `Phase`

`PHASE1` • `PHASE2` • `PHASE3` • `PHASE4` • `NA`

### 4.3  `Status` (overall & last‑known)

`RECRUITING` • `ACTIVE_NOT_RECRUITING` • `COMPLETED` • `TERMINATED` • `WITHDRAWN`

### 4.4  `InterventionType`

`DRUG` • `BIOLOGICAL` • `DEVICE` • `BEHAVIORAL` • `PROCEDURE` • `RADIATION` • `DIETARY_SUPPLEMENT`

*(Pipeline should ignore any other types or log & skip.)*

---

## 5  |  Reference JSON snippet (after flattening)

```jsonc
{
  "nct_id": "NCT00582660",
  "brief_title": "Evaluation of Surgically Resected Colorectal Adenomas ...",
  "official_title": "Randomized, Placebo‑Controlled, Phase 2B Evaluation ...",
  "study_type": "INTERVENTIONAL",
  "phase": "PHASE2",
  "overall_status": "COMPLETED",
  "start_date": "2001-12",
  "completion_date": "2008-06",
  "enrollment_count": 40,
  "enrollment_type": "ACTUAL",
  "conditions": ["Colorectal Adenoma", "Colorectal Carcinoma"],
  "interventions": [
    {"intervention_type": "DRUG", "intervention_name": "Celecoxib"},
    {"intervention_type": "DRUG", "intervention_name": "Placebo"}
  ],
  "lead_sponsor": "University of Alabama at Birmingham",
  "sponsor_class": "OTHER",
  "collaborators": ["Pfizer", "Pharmacia"],
  "geo_point": {"lat": 33.52066, "lon": -86.80249},
  "has_results": true,
  // enriched →
  "drug_modality": "SMALL_MOLECULE",
  "drug_target": "COX-2",
  "data_pull_timestamp": "2025-05-07T15:30:12Z"
}
```

---

## 6  |  Versioning & Change Management

* **Schema drift watch** Poll `/studies/metadata` daily; hash the response.
  *If the hash changes, raise a Prefect notification and run contract tests.*
* **Field additions** New fields in v2 will be silently ignored unless whitelisted in `config.py::ALLOWED_FIELDS`.
* **Deprecations** Legacy `/api/legacy/*` endpoints are logged but never called in production.
* **Backward compatibility** Downstream consumers must read *only* the fields in § 3.

---

## 7  |  Error handling (pipeline conventions)

| HTTP status           | Retry?  | Action                                           |
| --------------------- | ------- | ------------------------------------------------ |
| 429 Too Many Requests | **Yes** | Exponential back‑off, max 5 tries.               |
| 5xx Server Error      | **Yes** | Same back‑off.                                   |
| 4xx except 429        | **No**  | Log `nct_id` & skip study.                       |
| Malformed JSON        | —       | Capture raw body to `data/errors/` and continue. |

---

## 8  |  Checklist for implementation

1. **Unit test** `test_etl.py::test_required_fields_present`.
2. **Pydantic model** `StudyRecord` enforcing § 3 contract.
3. **Prefect task** `fetch_studies(disease, start_year, end_year, fields)`.
4. **CI rule** Fail build if § 3 field list drifts from `StudyRecord`.
5. **Data catalog** Register Parquet in AWS Glue / Athena with the same column names.

---

Place this document under **version control**; update it whenever you add a new column or depend on a new enum value. 
It doubles as both developer onboarding material *and* an implicit contract with any dashboards or ML models you build later.
