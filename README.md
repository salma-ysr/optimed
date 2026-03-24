# OPTI-MED

Encounter-relative medication review foundations for the OPTI-MED MVP. The repo now includes configurable MIMIC-IV loaders, a patient-first encounter index, canonical medication-event extraction, encounter-relative current-medication logic, and a local dossier UI/API that separates current medications from prior history. The active scorer remains an admission-row, rule-based benchmark.

## Scope implemented so far

- Python `src/` project layout
- pandas-based loading for core MIMIC-IV hospital tables
- Configurable dataset root and file extension
- Dual-demo discovery for both `data/external/mimic-iv-clinical-database-demo-2.2` and `data/external/mimic-iv-ed-demo-2.2`
- Dataset manifest reporting for present tables, row counts, key columns, and missing optional tables
- Required-column validation with clear error messages
- One CLI command that loads all core tables and prints basic shape summaries
- A canonical `encounter_index` build step that links ED stays to hospital admissions when `hadm_id` is available
- A canonical `medication_events` build step that merges home meds, ED meds, hospital orders, and hospital administrations
- A minimal cohort builder that joins `patients`, `admissions`, and `prescriptions`
- Older-adult restriction based on MIMIC-IV `anchor_age`
- Interim cohort output with one row per medication exposure during an admission
- A processed cohort builder with diagnosis, creatinine, current-vs-historical medication burden, and high-risk medication features
- A final rule-based scorer with a numeric score, priority label, and readable explanation per row
- A FastAPI backend for browsing scored rows and patient dossiers locally
- A lightweight React frontend for browsing scored records and opening a patient dossier focused on currently active medications

## Project structure

```text
.
├── pyproject.toml
├── README.md
├── data/
├── documentation/
└── src/
    └── opti_med/
        ├── cli/
        ├── config.py
        └── data_access/
```

## Requirements

- Python 3.10+
- Local MIMIC-IV demo data under `data/external/`

Expected clinical demo files for the current scoring pipeline:

- `hosp/patients.csv` or `hosp/patients.csv.gz`
- `hosp/admissions.csv` or `hosp/admissions.csv.gz`
- `hosp/prescriptions.csv` or `hosp/prescriptions.csv.gz`
- `hosp/diagnoses_icd.csv` or `hosp/diagnoses_icd.csv.gz`
- `hosp/labevents.csv` or `hosp/labevents.csv.gz`

Expected ED demo files for the new discovery and encounter-index layer:

- `ed/edstays.csv` or `ed/edstays.csv.gz`
- optional but discovered when present: `triage`, `vitalsign`, `medrecon`, `diagnosis`, `pyxis`

Optional clinical medication tables used by the medication-event layer when present:

- `hosp/pharmacy.csv` or `hosp/pharmacy.csv.gz`
- `hosp/emar.csv` or `hosp/emar.csv.gz`
- `hosp/emar_detail.csv` or `hosp/emar_detail.csv.gz`

## Setup

Create a virtual environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

By default, the code looks for both demo datasets already present in this repo:

```text
data/external/
├── mimic-iv-clinical-database-demo-2.2
└── mimic-iv-ed-demo-2.2
```

You can override settings with environment variables:

```bash
export OPTI_MED_EXTERNAL_DATA_ROOT=data/external
export OPTI_MED_DATA_ROOT=/path/to/mimic-iv-root
export OPTI_MED_CLINICAL_DATA_ROOT=/path/to/mimic-iv-clinical-database-demo
export OPTI_MED_ED_DATA_ROOT=/path/to/mimic-iv-ed-demo
export OPTI_MED_FILE_EXTENSION=.csv.gz
export OPTI_MED_INTERIM_ROOT=data/interim
export OPTI_MED_PROCESSED_ROOT=data/processed
export OPTI_MED_FINAL_ROOT=data/final
export OPTI_MED_ED_DIR=ed
export OPTI_MED_OLDER_ADULT_AGE_THRESHOLD=65
export OPTI_MED_POLYPHARMACY_THRESHOLD=5
export OPTI_MED_RENAL_RISK_CREATININE_THRESHOLD=1.5
export OPTI_MED_SERUM_CREATININE_ITEMIDS=50912,51081,51977,52546
export OPTI_MED_SNAPSHOT_STRATEGY=latest_available
```

Current assumption:

- `OPTI_MED_DATA_ROOT` and `OPTI_MED_CLINICAL_DATA_ROOT` refer to the clinical demo root used by the cohort, feature, scoring, and dossier pipeline
- `OPTI_MED_ED_DATA_ROOT` refers to the ED demo root used for ED encounter linkage, reconciliation, Pyxis, triage, and vitals
- all "current meds" logic is encounter-relative and stays inside the patient’s own shifted MIMIC timeline

Or pass the root path directly on the command line:

```bash
python3 -m opti_med.cli.load_core_tables --data-root /path/to/mimic-iv-root
```

If your files are uncompressed CSVs, also pass:

```bash
python3 -m opti_med.cli.load_core_tables --data-root /path/to/mimic-iv-root --file-extension .csv
```

## Usage

Run the core loader:

```bash
python3 -m opti_med.cli.load_core_tables
```

This now does two things:

- loads and validates the current clinical core tables used by the MVP
- discovers both demo datasets and prints a lightweight manifest with row counts, key columns, and missing optional tables

To also build the canonical patient-first encounter index:

```bash
python3 -m opti_med.cli.load_core_tables --build-encounter-index
```

Expected output format:

```text
Loaded core MIMIC-IV tables successfully.
Clinical data root: data/external/mimic-iv-clinical-database-demo-2.2
- patients: rows=..., columns=...
- admissions: rows=..., columns=...
- prescriptions: rows=..., columns=...
- diagnoses_icd: rows=..., columns=...
- labevents: rows=..., columns=...
- clinical: root=..., tables_present=.../...
- ed: root=..., tables_present=.../...
```

If a required file is missing or required columns are absent, the command exits with a clear error message.

Build the minimal cohort:

```bash
python3 -m opti_med.cli.build_cohort
```

Optional overrides:

```bash
python3 -m opti_med.cli.build_cohort \
  --data-root /path/to/mimic-iv-root \
  --file-extension .csv.gz \
  --age-threshold 65 \
  --output data/interim/older_adult_medication_cohort.csv
```

The cohort output contains:

- `subject_id`
- `hadm_id`
- `sex`
- `age_proxy`
- `age_group`
- `admission_type`
- `admittime`
- `dischtime`
- `length_of_stay_days`
- `drug`
- `starttime`
- `stoptime`

The cohort builder validates join keys, rejects duplicate patient or admission identifiers in the source tables, checks for invalid admission timestamps, and collapses repeated medication exposure rows found in the source prescriptions table.

The cohort builder is intentionally unchanged in grain:

- it still uses the clinical demo only
- it still builds one row per medication exposure during a hospital admission
- it does not yet consume ED tables directly

Build the processed cohort with MVP features:

```bash
python3 -m opti_med.cli.build_features
```

Optional overrides:

```bash
python3 -m opti_med.cli.build_features \
  --data-root data/external/mimic-iv-clinical-database-demo-2.2 \
  --file-extension .csv.gz \
  --age-threshold 65 \
  --polypharmacy-threshold 5 \
  --renal-risk-threshold 1.5 \
  --output data/processed/older_adult_medication_features.csv
```

The processed output adds:

- diagnosis flags: `ckd_flag`, `dementia_flag`, `delirium_flag`, `heart_failure_flag`, `diabetes_flag`
- creatinine features: `creatinine_first`, `creatinine_max`, `creatinine_mean`, `renal_risk_flag`
- current medication burden:
  - `current_medication_count`
  - `current_peak_concurrent_medication_count`
  - `current_polypharmacy_flag`
- historical/background medication burden for debugging and analytics:
  - `historical_medication_count`
  - `historical_polypharmacy_flag`
- compatibility aliases:
  - `total_medication_count` maps to the current encounter-active count
  - `polypharmacy_flag` maps to the current encounter-active polypharmacy flag
- high-risk medication flags: `benzodiazepine_flag`, `opioid_flag`, `anticholinergic_flag`, `ppi_flag`, `antipsychotic_flag`

The feature engineering layer now also builds a reusable admission-level patient context object and merges it into the processed cohort. This adds:

- diagnosis-derived risk categories using the curated ICD dictionary
- supplemental ED diagnosis risk flags when an ED stay links to the admission
- morphology and reserve fields from `patients` and `omr`:
  - weight
  - BMI
  - eGFR
  - Cockcroft-Gault when weight is available
- dynamic medication-relevant evidence from `labevents`, `triage`, and `vitalsign`:
  - creatinine trend
  - potassium summaries
  - sodium summaries
  - blood pressure summaries
  - heart-rate summaries
  - pain summaries when available
- explicit provenance and unavailable-reason fields so downstream API work can distinguish observed, derived, and unavailable values

Current guardrails:

- values are only emitted when supported by the demo data
- OMR morphology values are taken only from records on or before the encounter end
- if Cockcroft-Gault cannot be computed because weight is missing, the value remains null and the reason is exposed explicitly
- medication burden and polypharmacy are computed only from canonical medications that are active at the encounter review timestamp
- inactive or historical medications are retained separately and do not contribute to the main dossier score

Build the final scored cohort:

```bash
python3 -m opti_med.cli.build_scores
```

Optional overrides:

```bash
python3 -m opti_med.cli.build_scores \
  --data-root data/external/mimic-iv-clinical-database-demo-2.2 \
  --file-extension .csv.gz \
  --age-threshold 65 \
  --polypharmacy-threshold 5 \
  --renal-risk-threshold 1.5 \
  --output data/final/older_adult_medication_scores.csv
```

The scored output adds:

- `deprescribing_priority_score`
- `deprescribing_priority_label`
- `deprescribing_priority_explanation`

The scoring layer now uses a structured IPD model with three buckets:

- base medication risk
- terrain or aggravating patient context
- dynamic biologic and vital evidence

Each scored medication row now also carries:

- a short summary alert
- bucket-by-bucket score breakdown
- structured clinician-readable reasons
- structured evidence values used in the score

The normalized score remains interpretable on a capped 0-to-10 scale, and the medication-class flags from the earlier MVP are still reused as part of the base medication-risk bucket.

The scoring rules are intentionally transparent and live in [src/opti_med/scoring/rules.py](/Users/salmayousry/Desktop/optimed/src/opti_med/scoring/rules.py) so they can be inspected and revised without changing the rest of the pipeline.

## Encounter Index

The dual-demo refactor adds an interim patient-first file:

- `data/interim/encounter_index.csv`

This file links:

- clinical `admissions`
- ED `edstays`
- patient demographics from `patients`

Current encounter-index behavior:

- one row per linked ED stay or standalone hospital admission
- rows are keyed by `subject_id` and carry `hadm_id` and/or `stay_id`
- when an ED stay has a `hadm_id`, it is linked to the corresponding hospital admission when present
- unmatched inpatient admissions remain as `hospital_only`
- unmatched ED stays remain as `ed_only`

This is partial patient-first scaffolding, not a replacement for the current scored benchmark. The score list and admission summaries still come from the saved admission-row scored dataset, while dossier encounter selection now reuses the encounter index.

## Medication Events

The canonical medication history layer is written to:

- `data/interim/medication_events.csv`

Build it with:

```bash
python3 -m opti_med.cli.build_medication_events
```

What it merges:

- home meds from ED `medrecon`
- ED dispense/admin-like events from `pyxis`
- hospital medication orders from `prescriptions`
- hospital medication metadata from `pharmacy` when available
- hospital administration events from `emar`, with optional dose/route detail from `emar_detail` when available

Key properties of the current implementation:

- stable schema even when optional tables are missing
- simple medication normalization from raw drug strings into `medication_normalized`
- provenance flags on every row:
  - `source_home_medrecon`
  - `source_ed_pyxis`
  - `source_hospital_order`
  - `source_hospital_admin`
- inferred continuity fields:
  - `continued_from_home_inferred`
  - `newly_started_during_encounter_inferred`

The scorer does not read the saved `medication_events.csv` artifact, but the medication-event layer is already reused today:

- the scorer rebuilds medication events in memory to decide which cohort medications are current at encounter review time and to compute historical burden
- the API dossier prefers the saved `medication_events.csv` artifact when present, otherwise it rebuilds the layer on demand

## Medication Snapshot

The encounter-relative snapshot layer is written to:

- `data/interim/medication_snapshot.csv`

Build it with:

```bash
python3 -m opti_med.cli.build_medication_snapshot --snapshot-strategy latest_available
```

Supported snapshot strategies:

- `ed`
- `hospital`
- `latest_available`

Current snapshot behavior:

- review time is selected from encounter-local timestamps, never from the real-world current clock
- active medication logic is explicit and reusable
- interval-based rows such as `hospital_order` are considered active when their interval overlaps the selected review timestamp
- `home_medrecon` is historical by default and is only treated as current when the encounter provides continuation evidence
- point events such as `ed_pyxis` and `hospital_admin` are treated as active only at their exact event timestamp unless richer duration data is added later
- the final snapshot deduplicates to one row per `encounter_id` and `medication_normalized`
- when several active rows map to the same normalized medication, the snapshot keeps the most durable representative row and aggregates provenance flags across all active candidates
- the API dossier keeps current medications and previous medication history as separate collections

The helper logic for interval overlap, snapshot-time selection, activity checks, and snapshot filtering is intentionally pure and unit-testable.

The scorer reuses this encounter-review logic in process, but it does not read the saved `medication_snapshot.csv` artifact. The API dossier uses the saved snapshot as an accelerator when present and otherwise recomputes selected-encounter review rows.

## Backend API

Install dependencies, including FastAPI and Uvicorn:

```bash
pip install -e .
```

Run the API locally:

```bash
uvicorn opti_med.api.app:app --reload --host 127.0.0.1 --port 8000
```

The API is local-only by default because it binds to `127.0.0.1`.
Interactive docs are available at:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

- `GET /health`
  - simple health check plus whether the scored output file exists
- `GET /patients`
  - list patient-centered dossier summaries
- `GET /patients/{subject_id}`
  - return one patient dossier with:
    - `review_timestamp`
    - `current_medications`
    - `previous_medication_history`
    - encounter summaries
    - left-column context
    - top problem flashes
  - supports optional `hadm_id` to select a specific encounter
- `GET /patients/{subject_id}/medications`
  - return the current medication cards for one patient
- `GET /patients/{subject_id}/encounters`
  - return encounter summaries for one patient
- `GET /scores`
  - list scored medication rows
  - supports `limit`, `offset`, `subject_id`, `hadm_id`, and `risk_label`
- `GET /admissions`
  - return admission-level review summaries aggregated from the scored medication rows
- `GET /scores/admission`
  - return the admission-level details view payload for one `subject_id` and `hadm_id`, including medication review and recommended review sections
- `GET /scores/row`
  - retrieve one scored row using `subject_id`, `hadm_id`, `drug`, and `starttime`
- `GET /scores/latest`
  - return metadata for the latest saved scored output file
- `POST /scores/refresh`
  - rebuild the scored output and reload it for the API

Fast dossier path:

- the patient dossier endpoint prefers the prebuilt interim artifacts:
  - `data/interim/encounter_index.csv`
  - `data/interim/medication_events.csv`
  - `data/interim/medication_snapshot.csv`
- this keeps dossier requests fast and avoids rebuilding the full raw medication-review layer on every click
- list and admission-summary endpoints still read from `data/final/older_adult_medication_scores.csv`

Example requests:

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/patients?limit=10"
curl "http://127.0.0.1:8000/patients/${SUBJECT_ID}"
curl "http://127.0.0.1:8000/patients/${SUBJECT_ID}/medications"
curl "http://127.0.0.1:8000/patients/${SUBJECT_ID}/encounters"
curl "http://127.0.0.1:8000/admissions?limit=10"
curl "http://127.0.0.1:8000/scores?limit=5"
curl "http://127.0.0.1:8000/scores?risk_label=high&limit=10"
curl "http://127.0.0.1:8000/scores/admission?subject_id=${SUBJECT_ID}&hadm_id=${HADM_ID}"
curl "http://127.0.0.1:8000/scores/row?subject_id=${SUBJECT_ID}&hadm_id=${HADM_ID}&drug=${DRUG}&starttime=${STARTTIME}"
curl http://127.0.0.1:8000/scores/latest
curl -X POST http://127.0.0.1:8000/scores/refresh
```

## Frontend

The frontend is a lightweight Vite React TypeScript app in [frontend/](/Users/salmayousry/Desktop/optimed/frontend).

Install and run it locally:

```bash
cd frontend
npm install
npm run dev
```

By default, the frontend expects the backend at:

```text
http://127.0.0.1:8000
```

If you need a different backend URL, create a local environment variable before starting Vite:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Frontend pages in this phase:

- `/`
  - patient-centered dashboard backed by `/patients`
- `/patients/:subjectId`
  - patient dossier showing current medications, previous medication history, encounter summaries, and reused rule-based score details when a scored row exists
- `/records/:subjectId/:hadmId`
  - legacy route alias that renders the same dossier component

Recommended local workflow:

```bash
# terminal 1
uvicorn opti_med.api.app:app --reload --host 127.0.0.1 --port 8000

# terminal 2
cd frontend
npm install
npm run dev
```

## Notes

- This repo still does not implement downstream modeling.
- The current rule-based scorer remains the benchmark and fallback while the patient-first medication-history path continues to mature.
- The loaders are intentionally simple so later phases can extend them without rewriting path or schema logic.
