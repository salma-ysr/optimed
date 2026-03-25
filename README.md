# OPTI-MED

Patient-first medication-review and training-data foundations for OPTI-MED.

The repo now has two active paths:

- the current training-data path built on standardized Parquet artifacts, encounter-review medication state, RxNorm mapping, and first-pass class/burden enrichment
- the older benchmark path built on admission-row CSV outputs and a transparent rule-based scorer

The first ML scope is intentionally narrow. We ingest the full data footprint, but the initial class-and-burden layer is limited to:

- `benzodiazepine`
- `opioid`
- `anticholinergic`
- `ppi`
- `antipsychotic`

Anything outside that scope remains explicitly `unresolved` rather than being forced into a heuristic class.

## What Is Implemented

- raw clinical and ED discovery plus standardization to Parquet
- manifest-based full-data ingestion
- canonical `encounter_index`
- canonical `medication_events`
- encounter-review `encounter_medication_state`
- cache-first RxNorm mapping with persisted unresolved outcomes
- first-pass class semantics and burden artifacts for the initial supported classes
- legacy cohort, feature, and scoring builders for the rule-based benchmark
- local FastAPI + React dossier UI for browsing the current benchmark outputs

## Project Layout

```text
.
├── README.md
├── docs/
├── frontend/
├── pyproject.toml
├── src/
│   └── opti_med/
│       ├── api/
│       ├── cli/
│       ├── data_access/
│       ├── medication_semantics/
│       ├── pipeline/
│       ├── scoring/
│       └── standardized/
└── tests/
```

## Requirements

- Python 3.10+
- local MIMIC-IV data, either demo or full
- Node.js only if you want to run the frontend

Current defaults expect data roots such as:

```text
data/external/
├── mimic-iv-clinical-database-demo-2.2
├── mimic-iv-ed-demo-2.2
└── mimiciv/...
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional frontend setup:

```bash
cd frontend
npm install
```

## Configuration

Important roots:

```bash
export OPTI_MED_EXTERNAL_DATA_ROOT=data/external
export OPTI_MED_CLINICAL_DATA_ROOT=/path/to/mimic-iv-clinical
export OPTI_MED_ED_DATA_ROOT=/path/to/mimic-iv-ed
export OPTI_MED_RAW_CLINICAL_DATA_ROOT=/path/to/full/raw/clinical
export OPTI_MED_RAW_ED_DATA_ROOT=/path/to/full/raw/ed
export OPTI_MED_STANDARDIZED_ROOT=data/standardized
export OPTI_MED_ANALYTICAL_ROOT=data/analytical
export OPTI_MED_MANIFEST_ROOT=data/standardized/manifests
```

Other useful settings:

```bash
export OPTI_MED_FILE_EXTENSION=.csv.gz
export OPTI_MED_STANDARDIZED_FILE_EXTENSION=.parquet
export OPTI_MED_INGESTION_BEHAVIOR=incremental
export OPTI_MED_INGESTION_CHUNK_SIZE=100000
export OPTI_MED_ENCOUNTER_MEDICATION_QC_REPORT_PATH=docs/ml_pivot/11_encounter_and_medication_events_qc.md
export OPTI_MED_SNAPSHOT_STRATEGY=latest_available
export OPTI_MED_OLDER_ADULT_AGE_THRESHOLD=65
export OPTI_MED_POLYPHARMACY_THRESHOLD=5
export OPTI_MED_RENAL_RISK_CREATININE_THRESHOLD=1.5
export OPTI_MED_SERUM_CREATININE_ITEMIDS=50912,51081,51977,52546
```

Notes:

- `OPTI_MED_STANDARDIZED_ROOT` holds standardized raw-table parquet plus `encounter_index.parquet` and `medication_events.parquet`
- `OPTI_MED_ANALYTICAL_ROOT` holds review-time and semantics/burden parquet outputs
- `cache_first` RxNorm mode will call RxNav for cache misses
- `cache_only` is the offline replay mode when you already have a mapping cache

## Quick Sanity Check

For demo data inspection and light validation:

```bash
python3 -m opti_med.cli.load_core_tables
```

This is still useful for:

- confirming roots and file extensions
- validating the core clinical demo tables
- printing a small dataset manifest summary

Optional:

```bash
python3 -m opti_med.cli.load_core_tables --build-encounter-index
```

## Recommended Build Order

The current primary build path is the standardized/parquet training-data path.

### 1. Standardize Raw Data

Build standardized Parquet tables and a manifest:

```bash
python3 -m opti_med.cli.ingest_full_data \
  --clinical-root /path/to/raw/mimic-iv-clinical \
  --ed-root /path/to/raw/mimic-iv-ed \
  --standardized-root data/standardized \
  --manifest-root data/standardized/manifests \
  --incremental
```

Use `--overwrite` when you want to rebuild standardized outputs even if the latest manifest matches.

Primary outputs:

- `data/standardized/clinical/*.parquet`
- `data/standardized/ed/*.parquet`
- `data/standardized/manifests/latest.json`

### 2. Build Encounter Index

```bash
python3 -m opti_med.cli.build_encounter_index \
  --standardized-root data/standardized
```

Primary output:

- `data/standardized/encounter_index.parquet`

### 3. Build Medication Events

```bash
python3 -m opti_med.cli.build_medication_events \
  --standardized-root data/standardized \
  --qc-report docs/ml_pivot/11_encounter_and_medication_events_qc.md
```

Primary outputs:

- `data/standardized/medication_events.parquet`
- optional QC markdown report

This layer merges:

- home meds from `medrecon`
- ED medication events from `pyxis`
- hospital orders from `prescriptions`
- pharmacy enrichment when available
- hospital administration rows from `emar` and `emar_detail`

### 4. Build Encounter Medication State

```bash
python3 -m opti_med.cli.build_encounter_medication_state \
  --standardized-root data/standardized \
  --rxnorm-lookup-mode cache_first \
  --qc-report docs/ml_pivot/12_encounter_medication_state.md
```

Primary outputs:

- `data/analytical/encounter_medication_state.parquet`
- `data/analytical/medication_rxnorm_mapping.parquet`

Current defaults:

- review-time policy: `discharge_capped_latest_available`
- RxNorm lookup mode: `cache_first`

Supported RxNorm lookup modes:

- `disabled`
- `cache_first`
- `cache_only`

### 5. Build First-Scope Class And Burden Features

```bash
python3 -m opti_med.cli.build_first_scope_medication_features \
  --standardized-root data/standardized \
  --analytical-root data/analytical
```

Primary outputs:

- `data/analytical/encounter_medication_semantics.parquet`
- `data/analytical/encounter_medication_burden.parquet`

This layer adds:

- `medication_class_standardized` for the initial supported classes or explicit `unresolved`
- preserved heuristic class flags as debug anchors
- exact current medication count at review time
- class-specific current burden counts
- same-class duplicate-therapy signals
- continuation vs new-start indicators
- duration before review time where inferable
- `scheduled_vs_prn`
- opioid MME readiness scaffolding
- renal-dose mismatch readiness scaffolding

### 6. Build The Phase 2.5 Modeling Hand-Off

Materialize the modeling-ready dataset, deterministic subject-safe splits, feature-list JSON, and QC report for the canonical 65+ first-scope branch:

```bash
python3 -m opti_med.cli.build_first_scope_modeling_handoff \
  --analytical-root data/analytical \
  --feature-store-root data/feature_store \
  --label-root data/labels \
  --modeling-root data/modeling \
  --dataset-output data/modeling/encounter_medication_dataset_v1_65plus_first_scope.parquet \
  --feature-list-output data/modeling/feature_list_v1_65plus_first_scope.json \
  --splits-output data/modeling/splits_v1_65plus_first_scope.parquet \
  --qc-output docs/ml_pivot/17a_dataset_v1_65plus_first_scope_qc.md
```

The canonical training boundary remains:

- train only on rows where `meta__dataset_row_eligible_for_training_flag == 1`
- use only `feature__*` columns as trainable inputs
- keep `benchmark__*` columns for baseline comparison and fallback only

## Current Artifact Map

Standardized root:

- `clinical/*.parquet`
- `ed/*.parquet`
- `encounter_index.parquet`
- `medication_events.parquet`

Analytical root:

- `encounter_medication_state.parquet`
- `medication_rxnorm_mapping.parquet`
- `encounter_medication_semantics.parquet`
- `encounter_medication_burden.parquet`

Modeling root:

- `data/modeling/encounter_medication_dataset_v1_65plus_first_scope.parquet`
- `data/modeling/splits_v1_65plus_first_scope.parquet`
- `data/modeling/feature_list_v1_65plus_first_scope.json`

Legacy CSV outputs:

- `data/interim/older_adult_medication_cohort.csv`
- `data/processed/older_adult_medication_features.csv`
- `data/final/older_adult_medication_scores.csv`

## Legacy Benchmark Path

The older benchmark path is still available and still useful for baseline comparisons, UI support, and rules validation.

Build the minimal cohort:

```bash
python3 -m opti_med.cli.build_cohort
```

Build processed features:

```bash
python3 -m opti_med.cli.build_features
```

Build scored output:

```bash
python3 -m opti_med.cli.build_scores
```

This remains:

- admission-row
- clinically narrow and rule-based
- the active benchmark and fallback for the current local API

It is not the new training-data representation for the patient-first medication-state pipeline.

## API And Frontend

The local API and frontend are still centered on the legacy score output plus dossier-oriented artifacts.

Run the API:

```bash
uvicorn opti_med.api.app:app --reload --host 127.0.0.1 --port 8000
```

Run the frontend:

```bash
cd frontend
npm run dev
```

Default local URLs:

- API docs: `http://127.0.0.1:8000/docs`
- frontend dev server: Vite default local URL

Important limitation:

- the new `encounter_medication_semantics` and `encounter_medication_burden` artifacts are part of the training-data path
- the current local API/UI does not yet treat them as its primary data source

## Validation

Current targeted test command:

```bash
.venv/bin/python -m unittest \
  tests.test_medication_rxnorm_mapping \
  tests.test_encounter_medication_state \
  tests.test_encounter_medication_semantics \
  tests.test_standardized_analytical_artifacts
```

## Documentation

Useful docs under `docs/ml_pivot/`:

- `10_full_data_ingestion.md`
- `12_encounter_medication_state.md`
- `13_rxnorm_api_workflow.md`
- `14_first_scope_class_and_burden_features.md`

## Current Guardrails

- the project does not yet model the full medication universe
- the first-pass class layer is intentionally partial and keeps unsupported rows explicit as `unresolved`
- opioid MME is scaffolded, not fully implemented
- renal-dose mismatch is scaffolded, not clinically implemented
- the rule-based scorer remains the benchmark while the training-data path matures
