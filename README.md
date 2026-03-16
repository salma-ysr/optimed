# OPTI-MED

Minimal structured-data foundations for the OPTI-MED MVP. The repo now includes configurable MIMIC-IV loaders plus a minimal older-adult patient-medication-admission cohort builder.

## Scope implemented so far

- Python `src/` project layout
- pandas-based loading for core MIMIC-IV hospital tables
- Configurable dataset root and file extension
- Required-column validation with clear error messages
- One CLI command that loads all core tables and prints basic shape summaries
- A minimal cohort builder that joins `patients`, `admissions`, and `prescriptions`
- Older-adult restriction based on MIMIC-IV `anchor_age`
- Interim cohort output with one row per medication exposure during an admission
- A processed cohort builder with diagnosis, creatinine, medication burden, and high-risk medication features
- A final rule-based scorer with a numeric score, priority label, and readable explanation per row

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
- A local MIMIC-IV dataset root containing a `hosp/` folder

Expected files for this phase:

- `hosp/patients.csv` or `hosp/patients.csv.gz`
- `hosp/admissions.csv` or `hosp/admissions.csv.gz`
- `hosp/prescriptions.csv` or `hosp/prescriptions.csv.gz`
- `hosp/diagnoses_icd.csv` or `hosp/diagnoses_icd.csv.gz`
- `hosp/labevents.csv` or `hosp/labevents.csv.gz`

## Setup

Create a virtual environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

By default, the code looks for the demo dataset already present in this repo:

```text
data/external/mimic-iv-clinical-database-demo-2.2
```

You can override settings with environment variables:

```bash
export OPTI_MED_DATA_ROOT=/path/to/mimic-iv-root
export OPTI_MED_FILE_EXTENSION=.csv.gz
export OPTI_MED_INTERIM_ROOT=data/interim
export OPTI_MED_PROCESSED_ROOT=data/processed
export OPTI_MED_FINAL_ROOT=data/final
export OPTI_MED_OLDER_ADULT_AGE_THRESHOLD=65
export OPTI_MED_POLYPHARMACY_THRESHOLD=5
export OPTI_MED_RENAL_RISK_CREATININE_THRESHOLD=1.5
export OPTI_MED_SERUM_CREATININE_ITEMIDS=50912,51081,51977,52546
```

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

Expected output format:

```text
Loaded core MIMIC-IV tables successfully.
Data root: data/external/mimic-iv-clinical-database-demo-2.2
- patients: rows=..., columns=...
- admissions: rows=..., columns=...
- prescriptions: rows=..., columns=...
- diagnoses_icd: rows=..., columns=...
- labevents: rows=..., columns=...
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
- medication burden: `total_medication_count`, `polypharmacy_flag`
- high-risk medication flags: `benzodiazepine_flag`, `opioid_flag`, `anticholinergic_flag`, `ppi_flag`, `antipsychotic_flag`

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

The scoring rules are intentionally transparent and live in [src/opti_med/scoring/rules.py](/Users/salmayousry/Desktop/optimed/src/opti_med/scoring/rules.py) so they can be inspected and revised without changing the rest of the pipeline.

## Notes

- This repo still does not implement APIs or downstream modeling.
- The loaders are intentionally simple so later phases can extend them without rewriting path or schema logic.
