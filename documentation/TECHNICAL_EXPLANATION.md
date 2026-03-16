# OPTI-MED MVP Technical Explanation

This document explains the current structured-data MVP pipeline and the final scored output file:

- `data/final/older_adult_medication_scores.csv`

It covers:

- the current rule-based deprescribing priority logic
- what each output row represents
- how each feature was derived
- what the MIMIC-IV demo data represents and what this MVP uses from it

## 1. Pipeline Overview

The current MVP pipeline is a staged structured-data workflow:

1. Load core MIMIC-IV demo hospital tables.
2. Build an older-adult medication cohort.
3. Enrich that cohort with minimal diagnosis, lab, and medication burden features.
4. Apply a transparent rule-based deprescribing priority score.

The main intermediate and final outputs are:

- `data/interim/older_adult_medication_cohort.csv`
- `data/processed/older_adult_medication_features.csv`
- `data/final/older_adult_medication_scores.csv`

At this stage, the pipeline is intentionally simple:

- no NLP
- no API layer
- no machine learning model
- no FHIR export

The design goal is clarity and replaceability rather than clinical completeness.

## 2. Demo Data Overview

This MVP uses the MIMIC-IV demo dataset included in the repository. MIMIC-IV is a deidentified critical care and hospital dataset derived from real clinical care data. The demo version is a small sample with the same general table structure as the full dataset, which makes it suitable for building and testing pipeline logic.

For this MVP, only the following hospital tables are used:

- `hosp/patients`
- `hosp/admissions`
- `hosp/prescriptions`
- `hosp/diagnoses_icd`
- `hosp/labevents`

What these tables represent in practical terms:

- `patients`: one row per patient, including demographic anchor fields such as sex and anchor age
- `admissions`: one row per hospital admission, including admission type and timestamps
- `prescriptions`: medication order and exposure records associated with admissions
- `diagnoses_icd`: ICD-coded diagnoses associated with admissions
- `labevents`: lab measurements recorded during care, including creatinine values

What we used from each table:

- `patients`: `subject_id`, `gender`, `anchor_age`
- `admissions`: `subject_id`, `hadm_id`, `admission_type`, `admittime`, `dischtime`
- `prescriptions`: `subject_id`, `hadm_id`, `drug`, `starttime`, `stoptime`
- `diagnoses_icd`: `hadm_id`, `icd_code`, `icd_version`
- `labevents`: `hadm_id`, `itemid`, `charttime`, `valuenum`

The MVP assumes the demo schema is compatible with the full MIMIC-IV dataset, so the same logic should transfer later with minimal changes to configuration and mappings.

## 3. What Each Row Represents

Each row in `older_adult_medication_scores.csv` represents:

- one medication exposure
- for one patient
- during one hospital admission

The grain of the file is therefore:

- `patient x admission x medication exposure`

More concretely:

- `subject_id` identifies the patient
- `hadm_id` identifies the hospital admission
- `drug`, `starttime`, and `stoptime` identify the medication exposure row within that admission

This means a single admission can appear many times:

- once for each medication exposure retained in the cohort

And a single patient can appear across:

- multiple admissions
- multiple medication rows per admission

Important implication:

- admission-level features such as diagnosis flags, creatinine summaries, medication count, and polypharmacy are repeated across all medication rows for the same `hadm_id`
- medication-specific flags such as opioid or benzodiazepine exposure vary row by row because they are derived from that row’s `drug` field
- the final score is row-level, not admission-level, because it combines admission context with the specific medication on that row

## 4. Cohort Construction Logic

The cohort builder creates the row set that later features and scoring operate on.

Source tables used:

- `patients`
- `admissions`
- `prescriptions`

Join logic:

- `patients` joins to `admissions` on `subject_id`
- `prescriptions` joins to the admission-level cohort on `subject_id` and `hadm_id`

Older-adult restriction:

- the cohort includes only patients with `anchor_age >= 65`

Why `anchor_age` is used:

- MIMIC-IV uses age-related deidentification conventions
- `anchor_age` is the practical age proxy available in the dataset for MVP filtering

One row per medication exposure:

- after joining, the cohort keeps the medication exposure fields from `prescriptions`
- repeated exposure rows present in the source prescriptions table are collapsed using the row-defining columns:
  - `subject_id`
  - `hadm_id`
  - `drug`
  - `starttime`
  - `stoptime`

Length of stay:

- `length_of_stay_days` is calculated from:
  - `dischtime - admittime`

This is a derived field, not a source field.

## 5. Feature Derivation

This section explains each feature in the processed and scored output and whether it was:

- directly extracted from demo data
- derived from direct source values
- derived from curated placeholder logic

### 5.1 Directly Carried or Renamed Fields

These are taken directly from source tables, with little or no transformation:

- `subject_id`
  - direct from MIMIC-IV source tables
- `hadm_id`
  - direct from MIMIC-IV source tables
- `sex`
  - renamed from `patients.gender`
- `admission_type`
  - direct from `admissions.admission_type`
- `admittime`
  - direct from `admissions.admittime`
- `dischtime`
  - direct from `admissions.dischtime`
- `drug`
  - direct from `prescriptions.drug`
- `starttime`
  - direct from `prescriptions.starttime`
- `stoptime`
  - direct from `prescriptions.stoptime`
- `age_proxy`
  - renamed from `patients.anchor_age`

### 5.2 Derived Demographic and Admission Features

- `age_group`
  - derived from `age_proxy`
  - current bins:
    - `65-74`
    - `75-84`
    - `85+`
  - this is calculated, not directly stored in the source data

- `length_of_stay_days`
  - calculated from `dischtime - admittime`
  - expressed in days
  - derived, not directly extracted

### 5.3 Diagnosis Flags

Source:

- `diagnoses_icd`

Fields used:

- `hadm_id`
- `icd_code`
- `icd_version`

How diagnosis flags are built:

- diagnoses are grouped at the admission level
- each diagnosis flag is generated by checking whether any diagnosis code for that `hadm_id` starts with one of a curated set of ICD prefixes
- separate ICD-9 and ICD-10 prefixes are supported

Current diagnosis flags:

- `ckd_flag`
- `dementia_flag`
- `delirium_flag`
- `heart_failure_flag`
- `diabetes_flag`

These are not directly stored in the demo data as ready-made flags. They are derived from diagnosis codes using curated placeholder mappings.

Important limitation:

- these are MVP heuristic mappings
- they are meant to be transparent and replaceable, not final clinical definitions

### 5.4 Creatinine and Renal Risk Features

Source:

- `labevents`

Fields used:

- `hadm_id`
- `itemid`
- `charttime`
- `valuenum`

How serum creatinine is identified:

- the pipeline uses a configurable set of MIMIC lab `itemid` values intended to represent serum or blood creatinine
- current default item IDs are:
  - `50912`
  - `51081`
  - `51977`
  - `52546`

How creatinine features are built:

- lab rows are filtered to the configured creatinine item IDs
- valid numeric `valuenum` rows are kept
- labs are grouped by `hadm_id`

Derived admission-level creatinine features:

- `creatinine_first`
  - first creatinine value in time order within the admission
- `creatinine_max`
  - maximum creatinine value within the admission
- `creatinine_mean`
  - mean creatinine value within the admission

These are all derived from source lab rows and are not direct source columns.

Renal risk proxy:

- `renal_risk_flag`
  - currently set to `1` if `creatinine_max >= 1.5`
  - otherwise `0`

This is a simple placeholder renal impairment heuristic for MVP purposes. It is not intended to be a full clinical kidney function model.

### 5.5 Medication Burden Features

Source:

- derived from the medication cohort built from `prescriptions`

How medication burden is built:

- for each admission (`hadm_id`), drug names are lowercased and trimmed
- distinct normalized drug strings are counted

Derived features:

- `total_medication_count`
  - number of distinct medications observed in the admission
- `polypharmacy_flag`
  - set to `1` if `total_medication_count >= 5`
  - otherwise `0`

These features are calculated from other data points and are not directly present in demo source tables.

### 5.6 High-Risk Medication Class Flags

Source:

- derived from the `drug` text field in `prescriptions`

How these flags are built:

- each row’s `drug` string is lowercased
- the medication name is matched against curated keyword lists
- if any keyword matches, the corresponding class flag is set to `1`

Current row-level medication class flags:

- `benzodiazepine_flag`
- `opioid_flag`
- `anticholinergic_flag`
- `ppi_flag`
- `antipsychotic_flag`

These are derived features based on curated keyword heuristics. They are not extracted as ready-made classes from demo data.

Important limitation:

- this is string matching, not a full medication normalization or RxNorm-based classification layer
- the current mappings are intentionally simple and replaceable

## 6. Current Scoring Logic

The final score is a transparent additive rule-based score applied to each row.

The scorer uses:

- medication burden
- renal risk
- medication class flags
- diagnosis flags

Current rule definitions:

| Rule | Condition | Points | Explanation Text |
| --- | --- | ---: | --- |
| Very high medication burden | `total_medication_count >= 10` | 3 | `very high medication burden (10+ distinct medications)` |
| Polypharmacy | `polypharmacy_flag == 1` and medication count < 10 | 2 | `polypharmacy during admission` |
| Renal risk | `renal_risk_flag == 1` | 2 | `renal risk based on creatinine` |
| Benzodiazepine | `benzodiazepine_flag == 1` | 2 | `benzodiazepine exposure` |
| Opioid | `opioid_flag == 1` | 2 | `opioid exposure` |
| Anticholinergic | `anticholinergic_flag == 1` | 2 | `anticholinergic exposure` |
| Antipsychotic | `antipsychotic_flag == 1` | 2 | `antipsychotic exposure` |
| Proton pump inhibitor | `ppi_flag == 1` | 1 | `proton pump inhibitor exposure` |
| CKD | `ckd_flag == 1` | 1 | `chronic kidney disease diagnosis` |
| Dementia | `dementia_flag == 1` | 1 | `dementia diagnosis` |
| Delirium | `delirium_flag == 1` | 2 | `delirium diagnosis` |
| Heart failure | `heart_failure_flag == 1` | 1 | `heart failure diagnosis` |
| Diabetes | `diabetes_flag == 1` | 1 | `diabetes diagnosis` |

How the final numeric score is calculated:

- evaluate every rule for the row
- sum the points from all triggered rules
- bound the result to the range `1` to `10`

This means:

- rows with no triggered rules still receive score `1`
- rows with many triggered rules are capped at `10`

Risk label mapping:

- `low`: score `1` to `3`
- `medium`: score `4` to `6`
- `high`: score `7` to `10`

Explanation field:

- `deprescribing_priority_explanation` is a semicolon-separated list of the explanation text for each triggered rule
- if no rules trigger, the explanation is:
  - `no major rule triggered`

Example interpretation:

- a row with high medication burden, opioid exposure, and renal risk will accumulate points from each of those rules
- the explanation will explicitly list those reasons

## 7. What the Final Score Means

The final score is not a prediction of harm, mortality, or an adverse drug event. It is a simple prioritization heuristic intended for demo purposes.

What it currently means:

- higher scores indicate rows where the medication exposure may deserve more attention in a deprescribing-oriented review
- the score is driven by transparent rule triggers rather than learned model weights

What it does not mean:

- it is not clinical advice
- it is not a validated risk score
- it is not medication appropriateness logic
- it is not a substitute for clinician review

This MVP score is best understood as:

- a demo-friendly ranking signal
- a placeholder for future refinement
- a way to make the prototype interpretable during early development

## 8. Column-by-Column Meaning in `older_adult_medication_scores.csv`

### Core row identity and context

- `subject_id`: patient identifier from MIMIC-IV
- `hadm_id`: hospital admission identifier from MIMIC-IV
- `sex`: patient sex, carried from `patients.gender`
- `age_proxy`: MIMIC-IV `anchor_age`
- `age_group`: derived age band
- `admission_type`: type of hospital admission
- `admittime`: admission timestamp
- `dischtime`: discharge timestamp
- `length_of_stay_days`: derived admission duration in days

### Medication exposure fields

- `drug`: raw medication name from `prescriptions`
- `starttime`: medication start timestamp
- `stoptime`: medication stop timestamp

### Admission-level burden and diagnosis features

- `total_medication_count`: distinct medication count for the admission
- `polypharmacy_flag`: admission-level polypharmacy indicator
- `ckd_flag`: admission-level CKD diagnosis flag
- `dementia_flag`: admission-level dementia diagnosis flag
- `delirium_flag`: admission-level delirium diagnosis flag
- `heart_failure_flag`: admission-level heart failure diagnosis flag
- `diabetes_flag`: admission-level diabetes diagnosis flag

### Row-level medication class features

- `benzodiazepine_flag`: row drug matched benzodiazepine keywords
- `opioid_flag`: row drug matched opioid keywords
- `anticholinergic_flag`: row drug matched anticholinergic keywords
- `ppi_flag`: row drug matched proton pump inhibitor keywords
- `antipsychotic_flag`: row drug matched antipsychotic keywords

### Admission-level renal features

- `creatinine_first`: first creatinine value found for the admission
- `creatinine_max`: highest creatinine value for the admission
- `creatinine_mean`: mean creatinine value for the admission
- `renal_risk_flag`: simple renal risk proxy derived from creatinine

### Final scoring outputs

- `deprescribing_priority_score`: integer score from 1 to 10
- `deprescribing_priority_label`: `low`, `medium`, or `high`
- `deprescribing_priority_explanation`: readable list of triggered scoring rules

## 9. Current MVP Limitations

The current implementation is intentionally minimal. Important limitations include:

- diagnosis flags are based on simple ICD prefix matching
- medication classes are based on keyword matching, not full normalization
- renal risk is based on a simple creatinine threshold, not eGFR or AKI criteria
- the score is not clinically validated
- admission-level context is repeated across medication rows by design
- the demo dataset is small and may not reflect the full frequency patterns of real MIMIC-IV

These limitations are acceptable for the current MVP because the immediate goal is:

- a clear, modular structured-data prototype
- working logic on demo data
- a codebase that can be refined incrementally later

## 10. Where to Modify the Logic

If you want to revise the pipeline later, the main files are:

- cohort construction:
  - [src/opti_med/cohort/builder.py](/Users/salmayousry/Desktop/optimed/src/opti_med/cohort/builder.py)
- diagnosis feature mappings:
  - [src/opti_med/features/mappings/diagnoses.py](/Users/salmayousry/Desktop/optimed/src/opti_med/features/mappings/diagnoses.py)
- medication class mappings:
  - [src/opti_med/features/mappings/medications.py](/Users/salmayousry/Desktop/optimed/src/opti_med/features/mappings/medications.py)
- creatinine item IDs and lab mapping:
  - [src/opti_med/features/mappings/labs.py](/Users/salmayousry/Desktop/optimed/src/opti_med/features/mappings/labs.py)
- feature engineering:
  - [src/opti_med/features/builder.py](/Users/salmayousry/Desktop/optimed/src/opti_med/features/builder.py)
- scoring rules:
  - [src/opti_med/scoring/rules.py](/Users/salmayousry/Desktop/optimed/src/opti_med/scoring/rules.py)
- score application:
  - [src/opti_med/scoring/scorer.py](/Users/salmayousry/Desktop/optimed/src/opti_med/scoring/scorer.py)

That separation is intentional so future changes can be made without rewriting the entire pipeline.
