"""Column definitions for the current clinical and ED demo loaders."""

CLINICAL_CORE_TABLE_SCHEMAS: dict[str, list[str]] = {
    "patients": [
        "subject_id",
        "gender",
        "anchor_age",
    ],
    "admissions": [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
    ],
    "prescriptions": [
        "subject_id",
        "hadm_id",
        "starttime",
        "stoptime",
        "drug",
    ],
    "diagnoses_icd": [
        "subject_id",
        "hadm_id",
        "icd_code",
        "icd_version",
    ],
    "labevents": [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
    ],
}

CORE_TABLE_SCHEMAS = CLINICAL_CORE_TABLE_SCHEMAS

CLINICAL_OPTIONAL_TABLE_SCHEMAS: dict[str, list[str]] = {
    "transfers": ["subject_id", "hadm_id"],
    "services": ["subject_id", "hadm_id"],
    "pharmacy": ["subject_id", "hadm_id"],
    "emar": ["subject_id", "hadm_id"],
    "emar_detail": ["subject_id", "emar_id", "emar_seq"],
    "omr": ["subject_id"],
}

ED_CORE_TABLE_SCHEMAS: dict[str, list[str]] = {
    "edstays": [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "gender",
        "race",
        "arrival_transport",
        "disposition",
    ],
}

ED_OPTIONAL_TABLE_SCHEMAS: dict[str, list[str]] = {
    "triage": ["subject_id", "stay_id", "chiefcomplaint", "acuity"],
    "vitalsign": ["subject_id", "stay_id"],
    "medrecon": ["subject_id", "stay_id", "name"],
    "diagnosis": ["subject_id", "stay_id"],
    "pyxis": ["subject_id", "stay_id"],
}

TABLE_KEY_COLUMNS: dict[str, list[str]] = {
    "patients": ["subject_id"],
    "admissions": ["subject_id", "hadm_id"],
    "prescriptions": ["subject_id", "hadm_id", "drug", "starttime"],
    "diagnoses_icd": ["subject_id", "hadm_id", "icd_code"],
    "labevents": ["subject_id", "hadm_id", "itemid", "charttime"],
    "transfers": ["subject_id", "hadm_id"],
    "services": ["subject_id", "hadm_id"],
    "pharmacy": ["subject_id", "hadm_id", "pharmacy_id"],
    "emar": ["subject_id", "hadm_id"],
    "emar_detail": ["subject_id", "emar_id", "emar_seq", "pharmacy_id"],
    "omr": ["subject_id"],
    "edstays": ["subject_id", "stay_id", "hadm_id"],
    "triage": ["subject_id", "stay_id"],
    "vitalsign": ["subject_id", "stay_id"],
    "medrecon": ["subject_id", "stay_id", "name"],
    "diagnosis": ["subject_id", "stay_id"],
    "pyxis": ["subject_id", "stay_id"],
}
