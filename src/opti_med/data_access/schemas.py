"""Required column definitions for the MVP core tables."""

CORE_TABLE_SCHEMAS: dict[str, list[str]] = {
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
