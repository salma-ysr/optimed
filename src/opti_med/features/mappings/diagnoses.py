"""Curated ICD prefix mappings for MVP diagnosis flags."""

DIAGNOSIS_ICD_PREFIXES: dict[str, dict[int, tuple[str, ...]]] = {
    "ckd_flag": {
        9: ("585",),
        10: ("N18",),
    },
    "dementia_flag": {
        9: ("290", "2941", "3310"),
        10: ("F01", "F02", "F03", "G30"),
    },
    "delirium_flag": {
        9: ("2930", "2931", "78009"),
        10: ("F05", "R410", "R4182"),
    },
    "heart_failure_flag": {
        9: ("428",),
        10: ("I50",),
    },
    "diabetes_flag": {
        9: ("250",),
        10: ("E08", "E09", "E10", "E11", "E13"),
    },
}

DIAGNOSIS_RISK_CATEGORY_GROUPS: dict[str, tuple[str, ...]] = {
    "diagnosis_risk_renal_flag": ("ckd_flag",),
    "diagnosis_risk_cognitive_flag": ("dementia_flag", "delirium_flag"),
    "diagnosis_risk_cardiac_flag": ("heart_failure_flag",),
    "diagnosis_risk_metabolic_flag": ("diabetes_flag",),
}
