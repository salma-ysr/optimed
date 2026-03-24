"""Standardization specs for raw MIMIC-IV clinical and ED tables."""

from __future__ import annotations

from dataclasses import dataclass

from opti_med.data_access.schemas import (
    CLINICAL_CORE_TABLE_SCHEMAS,
    CLINICAL_OPTIONAL_TABLE_SCHEMAS,
    ED_CORE_TABLE_SCHEMAS,
    ED_OPTIONAL_TABLE_SCHEMAS,
    TABLE_KEY_COLUMNS,
)


SOURCE_MANIFEST_METADATA_COLUMNS: tuple[str, ...] = (
    "_source_dataset",
    "_source_table",
    "_source_file",
    "_source_file_size_bytes",
    "_source_last_modified_at",
    "_ingestion_run_id",
    "_ingested_at",
)

GLOBAL_INTEGER_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "hadm_id",
    "stay_id",
    "pharmacy_id",
    "emar_id",
    "emar_seq",
    "labevent_id",
    "specimen_id",
    "itemid",
    "seq_num",
    "icd_version",
    "anchor_age",
    "anchor_year",
    "hospital_expire_flag",
)

GLOBAL_FLOAT_COLUMNS: tuple[str, ...] = (
    "valuenum",
    "ref_range_lower",
    "ref_range_upper",
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "acuity",
)


@dataclass(frozen=True, slots=True)
class TableStandardizationSpec:
    """Normalization spec for one raw table."""

    dataset_name: str
    table_name: str
    source_subdir: str
    required: bool
    required_columns: tuple[str, ...]
    key_columns: tuple[str, ...]
    timestamp_columns: tuple[str, ...] = ()
    harmonized_optional_columns: tuple[str, ...] = ()
    integer_columns: tuple[str, ...] = ()
    float_columns: tuple[str, ...] = ()

    @property
    def expected_columns(self) -> tuple[str, ...]:
        """Return canonical columns to retain in stable order before passthrough extras."""
        seen: set[str] = set()
        ordered_columns: list[str] = []
        for column_name in (
            *self.required_columns,
            *self.harmonized_optional_columns,
        ):
            if column_name not in seen:
                ordered_columns.append(column_name)
                seen.add(column_name)
        return tuple(ordered_columns)


def _spec(
    dataset_name: str,
    table_name: str,
    *,
    source_subdir: str,
    required: bool,
    required_columns: list[str],
    timestamp_columns: tuple[str, ...] = (),
    harmonized_optional_columns: tuple[str, ...] = (),
    integer_columns: tuple[str, ...] = (),
    float_columns: tuple[str, ...] = (),
) -> TableStandardizationSpec:
    return TableStandardizationSpec(
        dataset_name=dataset_name,
        table_name=table_name,
        source_subdir=source_subdir,
        required=required,
        required_columns=tuple(required_columns),
        key_columns=tuple(TABLE_KEY_COLUMNS.get(table_name, [])),
        timestamp_columns=timestamp_columns,
        harmonized_optional_columns=harmonized_optional_columns,
        integer_columns=integer_columns,
        float_columns=float_columns,
    )


STANDARDIZED_TABLE_SPECS: dict[tuple[str, str], TableStandardizationSpec] = {
    ("clinical", "patients"): _spec(
        "clinical",
        "patients",
        source_subdir="hosp",
        required=True,
        required_columns=CLINICAL_CORE_TABLE_SCHEMAS["patients"],
        timestamp_columns=("dod",),
        harmonized_optional_columns=("anchor_year", "anchor_year_group", "dod"),
        integer_columns=("subject_id", "anchor_age", "anchor_year"),
    ),
    ("clinical", "admissions"): _spec(
        "clinical",
        "admissions",
        source_subdir="hosp",
        required=True,
        required_columns=CLINICAL_CORE_TABLE_SCHEMAS["admissions"],
        timestamp_columns=("admittime", "dischtime", "deathtime", "edregtime", "edouttime"),
        harmonized_optional_columns=(
            "deathtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "edregtime",
            "edouttime",
            "hospital_expire_flag",
        ),
        integer_columns=("subject_id", "hadm_id", "hospital_expire_flag"),
    ),
    ("clinical", "prescriptions"): _spec(
        "clinical",
        "prescriptions",
        source_subdir="hosp",
        required=True,
        required_columns=CLINICAL_CORE_TABLE_SCHEMAS["prescriptions"],
        timestamp_columns=("starttime", "stoptime"),
        harmonized_optional_columns=(
            "pharmacy_id",
            "drug_type",
            "drug",
            "gsn",
            "ndc",
            "prod_strength",
            "form_rx",
            "dose_val_rx",
            "dose_unit_rx",
            "route",
        ),
        integer_columns=("subject_id", "hadm_id", "pharmacy_id"),
    ),
    ("clinical", "diagnoses_icd"): _spec(
        "clinical",
        "diagnoses_icd",
        source_subdir="hosp",
        required=True,
        required_columns=CLINICAL_CORE_TABLE_SCHEMAS["diagnoses_icd"],
        harmonized_optional_columns=("seq_num",),
        integer_columns=("subject_id", "hadm_id", "seq_num", "icd_version"),
    ),
    ("clinical", "labevents"): _spec(
        "clinical",
        "labevents",
        source_subdir="hosp",
        required=True,
        required_columns=CLINICAL_CORE_TABLE_SCHEMAS["labevents"],
        timestamp_columns=("charttime", "storetime"),
        harmonized_optional_columns=(
            "labevent_id",
            "specimen_id",
            "order_provider_id",
            "storetime",
            "value",
            "valueuom",
            "ref_range_lower",
            "ref_range_upper",
            "flag",
            "priority",
            "comments",
        ),
        integer_columns=(
            "labevent_id",
            "subject_id",
            "hadm_id",
            "specimen_id",
            "itemid",
        ),
        float_columns=("valuenum", "ref_range_lower", "ref_range_upper"),
    ),
    ("clinical", "omr"): _spec(
        "clinical",
        "omr",
        source_subdir="hosp",
        required=False,
        required_columns=CLINICAL_OPTIONAL_TABLE_SCHEMAS["omr"],
        timestamp_columns=("chartdate",),
        harmonized_optional_columns=("chartdate", "seq_num", "result_name", "result_value"),
        integer_columns=("subject_id", "seq_num"),
    ),
    ("clinical", "pharmacy"): _spec(
        "clinical",
        "pharmacy",
        source_subdir="hosp",
        required=False,
        required_columns=CLINICAL_OPTIONAL_TABLE_SCHEMAS["pharmacy"],
        timestamp_columns=("starttime", "stoptime", "entertime", "verifiedtime"),
        harmonized_optional_columns=(
            "pharmacy_id",
            "poe_id",
            "proc_type",
            "status",
            "medication",
            "entertime",
            "verifiedtime",
            "route",
            "frequency",
            "disp_sched",
            "infusion_type",
            "sliding_scale",
            "lockout_interval",
            "basal_rate",
            "one_hr_max",
            "doses_per_24_hrs",
            "duration",
            "duration_interval",
            "expiration_value",
            "expiration_unit",
            "fill_quantity",
        ),
        integer_columns=("subject_id", "hadm_id", "pharmacy_id"),
    ),
    ("clinical", "emar"): _spec(
        "clinical",
        "emar",
        source_subdir="hosp",
        required=False,
        required_columns=CLINICAL_OPTIONAL_TABLE_SCHEMAS["emar"],
        timestamp_columns=("charttime", "scheduletime", "storetime"),
        harmonized_optional_columns=(
            "emar_id",
            "emar_seq",
            "poe_id",
            "pharmacy_id",
            "medication",
            "event_txt",
            "charttime",
            "scheduletime",
            "storetime",
        ),
        integer_columns=("subject_id", "hadm_id", "emar_id", "emar_seq", "pharmacy_id"),
    ),
    ("clinical", "emar_detail"): _spec(
        "clinical",
        "emar_detail",
        source_subdir="hosp",
        required=False,
        required_columns=CLINICAL_OPTIONAL_TABLE_SCHEMAS["emar_detail"],
        timestamp_columns=("storetime",),
        harmonized_optional_columns=(
            "parent_field_ordinal",
            "administration_type",
            "pharmacy_id",
            "barcode_type",
            "reason_for_no_barcode",
            "complete_dose_not_given",
            "dose_due",
            "dose_due_unit",
            "dose_given",
            "dose_given_unit",
            "will_remainder_of_dose_be_given",
            "product_amount_given",
            "product_unit",
            "storetime",
        ),
        integer_columns=("subject_id", "emar_id", "emar_seq", "pharmacy_id"),
    ),
    ("ed", "edstays"): _spec(
        "ed",
        "edstays",
        source_subdir="ed",
        required=True,
        required_columns=ED_CORE_TABLE_SCHEMAS["edstays"],
        timestamp_columns=("intime", "outtime"),
        integer_columns=("subject_id", "hadm_id", "stay_id"),
    ),
    ("ed", "diagnosis"): _spec(
        "ed",
        "diagnosis",
        source_subdir="ed",
        required=False,
        required_columns=ED_OPTIONAL_TABLE_SCHEMAS["diagnosis"],
        harmonized_optional_columns=("seq_num", "icd_code", "icd_title", "icd_version"),
        integer_columns=("subject_id", "stay_id", "seq_num", "icd_version"),
    ),
    ("ed", "triage"): _spec(
        "ed",
        "triage",
        source_subdir="ed",
        required=False,
        required_columns=ED_OPTIONAL_TABLE_SCHEMAS["triage"],
        timestamp_columns=("charttime",),
        harmonized_optional_columns=(
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            "pain",
            "arrival_transport",
            "charttime",
        ),
        integer_columns=("subject_id", "stay_id"),
        float_columns=("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"),
    ),
    ("ed", "vitalsign"): _spec(
        "ed",
        "vitalsign",
        source_subdir="ed",
        required=False,
        required_columns=ED_OPTIONAL_TABLE_SCHEMAS["vitalsign"],
        timestamp_columns=("charttime",),
        harmonized_optional_columns=(
            "charttime",
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
            "rhythm",
            "pain",
        ),
        integer_columns=("subject_id", "stay_id"),
        float_columns=("temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain"),
    ),
    ("ed", "medrecon"): _spec(
        "ed",
        "medrecon",
        source_subdir="ed",
        required=False,
        required_columns=ED_OPTIONAL_TABLE_SCHEMAS["medrecon"],
        timestamp_columns=("charttime",),
        harmonized_optional_columns=(
            "name",
            "gsn",
            "ndc",
            "etc_rn",
            "etccode",
            "etcdescription",
            "charttime",
        ),
        integer_columns=("subject_id", "stay_id"),
    ),
    ("ed", "pyxis"): _spec(
        "ed",
        "pyxis",
        source_subdir="ed",
        required=False,
        required_columns=ED_OPTIONAL_TABLE_SCHEMAS["pyxis"],
        timestamp_columns=("charttime",),
        harmonized_optional_columns=("charttime", "med_rn", "name", "gsn_rn"),
        integer_columns=("subject_id", "stay_id"),
    ),
}


def iter_table_specs() -> tuple[TableStandardizationSpec, ...]:
    """Return all known table specs in stable order."""
    return tuple(STANDARDIZED_TABLE_SPECS.values())


def get_table_spec(dataset_name: str, table_name: str) -> TableStandardizationSpec:
    """Return one standardization spec by dataset and table name."""
    try:
        return STANDARDIZED_TABLE_SPECS[(dataset_name, table_name)]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(f"Unknown table spec for '{dataset_name}.{table_name}'") from exc
