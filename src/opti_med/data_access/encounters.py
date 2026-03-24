"""Canonical patient-first encounter indexing from persisted standardized tables."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_INDEX_COLUMNS,
    ENCOUNTER_INDEX_CONTRACT_VERSION,
    validate_encounter_index_artifact,
)
from opti_med.data_access.provenance import (
    dumps_json,
    loads_json_or_none,
    source_provenance_payload,
)
from opti_med.standardized import SOURCE_MANIFEST_METADATA_COLUMNS, StandardizedParquetRepository


@dataclass(frozen=True)
class EncounterIndexBuildResult:
    """Built encounter index and its output path."""

    dataframe: pd.DataFrame
    output_path: Path


class EncounterIndexBuilder:
    """Build a canonical encounter index keyed by subject_id from standardized tables."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = StandardizedParquetRepository(settings)

    def build(self) -> pd.DataFrame:
        """Combine standardized hospital admissions and ED stays into one patient-first index."""
        build_run_id = f"encounter-index-{uuid4().hex[:12]}"
        patients = self.repository.load_source_table(
            "clinical",
            "patients",
            columns=["subject_id", "gender", "anchor_age", *SOURCE_MANIFEST_METADATA_COLUMNS],
        )
        admissions = self.repository.load_source_table(
            "clinical",
            "admissions",
            columns=[
                "subject_id",
                "hadm_id",
                "admission_type",
                "admittime",
                "dischtime",
                *SOURCE_MANIFEST_METADATA_COLUMNS,
            ],
        )
        edstays = self.repository.load_source_table(
            "ed",
            "edstays",
            columns=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "intime",
                "outtime",
                "arrival_transport",
                "disposition",
                *SOURCE_MANIFEST_METADATA_COLUMNS,
            ],
        )

        patients_index = _build_patient_index(
            patients.dataframe,
            standardized_reference=patients.standardized_reference,
        )
        admissions_index = _build_admission_index(
            admissions.dataframe,
            standardized_reference=admissions.standardized_reference,
        )
        ed_index = _build_ed_index(
            edstays.dataframe,
            standardized_reference=edstays.standardized_reference,
        )

        ed_linked = ed_index.loc[ed_index["hadm_id"].notna()].merge(
            admissions_index,
            how="left",
            on=["subject_id", "hadm_id"],
            suffixes=("_ed", "_hospital"),
        )
        ed_linked = ed_linked.assign(
            linked_ed_stay_flag=1,
            linked_hospital_admission_flag=ed_linked["admittime"].notna().astype(int),
            encounter_source=lambda frame: frame["admittime"].notna().map(
                {True: "ed_to_inpatient", False: "ed_only"}
            ),
        )

        matched_admission_keys = (
            ed_linked.loc[ed_linked["linked_hospital_admission_flag"] == 1, ["subject_id", "hadm_id"]]
            .drop_duplicates()
        )
        admissions_only = admissions_index.merge(
            matched_admission_keys.assign(_matched=1),
            how="left",
            on=["subject_id", "hadm_id"],
        )
        admissions_only = admissions_only.loc[admissions_only["_matched"].isna()].drop(columns="_matched")
        admissions_only = admissions_only.assign(
            stay_id=pd.NA,
            intime=pd.NaT,
            outtime=pd.NaT,
            ed_length_of_stay_hours=pd.NA,
            ed_disposition=pd.NA,
            arrival_transport=pd.NA,
            ed_source_provenance_json=pd.NA,
            linked_ed_stay_flag=0,
            linked_hospital_admission_flag=1,
            encounter_source="hospital_only",
        )

        ed_only = ed_index.loc[ed_index["hadm_id"].isna()].copy()
        ed_only = ed_only.assign(
            admission_type=pd.NA,
            admittime=pd.NaT,
            dischtime=pd.NaT,
            hospital_length_of_stay_days=pd.NA,
            admission_source_provenance_json=pd.NA,
            linked_ed_stay_flag=1,
            linked_hospital_admission_flag=0,
            encounter_source="ed_only",
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
                category=FutureWarning,
            )
            encounter_index = pd.concat(
                [
                    ed_linked.reindex(columns=ENCOUNTER_STAGE_COLUMNS),
                    admissions_only.reindex(columns=ENCOUNTER_STAGE_COLUMNS),
                    ed_only.reindex(columns=ENCOUNTER_STAGE_COLUMNS),
                ],
                ignore_index=True,
                sort=False,
            )

        encounter_index = encounter_index.merge(
            patients_index,
            how="left",
            on="subject_id",
            validate="many_to_one",
        )
        encounter_index["encounter_id"] = encounter_index.apply(_encounter_id_from_row, axis=1)
        encounter_index["encounter_start"] = encounter_index[["intime", "admittime"]].bfill(axis=1).iloc[:, 0]
        encounter_index["encounter_end"] = encounter_index[["dischtime", "outtime"]].bfill(axis=1).iloc[:, 0]
        encounter_index["source_tables_json"] = encounter_index.apply(
            _encounter_source_tables_json,
            axis=1,
        )
        encounter_index["source_record_provenance_json"] = encounter_index.apply(
            _encounter_source_provenance_json,
            axis=1,
        )
        encounter_index["encounter_index_build_run_id"] = build_run_id
        encounter_index["encounter_index_contract_version"] = ENCOUNTER_INDEX_CONTRACT_VERSION
        encounter_index = encounter_index.loc[:, ENCOUNTER_INDEX_COLUMNS].copy()

        for column in [
            "admittime",
            "dischtime",
            "intime",
            "outtime",
            "encounter_start",
            "encounter_end",
        ]:
            encounter_index[column] = _format_timestamp_series(encounter_index[column])

        encounter_index = encounter_index.sort_values(
            ["subject_id", "encounter_start", "hadm_id", "stay_id"],
            na_position="last",
        ).reset_index(drop=True)
        validate_encounter_index_artifact(encounter_index)
        return encounter_index

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: Path | None = None,
    ) -> EncounterIndexBuildResult:
        """Persist the encounter index to Parquet."""
        validate_encounter_index_artifact(dataframe)
        target_path = output_path or self.settings.encounter_index_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(target_path, index=False)
        return EncounterIndexBuildResult(dataframe=dataframe, output_path=target_path)


ENCOUNTER_STAGE_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "encounter_source",
    "linked_ed_stay_flag",
    "linked_hospital_admission_flag",
    "admission_type",
    "admittime",
    "dischtime",
    "hospital_length_of_stay_days",
    "intime",
    "outtime",
    "ed_length_of_stay_hours",
    "ed_disposition",
    "arrival_transport",
    "admission_source_provenance_json",
    "ed_source_provenance_json",
]


def summarize_encounter_index(dataframe: pd.DataFrame) -> list[str]:
    """Return compact encounter index summaries for CLI or debugging output."""
    return [
        f"rows={len(dataframe):,}, columns={dataframe.shape[1]}",
        f"unique_subjects={dataframe['subject_id'].nunique():,}",
        f"linked_ed_to_inpatient_rows={int((dataframe['encounter_source'] == 'ed_to_inpatient').sum()):,}",
        f"hospital_only_rows={int((dataframe['encounter_source'] == 'hospital_only').sum()):,}",
        f"ed_only_rows={int((dataframe['encounter_source'] == 'ed_only').sum()):,}",
    ]


def _build_patient_index(
    patients: pd.DataFrame,
    *,
    standardized_reference: dict[str, object] | None,
) -> pd.DataFrame:
    patient_index = patients.loc[:, ["subject_id", "gender", "anchor_age", *SOURCE_MANIFEST_METADATA_COLUMNS]].copy()
    patient_index = patient_index.rename(columns={"gender": "sex", "anchor_age": "age_proxy"})
    patient_index["age_group"] = patient_index["age_proxy"].map(_age_group_from_anchor_age)
    patient_index["patient_source_provenance_json"] = patient_index.apply(
        lambda row: _source_provenance_json(
            row,
            role_name="patients",
            standardized_reference=standardized_reference,
        ),
        axis=1,
    )
    return patient_index.loc[
        :,
        ["subject_id", "sex", "age_proxy", "age_group", "patient_source_provenance_json"],
    ]


def _build_admission_index(
    admissions: pd.DataFrame,
    *,
    standardized_reference: dict[str, object] | None,
) -> pd.DataFrame:
    admission_index = admissions.loc[
        :,
        ["subject_id", "hadm_id", "admission_type", "admittime", "dischtime", *SOURCE_MANIFEST_METADATA_COLUMNS],
    ].copy()
    admission_index["admittime"] = pd.to_datetime(admission_index["admittime"], errors="coerce")
    admission_index["dischtime"] = pd.to_datetime(admission_index["dischtime"], errors="coerce")
    admission_index["hospital_length_of_stay_days"] = (
        admission_index["dischtime"] - admission_index["admittime"]
    ).dt.total_seconds() / 86400.0
    admission_index["hospital_length_of_stay_days"] = admission_index[
        "hospital_length_of_stay_days"
    ].round(3)
    admission_index["admission_source_provenance_json"] = admission_index.apply(
        lambda row: _source_provenance_json(
            row,
            role_name="admissions",
            standardized_reference=standardized_reference,
        ),
        axis=1,
    )
    return admission_index.loc[
        :,
        [
            "subject_id",
            "hadm_id",
            "admission_type",
            "admittime",
            "dischtime",
            "hospital_length_of_stay_days",
            "admission_source_provenance_json",
        ],
    ]


def _build_ed_index(
    edstays: pd.DataFrame,
    *,
    standardized_reference: dict[str, object] | None,
) -> pd.DataFrame:
    ed_index = edstays.loc[
        :,
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "arrival_transport",
            "disposition",
            *SOURCE_MANIFEST_METADATA_COLUMNS,
        ],
    ].copy()
    ed_index["intime"] = pd.to_datetime(ed_index["intime"], errors="coerce")
    ed_index["outtime"] = pd.to_datetime(ed_index["outtime"], errors="coerce")
    ed_index["ed_length_of_stay_hours"] = (
        ed_index["outtime"] - ed_index["intime"]
    ).dt.total_seconds() / 3600.0
    ed_index["ed_length_of_stay_hours"] = ed_index["ed_length_of_stay_hours"].round(3)
    ed_index = ed_index.rename(columns={"disposition": "ed_disposition"})
    ed_index["ed_source_provenance_json"] = ed_index.apply(
        lambda row: _source_provenance_json(
            row,
            role_name="edstays",
            standardized_reference=standardized_reference,
        ),
        axis=1,
    )
    return ed_index.loc[
        :,
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "ed_length_of_stay_hours",
            "ed_disposition",
            "arrival_transport",
            "ed_source_provenance_json",
        ],
    ]


def _age_group_from_anchor_age(anchor_age: int | float) -> str:
    """Reuse the current age banding until the patient-first pipeline changes it."""
    if pd.isna(anchor_age):
        return "unknown"
    age = float(anchor_age)
    if age >= 85:
        return "85+"
    if age >= 75:
        return "75-84"
    if age >= 65:
        return "65-74"
    return "<65"


def _encounter_id_from_row(row: pd.Series) -> str:
    """Build a stable encounter identifier even when only one source key is present."""
    if pd.notna(row.get("hadm_id")):
        return f"hadm:{int(row['hadm_id'])}"
    if pd.notna(row.get("stay_id")):
        return f"stay:{int(row['stay_id'])}"
    return f"subject:{int(row['subject_id'])}"


def _encounter_source_tables_json(row: pd.Series) -> str:
    source_tables: list[str] = ["patients"]
    if pd.notna(row.get("admission_source_provenance_json")):
        source_tables.append("admissions")
    if pd.notna(row.get("ed_source_provenance_json")):
        source_tables.append("edstays")
    return dumps_json(source_tables)


def _encounter_source_provenance_json(row: pd.Series) -> str:
    payload = {
        "patients": loads_json_or_none(row.get("patient_source_provenance_json")),
        "admissions": loads_json_or_none(row.get("admission_source_provenance_json")),
        "edstays": loads_json_or_none(row.get("ed_source_provenance_json")),
    }
    return dumps_json(payload)


def _source_provenance_json(
    row: pd.Series,
    *,
    role_name: str,
    standardized_reference: dict[str, object] | None,
) -> str:
    return dumps_json(
        source_provenance_payload(
            row,
            role_name=role_name,
            standardized_reference=standardized_reference,
            source_metadata_columns=SOURCE_MANIFEST_METADATA_COLUMNS,
        )
    )


def _format_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
