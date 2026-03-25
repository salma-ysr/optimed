"""Persisted older-adult encounter eligibility for downstream semi-joins."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_INDEX_CONTRACT_VERSION,
    OLDER_ADULT_ELIGIBILITY_COLUMNS,
    OLDER_ADULT_ELIGIBILITY_CONTRACT_VERSION,
    validate_encounter_index_artifact,
    validate_older_adult_eligibility_artifact,
)
from opti_med.data_access.provenance import dumps_json, loads_json_or_none
from opti_med.standardized import StandardizedParquetRepository


OLDER_ADULT_ELIGIBILITY_MIN_AGE = 65
OLDER_ADULT_ELIGIBILITY_RULE_NAME = "age_proxy_gte_65"


@dataclass(frozen=True, slots=True)
class OlderAdultEncounterEligibilityBuildResult:
    """Built 65+ encounter eligibility artifact and its output path."""

    dataframe: pd.DataFrame
    output_path: Path


class OlderAdultEncounterEligibilityBuilder:
    """Build a persisted encounter-level 65+ eligibility branch from encounter_index."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = StandardizedParquetRepository(settings)

    def build(self, encounter_index: pd.DataFrame | None = None) -> pd.DataFrame:
        """Build the persisted 65+ encounter eligibility artifact."""
        encounter_index = (
            encounter_index
            if encounter_index is not None
            else self.repository.load_analytical_artifact("encounter_index")
        )
        validate_encounter_index_artifact(encounter_index)

        build_run_id = f"older-adult-eligibility-{uuid4().hex[:12]}"
        age_proxy_numeric = pd.to_numeric(encounter_index["age_proxy"], errors="coerce")
        eligible = encounter_index.loc[
            age_proxy_numeric >= OLDER_ADULT_ELIGIBILITY_MIN_AGE,
            [
                "subject_id",
                "hadm_id",
                "stay_id",
                "encounter_id",
                "age_proxy",
                "encounter_source",
                "encounter_index_build_run_id",
                "encounter_index_contract_version",
            ],
        ].copy()
        eligible["age_proxy"] = pd.to_numeric(eligible["age_proxy"], errors="coerce")
        eligible["age_group"] = eligible["age_proxy"].map(age_group_from_age_proxy)
        eligible["eligibility_flag"] = 1
        eligible["eligibility_rule_name"] = OLDER_ADULT_ELIGIBILITY_RULE_NAME
        eligible["eligibility_criteria_json"] = dumps_json(
            {
                "age_proxy_source_column": "patients.anchor_age",
                "eligibility_rule_name": OLDER_ADULT_ELIGIBILITY_RULE_NAME,
                "encounter_source_artifact": "encounter_index",
                "encounter_source_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
                "minimum_age_proxy_years": OLDER_ADULT_ELIGIBILITY_MIN_AGE,
            }
        )
        eligible["older_adult_eligibility_build_run_id"] = build_run_id
        eligible["older_adult_eligibility_contract_version"] = (
            OLDER_ADULT_ELIGIBILITY_CONTRACT_VERSION
        )
        eligible = eligible.loc[:, OLDER_ADULT_ELIGIBILITY_COLUMNS].copy()
        eligible = eligible.sort_values(
            ["subject_id", "encounter_id", "hadm_id", "stay_id"],
            na_position="last",
        ).reset_index(drop=True)
        validate_older_adult_eligibility_artifact(eligible)
        return eligible

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: Path | None = None,
    ) -> OlderAdultEncounterEligibilityBuildResult:
        """Persist the 65+ encounter eligibility artifact to Parquet."""
        validate_older_adult_eligibility_artifact(dataframe)
        target_path = output_path or self.settings.older_adult_eligibility_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(target_path, index=False)
        return OlderAdultEncounterEligibilityBuildResult(
            dataframe=dataframe,
            output_path=target_path,
        )

    def load(self, output_path: Path | None = None) -> pd.DataFrame:
        """Load the persisted 65+ encounter eligibility artifact from Parquet."""
        target_path = output_path or self.settings.older_adult_eligibility_output_path
        if not target_path.exists():
            raise FileNotFoundError(
                f"Expected older-adult eligibility artifact at '{target_path}', but it does not exist."
            )
        dataframe = pd.read_parquet(target_path)
        validate_older_adult_eligibility_artifact(dataframe)
        return dataframe


def limit_eligibility_to_subject_count(
    dataframe: pd.DataFrame,
    max_subjects: int | None,
) -> pd.DataFrame:
    """Deterministically subset eligibility to the first N unique subjects."""
    validate_older_adult_eligibility_artifact(dataframe)
    if max_subjects is None:
        return dataframe.copy()
    if max_subjects <= 0:
        raise ValueError("max_subjects must be a positive integer.")

    ordered_subject_ids = (
        pd.to_numeric(dataframe["subject_id"], errors="coerce")
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    selected_subject_ids = set(ordered_subject_ids[:max_subjects])
    limited = dataframe.loc[
        pd.to_numeric(dataframe["subject_id"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .isin(selected_subject_ids)
    ].copy()
    limited["eligibility_criteria_json"] = limited["eligibility_criteria_json"].apply(
        lambda value: _eligibility_criteria_with_subject_limit(
            value,
            max_subjects=max_subjects,
        )
    )
    limited = limited.sort_values(
        ["subject_id", "encounter_id", "hadm_id", "stay_id"],
        na_position="last",
    ).reset_index(drop=True)
    validate_older_adult_eligibility_artifact(limited)
    return limited


def age_group_from_age_proxy(age_proxy: int | float) -> str:
    """Convert anchor-age proxy into the scoped older-adult age bands."""
    if pd.isna(age_proxy):
        return "unknown"
    age_value = float(age_proxy)
    if age_value >= 85:
        return "85+"
    if age_value >= 75:
        return "75-84"
    if age_value >= 65:
        return "65-74"
    return "<65"


def _eligibility_criteria_with_subject_limit(
    value: object,
    *,
    max_subjects: int,
) -> str:
    payload = loads_json_or_none(value)
    criteria = payload if isinstance(payload, dict) else {}
    criteria["subject_limit"] = {
        "max_subjects": int(max_subjects),
        "selection_rule": "lowest_subject_id_ascending",
    }
    return dumps_json(criteria)


def semi_join_to_eligible_encounters(
    dataframe: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    encounter_id_column: str = "encounter_id",
) -> pd.DataFrame:
    """Filter one downstream dataframe to the persisted 65+ encounter set via semi-join."""
    if encounter_id_column not in dataframe:
        raise KeyError(
            f"Column '{encounter_id_column}' is required for a semi-join against eligible encounters."
        )
    validate_older_adult_eligibility_artifact(eligibility)
    eligible_encounter_ids = set(eligibility["encounter_id"].dropna().astype(str).tolist())
    join_values = dataframe[encounter_id_column].astype(str)
    return dataframe.loc[join_values.isin(eligible_encounter_ids)].copy()


def summarize_older_adult_eligibility(dataframe: pd.DataFrame) -> list[str]:
    """Return compact QA summaries for the persisted 65+ encounter eligibility branch."""
    validate_older_adult_eligibility_artifact(dataframe)
    return [
        f"rows={len(dataframe):,}, columns={dataframe.shape[1]}",
        f"unique_subjects={dataframe['subject_id'].nunique():,}" if not dataframe.empty else "unique_subjects=0",
        f"unique_encounters={dataframe['encounter_id'].nunique():,}" if not dataframe.empty else "unique_encounters=0",
        f"age_group_distribution={_distribution_summary(dataframe, 'age_group', ['65-74', '75-84', '85+'])}",
        f"encounter_source_distribution={_distribution_summary(dataframe, 'encounter_source', ['hospital_only', 'ed_to_inpatient', 'ed_only'])}",
    ]


def build_older_adult_eligibility_qc_report(dataframe: pd.DataFrame) -> str:
    """Render one Markdown QA report for the persisted 65+ encounter eligibility artifact."""
    validate_older_adult_eligibility_artifact(dataframe)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Older Adult Eligibility Branch QA",
        "",
        f"Generated at: {generated_at}",
        "",
        "## Row Counts",
        "",
        f"- eligible_encounters_65plus rows: {len(dataframe):,}",
        f"- unique subjects: {dataframe['subject_id'].nunique():,}" if not dataframe.empty else "- unique subjects: 0",
        f"- unique encounters: {dataframe['encounter_id'].nunique():,}" if not dataframe.empty else "- unique encounters: 0",
        "",
        "## Age Group Distribution",
        "",
    ]
    lines.extend(_value_count_lines(dataframe, "age_group", ["65-74", "75-84", "85+"]))
    lines.extend(
        [
            "",
            "## Encounter Source Distribution",
            "",
        ]
    )
    lines.extend(
        _value_count_lines(
            dataframe,
            "encounter_source",
            ["hospital_only", "ed_to_inpatient", "ed_only"],
        )
    )
    lines.extend(
        [
            "",
            "## Build Metadata",
            "",
            f"- encounter_index build run ids: {_distinct_values(dataframe, 'encounter_index_build_run_id')}",
            (
                "- older_adult_eligibility build run ids: "
                f"{_distinct_values(dataframe, 'older_adult_eligibility_build_run_id')}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_older_adult_eligibility_qc_report(
    *,
    dataframe: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write one Markdown QA report for the persisted 65+ encounter eligibility artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_older_adult_eligibility_qc_report(dataframe),
        encoding="utf-8",
    )
    return output_path


def _distribution_summary(
    dataframe: pd.DataFrame,
    column_name: str,
    ordered_values: list[str],
) -> str:
    counts = dataframe[column_name].fillna("null").astype(str).value_counts(dropna=False)
    return ", ".join(
        f"{value}:{int(counts.get(value, 0)):,}"
        for value in ordered_values
    )


def _value_count_lines(
    dataframe: pd.DataFrame,
    column_name: str,
    ordered_values: list[str],
) -> list[str]:
    counts = dataframe[column_name].fillna("null").astype(str).value_counts(dropna=False)
    return [f"- {column_name}={value}: {int(counts.get(value, 0)):,}" for value in ordered_values]


def _distinct_values(dataframe: pd.DataFrame, column_name: str) -> str:
    if dataframe.empty or column_name not in dataframe:
        return "[]"
    values = sorted({str(value) for value in dataframe[column_name].dropna().astype(str).tolist()})
    return "[" + ", ".join(values) + "]"
