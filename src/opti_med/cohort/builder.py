"""Minimal cohort builder for patient-medication-admission records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opti_med.cohort.exceptions import CohortBuildError
from opti_med.config import Settings
from opti_med.data_access.loaders import MimicCoreLoader


COHORT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "sex",
    "age_proxy",
    "age_group",
    "admission_type",
    "admittime",
    "dischtime",
    "length_of_stay_days",
    "drug",
    "starttime",
    "stoptime",
]


@dataclass(frozen=True)
class CohortBuildResult:
    """Built cohort and its target output path."""

    dataframe: pd.DataFrame
    output_path: Path
    dropped_duplicate_rows: int = 0


class OlderAdultMedicationCohortBuilder:
    """Build a minimal patient-medication-admission cohort."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = MimicCoreLoader(settings)
        self.last_dropped_duplicate_rows = 0

    def build(self) -> pd.DataFrame:
        """Load source tables and construct the cohort dataframe."""
        tables = self.loader.load_all()
        patients = tables["patients"].copy()
        admissions = tables["admissions"].copy()
        prescriptions = tables["prescriptions"].copy()

        self._validate_source_keys(patients, admissions, prescriptions)

        admissions_enriched = self._build_admission_frame(patients, admissions)
        cohort = self._build_medication_exposure_frame(admissions_enriched, prescriptions)
        cohort = self._format_output_columns(cohort)
        self.last_dropped_duplicate_rows = self._count_duplicate_output_rows(cohort)
        cohort = self._deduplicate_output_rows(cohort)
        self._validate_output_rows(cohort)
        return cohort

    def save(self, cohort: pd.DataFrame, output_path: Path | None = None) -> CohortBuildResult:
        """Persist the cohort dataframe to an interim CSV file."""
        target_path = output_path or self.settings.cohort_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        cohort.to_csv(target_path, index=False)
        return CohortBuildResult(
            dataframe=cohort,
            output_path=target_path,
            dropped_duplicate_rows=self.last_dropped_duplicate_rows,
        )

    def _build_admission_frame(
        self, patients: pd.DataFrame, admissions: pd.DataFrame
    ) -> pd.DataFrame:
        patients = patients.rename(columns={"gender": "sex", "anchor_age": "age_proxy"})
        older_adults = patients.loc[
            patients["age_proxy"] >= self.settings.older_adult_age_threshold
        ].copy()
        older_adults["age_group"] = older_adults["age_proxy"].map(_age_group_from_anchor_age)

        admissions_frame = admissions.loc[
            :, ["subject_id", "hadm_id", "admission_type", "admittime", "dischtime"]
        ].copy()
        admissions_frame["admittime"] = pd.to_datetime(admissions_frame["admittime"], errors="coerce")
        admissions_frame["dischtime"] = pd.to_datetime(admissions_frame["dischtime"], errors="coerce")

        joined = admissions_frame.merge(
            older_adults.loc[:, ["subject_id", "sex", "age_proxy", "age_group"]],
            how="inner",
            on="subject_id",
            validate="many_to_one",
        )
        joined["length_of_stay_days"] = (
            joined["dischtime"] - joined["admittime"]
        ).dt.total_seconds() / 86400.0

        if joined["length_of_stay_days"].isna().any():
            missing_count = int(joined["length_of_stay_days"].isna().sum())
            raise CohortBuildError(
                f"Found {missing_count} admissions with invalid admittime/dischtime values."
            )

        if (joined["length_of_stay_days"] < 0).any():
            negative_count = int((joined["length_of_stay_days"] < 0).sum())
            raise CohortBuildError(
                f"Found {negative_count} admissions with negative length of stay."
            )

        return joined

    def _build_medication_exposure_frame(
        self, admissions_enriched: pd.DataFrame, prescriptions: pd.DataFrame
    ) -> pd.DataFrame:
        prescriptions_frame = prescriptions.loc[
            :, ["subject_id", "hadm_id", "drug", "starttime", "stoptime"]
        ].copy()
        prescriptions_frame["starttime"] = pd.to_datetime(
            prescriptions_frame["starttime"], errors="coerce"
        )
        prescriptions_frame["stoptime"] = pd.to_datetime(
            prescriptions_frame["stoptime"], errors="coerce"
        )

        if prescriptions_frame[["subject_id", "hadm_id", "drug"]].isna().any().any():
            raise CohortBuildError(
                "Prescriptions contain null subject_id, hadm_id, or drug values required for cohort output."
            )

        cohort = prescriptions_frame.merge(
            admissions_enriched,
            how="inner",
            on=["subject_id", "hadm_id"],
            validate="many_to_one",
        )
        return cohort

    @staticmethod
    def _format_output_columns(cohort: pd.DataFrame) -> pd.DataFrame:
        formatted = cohort.loc[:, COHORT_COLUMNS].copy()
        for column in ["admittime", "dischtime", "starttime", "stoptime"]:
            formatted[column] = formatted[column].dt.strftime("%Y-%m-%d %H:%M:%S")
        formatted["length_of_stay_days"] = formatted["length_of_stay_days"].round(3)
        return formatted

    @staticmethod
    def _validate_source_keys(
        patients: pd.DataFrame, admissions: pd.DataFrame, prescriptions: pd.DataFrame
    ) -> None:
        if patients["subject_id"].isna().any():
            raise CohortBuildError("Patients table contains null subject_id values.")

        if admissions[["subject_id", "hadm_id"]].isna().any().any():
            raise CohortBuildError("Admissions table contains null subject_id or hadm_id values.")

        if admissions["hadm_id"].duplicated().any():
            duplicate_count = int(admissions["hadm_id"].duplicated().sum())
            raise CohortBuildError(
                f"Admissions table contains {duplicate_count} duplicate hadm_id values."
            )

        if patients["subject_id"].duplicated().any():
            duplicate_count = int(patients["subject_id"].duplicated().sum())
            raise CohortBuildError(
                f"Patients table contains {duplicate_count} duplicate subject_id values."
            )

        if prescriptions[["subject_id", "hadm_id"]].isna().any().any():
            raise CohortBuildError(
                "Prescriptions table contains null subject_id or hadm_id values."
            )

    @staticmethod
    def _validate_output_rows(cohort: pd.DataFrame) -> None:
        if cohort.empty:
            raise CohortBuildError(
                "Cohort build produced no rows after applying the older-adult restriction."
            )

    @classmethod
    def _deduplicate_output_rows(cls, cohort: pd.DataFrame) -> pd.DataFrame:
        """Drop repeated medication exposure rows created by source-table duplicates."""
        return cohort.drop_duplicates(subset=cls._output_deduplication_columns()).reset_index(
            drop=True
        )

    @classmethod
    def _count_duplicate_output_rows(cls, cohort: pd.DataFrame) -> int:
        """Count duplicate output rows before deduplication."""
        return int(cohort.duplicated(subset=cls._output_deduplication_columns()).sum())

    @staticmethod
    def _output_deduplication_columns() -> list[str]:
        return ["subject_id", "hadm_id", "drug", "starttime", "stoptime"]


def summarize_cohort(cohort: pd.DataFrame) -> list[str]:
    """Return compact cohort summary lines for CLI output."""
    unique_subjects = cohort["subject_id"].nunique()
    unique_admissions = cohort["hadm_id"].nunique()
    return [
        f"rows={len(cohort):,}, columns={cohort.shape[1]}",
        f"unique_subjects={unique_subjects:,}",
        f"unique_admissions={unique_admissions:,}",
    ]


def _age_group_from_anchor_age(anchor_age: int | float) -> str:
    """Convert MIMIC-IV anchor age into a simple older-adult age band."""
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
