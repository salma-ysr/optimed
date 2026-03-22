"""Minimal feature extraction pipeline for the OPTI-MED MVP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opti_med.cohort.builder import OlderAdultMedicationCohortBuilder
from opti_med.config import Settings
from opti_med.data_access.loaders import MimicDualDemoLoader
from opti_med.features.context import build_patient_context_features
from opti_med.features.mappings.diagnoses import DIAGNOSIS_ICD_PREFIXES
from opti_med.features.mappings.labs import serum_creatinine_itemids
from opti_med.features.mappings.medications import HIGH_RISK_MEDICATION_KEYWORDS


@dataclass(frozen=True)
class FeatureBuildResult:
    """Processed cohort and its target output path."""

    dataframe: pd.DataFrame
    output_path: Path
    dropped_duplicate_rows: int = 0


class MinimalFeatureBuilder:
    """Enrich the admission-medication cohort with MVP diagnosis, lab, and medication features."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cohort_builder = OlderAdultMedicationCohortBuilder(settings)
        self.dual_loader = MimicDualDemoLoader(settings)

    def build(self) -> pd.DataFrame:
        """Build the processed cohort dataframe."""
        tables = self.cohort_builder.loader.load_all()
        cohort = self.cohort_builder.build_from_tables(tables)
        omr_loaded = self.cohort_builder.loader.load_optional_table("omr")
        edstays = self.dual_loader.load_ed_table("edstays").dataframe
        ed_diagnosis_loaded = self.dual_loader.load_optional_ed_table("diagnosis")
        triage_loaded = self.dual_loader.load_optional_ed_table("triage")
        vitalsign_loaded = self.dual_loader.load_optional_ed_table("vitalsign")

        patient_context_features = build_patient_context_features(
            admissions=tables["admissions"],
            patients=tables["patients"],
            diagnoses_icd=tables["diagnoses_icd"],
            labevents=tables["labevents"],
            omr=omr_loaded.dataframe if omr_loaded else None,
            edstays=edstays,
            ed_diagnosis=ed_diagnosis_loaded.dataframe if ed_diagnosis_loaded else None,
            triage=triage_loaded.dataframe if triage_loaded else None,
            vitalsign=vitalsign_loaded.dataframe if vitalsign_loaded else None,
            creatinine_threshold=self.settings.renal_risk_creatinine_threshold,
            serum_creatinine_ids=self.settings.serum_creatinine_itemids,
        )
        medication_burden_features = build_medication_burden_features(cohort, self.settings)

        enriched = cohort.merge(
            patient_context_features,
            on=["subject_id", "hadm_id"],
            how="left",
            validate="many_to_one",
        )
        enriched = enriched.merge(
            medication_burden_features, on="hadm_id", how="left", validate="many_to_one"
        )
        enriched = add_high_risk_medication_flags(enriched)
        enriched = finalize_processed_dataframe(enriched)
        return enriched

    def save(
        self, dataframe: pd.DataFrame, output_path: Path | None = None
    ) -> FeatureBuildResult:
        """Persist the processed cohort dataframe to a CSV file."""
        target_path = output_path or self.settings.processed_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(target_path, index=False)
        return FeatureBuildResult(
            dataframe=dataframe,
            output_path=target_path,
            dropped_duplicate_rows=self.cohort_builder.last_dropped_duplicate_rows,
        )


def build_diagnosis_flags(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate diagnosis flags to the admission level using curated ICD prefixes."""
    diagnoses_frame = diagnoses.loc[:, ["hadm_id", "icd_code", "icd_version"]].copy()
    diagnoses_frame["icd_code"] = diagnoses_frame["icd_code"].astype(str).str.upper()
    diagnoses_frame["icd_version"] = pd.to_numeric(
        diagnoses_frame["icd_version"], errors="coerce"
    ).astype("Int64")
    diagnoses_frame = diagnoses_frame.dropna(subset=["hadm_id", "icd_code", "icd_version"])

    flags = diagnoses_frame.loc[:, ["hadm_id"]].drop_duplicates().reset_index(drop=True)
    for flag_name, version_mapping in DIAGNOSIS_ICD_PREFIXES.items():
        matches = pd.Series(False, index=diagnoses_frame.index)
        for icd_version, prefixes in version_mapping.items():
            version_mask = diagnoses_frame["icd_version"] == icd_version
            prefix_mask = diagnoses_frame["icd_code"].str.startswith(prefixes)
            matches = matches | (version_mask & prefix_mask)

        hadm_ids = diagnoses_frame.loc[matches, "hadm_id"].drop_duplicates()
        flags[flag_name] = flags["hadm_id"].isin(hadm_ids).astype(int)
    return flags


def build_creatinine_features(labevents: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Aggregate serum creatinine values to the admission level."""
    creatinine_itemids = serum_creatinine_itemids(settings)
    labs = labevents.loc[:, ["hadm_id", "itemid", "charttime", "valuenum"]].copy()
    labs = labs.dropna(subset=["hadm_id", "itemid", "valuenum"])
    labs["itemid"] = pd.to_numeric(labs["itemid"], errors="coerce")
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    labs = labs.dropna(subset=["itemid", "valuenum"])
    labs = labs.loc[labs["itemid"].isin(creatinine_itemids)].copy()

    if labs.empty:
        return pd.DataFrame(
            columns=[
                "hadm_id",
                "creatinine_first",
                "creatinine_max",
                "creatinine_mean",
                "renal_risk_flag",
            ]
        )

    labs = labs.sort_values(["hadm_id", "charttime", "itemid"])
    grouped = labs.groupby("hadm_id", as_index=False)
    features = grouped.agg(
        creatinine_first=("valuenum", "first"),
        creatinine_max=("valuenum", "max"),
        creatinine_mean=("valuenum", "mean"),
    )
    features["creatinine_mean"] = features["creatinine_mean"].round(3)
    features["renal_risk_flag"] = (
        features["creatinine_max"] >= settings.renal_risk_creatinine_threshold
    ).astype(int)
    return features


def build_medication_burden_features(cohort: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Aggregate medication burden from canonical medication episodes.

    ``total_medication_count`` counts collapsed episode rows per admission.
    ``peak_concurrent_medication_count`` counts how many canonical episodes overlap in time,
    which is the value used to decide the polypharmacy flag.
    """
    if cohort.empty:
        return pd.DataFrame(
            columns=[
                "hadm_id",
                "total_medication_count",
                "peak_concurrent_medication_count",
                "polypharmacy_flag",
            ]
        )

    burden = (
        cohort.groupby("hadm_id", as_index=False)
        .agg(total_medication_count=("medication_episode_id", "nunique"))
    )
    grouped_intervals = cohort.loc[:, ["hadm_id", "starttime", "stoptime", "dischtime"]].groupby(
        "hadm_id",
        sort=False,
    )
    try:
        concurrent_counts = (
            grouped_intervals.apply(_peak_concurrent_medication_count, include_groups=False)
            .rename("peak_concurrent_medication_count")
            .reset_index()
        )
    except TypeError:
        concurrent_counts = (
            grouped_intervals.apply(_peak_concurrent_medication_count)
            .rename("peak_concurrent_medication_count")
            .reset_index()
        )
    burden = burden.merge(concurrent_counts, on="hadm_id", how="left", validate="one_to_one")
    burden["polypharmacy_flag"] = (
        burden["peak_concurrent_medication_count"] >= settings.polypharmacy_threshold
    ).astype(int)
    return burden


def _peak_concurrent_medication_count(group: pd.DataFrame) -> int:
    events: list[tuple[pd.Timestamp, int, int]] = []
    for record in group.to_dict(orient="records"):
        start = pd.to_datetime(record.get("starttime"), errors="coerce")
        stop = pd.to_datetime(record.get("stoptime"), errors="coerce")
        discharge = pd.to_datetime(record.get("dischtime"), errors="coerce")
        if pd.isna(start):
            continue
        if pd.isna(stop):
            stop = discharge if pd.notna(discharge) else start
        events.append((start, 0, 1))
        events.append((stop, 1, -1))

    concurrent = 0
    peak = 0
    for _, _, delta in sorted(events):
        concurrent += delta
        peak = max(peak, concurrent)
    return peak


def add_high_risk_medication_flags(cohort: pd.DataFrame) -> pd.DataFrame:
    """Annotate each medication exposure with curated high-risk class flags."""
    enriched = cohort.copy()
    drug_series = enriched["drug"].fillna("").astype(str).str.lower()
    for flag_name, keywords in HIGH_RISK_MEDICATION_KEYWORDS.items():
        enriched[flag_name] = drug_series.str.contains("|".join(keywords), regex=True).astype(int)
    return enriched


def finalize_processed_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values and order processed columns for output."""
    diagnosis_and_risk_flags = [
        "ckd_flag",
        "dementia_flag",
        "delirium_flag",
        "heart_failure_flag",
        "diabetes_flag",
        "diagnosis_risk_renal_flag",
        "diagnosis_risk_cognitive_flag",
        "diagnosis_risk_cardiac_flag",
        "diagnosis_risk_metabolic_flag",
        "ed_ckd_flag",
        "ed_dementia_flag",
        "ed_delirium_flag",
        "ed_heart_failure_flag",
        "ed_diabetes_flag",
        "ed_diagnosis_risk_renal_flag",
        "ed_diagnosis_risk_cognitive_flag",
        "ed_diagnosis_risk_cardiac_flag",
        "ed_diagnosis_risk_metabolic_flag",
        "renal_risk_flag",
        "polypharmacy_flag",
        "benzodiazepine_flag",
        "opioid_flag",
        "anticholinergic_flag",
        "ppi_flag",
        "antipsychotic_flag",
    ]
    numeric_columns = [
        "creatinine_first",
        "creatinine_last",
        "creatinine_max",
        "creatinine_mean",
        "creatinine_delta",
        "potassium_first",
        "potassium_last",
        "potassium_min",
        "potassium_max",
        "potassium_mean",
        "potassium_delta",
        "sodium_first",
        "sodium_last",
        "sodium_min",
        "sodium_max",
        "sodium_mean",
        "sodium_delta",
        "total_medication_count",
        "peak_concurrent_medication_count",
        "weight_kg",
        "bmi",
        "egfr_ml_min_1_73m2",
        "cockcroft_gault_ml_min",
        "sbp_min",
        "sbp_max",
        "sbp_mean",
        "dbp_min",
        "dbp_max",
        "dbp_mean",
        "heart_rate_min",
        "heart_rate_max",
        "heart_rate_mean",
        "pain_min",
        "pain_max",
        "pain_mean",
    ]
    base_ordered_columns = [
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
        "drug_normalized",
        "starttime",
        "stoptime",
        "medication_episode_id",
        "prescription_segment_count",
        "prescription_segments_json",
        "total_medication_count",
        "peak_concurrent_medication_count",
        "polypharmacy_flag",
        "benzodiazepine_flag",
        "opioid_flag",
        "anticholinergic_flag",
        "ppi_flag",
        "antipsychotic_flag",
        "ckd_flag",
        "dementia_flag",
        "delirium_flag",
        "heart_failure_flag",
        "diabetes_flag",
        "diagnosis_risk_renal_flag",
        "diagnosis_risk_cognitive_flag",
        "diagnosis_risk_cardiac_flag",
        "diagnosis_risk_metabolic_flag",
        "ed_ckd_flag",
        "ed_dementia_flag",
        "ed_delirium_flag",
        "ed_heart_failure_flag",
        "ed_diabetes_flag",
        "ed_diagnosis_risk_renal_flag",
        "ed_diagnosis_risk_cognitive_flag",
        "ed_diagnosis_risk_cardiac_flag",
        "ed_diagnosis_risk_metabolic_flag",
        "diagnosis_context_provenance",
        "ed_diagnosis_context_provenance",
        "age_provenance",
        "sex_provenance",
        "weight_kg",
        "weight_kg_provenance",
        "weight_kg_unavailable_reason",
        "bmi",
        "bmi_provenance",
        "bmi_unavailable_reason",
        "creatinine_first",
        "creatinine_last",
        "creatinine_max",
        "creatinine_mean",
        "creatinine_delta",
        "creatinine_trend_direction",
        "creatinine_provenance",
        "renal_risk_flag",
        "potassium_first",
        "potassium_last",
        "potassium_min",
        "potassium_max",
        "potassium_mean",
        "potassium_delta",
        "potassium_trend_direction",
        "potassium_provenance",
        "sodium_first",
        "sodium_last",
        "sodium_min",
        "sodium_max",
        "sodium_mean",
        "sodium_delta",
        "sodium_trend_direction",
        "sodium_provenance",
        "egfr_ml_min_1_73m2",
        "egfr_provenance",
        "egfr_unavailable_reason",
        "cockcroft_gault_ml_min",
        "cockcroft_gault_provenance",
        "cockcroft_gault_unavailable_reason",
        "sbp_min",
        "sbp_max",
        "sbp_mean",
        "dbp_min",
        "dbp_max",
        "dbp_mean",
        "blood_pressure_provenance",
        "heart_rate_min",
        "heart_rate_max",
        "heart_rate_mean",
        "heart_rate_provenance",
        "pain_min",
        "pain_max",
        "pain_mean",
        "pain_provenance",
        "ed_triage_acuity",
        "ed_chiefcomplaint",
        "ed_triage_provenance",
    ]

    output = dataframe.copy()
    for column in diagnosis_and_risk_flags:
        output[column] = output[column].fillna(0).astype(int)
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    ordered_columns = [column for column in base_ordered_columns if column in output.columns]
    remaining_columns = [column for column in output.columns if column not in ordered_columns]
    return output.loc[:, [*ordered_columns, *remaining_columns]]


def summarize_processed_cohort(dataframe: pd.DataFrame) -> list[str]:
    """Return compact summaries for the processed cohort CLI."""
    admission_flags = (
        dataframe.loc[:, ["hadm_id", "polypharmacy_flag", "renal_risk_flag"]]
        .drop_duplicates(subset=["hadm_id"])
        .reset_index(drop=True)
    )
    polypharmacy_admissions = int(admission_flags["polypharmacy_flag"].sum())
    renal_risk_admissions = int(admission_flags["renal_risk_flag"].sum())
    return [
        f"rows={len(dataframe):,}, columns={dataframe.shape[1]}",
        f"unique_admissions={dataframe['hadm_id'].nunique():,}",
        f"polypharmacy_admissions={polypharmacy_admissions:,}",
        f"renal_risk_admissions={renal_risk_admissions:,}",
    ]
