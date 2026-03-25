"""Point-in-time-safe Feature Store v1 builders for the 65+ first-scope subset."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pandas as pd

from opti_med.data_access.artifact_schemas import (
    validate_encounter_medication_first_scope_artifact,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.provenance import dumps_json, loads_json_or_none
from opti_med.features.context import (
    BMI_PROVENANCE_DERIVED,
    BMI_PROVENANCE_OBSERVED,
    COCKCROFT_GAULT_PROVENANCE_DERIVED,
    ED_DIAGNOSIS_CONTEXT_PROVENANCE_OBSERVED,
    EGFR_PROVENANCE_DERIVED,
    PROVENANCE_UNAVAILABLE,
    WEIGHT_PROVENANCE_OBSERVED,
    build_diagnosis_flags_by_key,
    categorize_delta_trend,
    compute_cockcroft_gault,
    compute_egfr_ckd_epi_2021,
    derive_cockcroft_gault_unavailable_reason,
    derive_egfr_unavailable_reason,
)
from opti_med.features.contracts import (
    FEATURE_STORE_GRAIN_DESCRIPTION,
    FEATURE_STORE_PRIMARY_KEY,
)
from opti_med.features.mappings.diagnoses import (
    DIAGNOSIS_ICD_PREFIXES,
    DIAGNOSIS_RISK_CATEGORY_GROUPS,
)
from opti_med.features.mappings.labs import potassium_itemids, sodium_itemids


FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS = [
    "subject_id",
    "encounter_id",
    "hadm_id",
    "stay_id",
    "review_timestamp",
]
FIRST_SCOPE_FEATURE_KEY_COLUMNS = list(FEATURE_STORE_PRIMARY_KEY)

FIRST_SCOPE_PATIENT_CONTEXT_CONTRACT_VERSION = (
    "first_scope_patient_context_features_65plus.v1"
)
FIRST_SCOPE_TEMPORAL_CONTRACT_VERSION = "first_scope_temporal_features_65plus.v1"
FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION = (
    "encounter_medication_features_v1_65plus_first_scope.v1"
)

PATIENT_CONTEXT_SOURCE_TABLES = (
    "encounter_medication_first_scope_65plus",
    "clinical.patients",
    "clinical.admissions",
    "clinical.diagnoses_icd",
    "clinical.labevents",
    "clinical.omr",
    "ed.edstays",
    "ed.diagnosis",
)
TEMPORAL_SOURCE_TABLES = (
    "encounter_medication_first_scope_65plus",
    "clinical.labevents",
    "ed.edstays",
    "ed.triage",
    "ed.vitalsign",
)
FEATURE_STORE_SOURCE_TABLES = (
    "encounter_medication_first_scope_65plus",
    "encounter_medication_state_65plus_rxnorm",
    *PATIENT_CONTEXT_SOURCE_TABLES[1:],
    *TEMPORAL_SOURCE_TABLES[1:],
)

PATIENT_CONTEXT_PROVENANCE_PRIOR_UTILIZATION = (
    "derived_from_pre_review_admissions_and_edstays"
)
PATIENT_CONTEXT_PROVENANCE_PRIOR_DIAGNOSES = (
    "derived_from_prior_discharged_diagnoses_icd"
)
PATIENT_CONTEXT_PROVENANCE_ED_DIAGNOSES = "observed_ed_diagnosis_pre_review"
PATIENT_CONTEXT_PROVENANCE_AGE = "carried_encounter_medication_state_age_proxy"
PATIENT_CONTEXT_PROVENANCE_SEX = "observed_patients_gender"
LAB_PROVENANCE_PRE_REVIEW = "observed_labevents_pre_review"
VITAL_PROVENANCE_PRE_REVIEW = "observed_ed_triage_or_vitalsign_pre_review"
MISSINGNESS_UNAVAILABLE = "unavailable"
MISSINGNESS_OBSERVED = "observed"

MEDICATION_SEMANTICS_BURDEN_COLUMNS = [
    "medication_class_standardized",
    "medication_standardized_source",
    "ingredient_standardized",
    "ingredient_resolution_status",
    "mapping_confidence",
    "ambiguous_mapping_flag",
    "mapping_candidate_count",
    "active_at_review_flag",
    "medication_status_at_review",
    "continued_from_home_inferred",
    "newly_started_during_encounter_inferred",
    "medication_start_context",
    "duration_before_review_hours",
    "duration_before_review_inferable_flag",
    "duration_before_review_lower_bound_flag",
    "scheduled_vs_prn",
    "scheduled_vs_prn_inference_status",
    "dose_value",
    "dose_unit",
    "route",
    "frequency",
    "exact_current_medication_count",
    "current_benzodiazepine_count",
    "current_opioid_count",
    "current_anticholinergic_count",
    "current_ppi_count",
    "current_antipsychotic_count",
    "current_supported_class_count",
    "row_same_class_current_count",
    "same_class_duplicate_therapy_flag",
    "same_class_duplicate_therapy_signal_count",
]

STATE_PROVENANCE_COLUMNS = [
    *FIRST_SCOPE_FEATURE_KEY_COLUMNS,
    "review_time_validated_flag",
    "review_time_policy_name",
    "review_time_capped_to_discharge_flag",
    "source_home_medrecon_flag",
    "source_ed_pyxis_flag",
    "source_hospital_order_flag",
    "source_hospital_admin_flag",
    "source_tables_json",
]

PATIENT_CONTEXT_FEATURE_COLUMNS = [
    "age_context",
    "age_group",
    "age_provenance",
    "sex_context",
    "sex_provenance",
    "weight_kg",
    "weight_kg_provenance",
    "weight_kg_unavailable_reason",
    "bmi",
    "bmi_provenance",
    "bmi_unavailable_reason",
    "patient_context_latest_creatinine_time",
    "patient_context_latest_creatinine_value",
    "egfr_ml_min_1_73m2",
    "egfr_provenance",
    "egfr_unavailable_reason",
    "cockcroft_gault_ml_min",
    "cockcroft_gault_provenance",
    "cockcroft_gault_unavailable_reason",
    "prior_hospital_admission_count_all_time",
    "prior_hospital_admission_count_365d",
    "prior_ed_stay_count_all_time",
    "prior_ed_stay_count_365d",
    "prior_utilization_provenance",
    "prior_utilization_window_definition",
    "prior_ckd_flag",
    "prior_dementia_flag",
    "prior_delirium_flag",
    "prior_heart_failure_flag",
    "prior_diabetes_flag",
    "prior_diagnosis_risk_renal_flag",
    "prior_diagnosis_risk_cognitive_flag",
    "prior_diagnosis_risk_cardiac_flag",
    "prior_diagnosis_risk_metabolic_flag",
    "prior_diagnosis_context_provenance",
    "ed_ckd_flag",
    "ed_dementia_flag",
    "ed_delirium_flag",
    "ed_heart_failure_flag",
    "ed_diabetes_flag",
    "ed_diagnosis_risk_renal_flag",
    "ed_diagnosis_risk_cognitive_flag",
    "ed_diagnosis_risk_cardiac_flag",
    "ed_diagnosis_risk_metabolic_flag",
    "ed_diagnosis_context_provenance",
    "ed_diagnosis_context_missingness",
    "patient_context_window_end",
    "patient_context_feature_window_end_at_or_before_review_time_flag",
    "patient_context_available_as_of_review_time_flag",
    "patient_context_source_tables_json",
    "patient_context_build_run_id",
    "patient_context_contract_version",
]

TEMPORAL_FEATURE_COLUMNS = [
    "creatinine_observation_count_pre_review",
    "creatinine_first",
    "creatinine_last",
    "creatinine_min",
    "creatinine_max",
    "creatinine_mean",
    "creatinine_delta",
    "creatinine_trend_direction",
    "creatinine_latest_time",
    "creatinine_provenance",
    "creatinine_missingness",
    "potassium_observation_count_pre_review",
    "potassium_first",
    "potassium_last",
    "potassium_min",
    "potassium_max",
    "potassium_mean",
    "potassium_delta",
    "potassium_trend_direction",
    "potassium_latest_time",
    "potassium_provenance",
    "potassium_missingness",
    "sodium_observation_count_pre_review",
    "sodium_first",
    "sodium_last",
    "sodium_min",
    "sodium_max",
    "sodium_mean",
    "sodium_delta",
    "sodium_trend_direction",
    "sodium_latest_time",
    "sodium_provenance",
    "sodium_missingness",
    "sbp_observation_count_pre_review",
    "sbp_min",
    "sbp_max",
    "sbp_mean",
    "sbp_last",
    "dbp_observation_count_pre_review",
    "dbp_min",
    "dbp_max",
    "dbp_mean",
    "dbp_last",
    "heart_rate_observation_count_pre_review",
    "heart_rate_min",
    "heart_rate_max",
    "heart_rate_mean",
    "heart_rate_last",
    "pain_observation_count_pre_review",
    "pain_min",
    "pain_max",
    "pain_mean",
    "pain_last",
    "lab_window_start",
    "lab_window_end",
    "lab_window_end_at_or_before_review_time_flag",
    "vital_window_start",
    "vital_window_end",
    "vital_window_end_at_or_before_review_time_flag",
    "temporal_window_end",
    "temporal_feature_window_end_at_or_before_review_time_flag",
    "blood_pressure_provenance",
    "heart_rate_provenance",
    "pain_provenance",
    "temporal_available_as_of_review_time_flag",
    "temporal_source_tables_json",
    "temporal_build_run_id",
    "temporal_contract_version",
]

FINAL_PROVENANCE_COLUMNS = [
    "state_source_tables_json",
    "feature_available_as_of_review_time_flag",
    "feature_window_end_at_or_before_review_time_flag",
    "feature_source_tables_json",
    "missingness_profile_json",
    "first_scope_feature_store_build_run_id",
    "first_scope_feature_store_contract_version",
]

FIRST_SCOPE_FEATURE_STORE_COLUMNS = [
    *FIRST_SCOPE_FEATURE_KEY_COLUMNS,
    "hadm_id",
    "stay_id",
    "review_timestamp_source",
    *MEDICATION_SEMANTICS_BURDEN_COLUMNS,
    "review_time_validated_flag",
    "review_time_policy_name",
    "review_time_capped_to_discharge_flag",
    "source_home_medrecon_flag",
    "source_ed_pyxis_flag",
    "source_hospital_order_flag",
    "source_hospital_admin_flag",
    *[column for column in PATIENT_CONTEXT_FEATURE_COLUMNS if not column.endswith("_build_run_id") and not column.endswith("_contract_version")],
    *[column for column in TEMPORAL_FEATURE_COLUMNS if not column.endswith("_build_run_id") and not column.endswith("_contract_version")],
    *FINAL_PROVENANCE_COLUMNS,
]


@dataclass(frozen=True, slots=True)
class FirstScopeFeatureStoreBuildResult:
    """Built feature store and its metadata payload."""

    feature_store: pd.DataFrame
    metadata: dict[str, object]


def build_first_scope_patient_context_features(
    *,
    encounter_medication_first_scope: pd.DataFrame,
    patients: pd.DataFrame,
    admissions: pd.DataFrame,
    diagnoses_icd: pd.DataFrame,
    labevents: pd.DataFrame,
    omr: pd.DataFrame | None = None,
    edstays: pd.DataFrame | None = None,
    ed_diagnosis: pd.DataFrame | None = None,
    serum_creatinine_item_ids: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Build review-time-safe patient-context features at the encounter-review snapshot grain."""
    base = _prepare_snapshot_base(encounter_medication_first_scope)
    build_run_id = f"first-scope-patient-context-{uuid4().hex[:12]}"

    patient_frame = patients.loc[:, ["subject_id", "gender"]].drop_duplicates("subject_id").copy()
    context = base.merge(patient_frame, how="left", on="subject_id", validate="many_to_one")
    context["age_context"] = pd.to_numeric(context["age_proxy"], errors="coerce")
    context["age_provenance"] = PATIENT_CONTEXT_PROVENANCE_AGE
    context["sex_context"] = context["gender"]
    context["sex_provenance"] = context["sex_context"].notna().map(
        {True: PATIENT_CONTEXT_PROVENANCE_SEX, False: PROVENANCE_UNAVAILABLE}
    )
    context = context.drop(columns=["age_proxy", "gender"])

    weight_omr = _latest_subject_measurement_asof(
        context,
        omr,
        result_name="Weight (Lbs)",
        value_column="weight_kg",
        transform=lambda value: round(float(value) * 0.45359237, 3),
    )
    bmi_omr = _latest_subject_measurement_asof(
        context,
        omr,
        result_name="BMI (kg/m2)",
        value_column="bmi",
        transform=lambda value: round(float(value), 3),
    )
    height_omr = _latest_subject_measurement_asof(
        context,
        omr,
        result_name="Height (Inches)",
        value_column="height_inches_context",
        transform=lambda value: round(float(value), 3),
    )
    for frame in [weight_omr, bmi_omr, height_omr]:
        context = context.merge(
            frame,
            how="left",
            on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
            validate="one_to_one",
        )

    context["weight_kg_provenance"] = context["weight_kg"].notna().map(
        {True: WEIGHT_PROVENANCE_OBSERVED, False: PROVENANCE_UNAVAILABLE}
    )
    context["weight_kg_unavailable_reason"] = context["weight_kg"].notna().map(
        {True: pd.NA, False: "no_omr_weight_on_or_before_review"}
    )
    context["bmi_provenance"] = context["bmi"].notna().map(
        {True: BMI_PROVENANCE_OBSERVED, False: PROVENANCE_UNAVAILABLE}
    )
    context["bmi_unavailable_reason"] = context["bmi"].notna().map(
        {True: pd.NA, False: "no_omr_bmi_or_height_weight_on_or_before_review"}
    )

    bmi_derived_mask = (
        context["bmi"].isna()
        & context["weight_kg"].notna()
        & context["height_inches_context"].notna()
        & (pd.to_numeric(context["height_inches_context"], errors="coerce") > 0)
    )
    height_m = pd.to_numeric(context["height_inches_context"], errors="coerce") * 0.0254
    context.loc[bmi_derived_mask, "bmi"] = (
        pd.to_numeric(context.loc[bmi_derived_mask, "weight_kg"], errors="coerce")
        / (height_m.loc[bmi_derived_mask] ** 2)
    ).round(3)
    context.loc[bmi_derived_mask, "bmi_provenance"] = BMI_PROVENANCE_DERIVED
    context.loc[bmi_derived_mask, "bmi_unavailable_reason"] = pd.NA

    creatinine_context = _summarize_numeric_observations_for_snapshots(
        base=context,
        observations=_prepare_lab_events(
            labevents,
            item_ids=serum_creatinine_item_ids,
            value_column="valuenum",
            id_column="hadm_id",
            time_column="charttime",
        ),
        entity_id_column="hadm_id",
        prefix="patient_context_latest_creatinine",
        value_column="valuenum",
    )
    context = context.merge(
        creatinine_context.loc[
            :,
            FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS
            + [
                "patient_context_latest_creatinine_last",
                "patient_context_latest_creatinine_latest_time",
            ],
        ].rename(
            columns={
                "patient_context_latest_creatinine_last": (
                    "patient_context_latest_creatinine_value"
                ),
                "patient_context_latest_creatinine_latest_time": (
                    "patient_context_latest_creatinine_time"
                ),
            }
        ),
        how="left",
        on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        validate="one_to_one",
    )

    context["egfr_ml_min_1_73m2"] = context.apply(
        lambda row: compute_egfr_ckd_epi_2021(
            age=row.get("age_context"),
            sex=row.get("sex_context"),
            creatinine_mg_dl=row.get("patient_context_latest_creatinine_value"),
        ),
        axis=1,
    )
    context["egfr_provenance"] = context["egfr_ml_min_1_73m2"].notna().map(
        {True: EGFR_PROVENANCE_DERIVED, False: PROVENANCE_UNAVAILABLE}
    )
    context["egfr_unavailable_reason"] = context.apply(
        lambda row: derive_egfr_unavailable_reason(
            pd.Series(
                {
                    "egfr_ml_min_1_73m2": row.get("egfr_ml_min_1_73m2"),
                    "creatinine_first": row.get("patient_context_latest_creatinine_value"),
                    "age_context": row.get("age_context"),
                    "sex_context": row.get("sex_context"),
                }
            )
        ),
        axis=1,
    )
    context["cockcroft_gault_ml_min"] = context.apply(
        lambda row: compute_cockcroft_gault(
            age=row.get("age_context"),
            sex=row.get("sex_context"),
            weight_kg=row.get("weight_kg"),
            creatinine_mg_dl=row.get("patient_context_latest_creatinine_value"),
        ),
        axis=1,
    )
    context["cockcroft_gault_provenance"] = context["cockcroft_gault_ml_min"].notna().map(
        {True: COCKCROFT_GAULT_PROVENANCE_DERIVED, False: PROVENANCE_UNAVAILABLE}
    )
    context["cockcroft_gault_unavailable_reason"] = context.apply(
        lambda row: derive_cockcroft_gault_unavailable_reason(
            pd.Series(
                {
                    "cockcroft_gault_ml_min": row.get("cockcroft_gault_ml_min"),
                    "weight_kg": row.get("weight_kg"),
                    "creatinine_first": row.get("patient_context_latest_creatinine_value"),
                    "age_context": row.get("age_context"),
                    "sex_context": row.get("sex_context"),
                }
            )
        ),
        axis=1,
    )

    prior_utilization = _count_prior_events_for_snapshots(
        base=context,
        events=admissions.loc[:, ["subject_id", "dischtime"]].copy(),
        time_column="dischtime",
        all_time_column="prior_hospital_admission_count_all_time",
        lookback_column="prior_hospital_admission_count_365d",
        latest_time_column="prior_hospital_discharge_latest_time",
    )
    context = context.merge(
        prior_utilization,
        how="left",
        on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        validate="one_to_one",
    )

    prior_ed_utilization = _count_prior_events_for_snapshots(
        base=context,
        events=(
            edstays.loc[:, ["subject_id", "outtime"]].copy()
            if edstays is not None and not edstays.empty
            else pd.DataFrame(columns=["subject_id", "outtime"])
        ),
        time_column="outtime",
        all_time_column="prior_ed_stay_count_all_time",
        lookback_column="prior_ed_stay_count_365d",
        latest_time_column="prior_ed_outtime_latest_time",
    )
    context = context.merge(
        prior_ed_utilization,
        how="left",
        on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        validate="one_to_one",
    )
    context["prior_utilization_provenance"] = PATIENT_CONTEXT_PROVENANCE_PRIOR_UTILIZATION
    context["prior_utilization_window_definition"] = "all_time_and_365d_pre_review"

    prior_diagnosis = _build_prior_diagnosis_features(
        base=context,
        diagnoses_icd=diagnoses_icd,
        admissions=admissions,
    )
    context = context.merge(
        prior_diagnosis,
        how="left",
        on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        validate="one_to_one",
    )

    safe_ed_diagnosis = _build_current_ed_diagnosis_features(
        base=context,
        edstays=edstays,
        ed_diagnosis=ed_diagnosis,
    )
    context = context.merge(
        safe_ed_diagnosis,
        how="left",
        on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        validate="one_to_one",
    )

    for column_name in [
        "prior_hospital_admission_count_all_time",
        "prior_hospital_admission_count_365d",
        "prior_ed_stay_count_all_time",
        "prior_ed_stay_count_365d",
    ]:
        context[column_name] = (
            pd.to_numeric(context[column_name], errors="coerce").fillna(0).astype(int)
        )

    diagnosis_flag_columns = [
        f"prior_{column_name}" for column_name in DIAGNOSIS_ICD_PREFIXES
    ] + [
        f"prior_{column_name}" for column_name in DIAGNOSIS_RISK_CATEGORY_GROUPS
    ] + [
        f"ed_{column_name}" for column_name in DIAGNOSIS_ICD_PREFIXES
    ] + [
        f"ed_{column_name}" for column_name in DIAGNOSIS_RISK_CATEGORY_GROUPS
    ]
    for column_name in diagnosis_flag_columns:
        context[column_name] = (
            pd.to_numeric(context[column_name], errors="coerce").fillna(0).astype(int)
        )

    for column_name in [
        "prior_diagnosis_context_provenance",
        "ed_diagnosis_context_provenance",
    ]:
        context[column_name] = context[column_name].fillna(PROVENANCE_UNAVAILABLE)
    context["ed_diagnosis_context_missingness"] = context[
        "ed_diagnosis_context_missingness"
    ].fillna(MISSINGNESS_UNAVAILABLE)

    patient_window_sources = pd.concat(
        [
            pd.to_datetime(context["weight_kg_measurement_time"], errors="coerce"),
            pd.to_datetime(context["bmi_measurement_time"], errors="coerce"),
            pd.to_datetime(context["height_inches_measurement_time"], errors="coerce"),
            pd.to_datetime(context["patient_context_latest_creatinine_time"], errors="coerce"),
            pd.to_datetime(context["prior_hospital_discharge_latest_time"], errors="coerce"),
            pd.to_datetime(context["prior_ed_outtime_latest_time"], errors="coerce"),
            pd.to_datetime(context["prior_diagnosis_window_end"], errors="coerce"),
            pd.to_datetime(context["ed_diagnosis_window_end"], errors="coerce"),
        ],
        axis=1,
    )
    context["patient_context_window_end"] = patient_window_sources.max(axis=1)
    context["patient_context_feature_window_end_at_or_before_review_time_flag"] = (
        context["patient_context_window_end"].isna()
        | (
            pd.to_datetime(context["patient_context_window_end"], errors="coerce")
            <= pd.to_datetime(context["review_timestamp"], errors="coerce")
        )
    ).astype(int)
    context["patient_context_available_as_of_review_time_flag"] = 1
    context["patient_context_source_tables_json"] = dumps_json(PATIENT_CONTEXT_SOURCE_TABLES)
    context["patient_context_build_run_id"] = build_run_id
    context["patient_context_contract_version"] = (
        FIRST_SCOPE_PATIENT_CONTEXT_CONTRACT_VERSION
    )

    drop_columns = [
        "weight_kg_measurement_time",
        "bmi_measurement_time",
        "height_inches_context",
        "height_inches_measurement_time",
        "prior_hospital_discharge_latest_time",
        "prior_ed_outtime_latest_time",
        "prior_diagnosis_window_end",
        "ed_diagnosis_window_end",
    ]
    context = context.drop(columns=drop_columns, errors="ignore")
    context = context.loc[:, FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS + PATIENT_CONTEXT_FEATURE_COLUMNS].copy()
    context = context.sort_values(FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS).reset_index(drop=True)
    validate_first_scope_patient_context_features_artifact(context)
    return context


def build_first_scope_temporal_features(
    *,
    encounter_medication_first_scope: pd.DataFrame,
    labevents: pd.DataFrame,
    serum_creatinine_item_ids: tuple[int, ...],
    edstays: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build review-time-safe temporal physiology features at the encounter-review snapshot grain."""
    base = _prepare_snapshot_base(encounter_medication_first_scope)
    build_run_id = f"first-scope-temporal-{uuid4().hex[:12]}"

    creatinine = _summarize_numeric_observations_for_snapshots(
        base=base,
        observations=_prepare_lab_events(
            labevents,
            item_ids=serum_creatinine_item_ids,
            value_column="valuenum",
            id_column="hadm_id",
            time_column="charttime",
        ),
        entity_id_column="hadm_id",
        prefix="creatinine",
        value_column="valuenum",
    )
    potassium = _summarize_numeric_observations_for_snapshots(
        base=base,
        observations=_prepare_lab_events(
            labevents,
            item_ids=potassium_itemids(),
            value_column="valuenum",
            id_column="hadm_id",
            time_column="charttime",
        ),
        entity_id_column="hadm_id",
        prefix="potassium",
        value_column="valuenum",
    )
    sodium = _summarize_numeric_observations_for_snapshots(
        base=base,
        observations=_prepare_lab_events(
            labevents,
            item_ids=sodium_itemids(),
            value_column="valuenum",
            id_column="hadm_id",
            time_column="charttime",
        ),
        entity_id_column="hadm_id",
        prefix="sodium",
        value_column="valuenum",
    )

    temporal = base.copy()
    for frame in [creatinine, potassium, sodium]:
        temporal = temporal.merge(
            frame,
            how="left",
            on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
            validate="one_to_one",
        )

    lab_first_time_columns = [
        "creatinine_first_time",
        "potassium_first_time",
        "sodium_first_time",
    ]
    lab_last_time_columns = [
        "creatinine_latest_time",
        "potassium_latest_time",
        "sodium_latest_time",
    ]
    temporal["lab_window_start"] = pd.concat(
        [pd.to_datetime(temporal[column_name], errors="coerce") for column_name in lab_first_time_columns],
        axis=1,
    ).min(axis=1)
    temporal["lab_window_end"] = pd.concat(
        [pd.to_datetime(temporal[column_name], errors="coerce") for column_name in lab_last_time_columns],
        axis=1,
    ).max(axis=1)
    temporal["lab_window_end_at_or_before_review_time_flag"] = (
        temporal["lab_window_end"].isna()
        | (
            pd.to_datetime(temporal["lab_window_end"], errors="coerce")
            <= pd.to_datetime(temporal["review_timestamp"], errors="coerce")
        )
    ).astype(int)

    ed_signal_frame = _prepare_ed_signal_events(edstays=edstays, triage=triage, vitalsign=vitalsign)
    sbp = _summarize_numeric_observations_for_snapshots(
        base=base,
        observations=ed_signal_frame.loc[:, ["stay_id", "charttime", "sbp"]]
        if not ed_signal_frame.empty
        else pd.DataFrame(columns=["stay_id", "charttime", "sbp"]),
        entity_id_column="stay_id",
        prefix="sbp",
        value_column="sbp",
        time_column="charttime",
    )
    dbp = _summarize_numeric_observations_for_snapshots(
        base=base,
        observations=ed_signal_frame.loc[:, ["stay_id", "charttime", "dbp"]]
        if not ed_signal_frame.empty
        else pd.DataFrame(columns=["stay_id", "charttime", "dbp"]),
        entity_id_column="stay_id",
        prefix="dbp",
        value_column="dbp",
        time_column="charttime",
    )
    heart_rate = _summarize_numeric_observations_for_snapshots(
        base=base,
        observations=ed_signal_frame.loc[:, ["stay_id", "charttime", "heartrate"]]
        if not ed_signal_frame.empty
        else pd.DataFrame(columns=["stay_id", "charttime", "heartrate"]),
        entity_id_column="stay_id",
        prefix="heart_rate",
        value_column="heartrate",
        time_column="charttime",
    )
    pain = _summarize_numeric_observations_for_snapshots(
        base=base,
        observations=ed_signal_frame.loc[:, ["stay_id", "charttime", "pain"]]
        if not ed_signal_frame.empty
        else pd.DataFrame(columns=["stay_id", "charttime", "pain"]),
        entity_id_column="stay_id",
        prefix="pain",
        value_column="pain",
        time_column="charttime",
    )
    for frame in [sbp, dbp, heart_rate, pain]:
        temporal = temporal.merge(
            frame,
            how="left",
            on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
            validate="one_to_one",
        )

    vital_first_time_columns = [
        "sbp_first_time",
        "dbp_first_time",
        "heart_rate_first_time",
        "pain_first_time",
    ]
    vital_last_time_columns = [
        "sbp_latest_time",
        "dbp_latest_time",
        "heart_rate_latest_time",
        "pain_latest_time",
    ]
    temporal["vital_window_start"] = pd.concat(
        [pd.to_datetime(temporal[column_name], errors="coerce") for column_name in vital_first_time_columns],
        axis=1,
    ).min(axis=1)
    temporal["vital_window_end"] = pd.concat(
        [pd.to_datetime(temporal[column_name], errors="coerce") for column_name in vital_last_time_columns],
        axis=1,
    ).max(axis=1)
    temporal["vital_window_end_at_or_before_review_time_flag"] = (
        temporal["vital_window_end"].isna()
        | (
            pd.to_datetime(temporal["vital_window_end"], errors="coerce")
            <= pd.to_datetime(temporal["review_timestamp"], errors="coerce")
        )
    ).astype(int)
    temporal["temporal_window_end"] = pd.concat(
        [
            pd.to_datetime(temporal["lab_window_end"], errors="coerce"),
            pd.to_datetime(temporal["vital_window_end"], errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    temporal["temporal_feature_window_end_at_or_before_review_time_flag"] = (
        (
            pd.to_numeric(
                temporal["lab_window_end_at_or_before_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
        & (
            pd.to_numeric(
                temporal["vital_window_end_at_or_before_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
    ).astype(int)

    temporal["creatinine_provenance"] = temporal["creatinine_latest_time"].notna().map(
        {True: LAB_PROVENANCE_PRE_REVIEW, False: PROVENANCE_UNAVAILABLE}
    )
    temporal["creatinine_missingness"] = temporal["creatinine_latest_time"].notna().map(
        {True: MISSINGNESS_OBSERVED, False: MISSINGNESS_UNAVAILABLE}
    )
    temporal["potassium_provenance"] = temporal["potassium_latest_time"].notna().map(
        {True: LAB_PROVENANCE_PRE_REVIEW, False: PROVENANCE_UNAVAILABLE}
    )
    temporal["potassium_missingness"] = temporal["potassium_latest_time"].notna().map(
        {True: MISSINGNESS_OBSERVED, False: MISSINGNESS_UNAVAILABLE}
    )
    temporal["sodium_provenance"] = temporal["sodium_latest_time"].notna().map(
        {True: LAB_PROVENANCE_PRE_REVIEW, False: PROVENANCE_UNAVAILABLE}
    )
    temporal["sodium_missingness"] = temporal["sodium_latest_time"].notna().map(
        {True: MISSINGNESS_OBSERVED, False: MISSINGNESS_UNAVAILABLE}
    )
    temporal["blood_pressure_provenance"] = (
        temporal[["sbp_latest_time", "dbp_latest_time"]].notna().any(axis=1)
    ).map({True: VITAL_PROVENANCE_PRE_REVIEW, False: PROVENANCE_UNAVAILABLE})
    temporal["heart_rate_provenance"] = temporal["heart_rate_latest_time"].notna().map(
        {True: VITAL_PROVENANCE_PRE_REVIEW, False: PROVENANCE_UNAVAILABLE}
    )
    temporal["pain_provenance"] = temporal["pain_latest_time"].notna().map(
        {True: VITAL_PROVENANCE_PRE_REVIEW, False: PROVENANCE_UNAVAILABLE}
    )
    temporal["temporal_available_as_of_review_time_flag"] = 1
    temporal["temporal_source_tables_json"] = dumps_json(TEMPORAL_SOURCE_TABLES)
    temporal["temporal_build_run_id"] = build_run_id
    temporal["temporal_contract_version"] = FIRST_SCOPE_TEMPORAL_CONTRACT_VERSION

    temporal = temporal.drop(
        columns=[
            "creatinine_first_time",
            "potassium_first_time",
            "sodium_first_time",
            "sbp_first_time",
            "dbp_first_time",
            "heart_rate_first_time",
            "pain_first_time",
            "sbp_latest_time",
            "dbp_latest_time",
            "heart_rate_latest_time",
            "pain_latest_time",
        ],
        errors="ignore",
    )
    temporal = temporal.loc[:, FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS + TEMPORAL_FEATURE_COLUMNS].copy()
    temporal = temporal.sort_values(FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS).reset_index(drop=True)
    validate_first_scope_temporal_features_artifact(temporal)
    return temporal


def build_first_scope_feature_store(
    *,
    encounter_medication_first_scope: pd.DataFrame,
    patient_context_features: pd.DataFrame,
    temporal_features: pd.DataFrame,
    encounter_medication_state_provenance: pd.DataFrame,
) -> FirstScopeFeatureStoreBuildResult:
    """Build Feature Store v1 at the encounter-medication review grain."""
    validate_encounter_medication_first_scope_artifact(encounter_medication_first_scope)
    validate_first_scope_patient_context_features_artifact(patient_context_features)
    validate_first_scope_temporal_features_artifact(temporal_features)
    validate_first_scope_state_provenance_subset(encounter_medication_state_provenance)

    base = encounter_medication_first_scope.loc[
        :,
        FIRST_SCOPE_FEATURE_KEY_COLUMNS
        + ["hadm_id", "stay_id", "review_timestamp_source"]
        + MEDICATION_SEMANTICS_BURDEN_COLUMNS,
    ].copy()
    base["review_timestamp"] = pd.to_datetime(base["review_timestamp"], errors="coerce")
    base["hadm_id"] = pd.to_numeric(base["hadm_id"], errors="coerce").astype("Int64")
    base["stay_id"] = pd.to_numeric(base["stay_id"], errors="coerce").astype("Int64")

    state_subset = encounter_medication_state_provenance.loc[
        :,
        STATE_PROVENANCE_COLUMNS,
    ].copy()
    state_subset["review_timestamp"] = pd.to_datetime(
        state_subset["review_timestamp"],
        errors="coerce",
    )
    state_subset = state_subset.rename(columns={"source_tables_json": "state_source_tables_json"})

    first_scope_keys = base.loc[:, FIRST_SCOPE_FEATURE_KEY_COLUMNS].copy()
    state_subset = first_scope_keys.merge(
        state_subset,
        how="left",
        on=FIRST_SCOPE_FEATURE_KEY_COLUMNS,
        validate="one_to_one",
        indicator=True,
    )
    if not state_subset["_merge"].eq("both").all():
        raise DataLoadError(
            "First-scope feature-store build could not recover review-time provenance "
            "for every first-scope row from encounter_medication_state_65plus_rxnorm."
        )
    state_subset = state_subset.drop(columns="_merge")

    feature_store = base.merge(
        patient_context_features.drop(
            columns=["patient_context_build_run_id", "patient_context_contract_version"],
            errors="ignore",
        ),
        how="left",
        on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        validate="many_to_one",
    )
    feature_store = feature_store.merge(
        temporal_features.drop(
            columns=["temporal_build_run_id", "temporal_contract_version"],
            errors="ignore",
        ),
        how="left",
        on=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        validate="many_to_one",
    )
    feature_store = feature_store.merge(
        state_subset.loc[
            :,
            [
                *FIRST_SCOPE_FEATURE_KEY_COLUMNS,
                "review_time_validated_flag",
                "review_time_policy_name",
                "review_time_capped_to_discharge_flag",
                "source_home_medrecon_flag",
                "source_ed_pyxis_flag",
                "source_hospital_order_flag",
                "source_hospital_admin_flag",
                "state_source_tables_json",
            ],
        ],
        how="left",
        on=FIRST_SCOPE_FEATURE_KEY_COLUMNS,
        validate="one_to_one",
    )

    feature_store["feature_window_end_at_or_before_review_time_flag"] = (
        (
            pd.to_numeric(
                feature_store["patient_context_feature_window_end_at_or_before_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
        & (
            pd.to_numeric(
                feature_store["temporal_feature_window_end_at_or_before_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
    ).astype(int)
    feature_store["feature_available_as_of_review_time_flag"] = (
        (
            pd.to_numeric(
                feature_store["review_time_validated_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
        & (
            pd.to_numeric(
                feature_store["feature_window_end_at_or_before_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
    ).astype(int)
    feature_store["feature_source_tables_json"] = feature_store.apply(
        _build_feature_source_tables_json,
        axis=1,
    )
    feature_store["missingness_profile_json"] = feature_store.apply(
        _build_missingness_profile_json,
        axis=1,
    )
    feature_store["first_scope_feature_store_build_run_id"] = (
        f"first-scope-feature-store-{uuid4().hex[:12]}"
    )
    feature_store["first_scope_feature_store_contract_version"] = (
        FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION
    )

    feature_store = feature_store.loc[:, FIRST_SCOPE_FEATURE_STORE_COLUMNS].copy()
    feature_store = feature_store.sort_values(FIRST_SCOPE_FEATURE_KEY_COLUMNS).reset_index(drop=True)
    validate_first_scope_feature_store_artifact(feature_store)

    metadata = build_first_scope_feature_store_metadata(
        feature_store=feature_store,
        patient_context_features=patient_context_features,
        temporal_features=temporal_features,
    )
    return FirstScopeFeatureStoreBuildResult(
        feature_store=feature_store,
        metadata=metadata,
    )


def build_first_scope_feature_store_metadata(
    *,
    feature_store: pd.DataFrame,
    patient_context_features: pd.DataFrame,
    temporal_features: pd.DataFrame,
) -> dict[str, object]:
    """Build a feature dictionary / metadata payload for Feature Store v1."""
    feature_entries = [
        _build_feature_dictionary_entry(feature_store, column_name)
        for column_name in feature_store.columns
    ]
    return {
        "artifact_name": "encounter_medication_features_v1_65plus_first_scope",
        "feature_store_contract_version": FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION,
        "grain_description": FEATURE_STORE_GRAIN_DESCRIPTION,
        "primary_key": list(FEATURE_STORE_PRIMARY_KEY),
        "row_count": int(len(feature_store)),
        "snapshot_row_count": int(
            patient_context_features[FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS]
            .drop_duplicates()
            .shape[0]
        ),
        "feature_column_count": int(
            len(
                [
                    column_name
                    for column_name in feature_store.columns
                    if column_name not in FIRST_SCOPE_FEATURE_KEY_COLUMNS
                ]
            )
        ),
        "stage_contract_versions": {
            "patient_context": _first_non_null_value(
                patient_context_features.get("patient_context_contract_version")
            ),
            "temporal": _first_non_null_value(
                temporal_features.get("temporal_contract_version")
            ),
            "feature_store": FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION,
        },
        "excluded_legacy_or_retrospective_features": [
            "legacy CSV feature builder outputs under data/processed/",
            "current-encounter hospital diagnoses_icd flags without pre-review availability evidence",
            "discharge-capped legacy OMR selectors from the rules path",
            "retrospective admission-level lab summaries from src/opti_med/features/context.py",
            "placeholder opioid MME and renal-dose mismatch values from first-scope burden scaffolding",
        ],
        "features": feature_entries,
    }


def build_first_scope_feature_store_leakage_qc_report(
    *,
    encounter_medication_first_scope: pd.DataFrame,
    patient_context_features: pd.DataFrame,
    temporal_features: pd.DataFrame,
    feature_store: pd.DataFrame,
) -> str:
    """Render a Markdown leakage QC report for Feature Store v1."""
    validate_encounter_medication_first_scope_artifact(encounter_medication_first_scope)
    validate_first_scope_patient_context_features_artifact(patient_context_features)
    validate_first_scope_temporal_features_artifact(temporal_features)
    validate_first_scope_feature_store_artifact(feature_store)

    snapshot_count = int(
        encounter_medication_first_scope[FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS]
        .drop_duplicates()
        .shape[0]
    )
    patient_snapshot_count = int(
        patient_context_features[FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS]
        .drop_duplicates()
        .shape[0]
    )
    temporal_snapshot_count = int(
        temporal_features[FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS]
        .drop_duplicates()
        .shape[0]
    )
    patient_window_violations = int(
        (
            patient_context_features["patient_context_window_end"].notna()
            & (
                pd.to_datetime(
                    patient_context_features["patient_context_window_end"],
                    errors="coerce",
                )
                > pd.to_datetime(
                    patient_context_features["review_timestamp"],
                    errors="coerce",
                )
            )
        ).sum()
    )
    lab_window_violations = int(
        (
            temporal_features["lab_window_end"].notna()
            & (
                pd.to_datetime(temporal_features["lab_window_end"], errors="coerce")
                > pd.to_datetime(temporal_features["review_timestamp"], errors="coerce")
            )
        ).sum()
    )
    vital_window_violations = int(
        (
            temporal_features["vital_window_end"].notna()
            & (
                pd.to_datetime(temporal_features["vital_window_end"], errors="coerce")
                > pd.to_datetime(temporal_features["review_timestamp"], errors="coerce")
            )
        ).sum()
    )
    feature_window_violations = int(
        (
            pd.to_numeric(
                feature_store["feature_window_end_at_or_before_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            != 1
        ).sum()
    )
    feature_availability_failures = int(
        (
            pd.to_numeric(
                feature_store["feature_available_as_of_review_time_flag"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            != 1
        ).sum()
    )
    unexpected_class_rows = int(
        (~feature_store["medication_class_standardized"].isin(
            encounter_medication_first_scope["medication_class_standardized"].unique()
        )).sum()
    )
    missingness_counts = {
        "egfr_missing_rows": int(feature_store["egfr_ml_min_1_73m2"].isna().sum()),
        "cockcroft_gault_missing_rows": int(feature_store["cockcroft_gault_ml_min"].isna().sum()),
        "creatinine_missing_rows": int(
            (
                pd.to_numeric(
                    feature_store["creatinine_observation_count_pre_review"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
                == 0
            ).sum()
        ),
        "blood_pressure_missing_rows": int(
            (feature_store["sbp_mean"].isna() | feature_store["dbp_mean"].isna()).sum()
        ),
    }

    lines = [
        "# Feature Store Leakage QC 65plus First-Scope",
        "",
        "## Scope",
        "",
        "- Feature store target: `data/feature_store/encounter_medication_features_v1_65plus_first_scope.parquet`",
        "- Subset input: `data/analytical/encounter_medication_first_scope_65plus.parquet`",
        "- Leakage rule: every feature timestamp and window end must be `<= review_timestamp`",
        "",
        "## Row Counts",
        "",
        f"- first_scope rows: {len(encounter_medication_first_scope):,}",
        f"- first_scope unique encounter-review snapshots: {snapshot_count:,}",
        f"- patient_context rows: {len(patient_context_features):,}",
        f"- patient_context unique snapshots: {patient_snapshot_count:,}",
        f"- temporal rows: {len(temporal_features):,}",
        f"- temporal unique snapshots: {temporal_snapshot_count:,}",
        f"- feature_store rows: {len(feature_store):,}",
        f"- feature_store unique review-time medication keys: {feature_store[FIRST_SCOPE_FEATURE_KEY_COLUMNS].drop_duplicates().shape[0]:,}",
        "",
        "## Leakage Checks",
        "",
        f"- patient_context_window_end > review_timestamp rows: {patient_window_violations:,}",
        f"- lab_window_end > review_timestamp rows: {lab_window_violations:,}",
        f"- vital_window_end > review_timestamp rows: {vital_window_violations:,}",
        f"- final feature_window_end_at_or_before_review_time_flag != 1 rows: {feature_window_violations:,}",
        f"- final feature_available_as_of_review_time_flag != 1 rows: {feature_availability_failures:,}",
        f"- unexpected medication classes outside first-scope subset rows: {unexpected_class_rows:,}",
        "",
        "## Missingness Snapshot",
        "",
    ]
    for metric_name, metric_value in missingness_counts.items():
        lines.append(f"- {metric_name}: {metric_value:,}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Zero leakage-check counts are required before promoting the artifact to later ML phases.",
            "- Non-zero missingness counts are expected for source-conditional features and are tracked explicitly in provenance.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_first_scope_patient_context_features_artifact(dataframe: pd.DataFrame) -> None:
    """Validate the persisted patient-context stage artifact."""
    _assert_required_columns(
        dataframe,
        required_columns=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS + PATIENT_CONTEXT_FEATURE_COLUMNS,
        artifact_name="first_scope_patient_context_features_65plus",
    )
    _assert_unique(
        dataframe,
        key_columns=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        artifact_name="first_scope_patient_context_features_65plus",
    )
    _assert_binary_columns(
        dataframe,
        column_names=[
            "patient_context_feature_window_end_at_or_before_review_time_flag",
            "patient_context_available_as_of_review_time_flag",
        ],
        artifact_name="first_scope_patient_context_features_65plus",
    )
    if (
        dataframe["patient_context_contract_version"].dropna().astype(str)
        != FIRST_SCOPE_PATIENT_CONTEXT_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Patient-context stage artifact must use contract version "
            f"'{FIRST_SCOPE_PATIENT_CONTEXT_CONTRACT_VERSION}'."
        )


def validate_first_scope_temporal_features_artifact(dataframe: pd.DataFrame) -> None:
    """Validate the persisted temporal stage artifact."""
    _assert_required_columns(
        dataframe,
        required_columns=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS + TEMPORAL_FEATURE_COLUMNS,
        artifact_name="first_scope_temporal_features_65plus",
    )
    _assert_unique(
        dataframe,
        key_columns=FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS,
        artifact_name="first_scope_temporal_features_65plus",
    )
    _assert_binary_columns(
        dataframe,
        column_names=[
            "lab_window_end_at_or_before_review_time_flag",
            "vital_window_end_at_or_before_review_time_flag",
            "temporal_feature_window_end_at_or_before_review_time_flag",
            "temporal_available_as_of_review_time_flag",
        ],
        artifact_name="first_scope_temporal_features_65plus",
    )
    if (
        dataframe["temporal_contract_version"].dropna().astype(str)
        != FIRST_SCOPE_TEMPORAL_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Temporal stage artifact must use contract version "
            f"'{FIRST_SCOPE_TEMPORAL_CONTRACT_VERSION}'."
        )


def validate_first_scope_state_provenance_subset(dataframe: pd.DataFrame) -> None:
    """Validate the keyed state-provenance subset used by the final feature-store join."""
    _assert_required_columns(
        dataframe,
        required_columns=STATE_PROVENANCE_COLUMNS,
        artifact_name="encounter_medication_state_65plus_rxnorm_subset",
    )
    _assert_unique(
        dataframe,
        key_columns=FIRST_SCOPE_FEATURE_KEY_COLUMNS,
        artifact_name="encounter_medication_state_65plus_rxnorm_subset",
    )
    _assert_binary_columns(
        dataframe,
        column_names=[
            "review_time_validated_flag",
            "review_time_capped_to_discharge_flag",
            "source_home_medrecon_flag",
            "source_ed_pyxis_flag",
            "source_hospital_order_flag",
            "source_hospital_admin_flag",
        ],
        artifact_name="encounter_medication_state_65plus_rxnorm_subset",
    )


def validate_first_scope_feature_store_artifact(dataframe: pd.DataFrame) -> None:
    """Validate the final Feature Store v1 artifact."""
    _assert_required_columns(
        dataframe,
        required_columns=FIRST_SCOPE_FEATURE_STORE_COLUMNS,
        artifact_name="encounter_medication_features_v1_65plus_first_scope",
    )
    _assert_unique(
        dataframe,
        key_columns=FIRST_SCOPE_FEATURE_KEY_COLUMNS,
        artifact_name="encounter_medication_features_v1_65plus_first_scope",
    )
    _assert_binary_columns(
        dataframe,
        column_names=[
            "review_time_validated_flag",
            "review_time_capped_to_discharge_flag",
            "source_home_medrecon_flag",
            "source_ed_pyxis_flag",
            "source_hospital_order_flag",
            "source_hospital_admin_flag",
            "patient_context_feature_window_end_at_or_before_review_time_flag",
            "patient_context_available_as_of_review_time_flag",
            "lab_window_end_at_or_before_review_time_flag",
            "vital_window_end_at_or_before_review_time_flag",
            "temporal_feature_window_end_at_or_before_review_time_flag",
            "temporal_available_as_of_review_time_flag",
            "feature_available_as_of_review_time_flag",
            "feature_window_end_at_or_before_review_time_flag",
        ],
        artifact_name="encounter_medication_features_v1_65plus_first_scope",
    )
    if (
        dataframe["first_scope_feature_store_contract_version"].dropna().astype(str)
        != FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION
    ).any():
        raise DataLoadError(
            "Feature Store v1 artifact must use contract version "
            f"'{FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION}'."
        )


def _prepare_snapshot_base(encounter_medication_first_scope: pd.DataFrame) -> pd.DataFrame:
    validate_encounter_medication_first_scope_artifact(encounter_medication_first_scope)
    base = encounter_medication_first_scope.loc[
        :,
        FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS + ["age_proxy", "age_group"],
    ].copy()
    base["review_timestamp"] = pd.to_datetime(base["review_timestamp"], errors="coerce")
    base["hadm_id"] = pd.to_numeric(base["hadm_id"], errors="coerce").astype("Int64")
    base["stay_id"] = pd.to_numeric(base["stay_id"], errors="coerce").astype("Int64")
    base = (
        base.sort_values(FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS)
        .drop_duplicates(FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS)
        .reset_index(drop=True)
    )
    return base


def _prepare_lab_events(
    dataframe: pd.DataFrame,
    *,
    item_ids: tuple[int, ...],
    value_column: str,
    id_column: str,
    time_column: str,
) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame(columns=[id_column, time_column, value_column])
    prepared = dataframe.loc[:, [id_column, "itemid", time_column, value_column]].copy()
    prepared[id_column] = pd.to_numeric(prepared[id_column], errors="coerce").astype("Int64")
    prepared["itemid"] = pd.to_numeric(prepared["itemid"], errors="coerce").astype("Int64")
    prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
    prepared[value_column] = pd.to_numeric(prepared[value_column], errors="coerce")
    prepared = prepared.dropna(subset=[id_column, "itemid", time_column, value_column])
    prepared = prepared.loc[prepared["itemid"].isin(item_ids)].copy()
    return prepared.loc[:, [id_column, time_column, value_column]]


def _latest_subject_measurement_asof(
    base: pd.DataFrame,
    omr: pd.DataFrame | None,
    *,
    result_name: str,
    value_column: str,
    transform,
) -> pd.DataFrame:
    output = base.loc[:, FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS].copy()
    output[value_column] = pd.NA
    time_column_name = f"{value_column.split('_')[0]}_measurement_time"
    if value_column == "height_inches_context":
        time_column_name = "height_inches_measurement_time"
    if value_column == "weight_kg":
        time_column_name = "weight_kg_measurement_time"
    if value_column == "bmi":
        time_column_name = "bmi_measurement_time"
    output[time_column_name] = pd.NaT
    if omr is None or omr.empty:
        return output

    measurements = omr.loc[
        :,
        ["subject_id", "chartdate", "result_name", "result_value"],
    ].copy()
    measurements = measurements.loc[measurements["result_name"] == result_name].copy()
    if measurements.empty:
        return output

    measurements["chartdate"] = pd.to_datetime(measurements["chartdate"], errors="coerce")
    measurements["subject_id"] = pd.to_numeric(
        measurements["subject_id"],
        errors="coerce",
    ).astype("int64")
    measurements["result_value_numeric"] = pd.to_numeric(
        measurements["result_value"],
        errors="coerce",
    )
    measurements = measurements.dropna(
        subset=["subject_id", "chartdate", "result_value_numeric"]
    )
    if measurements.empty:
        return output

    grouped_measurements = {
        int(subject_id): group.sort_values("chartdate").reset_index(drop=True)
        for subject_id, group in measurements.groupby("subject_id", sort=False)
        if not pd.isna(subject_id)
    }

    for subject_id, group in output.groupby("subject_id", sort=False):
        if pd.isna(subject_id):
            continue
        measurement_group = grouped_measurements.get(int(subject_id))
        if measurement_group is None or measurement_group.empty:
            continue
        measurement_times = measurement_group["chartdate"].to_numpy(dtype="datetime64[ns]")
        measurement_values = measurement_group["result_value_numeric"].to_numpy(dtype=float)
        review_times = pd.to_datetime(group["review_timestamp"], errors="coerce").to_numpy(
            dtype="datetime64[ns]"
        )
        positions = np.searchsorted(measurement_times, review_times, side="right") - 1
        valid_mask = positions >= 0
        if not valid_mask.any():
            continue
        valid_index = group.index[valid_mask]
        output.loc[valid_index, value_column] = [
            transform(measurement_values[position]) for position in positions[valid_mask]
        ]
        output.loc[valid_index, time_column_name] = pd.to_datetime(
            measurement_times[positions[valid_mask]]
        )
    return output


def _summarize_numeric_observations_for_snapshots(
    *,
    base: pd.DataFrame,
    observations: pd.DataFrame,
    entity_id_column: str,
    prefix: str,
    value_column: str | None = None,
    time_column: str = "charttime",
) -> pd.DataFrame:
    value_column = value_column or prefix
    output = base.loc[:, FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS].copy()
    output[f"{prefix}_observation_count_pre_review"] = 0
    output[f"{prefix}_first"] = np.nan
    output[f"{prefix}_last"] = np.nan
    output[f"{prefix}_min"] = np.nan
    output[f"{prefix}_max"] = np.nan
    output[f"{prefix}_mean"] = np.nan
    output[f"{prefix}_delta"] = np.nan
    output[f"{prefix}_trend_direction"] = "unavailable"
    output[f"{prefix}_first_time"] = pd.NaT
    output[f"{prefix}_latest_time"] = pd.NaT

    if observations.empty:
        return output

    observed = observations.loc[:, [entity_id_column, time_column, value_column]].copy()
    observed[entity_id_column] = pd.to_numeric(
        observed[entity_id_column],
        errors="coerce",
    ).astype("Int64")
    observed[time_column] = pd.to_datetime(observed[time_column], errors="coerce")
    observed[value_column] = pd.to_numeric(observed[value_column], errors="coerce")
    observed = observed.dropna(subset=[entity_id_column, time_column, value_column])
    if observed.empty:
        return output

    grouped_observations: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for entity_value, group in observed.groupby(entity_id_column, sort=False):
        if pd.isna(entity_value):
            continue
        entity_key = int(entity_value)
        group = group.sort_values(time_column, na_position="last").reset_index(drop=True)
        times = group[time_column].to_numpy(dtype="datetime64[ns]")
        values = group[value_column].to_numpy(dtype=float)
        if len(times) == 0:
            continue
        cumulative_min = np.minimum.accumulate(values)
        cumulative_max = np.maximum.accumulate(values)
        cumulative_sum = np.cumsum(values)
        grouped_observations[entity_key] = (
            times,
            values,
            cumulative_min,
            cumulative_max,
            cumulative_sum,
        )

    for entity_value, group in base.loc[base[entity_id_column].notna()].groupby(
        entity_id_column,
        sort=False,
    ):
        entity_key = int(entity_value)
        payload = grouped_observations.get(entity_key)
        if payload is None:
            continue
        times, values, cumulative_min, cumulative_max, cumulative_sum = payload
        review_times = group["review_timestamp"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(times, review_times, side="right") - 1
        valid_mask = positions >= 0
        if not valid_mask.any():
            continue

        index = group.index[valid_mask]
        selected_positions = positions[valid_mask]
        counts = selected_positions + 1
        means = (cumulative_sum[selected_positions] / counts).round(3)
        first_values = values[0]
        last_values = values[selected_positions]
        output.loc[index, f"{prefix}_observation_count_pre_review"] = counts.astype(int)
        output.loc[index, f"{prefix}_first"] = np.repeat(first_values, len(index))
        output.loc[index, f"{prefix}_last"] = last_values
        output.loc[index, f"{prefix}_min"] = cumulative_min[selected_positions]
        output.loc[index, f"{prefix}_max"] = cumulative_max[selected_positions]
        output.loc[index, f"{prefix}_mean"] = means
        output.loc[index, f"{prefix}_delta"] = np.round(last_values - first_values, 3)
        output.loc[index, f"{prefix}_trend_direction"] = [
            categorize_delta_trend(value) for value in output.loc[index, f"{prefix}_delta"]
        ]
        output.loc[index, f"{prefix}_first_time"] = pd.to_datetime(times[0])
        output.loc[index, f"{prefix}_latest_time"] = pd.to_datetime(times[selected_positions])

    return output


def _count_prior_events_for_snapshots(
    *,
    base: pd.DataFrame,
    events: pd.DataFrame,
    time_column: str,
    all_time_column: str,
    lookback_column: str,
    latest_time_column: str,
    lookback_days: int = 365,
) -> pd.DataFrame:
    output = base.loc[:, FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS].copy()
    output[all_time_column] = 0
    output[lookback_column] = 0
    output[latest_time_column] = pd.NaT
    if events.empty:
        return output

    prepared = events.loc[:, ["subject_id", time_column]].copy()
    prepared["subject_id"] = pd.to_numeric(prepared["subject_id"], errors="coerce").astype("Int64")
    prepared[time_column] = pd.to_datetime(prepared[time_column], errors="coerce")
    prepared = prepared.dropna(subset=["subject_id", time_column]).sort_values(
        ["subject_id", time_column]
    )
    grouped_events = {
        int(subject_id): group[time_column].to_numpy(dtype="datetime64[ns]")
        for subject_id, group in prepared.groupby("subject_id", sort=False)
        if not pd.isna(subject_id)
    }
    lookback_delta = np.timedelta64(lookback_days, "D")

    for subject_id, group in base.groupby("subject_id", sort=False):
        if pd.isna(subject_id):
            continue
        event_times = grouped_events.get(int(subject_id))
        if event_times is None or len(event_times) == 0:
            continue
        review_times = group["review_timestamp"].to_numpy(dtype="datetime64[ns]")
        left_positions = np.searchsorted(event_times, review_times, side="left")
        lookback_positions = np.searchsorted(
            event_times,
            review_times - lookback_delta,
            side="left",
        )
        output.loc[group.index, all_time_column] = left_positions.astype(int)
        output.loc[group.index, lookback_column] = (
            left_positions - lookback_positions
        ).astype(int)
        valid_mask = left_positions > 0
        if valid_mask.any():
            output.loc[group.index[valid_mask], latest_time_column] = pd.to_datetime(
                event_times[left_positions[valid_mask] - 1]
            )
    return output


def _build_prior_diagnosis_features(
    *,
    base: pd.DataFrame,
    diagnoses_icd: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.DataFrame:
    output = base.loc[:, FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS].copy()
    for column_name in DIAGNOSIS_ICD_PREFIXES:
        output[f"prior_{column_name}"] = 0
    for column_name in DIAGNOSIS_RISK_CATEGORY_GROUPS:
        output[f"prior_{column_name}"] = 0
    output["prior_diagnosis_context_provenance"] = PATIENT_CONTEXT_PROVENANCE_PRIOR_DIAGNOSES
    output["prior_diagnosis_window_end"] = pd.NaT

    if diagnoses_icd.empty or admissions.empty:
        output["prior_diagnosis_context_provenance"] = PROVENANCE_UNAVAILABLE
        return output

    diagnosis_flags = build_diagnosis_flags_by_key(
        diagnoses_icd,
        key_columns=["subject_id", "hadm_id"],
    )
    if diagnosis_flags.empty:
        return output

    admissions_frame = admissions.loc[:, ["subject_id", "hadm_id", "dischtime"]].copy()
    admissions_frame["subject_id"] = pd.to_numeric(
        admissions_frame["subject_id"],
        errors="coerce",
    ).astype("Int64")
    admissions_frame["hadm_id"] = pd.to_numeric(
        admissions_frame["hadm_id"],
        errors="coerce",
    ).astype("Int64")
    admissions_frame["dischtime"] = pd.to_datetime(
        admissions_frame["dischtime"],
        errors="coerce",
    )
    diagnosis_flags = diagnosis_flags.merge(
        admissions_frame,
        how="left",
        on=["subject_id", "hadm_id"],
        validate="one_to_one",
    )
    diagnosis_flags = diagnosis_flags.dropna(subset=["subject_id", "dischtime"]).sort_values(
        ["subject_id", "dischtime"]
    )
    if diagnosis_flags.empty:
        return output

    flag_columns = [f"prior_{column_name}" for column_name in DIAGNOSIS_ICD_PREFIXES] + [
        f"prior_{column_name}" for column_name in DIAGNOSIS_RISK_CATEGORY_GROUPS
    ]
    rename_map = {
        column_name.replace("prior_", ""): column_name
        for column_name in flag_columns
    }
    diagnosis_flags = diagnosis_flags.rename(columns=rename_map)
    grouped_payload: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for subject_id, group in diagnosis_flags.groupby("subject_id", sort=False):
        if pd.isna(subject_id):
            continue
        discharge_times = group["dischtime"].to_numpy(dtype="datetime64[ns]")
        cumulative_flags = group.loc[:, flag_columns].cummax().to_numpy(dtype=int)
        grouped_payload[int(subject_id)] = (discharge_times, cumulative_flags)

    for subject_id, group in base.groupby("subject_id", sort=False):
        if pd.isna(subject_id):
            continue
        payload = grouped_payload.get(int(subject_id))
        if payload is None:
            continue
        discharge_times, cumulative_flags = payload
        review_times = group["review_timestamp"].to_numpy(dtype="datetime64[ns]")
        positions = np.searchsorted(discharge_times, review_times, side="left") - 1
        valid_mask = positions >= 0
        if not valid_mask.any():
            continue
        valid_index = group.index[valid_mask]
        selected = cumulative_flags[positions[valid_mask]]
        output.loc[valid_index, flag_columns] = selected
        output.loc[valid_index, "prior_diagnosis_window_end"] = pd.to_datetime(
            discharge_times[positions[valid_mask]]
        )
    return output


def _build_current_ed_diagnosis_features(
    *,
    base: pd.DataFrame,
    edstays: pd.DataFrame | None,
    ed_diagnosis: pd.DataFrame | None,
) -> pd.DataFrame:
    output = base.loc[:, FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS].copy()
    flag_columns = [f"ed_{column_name}" for column_name in DIAGNOSIS_ICD_PREFIXES] + [
        f"ed_{column_name}" for column_name in DIAGNOSIS_RISK_CATEGORY_GROUPS
    ]
    for column_name in flag_columns:
        output[column_name] = 0
    output["ed_diagnosis_context_provenance"] = PROVENANCE_UNAVAILABLE
    output["ed_diagnosis_context_missingness"] = MISSINGNESS_UNAVAILABLE
    output["ed_diagnosis_window_end"] = pd.NaT
    if edstays is None or edstays.empty or ed_diagnosis is None or ed_diagnosis.empty:
        return output

    edstays_frame = edstays.loc[:, ["subject_id", "stay_id", "outtime"]].copy()
    edstays_frame["subject_id"] = pd.to_numeric(edstays_frame["subject_id"], errors="coerce").astype("Int64")
    edstays_frame["stay_id"] = pd.to_numeric(edstays_frame["stay_id"], errors="coerce").astype("Int64")
    edstays_frame["outtime"] = pd.to_datetime(edstays_frame["outtime"], errors="coerce")

    linked = ed_diagnosis.loc[:, ["subject_id", "stay_id", "icd_code", "icd_version"]].copy()
    linked["subject_id"] = pd.to_numeric(linked["subject_id"], errors="coerce").astype("Int64")
    linked["stay_id"] = pd.to_numeric(linked["stay_id"], errors="coerce").astype("Int64")
    diagnosis_flags = build_diagnosis_flags_by_key(
        linked,
        key_columns=["subject_id", "stay_id"],
    )
    if diagnosis_flags.empty:
        return output

    diagnosis_flags = diagnosis_flags.merge(
        edstays_frame,
        how="left",
        on=["subject_id", "stay_id"],
        validate="many_to_one",
    )
    rename_map = {
        column_name: f"ed_{column_name}"
        for column_name in diagnosis_flags.columns
        if column_name not in {"subject_id", "stay_id", "outtime"}
    }
    diagnosis_flags = diagnosis_flags.rename(columns=rename_map)
    output = output.merge(
        diagnosis_flags,
        how="left",
        on=["subject_id", "stay_id"],
        validate="many_to_one",
        suffixes=("", "_ed"),
    )
    for column_name in flag_columns:
        suffixed_column = f"{column_name}_ed"
        if suffixed_column in output.columns:
            output[column_name] = output[suffixed_column]
            output = output.drop(columns=suffixed_column)
    if "outtime" in output.columns:
        output["ed_diagnosis_window_end"] = output["outtime"]
        output = output.drop(columns="outtime")
    safe_mask = (
        output["ed_diagnosis_window_end"].notna()
        & (
            pd.to_datetime(output["ed_diagnosis_window_end"], errors="coerce")
            <= pd.to_datetime(output["review_timestamp"], errors="coerce")
        )
    )
    for column_name in flag_columns:
        output[column_name] = np.where(
            safe_mask,
            pd.to_numeric(output[column_name], errors="coerce").fillna(0).astype(int),
            0,
        )
    output["ed_diagnosis_context_provenance"] = safe_mask.map(
        {True: PATIENT_CONTEXT_PROVENANCE_ED_DIAGNOSES, False: PROVENANCE_UNAVAILABLE}
    )
    output["ed_diagnosis_context_missingness"] = safe_mask.map(
        {True: MISSINGNESS_OBSERVED, False: MISSINGNESS_UNAVAILABLE}
    )
    output.loc[~safe_mask, "ed_diagnosis_window_end"] = pd.NaT
    return output.loc[:, FIRST_SCOPE_SNAPSHOT_KEY_COLUMNS + flag_columns + [
        "ed_diagnosis_context_provenance",
        "ed_diagnosis_context_missingness",
        "ed_diagnosis_window_end",
    ]]


def _prepare_ed_signal_events(
    *,
    edstays: pd.DataFrame | None,
    triage: pd.DataFrame | None,
    vitalsign: pd.DataFrame | None,
) -> pd.DataFrame:
    if edstays is None or edstays.empty:
        return pd.DataFrame(
            columns=["stay_id", "charttime", "sbp", "dbp", "heartrate", "pain"]
        )

    edstays_frame = edstays.loc[:, ["subject_id", "stay_id", "intime"]].copy()
    edstays_frame["subject_id"] = pd.to_numeric(
        edstays_frame["subject_id"], errors="coerce"
    ).astype("Int64")
    edstays_frame["stay_id"] = pd.to_numeric(edstays_frame["stay_id"], errors="coerce").astype("Int64")
    edstays_frame["intime"] = pd.to_datetime(edstays_frame["intime"], errors="coerce")

    frames: list[pd.DataFrame] = []
    if triage is not None and not triage.empty:
        triage_frame = triage.loc[
            :,
            ["subject_id", "stay_id", "charttime", "sbp", "dbp", "heartrate", "pain"],
        ].copy()
        triage_frame["subject_id"] = pd.to_numeric(
            triage_frame["subject_id"], errors="coerce"
        ).astype("Int64")
        triage_frame["stay_id"] = pd.to_numeric(
            triage_frame["stay_id"], errors="coerce"
        ).astype("Int64")
        triage_frame["charttime"] = pd.to_datetime(
            triage_frame["charttime"],
            errors="coerce",
        )
        triage_frame = triage_frame.merge(
            edstays_frame,
            how="left",
            on=["subject_id", "stay_id"],
            validate="many_to_one",
        )
        triage_frame["charttime"] = triage_frame["charttime"].combine_first(
            triage_frame["intime"]
        )
        triage_frame = triage_frame.drop(columns=["intime"])
        frames.append(triage_frame)

    if vitalsign is not None and not vitalsign.empty:
        vitals_frame = vitalsign.loc[
            :,
            ["subject_id", "stay_id", "charttime", "sbp", "dbp", "heartrate", "pain"],
        ].copy()
        vitals_frame["subject_id"] = pd.to_numeric(
            vitals_frame["subject_id"], errors="coerce"
        ).astype("Int64")
        vitals_frame["stay_id"] = pd.to_numeric(
            vitals_frame["stay_id"], errors="coerce"
        ).astype("Int64")
        vitals_frame["charttime"] = pd.to_datetime(
            vitals_frame["charttime"],
            errors="coerce",
        )
        frames.append(vitals_frame)

    if not frames:
        return pd.DataFrame(
            columns=["stay_id", "charttime", "sbp", "dbp", "heartrate", "pain"]
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column_name in ["sbp", "dbp", "heartrate", "pain"]:
        combined[column_name] = pd.to_numeric(combined[column_name], errors="coerce")
    combined = combined.dropna(subset=["stay_id", "charttime"], how="any")
    return combined.loc[:, ["stay_id", "charttime", "sbp", "dbp", "heartrate", "pain"]]


def _build_feature_source_tables_json(row: pd.Series) -> str:
    source_tables: set[str] = set()
    for value in [
        row.get("state_source_tables_json"),
        row.get("patient_context_source_tables_json"),
        row.get("temporal_source_tables_json"),
    ]:
        payload = loads_json_or_none(value)
        if isinstance(payload, list):
            source_tables.update(str(item) for item in payload)
    source_tables.update(FEATURE_STORE_SOURCE_TABLES)
    return dumps_json(sorted(source_tables))


def _build_missingness_profile_json(row: pd.Series) -> str:
    payload = {
        "blood_pressure_missing": bool(
            pd.isna(row.get("sbp_mean")) or pd.isna(row.get("dbp_mean"))
        ),
        "bmi_missing": bool(pd.isna(row.get("bmi"))),
        "cockcroft_gault_missing": bool(pd.isna(row.get("cockcroft_gault_ml_min"))),
        "creatinine_missing": bool(
            pd.to_numeric(
                pd.Series([row.get("creatinine_observation_count_pre_review")]),
                errors="coerce",
            )
            .fillna(0)
            .iloc[0]
            == 0
        ),
        "egfr_missing": bool(pd.isna(row.get("egfr_ml_min_1_73m2"))),
        "heart_rate_missing": bool(pd.isna(row.get("heart_rate_mean"))),
        "pain_missing": bool(pd.isna(row.get("pain_mean"))),
        "potassium_missing": bool(
            pd.to_numeric(
                pd.Series([row.get("potassium_observation_count_pre_review")]),
                errors="coerce",
            )
            .fillna(0)
            .iloc[0]
            == 0
        ),
        "sodium_missing": bool(
            pd.to_numeric(
                pd.Series([row.get("sodium_observation_count_pre_review")]),
                errors="coerce",
            )
            .fillna(0)
            .iloc[0]
            == 0
        ),
        "weight_kg_missing": bool(pd.isna(row.get("weight_kg"))),
    }
    return dumps_json(payload)


def _build_feature_dictionary_entry(
    feature_store: pd.DataFrame,
    column_name: str,
) -> dict[str, object]:
    role = "feature"
    stage_name = "final_feature_store"
    feature_group = "provenance_missingness"
    source_tables: tuple[str, ...] = FEATURE_STORE_SOURCE_TABLES
    expected_missingness_behavior = "source_conditional"
    provenance_columns: list[str] = []
    description = column_name.replace("_", " ")

    if column_name in FIRST_SCOPE_FEATURE_KEY_COLUMNS or column_name in {"hadm_id", "stay_id"}:
        role = "key"
        feature_group = "identifiers"
        source_tables = ("encounter_medication_first_scope_65plus",)
        expected_missingness_behavior = "required_if_state_exists"
        description = f"Identifier column '{column_name}'."
    elif column_name in MEDICATION_SEMANTICS_BURDEN_COLUMNS or column_name in {
        "review_timestamp_source",
    }:
        stage_name = "first_scope_base"
        feature_group = "medication_semantics_burden"
        source_tables = ("encounter_medication_first_scope_65plus",)
        expected_missingness_behavior = "required_if_state_exists"
        description = _base_feature_description(column_name)
    elif column_name in PATIENT_CONTEXT_FEATURE_COLUMNS:
        stage_name = "patient_context"
        feature_group = "patient_context"
        source_tables = PATIENT_CONTEXT_SOURCE_TABLES
        description = _patient_context_description(column_name)
        provenance_columns = _patient_context_provenance_columns(column_name)
    elif column_name in TEMPORAL_FEATURE_COLUMNS:
        stage_name = "temporal"
        feature_group = "temporal_physiology"
        source_tables = TEMPORAL_SOURCE_TABLES
        expected_missingness_behavior = "window_conditional"
        description = _temporal_feature_description(column_name)
        provenance_columns = _temporal_provenance_columns(column_name)
    elif column_name in FINAL_PROVENANCE_COLUMNS or column_name in {
        "review_time_validated_flag",
        "review_time_policy_name",
        "review_time_capped_to_discharge_flag",
        "source_home_medrecon_flag",
        "source_ed_pyxis_flag",
        "source_hospital_order_flag",
        "source_hospital_admin_flag",
    }:
        stage_name = "final_feature_store"
        feature_group = "provenance_missingness"
        source_tables = FEATURE_STORE_SOURCE_TABLES
        expected_missingness_behavior = "must_be_explicitly_tracked"
        description = _provenance_feature_description(column_name)
        provenance_columns = [column_name] if column_name.endswith("_json") else []

    return {
        "column_name": column_name,
        "column_role": role,
        "feature_group": feature_group,
        "stage_name": stage_name,
        "data_type": str(feature_store[column_name].dtype),
        "description": description,
        "source_tables": list(source_tables),
        "point_in_time_safe": True,
        "expected_missingness_behavior": expected_missingness_behavior,
        "provenance_columns": provenance_columns,
    }


def _base_feature_description(column_name: str) -> str:
    descriptions = {
        "review_timestamp_source": "Source used to resolve the encounter review timestamp.",
        "medication_class_standardized": "Standardized first-scope medication class at review time.",
        "medication_standardized_source": "Source used for medication standardization.",
        "ingredient_standardized": "Standardized medication ingredient carried from the enriched first-scope artifact.",
        "ingredient_resolution_status": "Resolution status for ingredient standardization.",
        "mapping_confidence": "Confidence label for medication standardization.",
        "ambiguous_mapping_flag": "Whether medication mapping remained ambiguous.",
        "mapping_candidate_count": "Number of mapping candidates considered for the medication.",
        "active_at_review_flag": "Whether the medication is active at review_timestamp.",
        "medication_status_at_review": "Categorical medication status at review time.",
        "continued_from_home_inferred": "Whether the medication appears to continue from home at review time.",
        "newly_started_during_encounter_inferred": "Whether the medication appears newly started during the encounter.",
        "medication_start_context": "Interpretation of medication start context relative to the encounter.",
        "duration_before_review_hours": "Observed or lower-bound duration from earliest pre-review medication evidence to review_timestamp.",
        "duration_before_review_inferable_flag": "Whether duration_before_review_hours could be inferred safely.",
        "duration_before_review_lower_bound_flag": "Whether duration_before_review_hours is a lower bound rather than an exact duration.",
        "scheduled_vs_prn": "Scheduled-vs-PRN semantic label inferred from pre-review medication evidence.",
        "scheduled_vs_prn_inference_status": "Status label describing how scheduled_vs_prn was inferred.",
        "dose_value": "Dose value observed on the representative pre-review medication event.",
        "dose_unit": "Dose unit observed on the representative pre-review medication event.",
        "route": "Medication route observed on or before review time.",
        "frequency": "Medication frequency observed on or before review time.",
        "exact_current_medication_count": "Count of active first-scope medications at the same review snapshot.",
        "current_benzodiazepine_count": "Count of active benzodiazepines at the same review snapshot.",
        "current_opioid_count": "Count of active opioids at the same review snapshot.",
        "current_anticholinergic_count": "Count of active anticholinergics at the same review snapshot.",
        "current_ppi_count": "Count of active PPIs at the same review snapshot.",
        "current_antipsychotic_count": "Count of active antipsychotics at the same review snapshot.",
        "current_supported_class_count": "Count of active supported-class medications at the same review snapshot.",
        "row_same_class_current_count": "Count of active medications in the same standardized class as the row medication.",
        "same_class_duplicate_therapy_flag": "Whether same-class duplicate therapy is present at the review snapshot.",
        "same_class_duplicate_therapy_signal_count": "Number of same-class medications contributing to the duplicate-therapy signal.",
    }
    return descriptions.get(column_name, f"First-scope base feature '{column_name}'.")


def _patient_context_description(column_name: str) -> str:
    if column_name.startswith("prior_") and column_name.endswith("_flag"):
        return (
            f"History flag '{column_name}' derived only from prior discharged diagnoses known before review time."
        )
    if column_name.startswith("ed_") and column_name.endswith("_flag"):
        return (
            f"Current-encounter ED diagnosis flag '{column_name}' used only when ED diagnosis evidence is available before review time."
        )
    descriptions = {
        "age_context": "Age proxy carried from the persisted 65+ review-time branch.",
        "age_group": "Age-band context carried from the persisted 65+ review-time branch.",
        "age_provenance": "Provenance for age_context.",
        "sex_context": "Patient sex observed from standardized patients data.",
        "sex_provenance": "Provenance for sex_context.",
        "weight_kg": "Latest OMR weight observed on or before review_timestamp.",
        "weight_kg_provenance": "Provenance for weight_kg.",
        "weight_kg_unavailable_reason": "Explicit reason weight_kg is unavailable.",
        "bmi": "Latest OMR BMI on or before review_timestamp, or BMI derived from pre-review height and weight.",
        "bmi_provenance": "Provenance for bmi.",
        "bmi_unavailable_reason": "Explicit reason bmi is unavailable.",
        "patient_context_latest_creatinine_time": "Latest creatinine timestamp used for renal-reserve context.",
        "patient_context_latest_creatinine_value": "Latest creatinine value observed on or before review_timestamp for renal-reserve context.",
        "egfr_ml_min_1_73m2": "eGFR derived from the latest pre-review creatinine plus age and sex context.",
        "egfr_provenance": "Provenance for egfr_ml_min_1_73m2.",
        "egfr_unavailable_reason": "Explicit reason eGFR could not be derived safely.",
        "cockcroft_gault_ml_min": "Cockcroft-Gault estimate derived from the latest pre-review creatinine plus age, sex, and weight context.",
        "cockcroft_gault_provenance": "Provenance for cockcroft_gault_ml_min.",
        "cockcroft_gault_unavailable_reason": "Explicit reason Cockcroft-Gault could not be derived safely.",
        "prior_hospital_admission_count_all_time": "Count of prior hospital admissions with dischtime strictly before review_timestamp.",
        "prior_hospital_admission_count_365d": "Count of prior hospital admissions in the 365 days before review_timestamp.",
        "prior_ed_stay_count_all_time": "Count of prior ED stays with outtime strictly before review_timestamp.",
        "prior_ed_stay_count_365d": "Count of prior ED stays in the 365 days before review_timestamp.",
        "prior_utilization_provenance": "Provenance for prior utilization counts.",
        "prior_utilization_window_definition": "Window definition used for prior utilization counts.",
        "prior_diagnosis_context_provenance": "Provenance for prior discharged diagnosis history features.",
        "ed_diagnosis_context_provenance": "Provenance for current-encounter ED diagnosis features.",
        "ed_diagnosis_context_missingness": "Missingness behavior for ED diagnosis context at review time.",
        "patient_context_window_end": "Latest timestamp contributing to the patient-context stage for the snapshot.",
        "patient_context_feature_window_end_at_or_before_review_time_flag": "Whether patient-context evidence ends at or before review_timestamp.",
        "patient_context_available_as_of_review_time_flag": "Whether the patient-context stage completed as-of review time.",
        "patient_context_source_tables_json": "Serialized list of source tables used by the patient-context stage.",
        "patient_context_build_run_id": "Build-run identifier for the patient-context stage artifact.",
        "patient_context_contract_version": "Contract version for the patient-context stage artifact.",
    }
    return descriptions.get(column_name, f"Patient-context feature '{column_name}'.")


def _patient_context_provenance_columns(column_name: str) -> list[str]:
    if column_name in {"age_context", "age_group"}:
        return ["age_provenance"]
    if column_name == "sex_context":
        return ["sex_provenance"]
    if column_name == "weight_kg":
        return ["weight_kg_provenance", "weight_kg_unavailable_reason"]
    if column_name == "bmi":
        return ["bmi_provenance", "bmi_unavailable_reason"]
    if column_name == "egfr_ml_min_1_73m2":
        return ["egfr_provenance", "egfr_unavailable_reason"]
    if column_name == "cockcroft_gault_ml_min":
        return ["cockcroft_gault_provenance", "cockcroft_gault_unavailable_reason"]
    if column_name.startswith("prior_"):
        return ["prior_diagnosis_context_provenance", "prior_utilization_provenance"]
    if column_name.startswith("ed_"):
        return ["ed_diagnosis_context_provenance", "ed_diagnosis_context_missingness"]
    return []


def _temporal_feature_description(column_name: str) -> str:
    if column_name.startswith("creatinine_"):
        return f"Pre-review creatinine summary '{column_name}'."
    if column_name.startswith("potassium_"):
        return f"Pre-review potassium summary '{column_name}'."
    if column_name.startswith("sodium_"):
        return f"Pre-review sodium summary '{column_name}'."
    if column_name.startswith("sbp_") or column_name.startswith("dbp_"):
        return f"Pre-review ED blood-pressure summary '{column_name}'."
    if column_name.startswith("heart_rate_"):
        return f"Pre-review ED heart-rate summary '{column_name}'."
    if column_name.startswith("pain_"):
        return f"Pre-review ED pain-score summary '{column_name}'."
    descriptions = {
        "lab_window_start": "Earliest pre-review lab timestamp contributing to temporal physiology.",
        "lab_window_end": "Latest pre-review lab timestamp contributing to temporal physiology.",
        "lab_window_end_at_or_before_review_time_flag": "Whether the lab window ends at or before review_timestamp.",
        "vital_window_start": "Earliest pre-review ED vital timestamp contributing to temporal physiology.",
        "vital_window_end": "Latest pre-review ED vital timestamp contributing to temporal physiology.",
        "vital_window_end_at_or_before_review_time_flag": "Whether the ED vital window ends at or before review_timestamp.",
        "temporal_window_end": "Latest timestamp across temporal physiology sources for the snapshot.",
        "temporal_feature_window_end_at_or_before_review_time_flag": "Whether all temporal windows end at or before review_timestamp.",
        "blood_pressure_provenance": "Provenance for ED blood-pressure summaries.",
        "heart_rate_provenance": "Provenance for ED heart-rate summaries.",
        "pain_provenance": "Provenance for ED pain summaries.",
        "temporal_available_as_of_review_time_flag": "Whether the temporal stage completed as-of review time.",
        "temporal_source_tables_json": "Serialized list of source tables used by the temporal stage.",
        "temporal_build_run_id": "Build-run identifier for the temporal stage artifact.",
        "temporal_contract_version": "Contract version for the temporal stage artifact.",
    }
    return descriptions.get(column_name, f"Temporal feature '{column_name}'.")


def _temporal_provenance_columns(column_name: str) -> list[str]:
    if column_name.startswith("creatinine_"):
        return ["creatinine_provenance", "creatinine_missingness"]
    if column_name.startswith("potassium_"):
        return ["potassium_provenance", "potassium_missingness"]
    if column_name.startswith("sodium_"):
        return ["sodium_provenance", "sodium_missingness"]
    if column_name.startswith("sbp_") or column_name.startswith("dbp_"):
        return ["blood_pressure_provenance"]
    if column_name.startswith("heart_rate_"):
        return ["heart_rate_provenance"]
    if column_name.startswith("pain_"):
        return ["pain_provenance"]
    return []


def _provenance_feature_description(column_name: str) -> str:
    descriptions = {
        "review_time_validated_flag": "Whether the review timestamp passed upstream review-time validation.",
        "review_time_policy_name": "Upstream review-time policy name carried from encounter_medication_state_65plus.",
        "review_time_capped_to_discharge_flag": "Whether the review timestamp was capped to discharge upstream.",
        "source_home_medrecon_flag": "Whether home medication reconciliation contributed upstream source evidence.",
        "source_ed_pyxis_flag": "Whether ED pyxis evidence contributed upstream source evidence.",
        "source_hospital_order_flag": "Whether hospital order evidence contributed upstream source evidence.",
        "source_hospital_admin_flag": "Whether hospital administration evidence contributed upstream source evidence.",
        "state_source_tables_json": "Serialized upstream source tables carried from encounter_medication_state_65plus_rxnorm.",
        "feature_available_as_of_review_time_flag": "Whether the final feature row is available as-of review time.",
        "feature_window_end_at_or_before_review_time_flag": "Whether all final feature windows end at or before review_timestamp.",
        "feature_source_tables_json": "Serialized union of source tables contributing to the final feature row.",
        "missingness_profile_json": "Explicit JSON summary of selected feature missingness indicators.",
        "first_scope_feature_store_build_run_id": "Build-run identifier for the final Feature Store v1 artifact.",
        "first_scope_feature_store_contract_version": "Contract version for the final Feature Store v1 artifact.",
    }
    return descriptions.get(column_name, f"Provenance feature '{column_name}'.")


def _assert_required_columns(
    dataframe: pd.DataFrame,
    *,
    required_columns: list[str],
    artifact_name: str,
) -> None:
    missing_columns = sorted(set(required_columns) - set(dataframe.columns))
    if missing_columns:
        raise DataLoadError(
            f"Artifact '{artifact_name}' is missing required columns: {missing_columns}"
        )


def _assert_unique(
    dataframe: pd.DataFrame,
    *,
    key_columns: list[str],
    artifact_name: str,
) -> None:
    duplicate_count = int(dataframe.duplicated(subset=key_columns).sum())
    if duplicate_count:
        raise DataLoadError(
            f"Artifact '{artifact_name}' must be unique on {key_columns}, found {duplicate_count:,} duplicate rows."
        )


def _assert_binary_columns(
    dataframe: pd.DataFrame,
    *,
    column_names: list[str],
    artifact_name: str,
) -> None:
    for column_name in column_names:
        values = pd.to_numeric(dataframe[column_name], errors="coerce").dropna().astype(int)
        if not values.isin([0, 1]).all():
            raise DataLoadError(
                f"Artifact '{artifact_name}' column '{column_name}' must be binary 0/1."
            )


def _first_non_null_value(series: pd.Series | None) -> object:
    if series is None:
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]
