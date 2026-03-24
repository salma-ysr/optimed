"""Reusable patient context and dynamic signal feature engineering.

This module remains part of the active saved-artifact path for the current
rules baseline. It builds admission-level context with explicit provenance, but
it is not yet the future point-in-time-safe feature store.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from opti_med.features.mappings.diagnoses import (
    DIAGNOSIS_ICD_PREFIXES,
    DIAGNOSIS_RISK_CATEGORY_GROUPS,
)
from opti_med.features.mappings.labs import potassium_itemids, serum_creatinine_itemids, sodium_itemids
from opti_med.time_semantics.constants import MISSINGNESS_OBSERVED, PROVENANCE_UNAVAILABLE


DIAGNOSIS_CONTEXT_PROVENANCE_OBSERVED = "observed_diagnoses_icd"
ED_DIAGNOSIS_CONTEXT_PROVENANCE_OBSERVED = "observed_ed_diagnosis"
AGE_PROVENANCE_OBSERVED = "observed_patients_anchor_age"
SEX_PROVENANCE_OBSERVED = "observed_patients_gender"
WEIGHT_PROVENANCE_OBSERVED = "observed_omr_weight"
BMI_PROVENANCE_OBSERVED = "observed_omr_bmi"
BMI_PROVENANCE_DERIVED = "derived_from_omr_height_weight"
LAB_PROVENANCE_OBSERVED = "observed_labevents"
EGFR_PROVENANCE_DERIVED = "derived_ckd_epi_2021_from_creatinine_first"
COCKCROFT_GAULT_PROVENANCE_DERIVED = (
    "derived_cockcroft_gault_from_weight_and_creatinine_first"
)
ED_SIGNAL_PROVENANCE_OBSERVED = "observed_ed_triage_or_vitalsign"
ED_TRIAGE_PROVENANCE_OBSERVED = "observed_ed_triage"


def build_patient_context_features(
    *,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    diagnoses_icd: pd.DataFrame,
    labevents: pd.DataFrame,
    omr: pd.DataFrame | None,
    edstays: pd.DataFrame | None,
    ed_diagnosis: pd.DataFrame | None,
    triage: pd.DataFrame | None,
    vitalsign: pd.DataFrame | None,
    creatinine_threshold: float,
    serum_creatinine_ids: tuple[int, ...],
) -> pd.DataFrame:
    """Build a reusable admission-level patient context object with provenance.

    TODO(ml-pivot): this is the active saved-artifact path for current scoring. Future review-time
    feature builders must not assume these admission-level summaries are point-in-time safe.
    """
    base = admissions.loc[:, ["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    base["admittime"] = pd.to_datetime(base["admittime"], errors="coerce")
    base["dischtime"] = pd.to_datetime(base["dischtime"], errors="coerce")

    diagnosis_features = build_diagnosis_context_features(diagnoses_icd)
    ed_diagnosis_features = build_ed_diagnosis_context_features(edstays, ed_diagnosis)
    morphology_features = build_morphology_features(
        admissions=base,
        patients=patients,
        omr=omr,
    )
    lab_features = build_dynamic_lab_features(
        morphology_features=morphology_features,
        labevents=labevents,
        creatinine_item_ids=serum_creatinine_ids,
        potassium_ids=potassium_itemids(),
        sodium_ids=sodium_itemids(),
        creatinine_threshold=creatinine_threshold,
    )
    ed_signal_features = build_dynamic_ed_signal_features(base, edstays, triage, vitalsign)

    context = base.loc[:, ["subject_id", "hadm_id"]].drop_duplicates().copy()
    for features in [
        diagnosis_features,
        ed_diagnosis_features,
        morphology_features,
        lab_features,
        ed_signal_features,
    ]:
        context = context.merge(features, how="left", on=["subject_id", "hadm_id"], validate="one_to_one")
    return fill_context_defaults(context)


def build_diagnosis_context_features(diagnoses_icd: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hospital diagnosis flags and higher-level risk categories."""
    if diagnoses_icd.empty:
        return _empty_context_frame(["subject_id", "hadm_id"], diagnosis_context_columns())

    flags = build_diagnosis_flags_by_key(
        diagnoses_icd,
        key_columns=["subject_id", "hadm_id"],
    )
    flags["diagnosis_context_provenance"] = DIAGNOSIS_CONTEXT_PROVENANCE_OBSERVED
    flags["diagnosis_context_missingness"] = MISSINGNESS_OBSERVED
    return flags


def build_ed_diagnosis_context_features(
    edstays: pd.DataFrame | None,
    ed_diagnosis: pd.DataFrame | None,
) -> pd.DataFrame:
    """Aggregate ED diagnosis signals linked back to hospital admissions when possible."""
    if edstays is None or edstays.empty or ed_diagnosis is None or ed_diagnosis.empty:
        return _empty_context_frame(["subject_id", "hadm_id"], ed_diagnosis_context_columns())

    linked = ed_diagnosis.loc[:, ["subject_id", "stay_id", "icd_code", "icd_version"]].merge(
        edstays.loc[:, ["subject_id", "stay_id", "hadm_id"]],
        how="left",
        on=["subject_id", "stay_id"],
        validate="many_to_one",
    )
    linked = linked.loc[linked["hadm_id"].notna()].copy()
    if linked.empty:
        return _empty_context_frame(["subject_id", "hadm_id"], ed_diagnosis_context_columns())

    flags = build_diagnosis_flags_by_key(linked, key_columns=["subject_id", "hadm_id"])
    rename_map = {
        column: f"ed_{column}"
        for column in flags.columns
        if column not in {"subject_id", "hadm_id"}
    }
    flags = flags.rename(columns=rename_map)
    flags["ed_diagnosis_context_provenance"] = ED_DIAGNOSIS_CONTEXT_PROVENANCE_OBSERVED
    flags["ed_diagnosis_context_missingness"] = MISSINGNESS_OBSERVED
    return flags


def build_diagnosis_flags_by_key(
    diagnoses: pd.DataFrame,
    *,
    key_columns: list[str],
) -> pd.DataFrame:
    """Aggregate diagnosis flags and risk categories by a reusable encounter key."""
    diagnoses_frame = diagnoses.loc[:, [*key_columns, "icd_code", "icd_version"]].copy()
    diagnoses_frame["icd_code"] = diagnoses_frame["icd_code"].astype(str).str.upper()
    diagnoses_frame["icd_version"] = pd.to_numeric(
        diagnoses_frame["icd_version"], errors="coerce"
    ).astype("Int64")
    diagnoses_frame = diagnoses_frame.dropna(subset=[*key_columns, "icd_code", "icd_version"])

    if diagnoses_frame.empty:
        return _empty_context_frame(key_columns, diagnosis_context_columns())

    flags = diagnoses_frame.loc[:, key_columns].drop_duplicates().reset_index(drop=True)
    for flag_name, version_mapping in DIAGNOSIS_ICD_PREFIXES.items():
        matches = pd.Series(False, index=diagnoses_frame.index)
        for icd_version, prefixes in version_mapping.items():
            version_mask = diagnoses_frame["icd_version"] == icd_version
            prefix_mask = diagnoses_frame["icd_code"].str.startswith(prefixes)
            matches = matches | (version_mask & prefix_mask)

        matched_keys = diagnoses_frame.loc[matches, key_columns].drop_duplicates()
        flags = flags.merge(
            matched_keys.assign(**{flag_name: 1}),
            how="left",
            on=key_columns,
        )
        flags[flag_name] = flags[flag_name].fillna(0).astype(int)

    for category_name, source_flags in DIAGNOSIS_RISK_CATEGORY_GROUPS.items():
        flags[category_name] = flags.loc[:, list(source_flags)].max(axis=1)
    return flags


def build_morphology_features(
    *,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
    omr: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build morphology and renal reserve context from patients and OMR."""
    base = admissions.loc[:, ["subject_id", "hadm_id", "dischtime"]].copy()
    patient_frame = patients.loc[:, ["subject_id", "gender", "anchor_age"]].copy()
    patient_frame = patient_frame.rename(columns={"gender": "sex_context", "anchor_age": "age_context"})
    base = base.merge(patient_frame, how="left", on="subject_id", validate="many_to_one")
    base["age_provenance"] = base["age_context"].notna().map(
        {True: AGE_PROVENANCE_OBSERVED, False: PROVENANCE_UNAVAILABLE}
    )
    base["sex_provenance"] = base["sex_context"].notna().map(
        {True: SEX_PROVENANCE_OBSERVED, False: PROVENANCE_UNAVAILABLE}
    )

    omr_features = select_baseline_omr_features(admissions=admissions, omr=omr)
    base = base.merge(omr_features, how="left", on=["subject_id", "hadm_id"], validate="one_to_one")
    return base.drop(columns="dischtime")


def select_baseline_omr_features(
    *,
    admissions: pd.DataFrame,
    omr: pd.DataFrame | None,
) -> pd.DataFrame:
    """Select the most recent OMR morphology values on or before the encounter end.

    TODO(ml-pivot): this helper is intentionally discharge-capped because it feeds the current
    rules-era saved artifact. Future review-time feature builders must replace it with a
    review-time-safe selector instead of reusing it implicitly.
    """
    base = admissions.loc[:, ["subject_id", "hadm_id", "dischtime"]].copy()
    base["weight_kg"] = pd.NA
    base["weight_kg_provenance"] = PROVENANCE_UNAVAILABLE
    base["weight_kg_unavailable_reason"] = "no_omr_weight_on_or_before_discharge"
    base["bmi"] = pd.NA
    base["bmi_provenance"] = PROVENANCE_UNAVAILABLE
    base["bmi_unavailable_reason"] = "no_omr_bmi_or_height_weight_on_or_before_discharge"
    base["height_inches_context"] = pd.NA

    if omr is None or omr.empty:
        return base.drop(columns="dischtime")

    omr_frame = omr.loc[:, ["subject_id", "chartdate", "result_name", "result_value"]].copy()
    omr_frame["chartdate"] = pd.to_datetime(omr_frame["chartdate"], errors="coerce")
    omr_frame["result_value_numeric"] = pd.to_numeric(omr_frame["result_value"], errors="coerce")
    omr_frame = omr_frame.dropna(subset=["subject_id", "chartdate", "result_name", "result_value_numeric"])

    for result_name, target_column, transform in [
        ("Weight (Lbs)", "weight_kg", lambda value: round(float(value) * 0.45359237, 3)),
        ("BMI (kg/m2)", "bmi", lambda value: round(float(value), 3)),
        ("Height (Inches)", "height_inches_context", lambda value: round(float(value), 3)),
    ]:
        values = select_latest_omr_value_for_result(
            admissions=admissions,
            omr_frame=omr_frame,
            result_name=result_name,
        )
        if values.empty:
            continue
        values = values.rename(columns={"result_value_numeric": target_column})
        values[target_column] = values[target_column].map(transform)
        base = base.merge(
            values.loc[:, ["subject_id", "hadm_id", target_column]],
            how="left",
            on=["subject_id", "hadm_id"],
            suffixes=("", "_new"),
        )
        if f"{target_column}_new" in base.columns:
            base[target_column] = base[f"{target_column}_new"].combine_first(base[target_column])
            base = base.drop(columns=f"{target_column}_new")

    weight_mask = base["weight_kg"].notna()
    base.loc[weight_mask, "weight_kg_provenance"] = WEIGHT_PROVENANCE_OBSERVED
    base.loc[weight_mask, "weight_kg_unavailable_reason"] = pd.NA

    bmi_observed_mask = base["bmi"].notna()
    base.loc[bmi_observed_mask, "bmi_provenance"] = BMI_PROVENANCE_OBSERVED
    base.loc[bmi_observed_mask, "bmi_unavailable_reason"] = pd.NA

    bmi_derived_mask = (
        base["bmi"].isna()
        & base["weight_kg"].notna()
        & base["height_inches_context"].notna()
        & (pd.to_numeric(base["height_inches_context"], errors="coerce") > 0)
    )
    height_m = pd.to_numeric(base["height_inches_context"], errors="coerce") * 0.0254
    base.loc[bmi_derived_mask, "bmi"] = (
        pd.to_numeric(base.loc[bmi_derived_mask, "weight_kg"], errors="coerce")
        / (height_m.loc[bmi_derived_mask] ** 2)
    ).round(3)
    base.loc[bmi_derived_mask, "bmi_provenance"] = BMI_PROVENANCE_DERIVED
    base.loc[bmi_derived_mask, "bmi_unavailable_reason"] = pd.NA
    return base.drop(columns="dischtime")


def select_latest_omr_value_for_result(
    *,
    admissions: pd.DataFrame,
    omr_frame: pd.DataFrame,
    result_name: str,
) -> pd.DataFrame:
    """Select the latest OMR result at or before hospital discharge for each admission."""
    filtered = omr_frame.loc[omr_frame["result_name"] == result_name].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "result_value_numeric"])

    candidates = admissions.loc[:, ["subject_id", "hadm_id", "dischtime"]].copy().merge(
        filtered,
        how="left",
        on="subject_id",
    )
    candidates["dischtime"] = pd.to_datetime(candidates["dischtime"], errors="coerce")
    candidates = candidates.loc[candidates["chartdate"] <= candidates["dischtime"]].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "result_value_numeric"])

    candidates = candidates.sort_values(["subject_id", "hadm_id", "chartdate"])
    latest = candidates.groupby(["subject_id", "hadm_id"], as_index=False).tail(1)
    return latest.loc[:, ["subject_id", "hadm_id", "result_value_numeric"]]


def build_dynamic_lab_features(
    *,
    morphology_features: pd.DataFrame,
    labevents: pd.DataFrame,
    creatinine_item_ids: tuple[int, ...],
    potassium_ids: tuple[int, ...],
    sodium_ids: tuple[int, ...],
    creatinine_threshold: float,
) -> pd.DataFrame:
    """Build admission-level lab summaries and dynamic signals.

    TODO(ml-pivot): these summaries intentionally collapse all observed admission labs for the
    current rules baseline. Future ML feature windows must be rebuilt with explicit review-time
    constraints instead of reusing this helper directly.
    """
    base = morphology_features.copy()
    creatinine = summarize_lab_family(labevents, item_ids=creatinine_item_ids, value_name="creatinine")
    potassium = summarize_lab_family(labevents, item_ids=potassium_ids, value_name="potassium")
    sodium = summarize_lab_family(labevents, item_ids=sodium_ids, value_name="sodium")

    context = base.merge(creatinine, how="left", on="hadm_id", validate="one_to_one")
    context = context.merge(potassium, how="left", on="hadm_id", validate="one_to_one")
    context = context.merge(sodium, how="left", on="hadm_id", validate="one_to_one")

    context["renal_risk_flag"] = (
        pd.to_numeric(context["creatinine_max"], errors="coerce") >= creatinine_threshold
    ).fillna(False).astype(int)

    context["egfr_ml_min_1_73m2"] = context.apply(
        lambda row: compute_egfr_ckd_epi_2021(
            age=row.get("age_context"),
            sex=row.get("sex_context"),
            creatinine_mg_dl=row.get("creatinine_first"),
        ),
        axis=1,
    )
    context["egfr_provenance"] = context["egfr_ml_min_1_73m2"].notna().map(
        {True: EGFR_PROVENANCE_DERIVED, False: PROVENANCE_UNAVAILABLE}
    )
    context["egfr_unavailable_reason"] = context.apply(
        lambda row: derive_egfr_unavailable_reason(row), axis=1
    )
    context["cockcroft_gault_ml_min"] = context.apply(
        lambda row: compute_cockcroft_gault(
            age=row.get("age_context"),
            sex=row.get("sex_context"),
            weight_kg=row.get("weight_kg"),
            creatinine_mg_dl=row.get("creatinine_first"),
        ),
        axis=1,
    )
    context["cockcroft_gault_provenance"] = context["cockcroft_gault_ml_min"].notna().map(
        {
            True: COCKCROFT_GAULT_PROVENANCE_DERIVED,
            False: PROVENANCE_UNAVAILABLE,
        }
    )
    context["cockcroft_gault_unavailable_reason"] = context.apply(
        lambda row: derive_cockcroft_gault_unavailable_reason(row), axis=1
    )
    return context.drop(
        columns=[
            "age_context",
            "sex_context",
            "height_inches_context",
            "age_provenance",
            "sex_provenance",
            "weight_kg",
            "weight_kg_provenance",
            "weight_kg_unavailable_reason",
            "bmi",
            "bmi_provenance",
            "bmi_unavailable_reason",
        ],
        errors="ignore",
    )


def summarize_lab_family(
    labevents: pd.DataFrame,
    *,
    item_ids: tuple[int, ...],
    value_name: str,
) -> pd.DataFrame:
    """Summarize one lab family to admission-level first/last/min/max/mean and provenance.

    Legacy-active note: this remains correct for the current saved artifact, but it is not a
    substitute for future point-in-time lab windows.
    """
    labs = labevents.loc[:, ["hadm_id", "itemid", "charttime", "valuenum"]].copy()
    labs["itemid"] = pd.to_numeric(labs["itemid"], errors="coerce")
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    labs = labs.dropna(subset=["hadm_id", "itemid", "valuenum"])
    labs = labs.loc[labs["itemid"].isin(item_ids)].copy()

    if labs.empty:
        return pd.DataFrame(
            columns=[
                "hadm_id",
                f"{value_name}_first",
                f"{value_name}_last",
                f"{value_name}_min",
                f"{value_name}_max",
                f"{value_name}_mean",
                f"{value_name}_delta",
                f"{value_name}_trend_direction",
                f"{value_name}_provenance",
                f"{value_name}_missingness",
            ]
        )

    labs = labs.sort_values(["hadm_id", "charttime", "itemid"])
    grouped = labs.groupby("hadm_id", as_index=False)
    features = grouped.agg(
        **{
            f"{value_name}_first": ("valuenum", "first"),
            f"{value_name}_last": ("valuenum", "last"),
            f"{value_name}_min": ("valuenum", "min"),
            f"{value_name}_max": ("valuenum", "max"),
            f"{value_name}_mean": ("valuenum", "mean"),
        }
    )
    features[f"{value_name}_mean"] = features[f"{value_name}_mean"].round(3)
    features[f"{value_name}_delta"] = (
        pd.to_numeric(features[f"{value_name}_last"], errors="coerce")
        - pd.to_numeric(features[f"{value_name}_first"], errors="coerce")
    ).round(3)
    features[f"{value_name}_trend_direction"] = features[f"{value_name}_delta"].map(
        lambda value: categorize_delta_trend(value)
    )
    features[f"{value_name}_provenance"] = LAB_PROVENANCE_OBSERVED
    features[f"{value_name}_missingness"] = MISSINGNESS_OBSERVED
    return features


def build_dynamic_ed_signal_features(
    admissions: pd.DataFrame,
    edstays: pd.DataFrame | None,
    triage: pd.DataFrame | None,
    vitalsign: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build ED-derived hemodynamic and pain summaries linked to admissions.

    These ED summaries are still admission-linked compatibility features. Future ML work should
    keep their provenance but rebuild them at review-time-safe windows when needed.
    """
    base = admissions.loc[:, ["subject_id", "hadm_id"]].drop_duplicates().copy()
    if edstays is None or edstays.empty:
        return _empty_context_frame(["subject_id", "hadm_id"], ed_signal_columns())

    ed_admissions = edstays.loc[:, ["subject_id", "hadm_id", "stay_id", "intime"]].copy()
    ed_admissions["intime"] = pd.to_datetime(ed_admissions["intime"], errors="coerce")
    ed_events = []

    if triage is not None and not triage.empty:
        triage_frame = triage.loc[
            :,
            ["subject_id", "stay_id", "temperature", "heartrate", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"],
        ].copy()
        triage_frame = triage_frame.merge(
            ed_admissions,
            how="left",
            on=["subject_id", "stay_id"],
            validate="many_to_one",
        )
        triage_frame["charttime"] = triage_frame["intime"]
        ed_events.append(triage_frame)

    if vitalsign is not None and not vitalsign.empty:
        vitals_frame = vitalsign.loc[
            :,
            ["subject_id", "stay_id", "charttime", "heartrate", "sbp", "dbp", "pain"],
        ].copy()
        vitals_frame = vitals_frame.merge(
            ed_admissions.loc[:, ["subject_id", "hadm_id", "stay_id"]],
            how="left",
            on=["subject_id", "stay_id"],
            validate="many_to_one",
        )
        vitals_frame["charttime"] = pd.to_datetime(vitals_frame["charttime"], errors="coerce")
        ed_events.append(vitals_frame)

    if not ed_events:
        return _empty_context_frame(["subject_id", "hadm_id"], ed_signal_columns())

    combined = pd.concat(ed_events, ignore_index=True, sort=False)
    combined = combined.loc[combined["hadm_id"].notna()].copy()
    if combined.empty:
        return _empty_context_frame(["subject_id", "hadm_id"], ed_signal_columns())

    for column in ["heartrate", "sbp", "dbp", "pain", "acuity"]:
        if column in combined.columns:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")

    summary = base.merge(
        summarize_ed_numeric_signal(combined, "sbp", "sbp"),
        how="left",
        on=["subject_id", "hadm_id"],
        validate="one_to_one",
    )
    summary = summary.merge(
        summarize_ed_numeric_signal(combined, "dbp", "dbp"),
        how="left",
        on=["subject_id", "hadm_id"],
        validate="one_to_one",
    )
    summary = summary.merge(
        summarize_ed_numeric_signal(combined, "heartrate", "heart_rate"),
        how="left",
        on=["subject_id", "hadm_id"],
        validate="one_to_one",
    )
    summary = summary.merge(
        summarize_ed_numeric_signal(combined, "pain", "pain"),
        how="left",
        on=["subject_id", "hadm_id"],
        validate="one_to_one",
    )

    triage_info = (
        combined.sort_values(["subject_id", "hadm_id", "charttime"])
        .groupby(["subject_id", "hadm_id"], as_index=False)
        .first()
        .loc[:, ["subject_id", "hadm_id", "acuity", "chiefcomplaint"]]
        .rename(columns={"acuity": "ed_triage_acuity", "chiefcomplaint": "ed_chiefcomplaint"})
    )
    summary = summary.merge(triage_info, how="left", on=["subject_id", "hadm_id"], validate="one_to_one")
    summary["blood_pressure_provenance"] = summary[["sbp_mean", "dbp_mean"]].notna().any(axis=1).map(
        {True: ED_SIGNAL_PROVENANCE_OBSERVED, False: PROVENANCE_UNAVAILABLE}
    )
    summary["heart_rate_provenance"] = summary["heart_rate_mean"].notna().map(
        {True: ED_SIGNAL_PROVENANCE_OBSERVED, False: PROVENANCE_UNAVAILABLE}
    )
    summary["pain_provenance"] = summary["pain_mean"].notna().map(
        {True: ED_SIGNAL_PROVENANCE_OBSERVED, False: PROVENANCE_UNAVAILABLE}
    )
    summary["ed_triage_provenance"] = summary["ed_triage_acuity"].notna().map(
        {True: ED_TRIAGE_PROVENANCE_OBSERVED, False: PROVENANCE_UNAVAILABLE}
    )
    return summary


def summarize_ed_numeric_signal(
    dataframe: pd.DataFrame,
    source_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Summarize one ED numeric signal to admission-level min/max/mean."""
    signal = dataframe.loc[:, ["subject_id", "hadm_id", source_column]].copy()
    signal[source_column] = pd.to_numeric(signal[source_column], errors="coerce")
    signal = signal.dropna(subset=[source_column])
    if signal.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "hadm_id",
                f"{prefix}_min",
                f"{prefix}_max",
                f"{prefix}_mean",
            ]
        )

    summary = signal.groupby(["subject_id", "hadm_id"], as_index=False).agg(
        **{
            f"{prefix}_min": (source_column, "min"),
            f"{prefix}_max": (source_column, "max"),
            f"{prefix}_mean": (source_column, "mean"),
        }
    )
    summary[f"{prefix}_mean"] = summary[f"{prefix}_mean"].round(3)
    return summary


def compute_egfr_ckd_epi_2021(
    *,
    age: object,
    sex: object,
    creatinine_mg_dl: object,
) -> float | None:
    """Estimate eGFR using the race-free CKD-EPI 2021 equation."""
    age_value = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    creatinine_value = pd.to_numeric(pd.Series([creatinine_mg_dl]), errors="coerce").iloc[0]
    if pd.isna(age_value) or pd.isna(creatinine_value) or creatinine_value <= 0:
        return None

    sex_normalized = str(sex).strip().upper()
    is_female = sex_normalized == "F"
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    ratio = creatinine_value / kappa
    estimate = (
        142
        * min(ratio, 1) ** alpha
        * max(ratio, 1) ** -1.200
        * 0.9938 ** float(age_value)
        * (1.012 if is_female else 1.0)
    )
    return round(float(estimate), 3)


def compute_cockcroft_gault(
    *,
    age: object,
    sex: object,
    weight_kg: object,
    creatinine_mg_dl: object,
) -> float | None:
    """Estimate creatinine clearance using Cockcroft-Gault when weight is available."""
    age_value = pd.to_numeric(pd.Series([age]), errors="coerce").iloc[0]
    weight_value = pd.to_numeric(pd.Series([weight_kg]), errors="coerce").iloc[0]
    creatinine_value = pd.to_numeric(pd.Series([creatinine_mg_dl]), errors="coerce").iloc[0]
    if (
        pd.isna(age_value)
        or pd.isna(weight_value)
        or pd.isna(creatinine_value)
        or weight_value <= 0
        or creatinine_value <= 0
    ):
        return None

    estimate = ((140 - float(age_value)) * float(weight_value)) / (72 * float(creatinine_value))
    if str(sex).strip().upper() == "F":
        estimate *= 0.85
    return round(float(estimate), 3)


def derive_egfr_unavailable_reason(row: pd.Series) -> str | pd.NA:
    """Explain why eGFR could not be estimated from observed inputs."""
    if pd.notna(row.get("egfr_ml_min_1_73m2")):
        return pd.NA
    if pd.isna(row.get("creatinine_first")):
        return "missing_creatinine_first"
    if pd.isna(row.get("age_context")):
        return "missing_age"
    if pd.isna(row.get("sex_context")):
        return "missing_sex"
    return "invalid_egfr_inputs"


def derive_cockcroft_gault_unavailable_reason(row: pd.Series) -> str | pd.NA:
    """Explain why Cockcroft-Gault could not be estimated."""
    if pd.notna(row.get("cockcroft_gault_ml_min")):
        return pd.NA
    if pd.isna(row.get("weight_kg")):
        return "missing_weight"
    if pd.isna(row.get("creatinine_first")):
        return "missing_creatinine_first"
    if pd.isna(row.get("age_context")):
        return "missing_age"
    if pd.isna(row.get("sex_context")):
        return "missing_sex"
    return "invalid_cockcroft_gault_inputs"


def categorize_delta_trend(value: object) -> str:
    """Map a numeric delta into a simple trend label."""
    numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric_value):
        return "unavailable"
    if numeric_value >= 0.3:
        return "rising"
    if numeric_value <= -0.3:
        return "falling"
    return "stable"


def fill_context_defaults(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Fill explicit defaults so 0 can be distinguished from unavailable provenance.

    TODO(ml-pivot): this fill step belongs to the current saved-artifact path. Future feature-store
    builders should preserve raw missingness signals until the contract-governed export step.
    """
    output = dataframe.copy()
    int_columns = [
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
    ]
    for column in int_columns:
        if column in output.columns:
            output[column] = output[column].fillna(0).astype(int)

    provenance_defaults = {
        "diagnosis_context_provenance": PROVENANCE_UNAVAILABLE,
        "diagnosis_context_missingness": PROVENANCE_UNAVAILABLE,
        "ed_diagnosis_context_provenance": PROVENANCE_UNAVAILABLE,
        "ed_diagnosis_context_missingness": PROVENANCE_UNAVAILABLE,
        "blood_pressure_provenance": PROVENANCE_UNAVAILABLE,
        "heart_rate_provenance": PROVENANCE_UNAVAILABLE,
        "pain_provenance": PROVENANCE_UNAVAILABLE,
        "ed_triage_provenance": PROVENANCE_UNAVAILABLE,
    }
    for column, default_value in provenance_defaults.items():
        if column in output.columns:
            output[column] = output[column].fillna(default_value)
    return output


def diagnosis_context_columns() -> list[str]:
    return [
        *DIAGNOSIS_ICD_PREFIXES.keys(),
        *DIAGNOSIS_RISK_CATEGORY_GROUPS.keys(),
        "diagnosis_context_provenance",
        "diagnosis_context_missingness",
    ]


def ed_diagnosis_context_columns() -> list[str]:
    return [
        *(f"ed_{column}" for column in DIAGNOSIS_ICD_PREFIXES.keys()),
        *(f"ed_{column}" for column in DIAGNOSIS_RISK_CATEGORY_GROUPS.keys()),
        "ed_diagnosis_context_provenance",
        "ed_diagnosis_context_missingness",
    ]


def ed_signal_columns() -> list[str]:
    return [
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
        "ed_triage_acuity",
        "ed_chiefcomplaint",
        "blood_pressure_provenance",
        "heart_rate_provenance",
        "pain_provenance",
        "ed_triage_provenance",
    ]


def _empty_context_frame(key_columns: list[str], feature_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=[*key_columns, *feature_columns])
