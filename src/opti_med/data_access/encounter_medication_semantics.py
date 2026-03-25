"""First-pass class and burden artifacts at the encounter-medication review-time grain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_FIRST_SCOPE_COLUMNS,
    ENCOUNTER_MEDICATION_FIRST_SCOPE_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_BURDEN_COLUMNS,
    ENCOUNTER_MEDICATION_BURDEN_CONTRACT_VERSION,
    ENCOUNTER_MEDICATION_SEMANTICS_COLUMNS,
    ENCOUNTER_MEDICATION_SEMANTICS_CONTRACT_VERSION,
    validate_encounter_index_artifact,
    validate_encounter_medication_burden_artifact,
    validate_encounter_medication_first_scope_artifact,
    validate_encounter_medication_semantics_artifact,
    validate_encounter_medication_state_artifact,
    validate_medication_events_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.encounter_medication_state import (
    apply_medication_rxnorm_mapping_to_events,
)
from opti_med.data_access.medication_snapshot import (
    coalesce_event_start,
    deduplicate_active_medication_group,
    filter_active_medication_events,
)
from opti_med.data_access.provenance import dumps_json
from opti_med.features.mappings.medications import HIGH_RISK_MEDICATION_KEYWORDS
from opti_med.medication_semantics import (
    FIRST_SCOPE_CLASS_SYSTEM,
    FIRST_SCOPE_CLASS_SYSTEM_VERSION,
    FIRST_SCOPE_SUPPORTED_CLASSES,
    RxNormBackedMedicationSemanticMapper,
    infer_prn_vs_scheduled,
)
from opti_med.standardized import StandardizedParquetRepository


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
HEURISTIC_FLAG_COLUMNS: dict[str, str] = {
    "benzodiazepine": "benzodiazepine_heuristic_flag",
    "opioid": "opioid_heuristic_flag",
    "anticholinergic": "anticholinergic_heuristic_flag",
    "ppi": "ppi_heuristic_flag",
    "antipsychotic": "antipsychotic_heuristic_flag",
}


@dataclass(frozen=True, slots=True)
class EncounterMedicationScopeArtifactsBuildResult:
    """Built semantics and burden artifacts plus their output paths."""

    encounter_medication_semantics: pd.DataFrame
    encounter_medication_burden: pd.DataFrame
    semantics_output_path: Path
    burden_output_path: Path


class EncounterMedicationScopeArtifactsBuilder:
    """Build first-pass class and burden artifacts from review-time medication state."""

    def __init__(
        self,
        settings: Settings,
        *,
        semantic_mapper: RxNormBackedMedicationSemanticMapper | None = None,
    ) -> None:
        self.settings = settings
        self.repository = StandardizedParquetRepository(settings)
        self.semantic_mapper = semantic_mapper or RxNormBackedMedicationSemanticMapper()

    def build(
        self,
        *,
        encounter_medication_state: pd.DataFrame | None = None,
        medication_events: pd.DataFrame | None = None,
        medication_rxnorm_mapping: pd.DataFrame | None = None,
        encounter_index: pd.DataFrame | None = None,
        labevents: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build the two first-pass review-time artifacts."""
        encounter_medication_state = (
            encounter_medication_state
            if encounter_medication_state is not None
            else self._load_preferred_encounter_medication_state()
        )
        medication_events = (
            medication_events
            if medication_events is not None
            else self.repository.load_analytical_artifact("medication_events")
        )
        medication_rxnorm_mapping = (
            medication_rxnorm_mapping
            if medication_rxnorm_mapping is not None
            else self._load_required_parquet(self.settings.medication_rxnorm_mapping_output_path)
        )
        encounter_index = (
            encounter_index
            if encounter_index is not None
            else self.repository.load_analytical_artifact("encounter_index")
        )
        if labevents is None:
            loaded = self.repository.load_optional_source_table(
                "clinical",
                "labevents",
                columns=["hadm_id", "itemid", "charttime", "valuenum"],
            )
            labevents = loaded.dataframe if loaded else None

        semantics = build_encounter_medication_semantics(
            encounter_medication_state=encounter_medication_state,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
        )
        burden = build_encounter_medication_burden(
            encounter_medication_semantics=semantics,
            medication_events=medication_events,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            encounter_index=encounter_index,
            labevents=labevents,
            settings=self.settings,
        )
        return semantics, burden

    def save(
        self,
        *,
        encounter_medication_semantics: pd.DataFrame,
        encounter_medication_burden: pd.DataFrame,
        semantics_output_path: Path | None = None,
        burden_output_path: Path | None = None,
    ) -> EncounterMedicationScopeArtifactsBuildResult:
        """Persist both artifacts to their standard output paths."""
        validate_encounter_medication_semantics_artifact(encounter_medication_semantics)
        validate_encounter_medication_burden_artifact(encounter_medication_burden)
        semantics_path = (
            semantics_output_path or self.settings.encounter_medication_semantics_output_path
        )
        burden_path = burden_output_path or self.settings.encounter_medication_burden_output_path
        semantics_path.parent.mkdir(parents=True, exist_ok=True)
        burden_path.parent.mkdir(parents=True, exist_ok=True)
        encounter_medication_semantics.to_parquet(semantics_path, index=False)
        encounter_medication_burden.to_parquet(burden_path, index=False)
        return EncounterMedicationScopeArtifactsBuildResult(
            encounter_medication_semantics=encounter_medication_semantics,
            encounter_medication_burden=encounter_medication_burden,
            semantics_output_path=semantics_path,
            burden_output_path=burden_path,
        )

    @staticmethod
    def _load_required_parquet(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Expected analytical artifact at '{path}', but it does not exist.")
        return pd.read_parquet(path)

    def _load_preferred_encounter_medication_state(self) -> pd.DataFrame:
        rxnorm_state_path = self.settings.encounter_medication_state_rxnorm_output_path
        if rxnorm_state_path.exists():
            return self._load_required_parquet(rxnorm_state_path)
        return self._load_required_parquet(self.settings.encounter_medication_state_output_path)


def build_encounter_medication_semantics(
    *,
    encounter_medication_state: pd.DataFrame,
    medication_rxnorm_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Build first-pass ontology-aware class semantics with heuristic anchors."""
    validate_encounter_medication_state_artifact(encounter_medication_state)
    validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
    if encounter_medication_state.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_SEMANTICS_COLUMNS)

    build_run_id = f"encounter-medication-semantics-{uuid4().hex[:12]}"
    semantics = encounter_medication_state.copy()
    semantics["medication_class_standardized"] = semantics["medication_class_standardized"].fillna(
        "unresolved"
    )
    semantics["class_assignment_status"] = semantics.apply(
        _class_assignment_status_from_state_row,
        axis=1,
    )
    semantics["class_system"] = FIRST_SCOPE_CLASS_SYSTEM
    semantics["class_system_version"] = FIRST_SCOPE_CLASS_SYSTEM_VERSION
    semantics["first_scope_supported_class_flag"] = (
        semantics["medication_class_standardized"].isin(FIRST_SCOPE_SUPPORTED_CLASSES)
    ).astype(int)
    heuristics = _legacy_heuristic_flags(
        semantics,
        medication_text_columns=[
            "medication_normalized",
            "medication_raw",
            "medication_standardized",
        ],
    )
    semantics = pd.concat([semantics.reset_index(drop=True), heuristics], axis=1)
    semantics["heuristic_any_supported_class_flag"] = (
        semantics[list(HEURISTIC_FLAG_COLUMNS.values())].sum(axis=1) > 0
    ).astype(int)
    semantics["standardized_vs_heuristic_class_agreement_flag"] = semantics.apply(
        _standardized_vs_heuristic_agreement_flag,
        axis=1,
    )
    semantics["encounter_medication_semantics_build_run_id"] = build_run_id
    semantics["encounter_medication_semantics_contract_version"] = (
        ENCOUNTER_MEDICATION_SEMANTICS_CONTRACT_VERSION
    )
    semantics = semantics.loc[:, ENCOUNTER_MEDICATION_SEMANTICS_COLUMNS].copy()
    semantics = semantics.sort_values(
        ["subject_id", "encounter_id", "review_timestamp", "medication_standardized"],
        na_position="last",
    ).reset_index(drop=True)
    validate_encounter_medication_semantics_artifact(semantics)
    return semantics


def build_encounter_medication_burden(
    *,
    encounter_medication_semantics: pd.DataFrame,
    medication_events: pd.DataFrame,
    medication_rxnorm_mapping: pd.DataFrame,
    encounter_index: pd.DataFrame,
    labevents: pd.DataFrame | None,
    settings: Settings,
) -> pd.DataFrame:
    """Build first-pass burden features from the review-time medication semantics layer."""
    validate_encounter_medication_semantics_artifact(encounter_medication_semantics)
    validate_medication_events_artifact(medication_events)
    validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
    validate_encounter_index_artifact(encounter_index)
    if encounter_medication_semantics.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_BURDEN_COLUMNS)

    build_run_id = f"encounter-medication-burden-{uuid4().hex[:12]}"
    events = medication_events.copy()
    events["_medication_candidate_key"] = events.apply(
        _medication_candidate_key_from_event,
        axis=1,
    )
    events = events.loc[events["_medication_candidate_key"].notna()].copy()
    events = apply_medication_rxnorm_mapping_to_events(
        medication_events=events,
        medication_rxnorm_mapping=medication_rxnorm_mapping,
    )
    events["_event_start"] = events.apply(coalesce_event_start, axis=1)
    events["_event_start"] = pd.to_datetime(events["_event_start"], errors="coerce")
    events["_dose_value_numeric"] = pd.to_numeric(events["dose_value"], errors="coerce")

    encounter_context = (
        encounter_index.loc[:, ["encounter_id", "hadm_id", "sex", "age_proxy"]]
        .drop_duplicates(subset=["encounter_id"])
        .copy()
    )
    context_by_encounter = {
        str(row["encounter_id"]): row
        for row in encounter_context.to_dict(orient="records")
    }

    creatinine_frame = _prepare_creatinine_frame(labevents=labevents, settings=settings)
    creatinine_by_hadm: dict[int, pd.DataFrame] = {
        int(hadm_id): group.sort_values("charttime", na_position="last").reset_index(drop=True)
        for hadm_id, group in creatinine_frame.groupby("hadm_id", sort=False)
    }
    event_groups = {
        (str(encounter_id), str(medication_key)): group.copy()
        for (encounter_id, medication_key), group in events.groupby(
            ["encounter_id", "_medication_standardized_key"],
            dropna=False,
            sort=False,
        )
        if medication_key is not None and not pd.isna(medication_key)
    }
    renal_context_cache: dict[tuple[int | None, str], dict[str, object]] = {}

    burden_rows: list[dict[str, object]] = []
    for row in encounter_medication_semantics.to_dict(orient="records"):
        encounter_id = str(row["encounter_id"])
        medication_standardized = str(row["medication_standardized"])
        review_timestamp = pd.to_datetime(row["review_timestamp"], errors="coerce")
        event_group = event_groups.get((encounter_id, medication_standardized), pd.DataFrame())
        representative_event = _representative_event_for_row(
            event_group=event_group,
            selected_medication_event_id=row.get("selected_medication_event_id"),
            review_timestamp=review_timestamp,
        )
        schedule_value, schedule_status = _scheduled_vs_prn_for_group(event_group)
        duration_hours, duration_inferable_flag, duration_lower_bound_flag = (
            _duration_before_review_for_group(
                event_group=event_group,
                review_timestamp=review_timestamp,
            )
        )
        context_row = context_by_encounter.get(encounter_id, {})
        renal_context = _renal_context_for_row(
            hadm_id=_int_or_none(row.get("hadm_id")),
            review_timestamp=review_timestamp,
            age_proxy=context_row.get("age_proxy"),
            sex=context_row.get("sex"),
            creatinine_by_hadm=creatinine_by_hadm,
            cache=renal_context_cache,
        )
        dose_value = representative_event.get("dose_value")
        dose_unit = representative_event.get("dose_unit")
        route = representative_event.get("route") or row.get("route")
        frequency = representative_event.get("frequency") or row.get("frequency")
        start_context = _medication_start_context(
            continued_from_home=row.get("continued_from_home_inferred"),
            newly_started=row.get("newly_started_during_encounter_inferred"),
        )
        mme_value, mme_status, mme_required_fields_flag, mme_missing_fields_json = (
            _opioid_mme_placeholder(
                medication_class_standardized=row.get("medication_class_standardized"),
                ingredient_standardized=row.get("ingredient_standardized"),
                dose_value=dose_value,
                dose_unit=dose_unit,
                route=route,
                frequency=frequency,
            )
        )
        (
            renal_dose_mismatch_value,
            renal_dose_mismatch_status,
            renal_dose_fields_available_flag,
            renal_dose_mismatch_ready_flag,
        ) = _renal_dose_mismatch_placeholder(
            active_at_review_flag=row.get("active_at_review_flag"),
            dose_value=dose_value,
            dose_unit=dose_unit,
            route=route,
            renal_context_available_flag=renal_context["renal_context_available_flag"],
        )

        burden_rows.append(
            {
                "subject_id": row["subject_id"],
                "encounter_id": encounter_id,
                "hadm_id": row.get("hadm_id"),
                "stay_id": row.get("stay_id"),
                "review_timestamp": row["review_timestamp"],
                "age_proxy": row.get("age_proxy"),
                "age_group": row.get("age_group"),
                "medication_standardized": medication_standardized,
                "medication_class_standardized": row.get("medication_class_standardized")
                or "unresolved",
                "medication_status_at_review": row.get("medication_status_at_review"),
                "active_at_review_flag": int(row.get("active_at_review_flag") or 0),
                "continued_from_home_inferred": int(
                    row.get("continued_from_home_inferred") or 0
                ),
                "newly_started_during_encounter_inferred": int(
                    row.get("newly_started_during_encounter_inferred") or 0
                ),
                "medication_start_context": start_context,
                "duration_before_review_hours": duration_hours,
                "duration_before_review_inferable_flag": duration_inferable_flag,
                "duration_before_review_lower_bound_flag": duration_lower_bound_flag,
                "scheduled_vs_prn": schedule_value,
                "scheduled_vs_prn_inference_status": schedule_status,
                "dose_value": dose_value,
                "dose_unit": dose_unit,
                "route": route,
                "frequency": frequency,
                "exact_current_medication_count": 0,
                "current_benzodiazepine_count": 0,
                "current_opioid_count": 0,
                "current_anticholinergic_count": 0,
                "current_ppi_count": 0,
                "current_antipsychotic_count": 0,
                "current_supported_class_count": 0,
                "row_same_class_current_count": 0,
                "same_class_duplicate_therapy_flag": 0,
                "same_class_duplicate_therapy_signal_count": 0,
                "opioid_mme": mme_value,
                "opioid_mme_status": mme_status,
                "opioid_mme_required_fields_present_flag": mme_required_fields_flag,
                "opioid_mme_missing_fields_json": mme_missing_fields_json,
                "renal_dose_mismatch": renal_dose_mismatch_value,
                "renal_dose_mismatch_status": renal_dose_mismatch_status,
                "renal_dose_fields_available_flag": renal_dose_fields_available_flag,
                "renal_context_available_flag": renal_context[
                    "renal_context_available_flag"
                ],
                "renal_context_age_available_flag": renal_context[
                    "renal_context_age_available_flag"
                ],
                "renal_context_sex_available_flag": renal_context[
                    "renal_context_sex_available_flag"
                ],
                "renal_context_latest_creatinine_time": renal_context[
                    "renal_context_latest_creatinine_time"
                ],
                "renal_context_latest_creatinine_value": renal_context[
                    "renal_context_latest_creatinine_value"
                ],
                "renal_dose_mismatch_ready_flag": renal_dose_mismatch_ready_flag,
            }
        )

    burden = pd.DataFrame(burden_rows)
    burden = _attach_current_burden_counts(burden)
    burden["encounter_medication_burden_build_run_id"] = build_run_id
    burden["encounter_medication_burden_contract_version"] = (
        ENCOUNTER_MEDICATION_BURDEN_CONTRACT_VERSION
    )
    burden = burden.loc[:, ENCOUNTER_MEDICATION_BURDEN_COLUMNS].copy()
    burden = burden.sort_values(
        ["subject_id", "encounter_id", "review_timestamp", "medication_standardized"],
        na_position="last",
    ).reset_index(drop=True)
    validate_encounter_medication_burden_artifact(burden)
    return burden


def build_encounter_medication_first_scope(
    *,
    encounter_medication_semantics: pd.DataFrame,
    encounter_medication_burden: pd.DataFrame,
) -> pd.DataFrame:
    """Build the canonical supported-class subset from full 65+ semantics and burden artifacts."""
    validate_encounter_medication_semantics_artifact(encounter_medication_semantics)
    validate_encounter_medication_burden_artifact(encounter_medication_burden)
    if encounter_medication_semantics.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_FIRST_SCOPE_COLUMNS)

    key_columns = [
        "subject_id",
        "encounter_id",
        "medication_standardized",
        "review_timestamp",
    ]
    burden_only_columns = [
        column_name
        for column_name in ENCOUNTER_MEDICATION_BURDEN_COLUMNS
        if column_name not in ENCOUNTER_MEDICATION_SEMANTICS_COLUMNS
    ]
    merged = encounter_medication_semantics.merge(
        encounter_medication_burden.loc[:, key_columns + burden_only_columns].copy(),
        how="outer",
        on=key_columns,
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        unmatched_rows = int((merged["_merge"] != "both").sum())
        raise DataLoadError(
            "Encounter-medication first-scope build requires one-to-one aligned full semantics "
            f"and burden artifacts, but found {unmatched_rows:,} unmatched rows."
        )
    merged = merged.drop(columns=["_merge"])
    first_scope = merged.loc[
        merged["medication_class_standardized"].isin(FIRST_SCOPE_SUPPORTED_CLASSES)
    ].copy()
    build_run_id = f"encounter-medication-first-scope-{uuid4().hex[:12]}"
    first_scope["encounter_medication_first_scope_build_run_id"] = build_run_id
    first_scope["encounter_medication_first_scope_contract_version"] = (
        ENCOUNTER_MEDICATION_FIRST_SCOPE_CONTRACT_VERSION
    )
    first_scope = first_scope.loc[:, ENCOUNTER_MEDICATION_FIRST_SCOPE_COLUMNS].copy()
    first_scope = first_scope.sort_values(
        ["subject_id", "encounter_id", "review_timestamp", "medication_standardized"],
        na_position="last",
    ).reset_index(drop=True)
    validate_encounter_medication_first_scope_artifact(first_scope)
    return first_scope


def calculate_first_scope_qc_metrics(
    *,
    encounter_medication_semantics: pd.DataFrame,
    encounter_medication_burden: pd.DataFrame,
) -> dict[str, object]:
    """Calculate QA metrics for the first-pass class and burden artifacts."""
    validate_encounter_medication_semantics_artifact(encounter_medication_semantics)
    validate_encounter_medication_burden_artifact(encounter_medication_burden)
    active_semantics = encounter_medication_semantics.loc[
        encounter_medication_semantics["active_at_review_flag"] == 1
    ].copy()
    total_rows = int(len(encounter_medication_semantics))
    unresolved_rows = int(
        (encounter_medication_semantics["medication_class_standardized"] == "unresolved").sum()
    )
    class_counts = (
        active_semantics["medication_class_standardized"]
        .fillna("unresolved")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    comparison_by_class: dict[str, dict[str, int]] = {}
    for class_name, flag_column in HEURISTIC_FLAG_COLUMNS.items():
        standardized_positive = (
            encounter_medication_semantics["medication_class_standardized"] == class_name
        )
        heuristic_positive = (
            pd.to_numeric(
                encounter_medication_semantics[flag_column],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            == 1
        )
        comparison_by_class[class_name] = {
            "standardized_positive": int(standardized_positive.sum()),
            "heuristic_positive": int(heuristic_positive.sum()),
            "overlap": int((standardized_positive & heuristic_positive).sum()),
            "standardized_only": int((standardized_positive & ~heuristic_positive).sum()),
            "heuristic_only": int((~standardized_positive & heuristic_positive).sum()),
        }
    duplicate_rows = int(
        (encounter_medication_burden["same_class_duplicate_therapy_flag"] == 1).sum()
    )
    duplicate_groups = int(
        encounter_medication_burden.loc[
            encounter_medication_burden["same_class_duplicate_therapy_flag"] == 1,
            ["encounter_id", "medication_class_standardized"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    return {
        "row_count": total_rows,
        "active_row_count": int(len(active_semantics)),
        "class_counts_active": {str(key): int(value) for key, value in class_counts.items()},
        "unresolved_class_rows": unresolved_rows,
        "unresolved_class_rate": (unresolved_rows / total_rows) if total_rows else 0.0,
        "heuristic_comparison_by_class": comparison_by_class,
        "exact_duplicate_therapy_signal_rows": duplicate_rows,
        "exact_duplicate_therapy_signal_groups": duplicate_groups,
    }


def calculate_first_scope_subset_metrics(
    *,
    encounter_medication_first_scope: pd.DataFrame,
) -> dict[str, object]:
    """Calculate compact metrics for the canonical supported-class subset artifact."""
    validate_encounter_medication_first_scope_artifact(encounter_medication_first_scope)
    active_rows = encounter_medication_first_scope.loc[
        encounter_medication_first_scope["active_at_review_flag"] == 1
    ].copy()
    return {
        "row_count": int(len(encounter_medication_first_scope)),
        "active_row_count": int(len(active_rows)),
        "unique_subject_count": int(
            encounter_medication_first_scope["subject_id"].nunique(dropna=True)
        ),
        "unique_encounter_count": int(
            encounter_medication_first_scope["encounter_id"].nunique(dropna=True)
        ),
        "row_counts_by_class": {
            str(key): int(value)
            for key, value in encounter_medication_first_scope["medication_class_standardized"]
            .fillna("unresolved")
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
            .items()
        },
        "active_row_counts_by_class": {
            str(key): int(value)
            for key, value in active_rows["medication_class_standardized"]
            .fillna("unresolved")
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
            .items()
        },
    }


def summarize_first_scope_artifacts(
    *,
    encounter_medication_semantics: pd.DataFrame,
    encounter_medication_burden: pd.DataFrame,
) -> list[str]:
    """Return compact QA summary lines for the first-pass class and burden artifacts."""
    metrics = calculate_first_scope_qc_metrics(
        encounter_medication_semantics=encounter_medication_semantics,
        encounter_medication_burden=encounter_medication_burden,
    )
    class_counts = ", ".join(
        f"{class_name}={count:,}"
        for class_name, count in metrics["class_counts_active"].items()
    ) or "none"
    lines = [
        f"rows={metrics['row_count']:,}, active_rows={metrics['active_row_count']:,}",
        f"class_coverage_active={class_counts}",
        (
            "unresolved_class_rate="
            f"{metrics['unresolved_class_rows']:,}/{metrics['row_count']:,} "
            f"({metrics['unresolved_class_rate']:.2%})"
        ),
    ]
    for class_name in FIRST_SCOPE_SUPPORTED_CLASSES:
        comparison = metrics["heuristic_comparison_by_class"][class_name]
        lines.append(
            "heuristic_comparison "
            f"{class_name}: standardized={comparison['standardized_positive']:,}, "
            f"heuristic={comparison['heuristic_positive']:,}, "
            f"overlap={comparison['overlap']:,}, "
            f"standardized_only={comparison['standardized_only']:,}, "
            f"heuristic_only={comparison['heuristic_only']:,}"
        )
    lines.append(
        "exact_duplicate_therapy_signals="
        f"rows:{metrics['exact_duplicate_therapy_signal_rows']:,}, "
        f"groups:{metrics['exact_duplicate_therapy_signal_groups']:,}"
    )
    return lines


def summarize_encounter_medication_first_scope(
    *,
    encounter_medication_first_scope: pd.DataFrame,
) -> list[str]:
    """Return compact summary lines for the canonical supported-class subset artifact."""
    metrics = calculate_first_scope_subset_metrics(
        encounter_medication_first_scope=encounter_medication_first_scope,
    )
    class_counts = ", ".join(
        f"{class_name}={count:,}"
        for class_name, count in metrics["row_counts_by_class"].items()
    ) or "none"
    active_class_counts = ", ".join(
        f"{class_name}={count:,}"
        for class_name, count in metrics["active_row_counts_by_class"].items()
    ) or "none"
    return [
        (
            "rows="
            f"{metrics['row_count']:,}, active_rows={metrics['active_row_count']:,}, "
            f"subjects={metrics['unique_subject_count']:,}, "
            f"encounters={metrics['unique_encounter_count']:,}"
        ),
        f"row_counts_by_class={class_counts}",
        f"active_row_counts_by_class={active_class_counts}",
    ]


def _class_assignment_status_from_state_row(row: pd.Series) -> str:
    medication_class = str(row.get("medication_class_standardized") or "unresolved")
    ingredient_status = str(row.get("ingredient_resolution_status") or "")
    if medication_class in FIRST_SCOPE_SUPPORTED_CLASSES:
        return "supported_scope_class_assigned"
    if ingredient_status in {"resolved_to_ingredient", "resolved_via_related_concept"}:
        return "supported_scope_unresolved"
    if ingredient_status == "resolved_term_only_no_ingredient":
        return "no_standardized_ingredient_available"
    return "class_enrichment_not_attempted"


def _legacy_heuristic_flags(
    dataframe: pd.DataFrame,
    *,
    medication_text_columns: list[str],
) -> pd.DataFrame:
    text_series = pd.Series([""] * len(dataframe), dtype="string")
    for column_name in medication_text_columns:
        if column_name not in dataframe:
            continue
        candidate = dataframe[column_name].fillna("").astype(str).str.lower()
        text_series = text_series.where(text_series.str.len() > 0, candidate)
    flags = pd.DataFrame(index=dataframe.index)
    for class_name, flag_column in HEURISTIC_FLAG_COLUMNS.items():
        keywords = HIGH_RISK_MEDICATION_KEYWORDS[f"{class_name}_flag"]
        flags[flag_column] = text_series.apply(
            lambda value: int(any(keyword in value for keyword in keywords))
        )
    return flags


def _standardized_vs_heuristic_agreement_flag(row: pd.Series) -> int:
    medication_class = str(row.get("medication_class_standardized") or "unresolved")
    if medication_class in FIRST_SCOPE_SUPPORTED_CLASSES:
        return int(bool(row.get(HEURISTIC_FLAG_COLUMNS[medication_class], 0)))
    return int(bool(row.get("heuristic_any_supported_class_flag", 0)) is False)


def _prepare_creatinine_frame(
    *,
    labevents: pd.DataFrame | None,
    settings: Settings,
) -> pd.DataFrame:
    if labevents is None or labevents.empty:
        return pd.DataFrame(columns=["hadm_id", "charttime", "valuenum"])
    creatinine = labevents.loc[
        labevents["itemid"].isin(settings.serum_creatinine_itemids),
        ["hadm_id", "charttime", "valuenum"],
    ].copy()
    creatinine["hadm_id"] = pd.to_numeric(creatinine["hadm_id"], errors="coerce").astype("Int64")
    creatinine["charttime"] = pd.to_datetime(creatinine["charttime"], errors="coerce")
    creatinine["valuenum"] = pd.to_numeric(creatinine["valuenum"], errors="coerce")
    return creatinine.dropna(subset=["hadm_id", "charttime"])


def _representative_event_for_row(
    *,
    event_group: pd.DataFrame,
    selected_medication_event_id: object,
    review_timestamp: pd.Timestamp,
) -> dict[str, object]:
    if event_group.empty:
        return {}
    if selected_medication_event_id is not None:
        selected = event_group.loc[
            event_group["medication_event_id"] == selected_medication_event_id
        ]
        if not selected.empty:
            return selected.iloc[0].to_dict()
    active_events = filter_active_medication_events(event_group, review_timestamp)
    representative_group = active_events if not active_events.empty else event_group
    return deduplicate_active_medication_group(representative_group).to_dict()


def _scheduled_vs_prn_for_group(
    event_group: pd.DataFrame,
) -> tuple[str, str]:
    if event_group.empty:
        return "unknown", "unknown"
    inferences = [
        infer_prn_vs_scheduled(
            frequency=_string_or_none(row.get("frequency")),
            status=_string_or_none(row.get("status")),
        )
        for row in event_group.to_dict(orient="records")
    ]
    if "prn" in inferences:
        return "prn", "explicit_prn"
    if "scheduled" in inferences:
        return "scheduled", "explicit_scheduled"
    return "unknown", "unknown"


def _duration_before_review_for_group(
    *,
    event_group: pd.DataFrame,
    review_timestamp: pd.Timestamp,
) -> tuple[float | None, int, int]:
    if event_group.empty or pd.isna(review_timestamp):
        return None, 0, 0
    starts = [
        start
        for start in pd.to_datetime(event_group["_event_start"], errors="coerce").tolist()
        if pd.notna(start) and start <= review_timestamp
    ]
    if not starts:
        return None, 0, 0
    earliest_start = min(starts)
    duration_hours = round(
        float((review_timestamp - earliest_start).total_seconds()) / 3600.0,
        3,
    )
    lower_bound_flag = int(
        bool(
            int(event_group["continued_from_home_inferred"].fillna(0).astype(int).max()) == 1
            and int(event_group["source_home_medrecon"].fillna(0).astype(int).max()) == 1
        )
    )
    return duration_hours, 1, lower_bound_flag


def _renal_context_for_row(
    *,
    hadm_id: int | None,
    review_timestamp: pd.Timestamp,
    age_proxy: object,
    sex: object,
    creatinine_by_hadm: dict[int, pd.DataFrame],
    cache: dict[tuple[int | None, str], dict[str, object]],
) -> dict[str, object]:
    review_key = review_timestamp.strftime(TIMESTAMP_FORMAT) if pd.notna(review_timestamp) else "null"
    cache_key = (hadm_id, review_key)
    if cache_key in cache:
        return cache[cache_key]

    age_available = int(pd.notna(pd.to_numeric(pd.Series([age_proxy]), errors="coerce").iloc[0]))
    sex_text = _string_or_none(sex)
    sex_available = int(sex_text in {"M", "F"})
    latest_creatinine_time: str | None = None
    latest_creatinine_value: float | None = None

    if hadm_id is not None and hadm_id in creatinine_by_hadm and pd.notna(review_timestamp):
        creatinine_group = creatinine_by_hadm[hadm_id]
        eligible = creatinine_group.loc[
            creatinine_group["charttime"] <= review_timestamp
        ].copy()
        if not eligible.empty:
            latest = eligible.sort_values("charttime", na_position="last").iloc[-1]
            latest_creatinine_time = pd.to_datetime(
                latest["charttime"],
                errors="coerce",
            ).strftime(TIMESTAMP_FORMAT)
            latest_value = pd.to_numeric(pd.Series([latest["valuenum"]]), errors="coerce").iloc[0]
            latest_creatinine_value = None if pd.isna(latest_value) else round(float(latest_value), 3)

    renal_context_available_flag = int(
        age_available == 1
        and sex_available == 1
        and latest_creatinine_time is not None
    )
    payload = {
        "renal_context_available_flag": renal_context_available_flag,
        "renal_context_age_available_flag": age_available,
        "renal_context_sex_available_flag": sex_available,
        "renal_context_latest_creatinine_time": latest_creatinine_time,
        "renal_context_latest_creatinine_value": latest_creatinine_value,
    }
    cache[cache_key] = payload
    return payload


def _medication_start_context(
    *,
    continued_from_home: object,
    newly_started: object,
) -> str:
    continued_flag = int(pd.to_numeric(pd.Series([continued_from_home]), errors="coerce").fillna(0).iloc[0])
    started_flag = int(pd.to_numeric(pd.Series([newly_started]), errors="coerce").fillna(0).iloc[0])
    if continued_flag == 1 and started_flag == 1:
        return "mixed_evidence"
    if continued_flag == 1:
        return "continued_from_home"
    if started_flag == 1:
        return "new_start_during_encounter"
    return "neither_or_unknown"


def _opioid_mme_placeholder(
    *,
    medication_class_standardized: object,
    ingredient_standardized: object,
    dose_value: object,
    dose_unit: object,
    route: object,
    frequency: object,
) -> tuple[float | None, str, int, str]:
    if _string_or_none(medication_class_standardized) != "opioid":
        return None, "not_opioid", 0, dumps_json([])
    missing_fields: list[str] = []
    if _string_or_none(ingredient_standardized) is None:
        missing_fields.append("ingredient_standardized")
    if pd.isna(pd.to_numeric(pd.Series([dose_value]), errors="coerce").iloc[0]):
        missing_fields.append("dose_value")
    if _string_or_none(dose_unit) is None:
        missing_fields.append("dose_unit")
    if _string_or_none(route) is None:
        missing_fields.append("route")
    if _string_or_none(frequency) is None:
        missing_fields.append("frequency")
    if missing_fields:
        return None, "missing_required_fields", 0, dumps_json(missing_fields)
    return None, "ready_for_future_conversion", 1, dumps_json([])


def _renal_dose_mismatch_placeholder(
    *,
    active_at_review_flag: object,
    dose_value: object,
    dose_unit: object,
    route: object,
    renal_context_available_flag: object,
) -> tuple[float | None, str, int, int]:
    dose_fields_available_flag = int(
        pd.notna(pd.to_numeric(pd.Series([dose_value]), errors="coerce").iloc[0])
        and _string_or_none(dose_unit) is not None
        and _string_or_none(route) is not None
    )
    renal_context_flag = int(pd.to_numeric(pd.Series([renal_context_available_flag]), errors="coerce").fillna(0).iloc[0])
    active_flag = int(pd.to_numeric(pd.Series([active_at_review_flag]), errors="coerce").fillna(0).iloc[0])
    ready_flag = int(active_flag == 1 and dose_fields_available_flag == 1 and renal_context_flag == 1)
    if active_flag != 1:
        return None, "not_current_medication", dose_fields_available_flag, 0
    if ready_flag == 1:
        return None, "ready_for_future_logic", dose_fields_available_flag, 1
    return None, "insufficient_inputs", dose_fields_available_flag, 0


def _attach_current_burden_counts(burden: pd.DataFrame) -> pd.DataFrame:
    keys = ["subject_id", "encounter_id", "review_timestamp"]
    active = burden.loc[burden["active_at_review_flag"] == 1].copy()
    exact_counts = (
        active.groupby(keys, as_index=False)
        .agg(exact_current_medication_count=("medication_standardized", "nunique"))
    )
    enriched = burden.merge(exact_counts, how="left", on=keys, suffixes=("", "_current"))
    if "exact_current_medication_count_current" in enriched:
        enriched["exact_current_medication_count"] = enriched[
            "exact_current_medication_count_current"
        ]
        enriched = enriched.drop(columns="exact_current_medication_count_current")
    enriched["exact_current_medication_count"] = (
        pd.to_numeric(enriched["exact_current_medication_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    for class_name in FIRST_SCOPE_SUPPORTED_CLASSES:
        column_name = f"current_{class_name}_count"
        class_counts = (
            active.loc[active["medication_class_standardized"] == class_name]
            .groupby(keys, as_index=False)
            .agg(**{column_name: ("medication_standardized", "count")})
        )
        enriched = enriched.merge(class_counts, how="left", on=keys, suffixes=("", "_count"))
        if f"{column_name}_count" in enriched:
            enriched[column_name] = enriched[f"{column_name}_count"]
            enriched = enriched.drop(columns=f"{column_name}_count")
        enriched[column_name] = (
            pd.to_numeric(enriched[column_name], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    enriched["current_supported_class_count"] = sum(
        enriched[f"current_{class_name}_count"] for class_name in FIRST_SCOPE_SUPPORTED_CLASSES
    )
    enriched["row_same_class_current_count"] = enriched.apply(
        _row_same_class_count,
        axis=1,
    )
    enriched["same_class_duplicate_therapy_flag"] = (
        (enriched["active_at_review_flag"] == 1)
        & (enriched["row_same_class_current_count"] > 1)
    ).astype(int)
    enriched["same_class_duplicate_therapy_signal_count"] = enriched.apply(
        lambda row: (
            int(row["row_same_class_current_count"])
            if int(row["same_class_duplicate_therapy_flag"]) == 1
            else 0
        ),
        axis=1,
    )
    return enriched


def _row_same_class_count(row: pd.Series) -> int:
    medication_class = _string_or_none(row.get("medication_class_standardized"))
    if medication_class not in FIRST_SCOPE_SUPPORTED_CLASSES:
        return 0
    value = row.get(f"current_{medication_class}_count", 0)
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0]
    return int(numeric)


def _medication_candidate_key_from_event(row: pd.Series) -> str | None:
    for value in [row.get("medication_standardized"), row.get("medication_normalized")]:
        text = _string_or_none(value)
        if text is not None:
            return text
    return None


def _int_or_none(value: object) -> int | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none"}:
        return None
    return text
