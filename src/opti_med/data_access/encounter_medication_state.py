"""Persisted encounter-medication-state builder at the review-time modeling grain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq

from opti_med.cohort import semi_join_to_eligible_encounters
from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_STATE_COLUMNS,
    ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION,
    validate_encounter_index_artifact,
    validate_encounter_medication_state_artifact,
    validate_medication_events_artifact,
    validate_medication_rxnorm_mapping_artifact,
    validate_older_adult_eligibility_artifact,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.medication_rxnorm_mapping import (
    LOOKUP_MODE_CACHE_FIRST,
    LOOKUP_MODE_CACHE_ONLY,
    MedicationRxNormMappingBuilder,
)
from opti_med.data_access.medication_snapshot import (
    classify_medication_status_at_review_time,
    deduplicate_active_medication_group,
    filter_active_medication_events,
    latest_valid_timestamp,
    select_review_timestamp_metadata_for_encounter,
)
from opti_med.data_access.provenance import dumps_json, loads_json_or_none
from opti_med.medication_semantics import RxNormBackedMedicationSemanticMapper
from opti_med.standardized import StandardizedParquetRepository
from opti_med.standardized.specs import get_table_spec
from opti_med.state import (
    EncounterMedicationStateBuilder,
    EncounterMedicationStateRow,
)
from opti_med.time_semantics import (
    MEDICATION_STATUS_ACTIVE_AT_REVIEW,
    MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW,
    MEDICATION_STATUS_INACTIVE_BEFORE_REVIEW,
    MEDICATION_STATUS_PRE_ADMISSION_ONLY,
    REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_BOUNDARY,
    ReviewTimePolicyName,
    ReviewTimeViolationError,
    assert_review_time_not_after_discharge_when_policy_requires,
    validate_review_time_policy,
)


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_ENCOUNTER_MEDICATION_STATE_OUTPUT_PATH = Path(
    "data/analytical/encounter_medication_state_65plus.parquet"
)
DEFAULT_ENCOUNTER_MEDICATION_STATE_POLICY = (
    ReviewTimePolicyName.DISCHARGE_CAPPED_LATEST_AVAILABLE
)
LOOKUP_MODE_DISABLED = "disabled"
STATE_BUILD_MEDICATION_EVENT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "encounter_id",
    "encounter_start",
    "encounter_end",
    "medication_event_id",
    "medication_event_type",
    "event_source_table",
    "raw_medication_name",
    "medication_name",
    "medication_normalized",
    "event_time",
    "starttime",
    "stoptime",
    "route",
    "frequency",
    "status",
    "pharmacy_enriched_flag",
    "continued_from_home_inferred",
    "newly_started_during_encounter_inferred",
    "source_home_medrecon",
    "source_ed_pyxis",
    "source_hospital_order",
    "source_hospital_admin",
    "source_tables_json",
    "source_record_provenance_json",
]
STATE_BUILD_LABEVENT_COLUMNS = ["hadm_id", "charttime"]


@dataclass(frozen=True, slots=True)
class EncounterMedicationStateBuildResult:
    """Built encounter-medication-state artifact and its output path."""

    dataframe: pd.DataFrame
    output_path: Path


@dataclass(frozen=True, slots=True)
class ReviewTimestampResolution:
    """Finalized review timestamp metadata for one encounter."""

    review_timestamp: pd.Timestamp | pd.NaT
    review_timestamp_source: str | None
    review_timestamp_candidate: pd.Timestamp | pd.NaT
    review_timestamp_candidate_source: str | None
    discharge_boundary: pd.Timestamp | pd.NaT
    review_time_policy_name: str
    review_time_capped_to_discharge_flag: bool
    review_time_validated_flag: bool


class EncounterMedicationStateArtifactBuilder(EncounterMedicationStateBuilder):
    """Build one row per medication candidate for eligible 65+ encounters at review time."""

    def __init__(
        self,
        settings: Settings,
        *,
        review_time_policy: ReviewTimePolicyName | str | None = None,
        semantic_lookup_mode: str = LOOKUP_MODE_DISABLED,
        semantic_mapper: RxNormBackedMedicationSemanticMapper | None = None,
    ) -> None:
        if semantic_lookup_mode not in {
            LOOKUP_MODE_DISABLED,
            LOOKUP_MODE_CACHE_FIRST,
            LOOKUP_MODE_CACHE_ONLY,
        }:
            allowed = ", ".join(
                [LOOKUP_MODE_DISABLED, LOOKUP_MODE_CACHE_FIRST, LOOKUP_MODE_CACHE_ONLY]
            )
            raise ValueError(
                f"Unsupported semantic lookup mode '{semantic_lookup_mode}'. Expected one of: {allowed}."
            )
        self.settings = settings
        self.repository = StandardizedParquetRepository(settings)
        self.review_time_policy = validate_review_time_policy(
            review_time_policy or DEFAULT_ENCOUNTER_MEDICATION_STATE_POLICY
        )
        self.semantic_lookup_mode = semantic_lookup_mode
        self.semantic_mapper = semantic_mapper or RxNormBackedMedicationSemanticMapper()
        self.last_medication_rxnorm_mapping: pd.DataFrame | None = None

    def build(
        self,
        *,
        encounter_index: pd.DataFrame | None = None,
        medication_events: pd.DataFrame | None = None,
        eligible_encounters: pd.DataFrame | None = None,
        medication_rxnorm_mapping: pd.DataFrame | None = None,
        labevents: pd.DataFrame | None = None,
        triage: pd.DataFrame | None = None,
        vitalsign: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the persisted 65+ encounter-medication-state artifact."""
        encounter_index = (
            encounter_index
            if encounter_index is not None
            else self.repository.load_analytical_artifact("encounter_index")
        )
        eligible_encounters = (
            eligible_encounters
            if eligible_encounters is not None
            else _load_required_parquet(self.settings.older_adult_eligibility_output_path)
        )
        validate_encounter_index_artifact(encounter_index)
        validate_older_adult_eligibility_artifact(eligible_encounters)
        encounter_index = semi_join_to_eligible_encounters(
            encounter_index,
            eligible_encounters,
        ).reset_index(drop=True)
        if encounter_index.empty:
            return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

        if medication_events is None:
            medication_event_columns = (
                None
                if self.semantic_lookup_mode != LOOKUP_MODE_DISABLED
                else STATE_BUILD_MEDICATION_EVENT_COLUMNS
            )
            medication_events = _load_filtered_medication_events_for_eligible_encounters(
                repository=self.repository,
                eligible_encounters=eligible_encounters,
                columns=medication_event_columns,
            )
            if medication_event_columns is None:
                validate_medication_events_artifact(medication_events)
            else:
                _validate_state_build_medication_event_input(medication_events)
        else:
            validate_medication_events_artifact(medication_events)
            medication_events = semi_join_to_eligible_encounters(
                medication_events,
                eligible_encounters,
            ).reset_index(drop=True)
        if medication_events.empty:
            return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

        if medication_rxnorm_mapping is None and self.semantic_lookup_mode != LOOKUP_MODE_DISABLED:
            mapping_builder = MedicationRxNormMappingBuilder(
                self.settings,
                semantic_mapper=self.semantic_mapper,
                lookup_mode=self.semantic_lookup_mode,
            )
            medication_rxnorm_mapping = mapping_builder.build(
                medication_events=medication_events,
            )
        if medication_rxnorm_mapping is not None:
            validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
        self.last_medication_rxnorm_mapping = medication_rxnorm_mapping

        supporting_review_time_encounters = identify_supporting_review_time_encounters(
            encounter_index=encounter_index,
            medication_events=medication_events,
        )
        if supporting_review_time_encounters.empty:
            labevents = None
            triage = None
            vitalsign = None
        else:
            if labevents is None:
                labevents = _load_filtered_labevents_for_encounters(
                    repository=self.repository,
                    encounter_index=supporting_review_time_encounters,
                )
            if triage is None:
                loaded = self.repository.load_optional_source_table(
                    "ed",
                    "triage",
                    columns=["subject_id", "stay_id", "charttime"],
                )
                triage = loaded.dataframe if loaded else None
            if vitalsign is None:
                loaded = self.repository.load_optional_source_table(
                    "ed",
                    "vitalsign",
                    columns=["subject_id", "stay_id", "charttime"],
                )
                vitalsign = loaded.dataframe if loaded else None
            labevents, triage, vitalsign = filter_supporting_review_time_inputs_to_encounters(
                encounter_index=supporting_review_time_encounters,
                labevents=labevents,
                triage=triage,
                vitalsign=vitalsign,
            )

        dataframe = build_encounter_medication_state(
            encounter_index=encounter_index,
            medication_events=medication_events,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            review_time_policy=self.review_time_policy,
            labevents=labevents,
            triage=triage,
            vitalsign=vitalsign,
        )
        validate_encounter_medication_state_artifact(dataframe)
        return dataframe

    def build_rows(
        self,
        *,
        encounter_index: pd.DataFrame | None = None,
        medication_events: pd.DataFrame | None = None,
        eligible_encounters: pd.DataFrame | None = None,
        medication_rxnorm_mapping: pd.DataFrame | None = None,
        labevents: pd.DataFrame | None = None,
        triage: pd.DataFrame | None = None,
        vitalsign: pd.DataFrame | None = None,
    ) -> list[EncounterMedicationStateRow]:
        """Build strongly typed rows for downstream ML-facing layers."""
        dataframe = self.build(
            encounter_index=encounter_index,
            medication_events=medication_events,
            eligible_encounters=eligible_encounters,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            labevents=labevents,
            triage=triage,
            vitalsign=vitalsign,
        )
        rows: list[EncounterMedicationStateRow] = []
        for record in dataframe.to_dict(orient="records"):
            provenance = {
                "review_time_policy_name": record.get("review_time_policy_name"),
                "review_timestamp_candidate": record.get("review_timestamp_candidate"),
                "review_timestamp_candidate_source": record.get(
                    "review_timestamp_candidate_source"
                ),
                "review_time_capped_to_discharge_flag": record.get(
                    "review_time_capped_to_discharge_flag"
                ),
                "review_time_validated_flag": record.get("review_time_validated_flag"),
                "discharge_boundary": record.get("discharge_boundary"),
                "selected_medication_event_id": record.get("selected_medication_event_id"),
                "selected_medication_event_type": record.get(
                    "selected_medication_event_type"
                ),
                "active_event_count_at_review": record.get("active_event_count_at_review"),
                "candidate_event_count": record.get("candidate_event_count"),
                "source_tables_json": record.get("source_tables_json"),
                "source_record_provenance_json": record.get(
                    "source_record_provenance_json"
                ),
            }
            rows.append(
                EncounterMedicationStateRow(
                    subject_id=int(record["subject_id"]),
                    encounter_id=str(record["encounter_id"]),
                    review_timestamp=pd.to_datetime(record["review_timestamp"]).to_pydatetime(),
                    hadm_id=_int_or_none(record.get("hadm_id")),
                    stay_id=_int_or_none(record.get("stay_id")),
                    review_timestamp_source=record.get("review_timestamp_source"),
                    age_proxy=_float_or_none(record.get("age_proxy")),
                    age_group=record.get("age_group"),
                    medication_raw=record.get("medication_raw"),
                    medication_normalized=record.get("medication_normalized"),
                    medication_standardized=record.get("medication_standardized"),
                    medication_standardized_source=record.get(
                        "medication_standardized_source"
                    ),
                    rxnorm_rxcui=record.get("rxnorm_rxcui"),
                    ingredient_standardized=record.get("ingredient_standardized"),
                    ingredient_resolution_status=record.get(
                        "ingredient_resolution_status"
                    ),
                    mapping_confidence=record.get("mapping_confidence"),
                    ambiguous_mapping_flag=_binary_as_bool(
                        record.get("ambiguous_mapping_flag")
                    ),
                    medication_class_standardized=record.get(
                        "medication_class_standardized"
                    ),
                    medication_status_at_review=record.get("medication_status_at_review"),
                    active_at_review_flag=_binary_as_bool(
                        record.get("active_at_review_flag")
                    ),
                    continued_from_home_inferred=_binary_as_bool(
                        record.get("continued_from_home_inferred")
                    ),
                    newly_started_during_encounter_inferred=_binary_as_bool(
                        record.get("newly_started_during_encounter_inferred")
                    ),
                    route=record.get("route"),
                    frequency=record.get("frequency"),
                    status=record.get("status"),
                    source_home_medrecon_flag=_binary_as_bool(
                        record.get("source_home_medrecon_flag")
                    ),
                    source_ed_pyxis_flag=_binary_as_bool(
                        record.get("source_ed_pyxis_flag")
                    ),
                    source_hospital_order_flag=_binary_as_bool(
                        record.get("source_hospital_order_flag")
                    ),
                    source_hospital_admin_flag=_binary_as_bool(
                        record.get("source_hospital_admin_flag")
                    ),
                    provenance=provenance,
                )
            )
        return rows

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: Path | None = None,
    ) -> EncounterMedicationStateBuildResult:
        """Persist the encounter-medication-state artifact to Parquet."""
        validate_encounter_medication_state_artifact(dataframe)
        target_path = output_path or self.settings.encounter_medication_state_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(target_path, index=False)
        return EncounterMedicationStateBuildResult(
            dataframe=dataframe,
            output_path=target_path,
        )


def build_encounter_medication_state(
    *,
    encounter_index: pd.DataFrame,
    medication_events: pd.DataFrame,
    medication_rxnorm_mapping: pd.DataFrame | None = None,
    review_time_policy: ReviewTimePolicyName | str = DEFAULT_ENCOUNTER_MEDICATION_STATE_POLICY,
    labevents: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row per review-time medication candidate for eligible encounters."""
    if encounter_index.empty or medication_events.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

    policy_name = validate_review_time_policy(review_time_policy)
    build_run_id = f"encounter-medication-state-65plus-{uuid4().hex[:12]}"
    encounter_rows = encounter_index.drop_duplicates(subset=["encounter_id"]).copy()
    encounter_rows = encounter_rows.sort_values(
        ["subject_id", "encounter_start", "encounter_id"],
        na_position="last",
    )

    events = medication_events.copy()
    events["_medication_candidate_key"] = _medication_candidate_keys(events)
    events = events.loc[events["_medication_candidate_key"].notna()].copy()
    if events.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)
    if medication_rxnorm_mapping is not None and not medication_rxnorm_mapping.empty:
        events = apply_medication_rxnorm_mapping_to_events(
            medication_events=events,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
        )
    else:
        events = _apply_unresolved_semantic_defaults(events)
    encounter_rows_by_id = {
        str(row["encounter_id"]): row for row in encounter_rows.to_dict(orient="records")
    }
    latest_lab_timestamp_by_hadm_id = _latest_lab_timestamp_by_hadm_id(labevents)
    latest_vitals_timestamp_by_subject_stay = _latest_vitals_timestamp_by_subject_stay(
        triage=triage,
        vitalsign=vitalsign,
    )
    review_resolutions = _resolve_review_timestamps_for_encounters(
        encounter_rows_by_id=encounter_rows_by_id,
        medication_events=events,
        labevents=labevents,
        triage=triage,
        vitalsign=vitalsign,
        review_time_policy=policy_name,
        latest_lab_timestamp_by_hadm_id=latest_lab_timestamp_by_hadm_id,
        latest_vitals_timestamp_by_subject_stay=latest_vitals_timestamp_by_subject_stay,
    )
    if review_resolutions.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

    review = _materialize_encounter_medication_state_rows(
        encounter_rows_by_id=encounter_rows_by_id,
        medication_events=events,
        review_resolutions=review_resolutions,
    )
    if review.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

    review["encounter_medication_state_build_run_id"] = build_run_id
    review["encounter_medication_state_contract_version"] = (
        ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION
    )
    review = review.loc[:, ENCOUNTER_MEDICATION_STATE_COLUMNS].copy()
    review = review.sort_values(
        ["subject_id", "encounter_id", "review_timestamp", "medication_standardized"],
        na_position="last",
    ).reset_index(drop=True)
    validate_encounter_medication_state_artifact(review)
    return review


def _resolve_review_timestamps_for_encounters(
    *,
    encounter_rows_by_id: dict[str, dict[str, object]],
    medication_events: pd.DataFrame,
    labevents: pd.DataFrame | None,
    triage: pd.DataFrame | None,
    vitalsign: pd.DataFrame | None,
    review_time_policy: ReviewTimePolicyName | str,
    latest_lab_timestamp_by_hadm_id: dict[int, pd.Timestamp] | None,
    latest_vitals_timestamp_by_subject_stay: dict[tuple[int, int], pd.Timestamp] | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for encounter_id, encounter_events in medication_events.groupby(
        "encounter_id",
        dropna=False,
        sort=False,
    ):
        encounter = encounter_rows_by_id.get(str(encounter_id))
        if encounter is None or encounter_events.empty:
            continue

        review_resolution = resolve_policy_safe_review_timestamp_for_encounter(
            encounter=encounter,
            medication_events=encounter_events,
            labevents=labevents,
            triage=triage,
            vitalsign=vitalsign,
            review_time_policy=review_time_policy,
            latest_lab_timestamp_by_hadm_id=latest_lab_timestamp_by_hadm_id,
            latest_vitals_timestamp_by_subject_stay=latest_vitals_timestamp_by_subject_stay,
        )
        if pd.isna(review_resolution.review_timestamp):
            continue

        rows.append(
            {
                "encounter_id": encounter_id,
                "review_timestamp": review_resolution.review_timestamp,
                "review_timestamp_source": review_resolution.review_timestamp_source,
                "review_time_policy_name": review_resolution.review_time_policy_name,
                "review_timestamp_candidate": review_resolution.review_timestamp_candidate,
                "review_timestamp_candidate_source": (
                    review_resolution.review_timestamp_candidate_source
                ),
                "review_time_capped_to_discharge_flag": int(
                    review_resolution.review_time_capped_to_discharge_flag
                ),
                "review_time_validated_flag": int(
                    review_resolution.review_time_validated_flag
                ),
                "discharge_boundary": review_resolution.discharge_boundary,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "encounter_id",
                "review_timestamp",
                "review_timestamp_source",
                "review_time_policy_name",
                "review_timestamp_candidate",
                "review_timestamp_candidate_source",
                "review_time_capped_to_discharge_flag",
                "review_time_validated_flag",
                "discharge_boundary",
            ]
        )
    return pd.DataFrame(rows)


def _materialize_encounter_medication_state_rows(
    *,
    encounter_rows_by_id: dict[str, dict[str, object]],
    medication_events: pd.DataFrame,
    review_resolutions: pd.DataFrame,
) -> pd.DataFrame:
    group_keys = ["encounter_id", "_medication_standardized_key"]
    review_lookup = {
        str(row["encounter_id"]): row for row in review_resolutions.to_dict(orient="records")
    }
    events = medication_events.merge(
        review_resolutions.loc[:, ["encounter_id", "review_timestamp"]],
        how="inner",
        on="encounter_id",
        validate="many_to_one",
    )
    if events.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

    events = _prepare_review_state_events(events)
    grouped = events.groupby(group_keys, dropna=False, sort=True)
    group_summary = _summarize_review_event_groups(grouped)
    if group_summary.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

    summary_lookup = {
        _review_group_key(row["encounter_id"], row["_medication_standardized_key"]): row
        for row in group_summary.to_dict(orient="records")
    }
    representative_lookup = _representative_review_event_lookup(
        events=events,
        group_summary=group_summary,
    )

    review_rows: list[dict[str, object]] = []
    for (encounter_id, medication_standardized_key), group in grouped:
        group_key = _review_group_key(encounter_id, medication_standardized_key)
        encounter = encounter_rows_by_id.get(group_key[0])
        representative = representative_lookup.get(group_key)
        summary = summary_lookup.get(group_key)
        review_resolution = review_lookup.get(group_key[0])
        if (
            encounter is None
            or representative is None
            or summary is None
            or review_resolution is None
        ):
            continue

        source_tables_json = _source_tables_json_for_group(group)
        source_record_provenance_json = _source_record_provenance_json_for_group(group)
        semantic_fields = _aggregate_semantic_fields(group, representative=representative)
        review_rows.append(
            {
                "subject_id": _coalesce_identifier(
                    representative.get("subject_id"),
                    encounter.get("subject_id"),
                ),
                "encounter_id": encounter["encounter_id"],
                "hadm_id": _coalesce_identifier(
                    representative.get("hadm_id"),
                    encounter.get("hadm_id"),
                ),
                "stay_id": _coalesce_identifier(
                    representative.get("stay_id"),
                    encounter.get("stay_id"),
                ),
                "review_timestamp": _format_timestamp(review_resolution["review_timestamp"]),
                "review_timestamp_source": review_resolution["review_timestamp_source"],
                "age_proxy": _float_or_none(encounter.get("age_proxy")),
                "age_group": encounter.get("age_group"),
                "review_time_policy_name": review_resolution["review_time_policy_name"],
                "review_timestamp_candidate": _format_timestamp(
                    review_resolution["review_timestamp_candidate"]
                ),
                "review_timestamp_candidate_source": review_resolution[
                    "review_timestamp_candidate_source"
                ],
                "review_time_capped_to_discharge_flag": int(
                    review_resolution["review_time_capped_to_discharge_flag"]
                ),
                "review_time_validated_flag": int(
                    review_resolution["review_time_validated_flag"]
                ),
                "discharge_boundary": _format_timestamp(
                    review_resolution["discharge_boundary"]
                ),
                "medication_raw": representative.get("raw_medication_name"),
                "medication_normalized": representative.get("medication_normalized"),
                "medication_standardized": str(medication_standardized_key),
                "medication_standardized_source": semantic_fields[
                    "medication_standardized_source"
                ],
                "rxnorm_rxcui": semantic_fields["rxnorm_rxcui"],
                "rxnorm_matched_term": semantic_fields["rxnorm_matched_term"],
                "rxnorm_term_type": semantic_fields["rxnorm_term_type"],
                "ingredient_standardized": semantic_fields["ingredient_standardized"],
                "ingredient_resolution_status": semantic_fields[
                    "ingredient_resolution_status"
                ],
                "mapping_confidence": semantic_fields["mapping_confidence"],
                "ambiguous_mapping_flag": semantic_fields["ambiguous_mapping_flag"],
                "mapping_candidate_count": semantic_fields["mapping_candidate_count"],
                "medication_mapping_lookup_strategy": semantic_fields[
                    "medication_mapping_lookup_strategy"
                ],
                "medication_class_standardized": semantic_fields[
                    "medication_class_standardized"
                ],
                "medication_status_at_review": summary["medication_status_at_review"],
                "active_at_review_flag": int(summary["active_at_review_flag"]),
                "continued_from_home_inferred": int(
                    summary["continued_from_home_inferred"]
                ),
                "newly_started_during_encounter_inferred": int(
                    summary["newly_started_during_encounter_inferred"]
                ),
                "route": representative.get("route"),
                "frequency": representative.get("frequency"),
                "status": representative.get("status"),
                "selected_medication_event_id": representative.get("medication_event_id"),
                "selected_medication_event_type": representative.get(
                    "medication_event_type"
                ),
                "active_event_count_at_review": int(
                    summary["active_event_count_at_review"]
                ),
                "candidate_event_count": int(summary["candidate_event_count"]),
                "source_home_medrecon_flag": int(summary["source_home_medrecon_flag"]),
                "source_ed_pyxis_flag": int(summary["source_ed_pyxis_flag"]),
                "source_hospital_order_flag": int(
                    summary["source_hospital_order_flag"]
                ),
                "source_hospital_admin_flag": int(
                    summary["source_hospital_admin_flag"]
                ),
                "source_tables_json": source_tables_json,
                "source_record_provenance_json": source_record_provenance_json,
            }
        )

    return pd.DataFrame(review_rows)


def _prepare_review_state_events(events: pd.DataFrame) -> pd.DataFrame:
    prepared = events.copy()
    prepared["review_timestamp"] = pd.to_datetime(
        prepared["review_timestamp"],
        errors="coerce",
        format=TIMESTAMP_FORMAT,
    )
    prepared["_event_start"] = _encounter_state_event_start_series(prepared)
    prepared["_event_stop"] = pd.to_datetime(
        prepared.get("stoptime", pd.Series(pd.NaT, index=prepared.index)),
        errors="coerce",
        format=TIMESTAMP_FORMAT,
    )
    prepared["_continued_from_home_int"] = _state_int_series(
        prepared.get("continued_from_home_inferred"),
        index=prepared.index,
    )
    prepared["_newly_started_int"] = _state_int_series(
        prepared.get("newly_started_during_encounter_inferred"),
        index=prepared.index,
    )
    prepared["_source_home_medrecon_int"] = _state_int_series(
        prepared.get("source_home_medrecon"),
        index=prepared.index,
    )
    prepared["_source_ed_pyxis_int"] = _state_int_series(
        prepared.get("source_ed_pyxis"),
        index=prepared.index,
    )
    prepared["_source_hospital_order_int"] = _state_int_series(
        prepared.get("source_hospital_order"),
        index=prepared.index,
    )
    prepared["_source_hospital_admin_int"] = _state_int_series(
        prepared.get("source_hospital_admin"),
        index=prepared.index,
    )
    prepared["_pharmacy_enriched_int"] = _state_int_series(
        prepared.get("pharmacy_enriched_flag"),
        index=prepared.index,
    )
    prepared["_priority_rank"] = _encounter_state_priority_rank_series(prepared)
    prepared["_active_at_review_event"] = _active_event_mask_for_review_rows(
        prepared
    ).astype(int)
    prepared["_non_home_event"] = (
        prepared["_source_home_medrecon_int"] != 1
    ).astype(int)
    prepared["_active_non_home_event"] = (
        (prepared["_active_at_review_event"] == 1)
        & (prepared["_non_home_event"] == 1)
    ).astype(int)
    prepared["_non_home_event_start"] = prepared["_event_start"].where(
        prepared["_non_home_event"] == 1
    )
    prepared["_non_home_event_stop"] = prepared["_event_stop"].where(
        prepared["_non_home_event"] == 1
    )
    return prepared


def _summarize_review_event_groups(
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
) -> pd.DataFrame:
    summary = grouped.agg(
        review_timestamp=("review_timestamp", "first"),
        active_event_count_at_review=("_active_at_review_event", "sum"),
        active_non_home_event=("_active_non_home_event", "max"),
        has_non_home=("_non_home_event", "max"),
        latest_non_home_start=("_non_home_event_start", "max"),
        latest_non_home_stop=("_non_home_event_stop", "max"),
        continued_from_home_inferred=("_continued_from_home_int", "max"),
        newly_started_during_encounter_inferred=("_newly_started_int", "max"),
        source_home_medrecon_flag=("_source_home_medrecon_int", "max"),
        source_ed_pyxis_flag=("_source_ed_pyxis_int", "max"),
        source_hospital_order_flag=("_source_hospital_order_int", "max"),
        source_hospital_admin_flag=("_source_hospital_admin_int", "max"),
    ).join(grouped.size().rename("candidate_event_count"))
    if summary.empty:
        return summary.reset_index()

    summary = summary.reset_index()
    summary["latest_non_home_time"] = summary.loc[
        :, ["latest_non_home_start", "latest_non_home_stop"]
    ].max(axis=1)
    status = pd.Series(
        MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW,
        index=summary.index,
        dtype="object",
    )
    status = status.mask(
        summary["active_non_home_event"] == 1,
        MEDICATION_STATUS_ACTIVE_AT_REVIEW,
    )
    status = status.mask(
        (summary["active_non_home_event"] != 1) & (summary["has_non_home"] != 1),
        MEDICATION_STATUS_PRE_ADMISSION_ONLY,
    )
    inactive_mask = (
        (status == MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW)
        & summary["latest_non_home_time"].notna()
        & summary["review_timestamp"].notna()
        & (summary["latest_non_home_time"] < summary["review_timestamp"])
    )
    status = status.mask(
        inactive_mask,
        MEDICATION_STATUS_INACTIVE_BEFORE_REVIEW,
    )
    summary["medication_status_at_review"] = status
    summary["active_at_review_flag"] = (
        summary["medication_status_at_review"] == MEDICATION_STATUS_ACTIVE_AT_REVIEW
    ).astype(int)
    summary["group_has_active_event"] = (
        summary["active_event_count_at_review"] > 0
    ).astype(int)
    return summary


def _representative_review_event_lookup(
    *,
    events: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> dict[tuple[str, str], dict[str, object]]:
    group_keys = ["encounter_id", "_medication_standardized_key"]
    ranked = events.merge(
        group_summary.loc[:, group_keys + ["group_has_active_event"]],
        how="left",
        on=group_keys,
        validate="many_to_one",
    )
    ranked["_representative_candidate_flag"] = (
        ((ranked["group_has_active_event"] == 1) & (ranked["_active_at_review_event"] == 1))
        | (ranked["group_has_active_event"] != 1)
    ).astype(int)
    ranked = ranked.sort_values(
        group_keys
        + [
            "_representative_candidate_flag",
            "_priority_rank",
            "_event_start",
            "_pharmacy_enriched_int",
            "medication_event_id",
        ],
        ascending=[True, True, False, False, False, False, True],
        na_position="last",
    )
    representatives = ranked.drop_duplicates(subset=group_keys, keep="first")
    return {
        _review_group_key(row["encounter_id"], row["_medication_standardized_key"]): row
        for row in representatives.to_dict(orient="records")
    }


def resolve_policy_safe_review_timestamp_for_encounter(
    *,
    encounter: dict | pd.Series,
    medication_events: pd.DataFrame | None,
    labevents: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
    review_time_policy: ReviewTimePolicyName | str = DEFAULT_ENCOUNTER_MEDICATION_STATE_POLICY,
    latest_lab_timestamp_by_hadm_id: dict[int, pd.Timestamp] | None = None,
    latest_vitals_timestamp_by_subject_stay: dict[tuple[int, int], pd.Timestamp] | None = None,
) -> ReviewTimestampResolution:
    """Resolve one encounter review timestamp under an explicit policy."""
    policy_name = validate_review_time_policy(review_time_policy)
    encounter_row = encounter if isinstance(encounter, dict) else encounter.to_dict()
    candidate_timestamp, candidate_source = select_review_timestamp_metadata_for_encounter(
        encounter_row,
        strategy="latest_available",
        medication_events=medication_events,
        labevents=labevents,
        triage=triage,
        vitalsign=vitalsign,
        latest_lab_timestamp_by_hadm_id=latest_lab_timestamp_by_hadm_id,
        latest_vitals_timestamp_by_subject_stay=latest_vitals_timestamp_by_subject_stay,
    )
    discharge_boundary = _encounter_discharge_boundary(encounter_row)

    final_timestamp = candidate_timestamp
    final_source = candidate_source
    capped_to_discharge = False

    if (
        policy_name == ReviewTimePolicyName.DISCHARGE_CAPPED_LATEST_AVAILABLE
        and pd.notna(candidate_timestamp)
        and pd.notna(discharge_boundary)
        and candidate_timestamp > discharge_boundary
    ):
        final_timestamp = discharge_boundary
        final_source = REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_BOUNDARY
        capped_to_discharge = True

    try:
        if pd.notna(final_timestamp):
            assert_review_time_not_after_discharge_when_policy_requires(
                final_timestamp,
                discharge_boundary,
                policy_name,
            )
        validated = True
    except ReviewTimeViolationError:
        validated = False
        raise

    return ReviewTimestampResolution(
        review_timestamp=final_timestamp,
        review_timestamp_source=final_source,
        review_timestamp_candidate=candidate_timestamp,
        review_timestamp_candidate_source=candidate_source,
        discharge_boundary=discharge_boundary,
        review_time_policy_name=policy_name.value,
        review_time_capped_to_discharge_flag=capped_to_discharge,
        review_time_validated_flag=validated,
    )


def calculate_encounter_medication_state_qc_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Compute QC metrics required for the review-time medication-state layer."""
    if dataframe.empty:
        return {
            "row_count": 0,
            "unique_encounter_count": 0,
            "review_timestamp_after_discharge_count": 0,
            "duplicate_key_count": 0,
            "counts_by_medication_status_at_review": {},
            "uncertain_state_count": 0,
        }

    review_timestamp = pd.to_datetime(dataframe["review_timestamp"], errors="coerce")
    discharge_boundary = pd.to_datetime(dataframe["discharge_boundary"], errors="coerce")
    after_discharge = discharge_boundary.notna() & review_timestamp.notna() & (
        review_timestamp > discharge_boundary
    )
    counts_by_status = (
        dataframe["medication_status_at_review"]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    return {
        "row_count": int(len(dataframe)),
        "unique_encounter_count": int(dataframe["encounter_id"].nunique()),
        "review_timestamp_after_discharge_count": int(after_discharge.sum()),
        "duplicate_key_count": int(
            dataframe.duplicated(
                subset=[
                    "subject_id",
                    "encounter_id",
                    "medication_standardized",
                    "review_timestamp",
                ],
                keep=False,
            ).sum()
        ),
        "counts_by_medication_status_at_review": {
            str(key): int(value) for key, value in counts_by_status.items()
        },
        "uncertain_state_count": int(
            (
                dataframe["medication_status_at_review"]
                == MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW
            ).sum()
        ),
    }


def summarize_encounter_medication_state(dataframe: pd.DataFrame) -> list[str]:
    """Return compact summaries for the encounter-medication-state artifact."""
    metrics = calculate_encounter_medication_state_qc_metrics(dataframe)
    return [
        f"rows={metrics['row_count']:,}, columns={dataframe.shape[1]}",
        (
            f"unique_subjects={dataframe['subject_id'].nunique():,}"
            if not dataframe.empty
            else "unique_subjects=0"
        ),
        (
            f"unique_encounters={dataframe['encounter_id'].nunique():,}"
            if not dataframe.empty
            else "unique_encounters=0"
        ),
        (
            f"active_rows={int((dataframe['active_at_review_flag'] == 1).sum()):,}"
            if not dataframe.empty
            else "active_rows=0"
        ),
        f"review_timestamp_after_discharge_rows={metrics['review_timestamp_after_discharge_count']:,}",
        f"duplicate_key_rows={metrics['duplicate_key_count']:,}",
        f"uncertain_state_rows={metrics['uncertain_state_count']:,}",
    ]


def filter_state_inputs_to_eligible_encounters(
    *,
    encounter_index: pd.DataFrame,
    medication_events: pd.DataFrame,
    eligible_encounters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Semi-join full encounter and medication inputs to the persisted 65+ encounter set."""
    validate_encounter_index_artifact(encounter_index)
    validate_medication_events_artifact(medication_events)
    validate_older_adult_eligibility_artifact(eligible_encounters)
    filtered_encounter_index = semi_join_to_eligible_encounters(
        encounter_index,
        eligible_encounters,
    )
    filtered_medication_events = semi_join_to_eligible_encounters(
        medication_events,
        eligible_encounters,
    )
    return (
        filtered_encounter_index.reset_index(drop=True),
        filtered_medication_events.reset_index(drop=True),
    )


def filter_supporting_review_time_inputs_to_encounters(
    *,
    encounter_index: pd.DataFrame,
    labevents: pd.DataFrame | None,
    triage: pd.DataFrame | None,
    vitalsign: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """Restrict supporting review-time evidence tables to the eligible encounter context."""
    hadm_ids = _identifier_set(encounter_index.get("hadm_id"))
    stay_ids = _identifier_set(encounter_index.get("stay_id"))
    subject_ids = _identifier_set(encounter_index.get("subject_id"))

    filtered_labevents = labevents
    if labevents is not None and not labevents.empty and "hadm_id" in labevents:
        filtered_labevents = labevents.loc[
            pd.to_numeric(labevents["hadm_id"], errors="coerce").isin(hadm_ids)
        ].copy()

    filtered_triage = triage
    if triage is not None and not triage.empty:
        filtered_triage = triage.copy()
        if "stay_id" in filtered_triage:
            filtered_triage = filtered_triage.loc[
                pd.to_numeric(filtered_triage["stay_id"], errors="coerce").isin(stay_ids)
            ].copy()
        if "subject_id" in filtered_triage:
            filtered_triage = filtered_triage.loc[
                pd.to_numeric(filtered_triage["subject_id"], errors="coerce").isin(subject_ids)
            ].copy()

    filtered_vitalsign = vitalsign
    if vitalsign is not None and not vitalsign.empty:
        filtered_vitalsign = vitalsign.copy()
        if "stay_id" in filtered_vitalsign:
            filtered_vitalsign = filtered_vitalsign.loc[
                pd.to_numeric(filtered_vitalsign["stay_id"], errors="coerce").isin(stay_ids)
            ].copy()
        if "subject_id" in filtered_vitalsign:
            filtered_vitalsign = filtered_vitalsign.loc[
                pd.to_numeric(filtered_vitalsign["subject_id"], errors="coerce").isin(subject_ids)
            ].copy()

    return filtered_labevents, filtered_triage, filtered_vitalsign


def identify_supporting_review_time_encounters(
    *,
    encounter_index: pd.DataFrame,
    medication_events: pd.DataFrame,
) -> pd.DataFrame:
    """Return the subset of encounters that need non-medication evidence for review time."""
    if encounter_index.empty or medication_events.empty:
        return encounter_index.iloc[0:0].copy()

    medication_supported_encounter_ids = _encounter_ids_with_medication_review_evidence(
        medication_events
    )
    if not medication_supported_encounter_ids:
        return encounter_index.copy()
    encounter_ids = encounter_index["encounter_id"].astype(str)
    return encounter_index.loc[
        ~encounter_ids.isin(medication_supported_encounter_ids)
    ].copy()


def _load_filtered_medication_events_for_eligible_encounters(
    *,
    repository: StandardizedParquetRepository,
    eligible_encounters: pd.DataFrame,
    columns: list[str] | None,
) -> pd.DataFrame:
    encounter_ids = {
        str(value).strip()
        for value in eligible_encounters["encounter_id"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    path = repository.analytical_artifact_path("medication_events")
    return _read_parquet_filtered_by_membership(
        path=path,
        membership_column="encounter_id",
        allowed_values=encounter_ids,
        columns=columns,
    )


def _load_filtered_labevents_for_encounters(
    *,
    repository: StandardizedParquetRepository,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame | None:
    hadm_ids = _identifier_set(encounter_index.get("hadm_id"))
    if not hadm_ids:
        return None
    path = repository.source_table_path(get_table_spec("clinical", "labevents"))
    if not path.exists():
        return None
    return _read_parquet_filtered_by_membership(
        path=path,
        membership_column="hadm_id",
        allowed_values=hadm_ids,
        columns=STATE_BUILD_LABEVENT_COLUMNS,
    )


def _read_parquet_filtered_by_membership(
    *,
    path: Path,
    membership_column: str,
    allowed_values: set[object],
    columns: list[str] | None,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected Parquet artifact at '{path}', but it does not exist.")
    if not allowed_values:
        requested_columns = columns or [membership_column]
        return pd.DataFrame(columns=requested_columns)

    parquet_file = pq.ParquetFile(path)
    filtered_batches: list[pd.DataFrame] = []
    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        batch_frame = batch.to_pandas()
        if membership_column not in batch_frame:
            raise DataLoadError(
                f"Expected column '{membership_column}' while filtering Parquet artifact '{path}'."
            )
        batch_filtered = batch_frame.loc[
            batch_frame[membership_column].isin(allowed_values)
        ].copy()
        if not batch_filtered.empty:
            filtered_batches.append(batch_filtered)

    if not filtered_batches:
        requested_columns = columns or parquet_file.schema.names
        return pd.DataFrame(columns=requested_columns)
    return pd.concat(filtered_batches, ignore_index=True, sort=False)


def _validate_state_build_medication_event_input(dataframe: pd.DataFrame) -> None:
    missing_columns = sorted(
        set(STATE_BUILD_MEDICATION_EVENT_COLUMNS) - set(dataframe.columns)
    )
    if missing_columns:
        raise DataLoadError(
            "State-build medication-event input is missing required columns: "
            + ", ".join(missing_columns)
        )


def _encounter_ids_with_medication_review_evidence(
    medication_events: pd.DataFrame,
) -> set[str]:
    if medication_events.empty:
        return set()

    encounter_ids: set[str] = set()
    if "encounter_id" not in medication_events:
        return encounter_ids

    administration_mask = medication_events["medication_event_type"].isin(
        ["hospital_admin", "ed_pyxis"]
    )
    if administration_mask.any():
        valid_administration_mask = administration_mask & pd.to_datetime(
            medication_events["event_time"],
            errors="coerce",
            format=TIMESTAMP_FORMAT,
        ).notna()
        encounter_ids.update(
            medication_events.loc[
                valid_administration_mask,
                "encounter_id",
            ].astype(str)
        )

    order_mask = medication_events["medication_event_type"] == "hospital_order"
    if order_mask.any():
        valid_order_mask = pd.Series(False, index=medication_events.index)
        for column in ["stoptime", "starttime", "event_time"]:
            if column not in medication_events:
                continue
            valid_order_mask = valid_order_mask | (
                order_mask
                & pd.to_datetime(
                    medication_events[column],
                    errors="coerce",
                    format=TIMESTAMP_FORMAT,
                ).notna()
            )
        encounter_ids.update(
            medication_events.loc[valid_order_mask, "encounter_id"].astype(str)
        )

    return encounter_ids


def _medication_candidate_keys(events: pd.DataFrame) -> pd.Series:
    standardized = _normalized_text_series(events.get("medication_standardized"), index=events.index)
    normalized = _normalized_text_series(events.get("medication_normalized"), index=events.index)
    return standardized.where(standardized.notna(), normalized).astype("object")


def _latest_lab_timestamp_by_hadm_id(
    labevents: pd.DataFrame | None,
) -> dict[int, pd.Timestamp]:
    if labevents is None or labevents.empty or "hadm_id" not in labevents:
        return {}
    labs = labevents.loc[:, ["hadm_id", "charttime"]].copy()
    labs["hadm_id"] = pd.to_numeric(labs["hadm_id"], errors="coerce")
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce", format=TIMESTAMP_FORMAT)
    labs = labs.dropna(subset=["hadm_id", "charttime"])
    if labs.empty:
        return {}
    latest = (
        labs.sort_values(["hadm_id", "charttime"], na_position="last")
        .drop_duplicates(subset=["hadm_id"], keep="last")
    )
    return {
        int(row.hadm_id): row.charttime
        for row in latest.itertuples(index=False)
    }


def _latest_vitals_timestamp_by_subject_stay(
    *,
    triage: pd.DataFrame | None,
    vitalsign: pd.DataFrame | None,
) -> dict[tuple[int, int], pd.Timestamp]:
    frames: list[pd.DataFrame] = []
    if triage is not None and not triage.empty:
        triage_time_column = "intime" if "intime" in triage.columns else "charttime"
        if triage_time_column in triage.columns:
            triage_frame = triage.loc[:, ["subject_id", "stay_id", triage_time_column]].copy()
            triage_frame = triage_frame.rename(columns={triage_time_column: "charttime"})
            frames.append(triage_frame)
    if vitalsign is not None and not vitalsign.empty and "charttime" in vitalsign:
        frames.append(vitalsign.loc[:, ["subject_id", "stay_id", "charttime"]].copy())
    if not frames:
        return {}
    vitals = pd.concat(frames, ignore_index=True, sort=False)
    vitals["subject_id"] = pd.to_numeric(vitals["subject_id"], errors="coerce")
    vitals["stay_id"] = pd.to_numeric(vitals["stay_id"], errors="coerce")
    vitals["charttime"] = pd.to_datetime(vitals["charttime"], errors="coerce", format=TIMESTAMP_FORMAT)
    vitals = vitals.dropna(subset=["subject_id", "stay_id", "charttime"])
    if vitals.empty:
        return {}
    latest = (
        vitals.sort_values(["subject_id", "stay_id", "charttime"], na_position="last")
        .drop_duplicates(subset=["subject_id", "stay_id"], keep="last")
    )
    return {
        (int(row.subject_id), int(row.stay_id)): row.charttime
        for row in latest.itertuples(index=False)
    }


def _encounter_medication_state_rows_for_encounter(
    *,
    encounter: dict[str, object],
    encounter_events: pd.DataFrame,
    review_resolution: ReviewTimestampResolution,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    review_timestamp = review_resolution.review_timestamp
    grouped = encounter_events.groupby("_medication_standardized_key", dropna=False, sort=True)
    for medication_standardized_key, group in grouped:
        if medication_standardized_key is None or pd.isna(medication_standardized_key):
            continue

        active_events = filter_active_medication_events(group, review_timestamp)
        representative_group = active_events if not active_events.empty else group
        representative = deduplicate_active_medication_group(representative_group)
        medication_status = classify_medication_status_at_review_time(group, review_timestamp)
        source_tables_json = _source_tables_json_for_group(group)
        source_record_provenance_json = _source_record_provenance_json_for_group(group)
        semantic_fields = _aggregate_semantic_fields(group, representative=representative)

        rows.append(
            {
                "subject_id": int(representative["subject_id"]),
                "encounter_id": encounter["encounter_id"],
                "hadm_id": _coalesce_identifier(
                    representative.get("hadm_id"),
                    encounter.get("hadm_id"),
                ),
                "stay_id": _coalesce_identifier(
                    representative.get("stay_id"),
                    encounter.get("stay_id"),
                ),
                "review_timestamp": _format_timestamp(review_timestamp),
                "review_timestamp_source": review_resolution.review_timestamp_source,
                "age_proxy": _float_or_none(encounter.get("age_proxy")),
                "age_group": encounter.get("age_group"),
                "review_time_policy_name": review_resolution.review_time_policy_name,
                "review_timestamp_candidate": _format_timestamp(
                    review_resolution.review_timestamp_candidate
                ),
                "review_timestamp_candidate_source": (
                    review_resolution.review_timestamp_candidate_source
                ),
                "review_time_capped_to_discharge_flag": int(
                    review_resolution.review_time_capped_to_discharge_flag
                ),
                "review_time_validated_flag": int(
                    review_resolution.review_time_validated_flag
                ),
                "discharge_boundary": _format_timestamp(
                    review_resolution.discharge_boundary
                ),
                "medication_raw": representative.get("raw_medication_name"),
                "medication_normalized": representative.get("medication_normalized"),
                "medication_standardized": str(medication_standardized_key),
                "medication_standardized_source": semantic_fields[
                    "medication_standardized_source"
                ],
                "rxnorm_rxcui": semantic_fields["rxnorm_rxcui"],
                "rxnorm_matched_term": semantic_fields["rxnorm_matched_term"],
                "rxnorm_term_type": semantic_fields["rxnorm_term_type"],
                "ingredient_standardized": semantic_fields["ingredient_standardized"],
                "ingredient_resolution_status": semantic_fields[
                    "ingredient_resolution_status"
                ],
                "mapping_confidence": semantic_fields["mapping_confidence"],
                "ambiguous_mapping_flag": semantic_fields["ambiguous_mapping_flag"],
                "mapping_candidate_count": semantic_fields["mapping_candidate_count"],
                "medication_mapping_lookup_strategy": semantic_fields[
                    "medication_mapping_lookup_strategy"
                ],
                "medication_class_standardized": semantic_fields[
                    "medication_class_standardized"
                ],
                "medication_status_at_review": medication_status,
                "active_at_review_flag": int(
                    medication_status == MEDICATION_STATUS_ACTIVE_AT_REVIEW
                ),
                "continued_from_home_inferred": int(
                    pd.to_numeric(
                        group["continued_from_home_inferred"],
                        errors="coerce",
                    ).fillna(0).astype(int).max()
                ),
                "newly_started_during_encounter_inferred": int(
                    pd.to_numeric(
                        group["newly_started_during_encounter_inferred"],
                        errors="coerce",
                    ).fillna(0).astype(int).max()
                ),
                "route": representative.get("route"),
                "frequency": representative.get("frequency"),
                "status": representative.get("status"),
                "selected_medication_event_id": representative.get("medication_event_id"),
                "selected_medication_event_type": representative.get(
                    "medication_event_type"
                ),
                "active_event_count_at_review": int(len(active_events)),
                "candidate_event_count": int(len(group)),
                "source_home_medrecon_flag": int(
                    pd.to_numeric(
                        group["source_home_medrecon"],
                        errors="coerce",
                    ).fillna(0).astype(int).max()
                ),
                "source_ed_pyxis_flag": int(
                    pd.to_numeric(
                        group["source_ed_pyxis"],
                        errors="coerce",
                    ).fillna(0).astype(int).max()
                ),
                "source_hospital_order_flag": int(
                    pd.to_numeric(
                        group["source_hospital_order"],
                        errors="coerce",
                    ).fillna(0).astype(int).max()
                ),
                "source_hospital_admin_flag": int(
                    pd.to_numeric(
                        group["source_hospital_admin"],
                        errors="coerce",
                    ).fillna(0).astype(int).max()
                ),
                "source_tables_json": source_tables_json,
                "source_record_provenance_json": source_record_provenance_json,
            }
        )
    return rows


def _source_tables_json_for_group(group: pd.DataFrame) -> str:
    source_tables: set[str] = set()
    for value in group["source_tables_json"].dropna().tolist():
        payload = loads_json_or_none(value)
        if isinstance(payload, list):
            source_tables.update(str(item) for item in payload if item is not None)
    return dumps_json(sorted(source_tables))


def _source_record_provenance_json_for_group(group: pd.DataFrame) -> str:
    provenance_rows: list[dict[str, object]] = []
    group_sorted = group.sort_values(["medication_event_id"], na_position="last")
    for row in group_sorted.to_dict(orient="records"):
        provenance_rows.append(
            {
                "medication_event_id": row.get("medication_event_id"),
                "medication_event_type": row.get("medication_event_type"),
                "event_source_table": row.get("event_source_table"),
                "source_record_provenance": loads_json_or_none(
                    row.get("source_record_provenance_json")
                ),
            }
        )
    return dumps_json(provenance_rows)


def apply_medication_rxnorm_mapping_to_events(
    *,
    medication_events: pd.DataFrame,
    medication_rxnorm_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Attach RxNorm mapping results to candidate events before final state grouping."""
    validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
    mapping = medication_rxnorm_mapping.loc[
        :,
        [
            "medication_query_key",
            "medication_standardized",
            "medication_standardized_source",
            "rxnorm_rxcui",
            "matched_term",
            "matched_term_type",
            "ingredient_standardized",
            "ingredient_resolution_status",
            "mapping_confidence",
            "ambiguous_match_flag",
            "candidate_match_count",
            "lookup_strategy_used",
            "class_assignment_status",
            "class_ids_json",
            "class_labels_json",
        ],
    ].copy()
    mapping = mapping.rename(
        columns={
            "medication_query_key": "_medication_candidate_key",
            "matched_term": "_rxnorm_matched_term",
            "matched_term_type": "_rxnorm_term_type",
            "candidate_match_count": "_mapping_candidate_count",
            "lookup_strategy_used": "_medication_mapping_lookup_strategy",
            "medication_standardized": "_resolved_medication_standardized",
            "medication_standardized_source": "_resolved_medication_standardized_source",
            "rxnorm_rxcui": "_resolved_rxnorm_rxcui",
            "ingredient_standardized": "_resolved_ingredient_standardized",
            "ingredient_resolution_status": "_resolved_ingredient_resolution_status",
            "mapping_confidence": "_resolved_mapping_confidence",
            "ambiguous_match_flag": "_resolved_ambiguous_mapping_flag",
            "class_assignment_status": "_resolved_class_assignment_status",
            "class_ids_json": "_resolved_class_ids_json",
            "class_labels_json": "_resolved_class_labels_json",
        }
    )
    enriched = medication_events.merge(
        mapping,
        how="left",
        on="_medication_candidate_key",
        validate="many_to_one",
    )
    return _apply_unresolved_semantic_defaults(enriched)


def _apply_unresolved_semantic_defaults(medication_events: pd.DataFrame) -> pd.DataFrame:
    enriched = medication_events.copy()
    resolved_standardized = enriched.get("_resolved_medication_standardized")
    if resolved_standardized is None:
        resolved_standardized = pd.Series(pd.NA, index=enriched.index, dtype="object")
    enriched["_medication_standardized_key"] = resolved_standardized.where(
        resolved_standardized.notna(),
        enriched["_medication_candidate_key"],
    )
    enriched["_resolved_medication_standardized_source"] = (
        enriched.get("_resolved_medication_standardized_source", pd.Series(pd.NA, index=enriched.index))
        .where(
            enriched.get("_resolved_medication_standardized_source", pd.Series(pd.NA, index=enriched.index)).notna(),
            "normalized_text_fallback",
        )
    )
    enriched["_resolved_ingredient_resolution_status"] = (
        enriched.get("_resolved_ingredient_resolution_status", pd.Series(pd.NA, index=enriched.index))
        .where(
            enriched.get("_resolved_ingredient_resolution_status", pd.Series(pd.NA, index=enriched.index)).notna(),
            "unresolved_no_match",
        )
    )
    enriched["_resolved_mapping_confidence"] = (
        enriched.get("_resolved_mapping_confidence", pd.Series(pd.NA, index=enriched.index))
        .where(
            enriched.get("_resolved_mapping_confidence", pd.Series(pd.NA, index=enriched.index)).notna(),
            "none",
        )
    )
    enriched["_resolved_ambiguous_mapping_flag"] = (
        pd.to_numeric(
            enriched.get(
                "_resolved_ambiguous_mapping_flag",
                pd.Series(0, index=enriched.index),
            ),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    enriched["_mapping_candidate_count"] = (
        pd.to_numeric(
            enriched.get("_mapping_candidate_count", pd.Series(0, index=enriched.index)),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    if "_medication_mapping_lookup_strategy" not in enriched:
        enriched["_medication_mapping_lookup_strategy"] = "mapping_not_attempted"
    if "_resolved_rxnorm_rxcui" not in enriched:
        enriched["_resolved_rxnorm_rxcui"] = pd.NA
    if "_rxnorm_matched_term" not in enriched:
        enriched["_rxnorm_matched_term"] = pd.NA
    if "_rxnorm_term_type" not in enriched:
        enriched["_rxnorm_term_type"] = pd.NA
    if "_resolved_ingredient_standardized" not in enriched:
        enriched["_resolved_ingredient_standardized"] = pd.NA
    if "_resolved_class_assignment_status" not in enriched:
        enriched["_resolved_class_assignment_status"] = "class_enrichment_not_attempted"
    if "_resolved_class_ids_json" not in enriched:
        enriched["_resolved_class_ids_json"] = "[]"
    if "_resolved_class_labels_json" not in enriched:
        enriched["_resolved_class_labels_json"] = "[]"
    return enriched


def _aggregate_semantic_fields(
    group: pd.DataFrame,
    *,
    representative: pd.Series,
) -> dict[str, object]:
    unique_rxcuis = {
        str(value)
        for value in group["_resolved_rxnorm_rxcui"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    unique_statuses = {
        str(value)
        for value in group["_resolved_ingredient_resolution_status"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    class_assignment_statuses = {
        str(value)
        for value in group["_resolved_class_assignment_status"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    strategies = {
        str(value)
        for value in group["_medication_mapping_lookup_strategy"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    class_labels = _class_labels_for_group(group)
    return {
        "medication_standardized_source": _first_non_empty(
            [representative.get("_resolved_medication_standardized_source")]
            + group["_resolved_medication_standardized_source"].astype(str).tolist()
        )
        or "normalized_text_fallback",
        "rxnorm_rxcui": (
            next(iter(sorted(unique_rxcuis))) if len(unique_rxcuis) == 1 else representative.get("_resolved_rxnorm_rxcui")
        ),
        "rxnorm_matched_term": representative.get("_rxnorm_matched_term"),
        "rxnorm_term_type": representative.get("_rxnorm_term_type"),
        "ingredient_standardized": _first_non_empty(
            [representative.get("_resolved_ingredient_standardized")]
            + group["_resolved_ingredient_standardized"].astype(str).tolist()
        ),
        "ingredient_resolution_status": _preferred_value(
            values=unique_statuses,
            precedence=[
                "resolved_to_ingredient",
                "resolved_via_related_concept",
                "resolved_term_only_no_ingredient",
                "unresolved_ambiguous_multi_hit",
                "unresolved_cache_only_miss",
                "unresolved_api_error",
                "unresolved_no_match",
            ],
            default="unresolved_no_match",
        ),
        "mapping_confidence": _preferred_value(
            values={
                str(value)
                for value in group["_resolved_mapping_confidence"].dropna().astype(str).tolist()
                if str(value).strip()
            },
            precedence=["high", "medium", "low", "none"],
            default="none",
        ),
        "ambiguous_mapping_flag": int(
            pd.to_numeric(group["_resolved_ambiguous_mapping_flag"], errors="coerce")
            .fillna(0)
            .astype(int)
            .max()
        ),
        "mapping_candidate_count": int(
            max(
                pd.to_numeric(group["_mapping_candidate_count"], errors="coerce")
                .fillna(0)
                .astype(int)
                .max(),
                group["_medication_candidate_key"].nunique(),
            )
        ),
        "medication_class_standardized": (
            "|".join(sorted(class_labels)) if class_labels else "unresolved"
        ),
        "class_assignment_status": _preferred_value(
            values=class_assignment_statuses,
            precedence=[
                "supported_scope_class_assigned",
                "supported_scope_unresolved",
                "no_standardized_ingredient_available",
                "class_enrichment_not_attempted",
            ],
            default="class_enrichment_not_attempted",
        ),
        "medication_mapping_lookup_strategy": (
            next(iter(strategies)) if len(strategies) == 1 else "multiple_query_strategies"
        ),
    }


def _encounter_discharge_boundary(encounter: dict[str, object]) -> pd.Timestamp | pd.NaT:
    return latest_valid_timestamp(
        [
            pd.to_datetime(encounter.get("dischtime"), errors="coerce"),
            pd.to_datetime(encounter.get("outtime"), errors="coerce"),
            pd.to_datetime(encounter.get("encounter_end"), errors="coerce"),
        ]
    )


def _medication_candidate_key(row: pd.Series) -> str | None:
    standardized_value = row.get("medication_standardized")
    normalized_value = row.get("medication_normalized")
    for value in [standardized_value, normalized_value]:
        if value is None:
            continue
        if value is pd.NA:
            continue
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _normalized_text_series(
    series: pd.Series | None,
    *,
    index: pd.Index,
) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="string")
    return series.astype("string").str.strip().replace("", pd.NA)


def _review_group_key(
    encounter_id: object,
    medication_standardized_key: object,
) -> tuple[str, str]:
    return str(encounter_id), str(medication_standardized_key)


def _state_int_series(
    series: pd.Series | None,
    *,
    index: pd.Index,
) -> pd.Series:
    if series is None:
        return pd.Series(0, index=index, dtype="int64")
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def _encounter_state_event_start_series(group: pd.DataFrame) -> pd.Series:
    start_columns: list[pd.Series] = []
    for column in ["starttime", "event_time", "encounter_start"]:
        if column in group:
            start_columns.append(
                pd.to_datetime(
                    group[column],
                    errors="coerce",
                    format=TIMESTAMP_FORMAT,
                )
            )
        else:
            start_columns.append(pd.Series(pd.NaT, index=group.index))
    return pd.concat(start_columns, axis=1).bfill(axis=1).iloc[:, 0]


def _encounter_state_priority_rank_series(group: pd.DataFrame) -> pd.Series:
    ranks = pd.Series(0, index=group.index, dtype="int64")
    ranks = ranks.mask(
        _state_int_series(group.get("source_ed_pyxis"), index=group.index) == 1,
        1,
    )
    ranks = ranks.mask(
        _state_int_series(group.get("source_hospital_admin"), index=group.index) == 1,
        2,
    )
    ranks = ranks.mask(
        _state_int_series(group.get("source_home_medrecon"), index=group.index) == 1,
        3,
    )
    ranks = ranks.mask(
        _state_int_series(group.get("source_hospital_order"), index=group.index) == 1,
        4,
    )
    return ranks


def _active_event_mask_for_review_rows(group: pd.DataFrame) -> pd.Series:
    event_types = group["medication_event_type"].astype("string")
    review_times = group["review_timestamp"]
    start_times = group["_event_start"]
    stop_times = group["_event_stop"]
    interval_active = (
        review_times.notna()
        & start_times.notna()
        & (start_times <= review_times)
        & (stop_times.isna() | (stop_times >= review_times))
    )
    point_event_mask = event_types.isin(["ed_pyxis", "hospital_admin"])
    point_active = review_times.notna() & start_times.notna() & (start_times == review_times)
    home_event_mask = event_types == "home_medrecon"
    continued_home = _state_int_series(
        group.get("continued_from_home_inferred"),
        index=group.index,
    ) == 1
    return (
        (point_event_mask & point_active)
        | (home_event_mask & continued_home & interval_active)
        | (~point_event_mask & ~home_event_mask & interval_active)
    )


def _format_timestamp(value: object) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.strftime(TIMESTAMP_FORMAT)


def _coalesce_identifier(*values: object) -> int | None:
    for value in values:
        coerced = _int_or_none(value)
        if coerced is not None:
            return coerced
    return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _binary_as_bool(value: object) -> bool:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return False
    return bool(int(numeric))


def _first_non_empty(values: list[object]) -> str | None:
    for value in values:
        if value is None or value is pd.NA:
            continue
        try:
            if pd.isna(value):
                continue
        except TypeError:
            pass
        text = str(value).strip()
        if text and text.lower() not in {"nan", "<na>", "none"}:
            return text
    return None


def _preferred_value(
    *,
    values: set[str],
    precedence: list[str],
    default: str,
) -> str:
    for candidate in precedence:
        if candidate in values:
            return candidate
    return default


def _class_labels_for_group(group: pd.DataFrame) -> set[str]:
    labels: set[str] = set()
    for value in group["_resolved_class_labels_json"].dropna().tolist():
        payload = loads_json_or_none(value)
        if not isinstance(payload, list):
            continue
        for item in payload:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                labels.add(text)
    return labels


def _identifier_set(series: pd.Series | None) -> set[int]:
    if series is None:
        return set()
    return {
        int(value)
        for value in pd.to_numeric(series, errors="coerce").dropna().astype(int).tolist()
    }


def _load_required_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected analytical artifact at '{path}', but it does not exist.")
    return pd.read_parquet(path)
