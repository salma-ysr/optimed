"""Persisted encounter-medication-state builder at the review-time modeling grain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_STATE_COLUMNS,
    ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION,
    validate_encounter_index_artifact,
    validate_encounter_medication_state_artifact,
    validate_medication_events_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
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
from opti_med.state import (
    EncounterMedicationStateBuilder,
    EncounterMedicationStateRow,
)
from opti_med.time_semantics import (
    MEDICATION_STATUS_ACTIVE_AT_REVIEW,
    MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW,
    REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_BOUNDARY,
    ReviewTimePolicyName,
    ReviewTimeViolationError,
    assert_review_time_not_after_discharge_when_policy_requires,
    validate_review_time_policy,
)


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_ENCOUNTER_MEDICATION_STATE_OUTPUT_PATH = Path(
    "data/analytical/encounter_medication_state.parquet"
)
DEFAULT_ENCOUNTER_MEDICATION_STATE_POLICY = (
    ReviewTimePolicyName.DISCHARGE_CAPPED_LATEST_AVAILABLE
)
LOOKUP_MODE_DISABLED = "disabled"


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
    """Build one row per encounter medication candidate at review time."""

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
        medication_rxnorm_mapping: pd.DataFrame | None = None,
        labevents: pd.DataFrame | None = None,
        triage: pd.DataFrame | None = None,
        vitalsign: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build the persisted encounter-medication-state artifact."""
        encounter_index = (
            encounter_index
            if encounter_index is not None
            else self.repository.load_analytical_artifact("encounter_index")
        )
        medication_events = (
            medication_events
            if medication_events is not None
            else self.repository.load_analytical_artifact("medication_events")
        )
        validate_encounter_index_artifact(encounter_index)
        validate_medication_events_artifact(medication_events)

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

        if labevents is None:
            loaded = self.repository.load_optional_source_table(
                "clinical",
                "labevents",
                columns=["hadm_id", "charttime"],
            )
            labevents = loaded.dataframe if loaded else None
        if triage is None:
            loaded = self.repository.load_optional_source_table(
                "ed",
                "triage",
                columns=["subject_id", "stay_id", "intime"],
            )
            triage = loaded.dataframe if loaded else None
        if vitalsign is None:
            loaded = self.repository.load_optional_source_table(
                "ed",
                "vitalsign",
                columns=["subject_id", "stay_id", "charttime"],
            )
            vitalsign = loaded.dataframe if loaded else None

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
        medication_rxnorm_mapping: pd.DataFrame | None = None,
        labevents: pd.DataFrame | None = None,
        triage: pd.DataFrame | None = None,
        vitalsign: pd.DataFrame | None = None,
    ) -> list[EncounterMedicationStateRow]:
        """Build strongly typed rows for downstream ML-facing layers."""
        dataframe = self.build(
            encounter_index=encounter_index,
            medication_events=medication_events,
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
    """Build one row per encounter-review-time medication candidate."""
    if encounter_index.empty or medication_events.empty:
        return pd.DataFrame(columns=ENCOUNTER_MEDICATION_STATE_COLUMNS)

    policy_name = validate_review_time_policy(review_time_policy)
    build_run_id = f"encounter-medication-state-{uuid4().hex[:12]}"
    encounter_rows = encounter_index.drop_duplicates(subset=["encounter_id"]).copy()
    encounter_rows = encounter_rows.sort_values(
        ["subject_id", "encounter_start", "encounter_id"],
        na_position="last",
    )

    events = medication_events.copy()
    events["_medication_candidate_key"] = events.apply(
        _medication_candidate_key,
        axis=1,
    )
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

    review_rows: list[dict[str, object]] = []
    for encounter in encounter_rows.to_dict(orient="records"):
        encounter_events = events.loc[
            events["encounter_id"] == encounter["encounter_id"]
        ].copy()
        if encounter_events.empty:
            continue

        review_resolution = resolve_policy_safe_review_timestamp_for_encounter(
            encounter=encounter,
            medication_events=events,
            labevents=labevents,
            triage=triage,
            vitalsign=vitalsign,
            review_time_policy=policy_name,
        )
        if pd.isna(review_resolution.review_timestamp):
            continue

        review_rows.extend(
            _encounter_medication_state_rows_for_encounter(
                encounter=encounter,
                encounter_events=encounter_events,
                review_resolution=review_resolution,
            )
        )

    review = pd.DataFrame(review_rows)
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


def resolve_policy_safe_review_timestamp_for_encounter(
    *,
    encounter: dict | pd.Series,
    medication_events: pd.DataFrame | None,
    labevents: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
    review_time_policy: ReviewTimePolicyName | str = DEFAULT_ENCOUNTER_MEDICATION_STATE_POLICY,
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
