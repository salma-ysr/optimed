"""Encounter-relative medication snapshot helpers and builder.

The dossier's "current medications" view is encounter-relative by design.
MIMIC timestamps are patient-shifted, so review-time logic must stay inside the
selected encounter instead of comparing against the real-world clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.encounters import EncounterIndexBuilder
from opti_med.data_access.medication_events import CanonicalMedicationEventBuilder
from opti_med.standardized import StandardizedParquetRepository
from opti_med.time_semantics.constants import (
    MEDICATION_STATUS_ACTIVE_AT_REVIEW,
    MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW,
    MEDICATION_STATUS_INACTIVE_BEFORE_REVIEW,
    MEDICATION_STATUS_PRE_ADMISSION_ONLY,
    REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_BOUNDARY,
    REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_END,
    REVIEW_TIMESTAMP_SOURCE_LAB,
    REVIEW_TIMESTAMP_SOURCE_MEDICATION_ADMINISTRATION,
    REVIEW_TIMESTAMP_SOURCE_MEDICATION_ORDER,
    REVIEW_TIMESTAMP_SOURCE_VITALS,
)


SnapshotStrategy = Literal["ed", "hospital", "latest_available"]
# Keep this type-level literal set aligned with the shared runtime labels in
# `opti_med.time_semantics.constants`.
MedicationReviewStatus = Literal[
    "active_at_review_time",
    "inactive_before_review_time",
    "pre_admission_only",
    "activity_uncertain_at_review_time",
]

MEDICATION_SNAPSHOT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "encounter_id",
    "encounter_source",
    "snapshot_strategy",
    "review_timestamp",
    "review_timestamp_source",
    "snapshot_time",
    "medication_normalized",
    "medication_name",
    "raw_medication_name",
    "selected_event_id",
    "selected_event_type",
    "active_event_count",
    "source_home_medrecon",
    "source_ed_pyxis",
    "source_hospital_order",
    "source_hospital_admin",
    "continued_from_home_inferred",
    "newly_started_during_encounter_inferred",
    "route",
    "frequency",
    "status",
    "dose_value",
    "dose_unit",
]

MEDICATION_REVIEW_COLUMNS = [
    *MEDICATION_SNAPSHOT_COLUMNS,
    "medication_status",
    "last_active_time",
    "all_event_count",
    "selected_event_count",
]

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class MedicationSnapshotBuildResult:
    """Built medication snapshot and its output path."""

    dataframe: pd.DataFrame
    output_path: Path


class MedicationSnapshotBuilder:
    """Build one encounter-relative medication snapshot row per patient-medication."""

    def __init__(self, settings: Settings, snapshot_strategy: SnapshotStrategy | None = None) -> None:
        self.settings = settings
        self.snapshot_strategy = validate_snapshot_strategy(
            snapshot_strategy or settings.snapshot_strategy
        )
        self.repository = StandardizedParquetRepository(settings)
        self.encounter_builder = EncounterIndexBuilder(settings)
        self.medication_event_builder = CanonicalMedicationEventBuilder(settings)

    def build(self) -> pd.DataFrame:
        """Build the medication snapshot using the configured encounter-relative strategy."""
        encounter_index = self.encounter_builder.build()
        medication_events = self.medication_event_builder.build(encounter_index=encounter_index)
        labevents_loaded = self.repository.load_source_table("clinical", "labevents")
        triage_loaded = self.repository.load_optional_source_table("ed", "triage")
        vitalsign_loaded = self.repository.load_optional_source_table("ed", "vitalsign")
        return build_medication_snapshot(
            medication_events=medication_events,
            encounter_index=encounter_index,
            snapshot_strategy=self.snapshot_strategy,
            labevents=labevents_loaded.dataframe if labevents_loaded else None,
            triage=triage_loaded.dataframe if triage_loaded else None,
            vitalsign=vitalsign_loaded.dataframe if vitalsign_loaded else None,
        )

    def save(
        self, dataframe: pd.DataFrame, output_path: Path | None = None
    ) -> MedicationSnapshotBuildResult:
        """Persist the medication snapshot to a CSV file."""
        target_path = output_path or self.settings.medication_snapshot_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(target_path, index=False)
        return MedicationSnapshotBuildResult(dataframe=dataframe, output_path=target_path)


def validate_snapshot_strategy(value: str) -> SnapshotStrategy:
    """Validate the supported encounter-relative snapshot strategies."""
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in {"ed", "hospital", "latest_available"}:
        raise ValueError(
            "Unsupported snapshot strategy. Expected one of: ed, hospital, latest_available."
        )
    return normalized  # type: ignore[return-value]


def select_snapshot_time_for_encounter(
    encounter_row: pd.Series | dict,
    strategy: SnapshotStrategy,
) -> pd.Timestamp | pd.NaT:
    """Backward-compatible wrapper around encounter-relative review timestamp selection."""
    return select_review_timestamp_for_encounter(encounter_row, strategy=strategy)


def select_review_timestamp_for_encounter(
    encounter_row: pd.Series | dict,
    *,
    strategy: SnapshotStrategy,
    medication_events: pd.DataFrame | None = None,
    labevents: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
) -> pd.Timestamp | pd.NaT:
    """Select the dossier review timestamp using only encounter-local evidence.

    Priority order is explicit and conservative:
    1. medication administrations
    2. medication orders
    3. labs
    4. vitals / triage
    5. encounter end markers
    """
    row = encounter_row if isinstance(encounter_row, dict) else encounter_row.to_dict()
    if strategy == "latest_available":
        for candidate in [
            _latest_medication_administration_timestamp(row, medication_events),
            _latest_medication_order_timestamp(row, medication_events),
            _latest_lab_timestamp(row, labevents),
            _latest_vitals_timestamp(row, triage=triage, vitalsign=vitalsign),
            _latest_encounter_boundary_timestamp(row),
        ]:
            if pd.notna(candidate):
                return candidate

    intime = pd.to_datetime(row.get("intime"), errors="coerce")
    outtime = pd.to_datetime(row.get("outtime"), errors="coerce")
    admittime = pd.to_datetime(row.get("admittime"), errors="coerce")
    dischtime = pd.to_datetime(row.get("dischtime"), errors="coerce")
    encounter_end = pd.to_datetime(row.get("encounter_end"), errors="coerce")
    encounter_start = pd.to_datetime(row.get("encounter_start"), errors="coerce")

    if strategy == "ed":
        return first_valid_timestamp([outtime, intime, encounter_end, encounter_start])
    if strategy == "hospital":
        return first_valid_timestamp([dischtime, admittime, outtime, encounter_end, encounter_start])
    return latest_valid_timestamp([intime, outtime, admittime, dischtime, encounter_end, encounter_start])


def select_review_timestamp_metadata_for_encounter(
    encounter_row: pd.Series | dict,
    *,
    strategy: SnapshotStrategy,
    medication_events: pd.DataFrame | None = None,
    labevents: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
) -> tuple[pd.Timestamp | pd.NaT, str]:
    """Return both the encounter-relative review timestamp and the winning evidence source."""
    row = encounter_row if isinstance(encounter_row, dict) else encounter_row.to_dict()
    if strategy == "latest_available":
        priority_candidates = [
            (
                REVIEW_TIMESTAMP_SOURCE_MEDICATION_ADMINISTRATION,
                _latest_medication_administration_timestamp(row, medication_events),
            ),
            (
                REVIEW_TIMESTAMP_SOURCE_MEDICATION_ORDER,
                _latest_medication_order_timestamp(row, medication_events),
            ),
            (REVIEW_TIMESTAMP_SOURCE_LAB, _latest_lab_timestamp(row, labevents)),
            (
                REVIEW_TIMESTAMP_SOURCE_VITALS,
                _latest_vitals_timestamp(row, triage=triage, vitalsign=vitalsign),
            ),
            (
                REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_END,
                _latest_encounter_boundary_timestamp(row),
            ),
        ]
        for source_name, candidate in priority_candidates:
            if pd.notna(candidate):
                return candidate, source_name

    return (
        select_review_timestamp_for_encounter(row, strategy=strategy),
        REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_BOUNDARY,
    )


def first_valid_timestamp(values: list[pd.Timestamp | pd.NaT]) -> pd.Timestamp | pd.NaT:
    """Return the first non-null timestamp in order."""
    for value in values:
        if pd.notna(value):
            return value
    return pd.NaT


def latest_valid_timestamp(values: list[pd.Timestamp | pd.NaT]) -> pd.Timestamp | pd.NaT:
    """Return the latest non-null timestamp in the list."""
    valid_values = [value for value in values if pd.notna(value)]
    if not valid_values:
        return pd.NaT
    return max(valid_values)


def interval_overlaps_snapshot(
    start_time: object,
    stop_time: object,
    snapshot_time: object,
) -> bool:
    """Return whether a closed interval overlaps a single review timestamp."""
    start = pd.to_datetime(start_time, errors="coerce")
    stop = pd.to_datetime(stop_time, errors="coerce")
    snapshot = pd.to_datetime(snapshot_time, errors="coerce")
    if pd.isna(snapshot) or pd.isna(start):
        return False
    if snapshot < start:
        return False
    if pd.notna(stop) and snapshot > stop:
        return False
    return True


def interval_overlaps_review_timestamp(
    start_time: object,
    stop_time: object,
    review_timestamp: object,
) -> bool:
    """Return whether an interval overlaps the encounter-relative review timestamp."""
    return interval_overlaps_snapshot(start_time, stop_time, review_timestamp)


def is_medication_event_active_at_snapshot(
    event_row: pd.Series | dict,
    snapshot_time: object,
) -> bool:
    """Determine whether a medication event should count as active at review time."""
    row = event_row if isinstance(event_row, dict) else event_row.to_dict()
    event_type = str(row.get("medication_event_type", ""))
    start_time = coalesce_event_start(row)
    stop_time = row.get("stoptime")
    snapshot = pd.to_datetime(snapshot_time, errors="coerce")

    if pd.isna(snapshot):
        return False

    if event_type in {"ed_pyxis", "hospital_admin"}:
        # Point events are only active exactly at the event timestamp unless a richer duration is added later.
        point_time = pd.to_datetime(start_time, errors="coerce")
        return pd.notna(point_time) and point_time == snapshot

    if event_type == "home_medrecon":
        # Medication reconciliation is historical by default. Treat it as current only when
        # the encounter provides explicit continuation evidence.
        return bool(int(row.get("continued_from_home_inferred", 0)) == 1) and interval_overlaps_review_timestamp(
            start_time,
            stop_time,
            snapshot,
        )

    return interval_overlaps_review_timestamp(start_time, stop_time, snapshot)


def coalesce_event_start(event_row: pd.Series | dict) -> object:
    """Pick the most useful start timestamp for interval-based snapshot logic."""
    row = event_row if isinstance(event_row, dict) else event_row.to_dict()
    for column in ["starttime", "event_time", "encounter_start"]:
        value = pd.to_datetime(row.get(column), errors="coerce")
        if pd.notna(value):
            return value
    return pd.NaT


def filter_active_medication_events(
    medication_events: pd.DataFrame,
    snapshot_time: object,
) -> pd.DataFrame:
    """Filter a medication event table down to rows active at the given snapshot."""
    if medication_events.empty:
        return medication_events.copy()
    active_mask = medication_events.apply(
        lambda row: is_medication_event_active_at_snapshot(row, snapshot_time),
        axis=1,
    )
    return medication_events.loc[active_mask].copy()


def build_medication_snapshot(
    *,
    medication_events: pd.DataFrame,
    encounter_index: pd.DataFrame,
    snapshot_strategy: SnapshotStrategy,
    labevents: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a deduplicated encounter-relative current-medication snapshot."""
    # The saved snapshot artifact is the active source for dossier-style "current medication"
    # views. Future encounter-medication-state builders should target the dedicated state
    # interface instead of extending this backward-compatible CSV shape.
    review = build_encounter_medication_review(
        medication_events=medication_events,
        encounter_index=encounter_index,
        snapshot_strategy=snapshot_strategy,
        labevents=labevents,
        triage=triage,
        vitalsign=vitalsign,
    )
    if review.empty:
        return pd.DataFrame(columns=MEDICATION_SNAPSHOT_COLUMNS)
    snapshot = review.loc[
        review["medication_status"] == MEDICATION_STATUS_ACTIVE_AT_REVIEW
    ].copy()
    if snapshot.empty:
        return pd.DataFrame(columns=MEDICATION_SNAPSHOT_COLUMNS)
    snapshot["snapshot_time"] = snapshot["review_timestamp"]
    return snapshot.loc[:, MEDICATION_SNAPSHOT_COLUMNS].reset_index(drop=True)


def build_encounter_medication_review(
    *,
    medication_events: pd.DataFrame,
    encounter_index: pd.DataFrame,
    snapshot_strategy: SnapshotStrategy,
    labevents: pd.DataFrame | None = None,
    triage: pd.DataFrame | None = None,
    vitalsign: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row per encounter medication with status at the review timestamp.

    This is the active saved-artifact path for encounter-relative medication review. It
    predates the future encounter-medication-state interface and therefore still returns a
    backward-compatible CSV-oriented shape.
    """
    if medication_events.empty or encounter_index.empty:
        return pd.DataFrame(columns=MEDICATION_REVIEW_COLUMNS)

    review_rows: list[dict] = []
    encounter_rows = encounter_index.drop_duplicates(subset=["encounter_id"]).copy()
    for encounter in encounter_rows.to_dict(orient="records"):
        review_timestamp, review_timestamp_source = select_review_timestamp_metadata_for_encounter(
            encounter,
            strategy=snapshot_strategy,
            medication_events=medication_events,
            labevents=labevents,
            triage=triage,
            vitalsign=vitalsign,
        )
        if pd.isna(review_timestamp):
            continue

        encounter_events = medication_events.loc[
            medication_events["encounter_id"] == encounter["encounter_id"]
        ].copy()
        if encounter_events.empty:
            continue

        review_rows.extend(
            _review_rows_for_encounter(
                encounter_events=encounter_events,
                encounter=encounter,
                review_timestamp=review_timestamp,
                review_timestamp_source=review_timestamp_source,
                snapshot_strategy=snapshot_strategy,
            )
        )

    review = pd.DataFrame(review_rows)
    if review.empty:
        return pd.DataFrame(columns=MEDICATION_REVIEW_COLUMNS)
    return review.loc[:, MEDICATION_REVIEW_COLUMNS].sort_values(
        ["subject_id", "review_timestamp", "medication_normalized"],
        na_position="last",
    ).reset_index(drop=True)


def build_review_rows_for_encounter(
    *,
    encounter_events: pd.DataFrame,
    encounter: dict,
    review_timestamp: object,
    review_timestamp_source: str,
    snapshot_strategy: SnapshotStrategy,
) -> pd.DataFrame:
    """Build medication review rows for one selected encounter."""
    review = pd.DataFrame(
        _review_rows_for_encounter(
            encounter_events=encounter_events,
            encounter=encounter,
            review_timestamp=pd.to_datetime(review_timestamp, errors="coerce"),
            review_timestamp_source=review_timestamp_source,
            snapshot_strategy=snapshot_strategy,
        )
    )
    if review.empty:
        return pd.DataFrame(columns=MEDICATION_REVIEW_COLUMNS)
    return review.loc[:, MEDICATION_REVIEW_COLUMNS].reset_index(drop=True)


def _review_rows_for_encounter(
    *,
    encounter_events: pd.DataFrame,
    encounter: dict,
    review_timestamp: pd.Timestamp,
    review_timestamp_source: str,
    snapshot_strategy: SnapshotStrategy,
) -> list[dict]:
    review_rows: list[dict] = []
    grouped = encounter_events.groupby("medication_normalized", dropna=False, sort=True)
    for medication_normalized, group in grouped:
        if medication_normalized is None or pd.isna(medication_normalized):
            continue
        medication_status = classify_medication_status_at_review_time(group, review_timestamp)
        selected_events = filter_active_medication_events(group, review_timestamp)
        representative_group = selected_events if not selected_events.empty else group
        representative = deduplicate_active_medication_group(representative_group)
        review_rows.append(
            {
                "subject_id": int(representative["subject_id"]),
                "hadm_id": representative.get("hadm_id"),
                "stay_id": representative.get("stay_id"),
                "encounter_id": encounter["encounter_id"],
                "encounter_source": encounter["encounter_source"],
                "snapshot_strategy": snapshot_strategy,
                "review_timestamp": review_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "review_timestamp_source": review_timestamp_source,
                "snapshot_time": review_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "medication_normalized": representative["medication_normalized"],
                "medication_name": representative.get("medication_name"),
                "raw_medication_name": representative.get("raw_medication_name"),
                "selected_event_id": representative["medication_event_id"],
                "selected_event_type": representative["medication_event_type"],
                "active_event_count": len(selected_events),
                "source_home_medrecon": int(group["source_home_medrecon"].max()),
                "source_ed_pyxis": int(group["source_ed_pyxis"].max()),
                "source_hospital_order": int(group["source_hospital_order"].max()),
                "source_hospital_admin": int(group["source_hospital_admin"].max()),
                "continued_from_home_inferred": int(group["continued_from_home_inferred"].max()),
                "newly_started_during_encounter_inferred": int(
                    group["newly_started_during_encounter_inferred"].max()
                ),
                "route": representative.get("route"),
                "frequency": representative.get("frequency"),
                "status": representative.get("status"),
                "dose_value": representative.get("dose_value"),
                "dose_unit": representative.get("dose_unit"),
                "medication_status": medication_status,
                "last_active_time": _last_active_time_for_group(group),
                "all_event_count": len(group),
                "selected_event_count": len(selected_events),
            }
        )
    return review_rows


def classify_medication_status_at_review_time(
    medication_group: pd.DataFrame,
    review_timestamp: object,
) -> MedicationReviewStatus:
    """Classify one encounter medication relative to the encounter review timestamp."""
    if medication_group.empty:
        return MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW

    active_events = filter_active_medication_events(medication_group, review_timestamp)
    active_non_home = active_events.loc[active_events["source_home_medrecon"] != 1]
    if not active_non_home.empty:
        return MEDICATION_STATUS_ACTIVE_AT_REVIEW

    has_non_home = bool((medication_group["source_home_medrecon"] != 1).any())
    if not has_non_home:
        return MEDICATION_STATUS_PRE_ADMISSION_ONLY

    latest_non_home_stop = latest_valid_timestamp(
        [
            pd.to_datetime(row.get("stoptime"), errors="coerce")
            for row in medication_group.to_dict(orient="records")
            if int(row.get("source_home_medrecon", 0)) != 1
        ]
    )
    latest_non_home_time = latest_valid_timestamp(
        [
            pd.to_datetime(coalesce_event_start(row), errors="coerce")
            for row in medication_group.to_dict(orient="records")
            if int(row.get("source_home_medrecon", 0)) != 1
        ]
        + [latest_non_home_stop]
    )
    review = pd.to_datetime(review_timestamp, errors="coerce")
    if pd.notna(review) and pd.notna(latest_non_home_time) and latest_non_home_time < review:
        return MEDICATION_STATUS_INACTIVE_BEFORE_REVIEW
    return MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW


def deduplicate_active_medication_group(group: pd.DataFrame) -> pd.Series:
    """Choose a single representative row for one active normalized medication."""
    ranked = group.copy()
    ranked["priority_rank"] = ranked.apply(_snapshot_priority_rank, axis=1)
    ranked["sort_time"] = ranked.apply(coalesce_event_start, axis=1)
    ranked = ranked.sort_values(
        ["priority_rank", "sort_time", "pharmacy_enriched_flag", "medication_event_id"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return ranked.iloc[0]


def _snapshot_priority_rank(row: pd.Series) -> int:
    """Prefer durable hospital state over point events when selecting one snapshot row."""
    if int(row.get("source_hospital_order", 0)) == 1:
        return 4
    if int(row.get("source_home_medrecon", 0)) == 1:
        return 3
    if int(row.get("source_hospital_admin", 0)) == 1:
        return 2
    if int(row.get("source_ed_pyxis", 0)) == 1:
        return 1
    return 0


def summarize_medication_snapshot(dataframe: pd.DataFrame) -> list[str]:
    """Return compact summaries for the medication snapshot table."""
    return [
        f"rows={len(dataframe):,}, columns={dataframe.shape[1]}",
        f"unique_subjects={dataframe['subject_id'].nunique():,}" if not dataframe.empty else "unique_subjects=0",
        (
            f"continued_from_home_rows={int(dataframe['continued_from_home_inferred'].sum()):,}"
            if not dataframe.empty
            else "continued_from_home_rows=0"
        ),
        (
            f"newly_started_rows={int(dataframe['newly_started_during_encounter_inferred'].sum()):,}"
            if not dataframe.empty
            else "newly_started_rows=0"
        ),
    ]


def _last_active_time_for_group(group: pd.DataFrame) -> str | None:
    """Return the latest encounter-local time suggesting the medication was active."""
    candidates: list[pd.Timestamp | pd.NaT] = []
    for row in group.to_dict(orient="records"):
        candidates.append(pd.to_datetime(row.get("stoptime"), errors="coerce"))
        candidates.append(pd.to_datetime(coalesce_event_start(row), errors="coerce"))
        candidates.append(pd.to_datetime(row.get("event_time"), errors="coerce"))
    latest = latest_valid_timestamp(candidates)
    if pd.isna(latest):
        return None
    return latest.strftime(TIMESTAMP_FORMAT)


def _latest_medication_administration_timestamp(
    encounter_row: pd.Series | dict,
    medication_events: pd.DataFrame | None,
) -> pd.Timestamp | pd.NaT:
    if medication_events is None or medication_events.empty:
        return pd.NaT
    encounter_id = (encounter_row if isinstance(encounter_row, dict) else encounter_row.to_dict()).get("encounter_id")
    events = medication_events.loc[
        (medication_events["encounter_id"] == encounter_id)
        & (medication_events["medication_event_type"].isin(["hospital_admin", "ed_pyxis"]))
    ].copy()
    return latest_valid_timestamp(
        _coerce_timestamp_series(events.get("event_time", pd.Series(dtype=object))).tolist()
    )


def _latest_medication_order_timestamp(
    encounter_row: pd.Series | dict,
    medication_events: pd.DataFrame | None,
) -> pd.Timestamp | pd.NaT:
    if medication_events is None or medication_events.empty:
        return pd.NaT
    encounter_id = (encounter_row if isinstance(encounter_row, dict) else encounter_row.to_dict()).get("encounter_id")
    events = medication_events.loc[
        (medication_events["encounter_id"] == encounter_id)
        & (medication_events["medication_event_type"] == "hospital_order")
    ].copy()
    candidates: list[pd.Timestamp | pd.NaT] = []
    for column in ["stoptime", "starttime", "event_time"]:
        if column in events.columns:
            candidates.extend(_coerce_timestamp_series(events[column]).tolist())
    return latest_valid_timestamp(candidates)


def _latest_lab_timestamp(
    encounter_row: pd.Series | dict,
    labevents: pd.DataFrame | None,
) -> pd.Timestamp | pd.NaT:
    row = encounter_row if isinstance(encounter_row, dict) else encounter_row.to_dict()
    hadm_id = row.get("hadm_id")
    if labevents is None or labevents.empty or hadm_id is None or pd.isna(hadm_id):
        return pd.NaT
    labs = labevents.loc[labevents["hadm_id"] == hadm_id, ["charttime"]].copy()
    return latest_valid_timestamp(_coerce_timestamp_series(labs["charttime"]).tolist())


def _latest_vitals_timestamp(
    encounter_row: pd.Series | dict,
    *,
    triage: pd.DataFrame | None,
    vitalsign: pd.DataFrame | None,
) -> pd.Timestamp | pd.NaT:
    row = encounter_row if isinstance(encounter_row, dict) else encounter_row.to_dict()
    subject_id = row.get("subject_id")
    stay_id = row.get("stay_id")
    candidates: list[pd.Timestamp | pd.NaT] = []
    if triage is not None and not triage.empty and subject_id is not None and stay_id is not None:
        triage_rows = triage.loc[
            (triage["subject_id"] == subject_id) & (triage["stay_id"] == stay_id)
        ].copy()
        if "intime" in triage_rows.columns:
            candidates.extend(_coerce_timestamp_series(triage_rows["intime"]).tolist())
    if vitalsign is not None and not vitalsign.empty and subject_id is not None and stay_id is not None:
        vitals_rows = vitalsign.loc[
            (vitalsign["subject_id"] == subject_id) & (vitalsign["stay_id"] == stay_id)
        ].copy()
        if "charttime" in vitals_rows.columns:
            candidates.extend(_coerce_timestamp_series(vitals_rows["charttime"]).tolist())
    return latest_valid_timestamp(candidates)


def _latest_encounter_boundary_timestamp(encounter_row: pd.Series | dict) -> pd.Timestamp | pd.NaT:
    row = encounter_row if isinstance(encounter_row, dict) else encounter_row.to_dict()
    return latest_valid_timestamp(
        [
            pd.to_datetime(row.get("dischtime"), errors="coerce"),
            pd.to_datetime(row.get("outtime"), errors="coerce"),
            pd.to_datetime(row.get("encounter_end"), errors="coerce"),
            pd.to_datetime(row.get("admittime"), errors="coerce"),
            pd.to_datetime(row.get("intime"), errors="coerce"),
            pd.to_datetime(row.get("encounter_start"), errors="coerce"),
        ]
    )


def _coerce_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse the canonical MIMIC timestamp string format without per-element inference."""
    return pd.to_datetime(series, errors="coerce", format=TIMESTAMP_FORMAT)
