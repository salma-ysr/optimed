"""Encounter-relative medication snapshot helpers and builder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.encounters import EncounterIndexBuilder
from opti_med.data_access.medication_events import CanonicalMedicationEventBuilder


SnapshotStrategy = Literal["ed", "hospital", "latest_available"]

MEDICATION_SNAPSHOT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "encounter_id",
    "encounter_source",
    "snapshot_strategy",
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
        self.encounter_builder = EncounterIndexBuilder(settings)
        self.medication_event_builder = CanonicalMedicationEventBuilder(settings)

    def build(self) -> pd.DataFrame:
        """Build the medication snapshot using the configured encounter-relative strategy."""
        encounter_index = self.encounter_builder.build()
        medication_events = self.medication_event_builder.build()
        return build_medication_snapshot(
            medication_events=medication_events,
            encounter_index=encounter_index,
            snapshot_strategy=self.snapshot_strategy,
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
    """Select an encounter-relative snapshot time without using the real-world clock."""
    row = encounter_row if isinstance(encounter_row, dict) else encounter_row.to_dict()
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
    """Return whether a closed interval overlaps a single snapshot timestamp."""
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


def is_medication_event_active_at_snapshot(
    event_row: pd.Series | dict,
    snapshot_time: object,
) -> bool:
    """Determine whether a medication event should count as active at the selected snapshot."""
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

    return interval_overlaps_snapshot(start_time, stop_time, snapshot)


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
) -> pd.DataFrame:
    """Build a deduplicated encounter-relative medication snapshot."""
    if medication_events.empty or encounter_index.empty:
        return pd.DataFrame(columns=MEDICATION_SNAPSHOT_COLUMNS)

    snapshot_rows: list[dict] = []
    encounter_rows = encounter_index.drop_duplicates(subset=["encounter_id"]).copy()
    for encounter in encounter_rows.to_dict(orient="records"):
        snapshot_time = select_snapshot_time_for_encounter(encounter, snapshot_strategy)
        if pd.isna(snapshot_time):
            continue

        encounter_events = medication_events.loc[
            medication_events["encounter_id"] == encounter["encounter_id"]
        ].copy()
        active_events = filter_active_medication_events(encounter_events, snapshot_time)
        if active_events.empty:
            continue

        snapshot_rows.extend(
            _snapshot_rows_for_encounter(
                active_events=active_events,
                encounter=encounter,
                snapshot_time=snapshot_time,
                snapshot_strategy=snapshot_strategy,
            )
        )

    snapshot = pd.DataFrame(snapshot_rows)
    if snapshot.empty:
        return pd.DataFrame(columns=MEDICATION_SNAPSHOT_COLUMNS)
    return snapshot.loc[:, MEDICATION_SNAPSHOT_COLUMNS].sort_values(
        ["subject_id", "snapshot_time", "medication_normalized"],
        na_position="last",
    ).reset_index(drop=True)


def _snapshot_rows_for_encounter(
    *,
    active_events: pd.DataFrame,
    encounter: dict,
    snapshot_time: pd.Timestamp,
    snapshot_strategy: SnapshotStrategy,
) -> list[dict]:
    snapshot_rows: list[dict] = []
    grouped = active_events.groupby("medication_normalized", dropna=False, sort=True)
    for medication_normalized, group in grouped:
        if medication_normalized is None or pd.isna(medication_normalized):
            continue
        representative = deduplicate_active_medication_group(group)
        snapshot_rows.append(
            {
                "subject_id": int(representative["subject_id"]),
                "hadm_id": representative.get("hadm_id"),
                "stay_id": representative.get("stay_id"),
                "encounter_id": encounter["encounter_id"],
                "encounter_source": encounter["encounter_source"],
                "snapshot_strategy": snapshot_strategy,
                "snapshot_time": snapshot_time.strftime("%Y-%m-%d %H:%M:%S"),
                "medication_normalized": representative["medication_normalized"],
                "medication_name": representative.get("medication_name"),
                "raw_medication_name": representative.get("raw_medication_name"),
                "selected_event_id": representative["medication_event_id"],
                "selected_event_type": representative["medication_event_type"],
                "active_event_count": len(group),
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
            }
        )
    return snapshot_rows


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
