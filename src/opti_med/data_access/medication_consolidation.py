"""Helpers for collapsing repeated medication rows into canonical continuation episodes."""

from __future__ import annotations

import json
import re

import pandas as pd


NORMALIZATION_STOPWORDS = {
    "dr",
    "er",
    "xr",
    "sr",
    "cr",
    "ec",
    "ir",
    "mg",
    "mcg",
    "g",
    "ml",
    "tablet",
    "tablets",
    "tab",
    "tabs",
    "capsule",
    "capsules",
    "cap",
    "caps",
}

EPISODE_CONTINUATION_GAP = pd.Timedelta(hours=12)
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def normalize_medication_name(raw_value: object) -> str | None:
    """Normalize raw medication strings into a stable lower-case matching key."""
    if raw_value is None or pd.isna(raw_value):
        return None
    text = str(raw_value).strip().lower()
    if not text or text in {"0", "nan", "none", "___"}:
        return None
    text = text.lstrip("*")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token and token not in NORMALIZATION_STOPWORDS]
    if not tokens:
        return None
    return " ".join(tokens)


def collapse_continuation_intervals(
    dataframe: pd.DataFrame,
    *,
    group_columns: list[str],
    start_column: str,
    stop_column: str,
    segment_fields: list[str],
    episode_id_prefix: str,
    gap_tolerance: pd.Timedelta = EPISODE_CONTINUATION_GAP,
) -> pd.DataFrame:
    """Collapse exact-name rows whose intervals overlap, touch, or are separated by a tiny gap.

    The rule is intentionally conservative:
    - exact match only on the caller-provided grouping key
    - intervals must overlap, touch, or be within ``gap_tolerance``
    - the merged row keeps canonical start/stop and preserves underlying segments as JSON
    """
    if dataframe.empty:
        return dataframe.copy()

    working = dataframe.copy()
    working[start_column] = pd.to_datetime(working[start_column], errors="coerce")
    working[stop_column] = pd.to_datetime(working[stop_column], errors="coerce")

    collapsed_rows: list[dict] = []
    episode_index = 0
    for _, group in working.groupby(group_columns, dropna=False, sort=False):
        sorted_group = group.sort_values([start_column, stop_column], na_position="last").reset_index(drop=True)

        current_rows: list[dict] = []
        current_start = pd.NaT
        current_stop = pd.NaT

        def flush_current() -> None:
            nonlocal episode_index, current_rows, current_start, current_stop
            if not current_rows:
                return

            representative = _select_representative_row(pd.DataFrame(current_rows))
            representative[start_column] = _format_timestamp(current_start)
            representative[stop_column] = _format_timestamp(current_stop)
            representative["medication_episode_id"] = f"{episode_id_prefix}-{episode_index}"
            representative["prescription_segment_count"] = len(current_rows)
            representative["prescription_segments_json"] = json.dumps(
                [
                    {
                        field: _json_ready_value(row.get(field))
                        for field in segment_fields
                    }
                    for row in current_rows
                ],
                sort_keys=True,
            )
            collapsed_rows.append(representative)
            episode_index += 1
            current_rows = []
            current_start = pd.NaT
            current_stop = pd.NaT

        for row in sorted_group.to_dict(orient="records"):
            row_start = pd.to_datetime(row.get(start_column), errors="coerce")
            row_stop = pd.to_datetime(row.get(stop_column), errors="coerce")
            if pd.isna(row_start):
                row_start = row_stop
            if pd.isna(row_stop):
                row_stop = row_start

            if not current_rows:
                current_rows = [row]
                current_start = row_start
                current_stop = row_stop
                continue

            if _intervals_represent_continuation(
                current_stop=current_stop,
                next_start=row_start,
                gap_tolerance=gap_tolerance,
            ):
                current_rows.append(row)
                current_start = _min_timestamp(current_start, row_start)
                current_stop = _max_timestamp(current_stop, row_stop)
            else:
                flush_current()
                current_rows = [row]
                current_start = row_start
                current_stop = row_stop

        flush_current()

    return pd.DataFrame(collapsed_rows)


def _intervals_represent_continuation(
    *,
    current_stop: pd.Timestamp | pd.NaT,
    next_start: pd.Timestamp | pd.NaT,
    gap_tolerance: pd.Timedelta,
) -> bool:
    if pd.isna(next_start):
        return False
    if pd.isna(current_stop):
        return True
    return next_start <= current_stop + gap_tolerance


def _select_representative_row(group: pd.DataFrame) -> dict:
    ranked = group.copy()
    ranked["data_completeness"] = ranked.notna().sum(axis=1)
    ranked = ranked.sort_values("data_completeness", ascending=False).reset_index(drop=True)
    return ranked.iloc[0].drop(labels=["data_completeness"]).to_dict()


def _format_timestamp(value: pd.Timestamp | pd.NaT) -> str | None:
    if pd.isna(value):
        return None
    return value.strftime(TIMESTAMP_FORMAT)


def _json_ready_value(value: object) -> object:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.notna(timestamp):
        return timestamp.strftime(TIMESTAMP_FORMAT)
    if value is None or pd.isna(value):
        return None
    return value


def _min_timestamp(left: pd.Timestamp | pd.NaT, right: pd.Timestamp | pd.NaT) -> pd.Timestamp | pd.NaT:
    if pd.isna(left):
        return right
    if pd.isna(right):
        return left
    return min(left, right)


def _max_timestamp(left: pd.Timestamp | pd.NaT, right: pd.Timestamp | pd.NaT) -> pd.Timestamp | pd.NaT:
    if pd.isna(left) or pd.isna(right):
        return pd.NaT
    return max(left, right)
