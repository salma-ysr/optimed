"""Helpers for collapsing repeated medication rows into canonical continuation episodes."""

from __future__ import annotations

from datetime import date, datetime
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
    working = working.sort_values(
        [*group_columns, start_column, stop_column],
        na_position="last",
        kind="stable",
    ).reset_index(drop=True)

    collapsed_rows: list[dict] = []
    episode_index = 0
    current_group_key: tuple[object, ...] | None = None
    current_segment_payloads: list[dict[str, object]] = []
    current_segment_count = 0
    current_representative: dict[str, object] | None = None
    current_representative_completeness = -1
    current_start = pd.NaT
    current_stop = pd.NaT

    def flush_current() -> None:
        nonlocal episode_index
        nonlocal current_group_key
        nonlocal current_segment_payloads
        nonlocal current_segment_count
        nonlocal current_representative
        nonlocal current_representative_completeness
        nonlocal current_start
        nonlocal current_stop
        if current_representative is None:
            return

        representative = dict(current_representative)
        representative[start_column] = _format_timestamp(current_start)
        representative[stop_column] = _format_timestamp(current_stop)
        representative["medication_episode_id"] = f"{episode_id_prefix}-{episode_index}"
        representative["prescription_segment_count"] = current_segment_count
        representative["prescription_segments_json"] = json.dumps(
            current_segment_payloads,
            sort_keys=True,
        )
        collapsed_rows.append(representative)
        episode_index += 1
        current_group_key = None
        current_segment_payloads = []
        current_segment_count = 0
        current_representative = None
        current_representative_completeness = -1
        current_start = pd.NaT
        current_stop = pd.NaT

    columns = list(working.columns)
    for row_values in working.itertuples(index=False, name=None):
        row = dict(zip(columns, row_values))
        group_key = tuple(_group_key_value(row.get(column_name)) for column_name in group_columns)
        row_start = pd.to_datetime(row.get(start_column), errors="coerce")
        row_stop = pd.to_datetime(row.get(stop_column), errors="coerce")
        if pd.isna(row_start):
            row_start = row_stop
        if pd.isna(row_stop):
            row_stop = row_start

        if current_group_key is None:
            current_group_key = group_key
        elif group_key != current_group_key:
            flush_current()
            current_group_key = group_key

        if current_representative is not None and not _intervals_represent_continuation(
            current_stop=current_stop,
            next_start=row_start,
            gap_tolerance=gap_tolerance,
        ):
            flush_current()
            current_group_key = group_key

        row_completeness = _count_non_null_values(row)
        if current_representative is None or row_completeness > current_representative_completeness:
            current_representative = row
            current_representative_completeness = row_completeness

        current_segment_payloads.append(
            {
                field: _json_ready_value(row.get(field))
                for field in segment_fields
            }
        )
        current_segment_count += 1
        current_start = _min_timestamp(current_start, row_start)
        current_stop = _max_timestamp(current_stop, row_stop)

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


def _count_non_null_values(row: dict[str, object]) -> int:
    return sum(1 for value in row.values() if _value_is_present(value))


def _format_timestamp(value: pd.Timestamp | pd.NaT) -> str | None:
    if pd.isna(value):
        return None
    return value.strftime(TIMESTAMP_FORMAT)


def _json_ready_value(value: object) -> object:
    if not _value_is_present(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime(TIMESTAMP_FORMAT)
    if isinstance(value, datetime):
        return pd.Timestamp(value).strftime(TIMESTAMP_FORMAT)
    if isinstance(value, date):
        return pd.Timestamp(value).strftime(TIMESTAMP_FORMAT)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except ValueError:
            pass
        except TypeError:
            pass
    return value


def _group_key_value(value: object) -> object:
    if not _value_is_present(value):
        return None
    return value


def _min_timestamp(left: pd.Timestamp | pd.NaT, right: pd.Timestamp | pd.NaT) -> pd.Timestamp | pd.NaT:
    if pd.isna(left):
        return right
    if pd.isna(right):
        return left
    return min(left, right)


def _max_timestamp(left: pd.Timestamp | pd.NaT, right: pd.Timestamp | pd.NaT) -> pd.Timestamp | pd.NaT:
    if pd.isna(left):
        return right
    if pd.isna(right):
        return left
    return max(left, right)


def _value_is_present(value: object) -> bool:
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except TypeError:
        return True
    except ValueError:
        return True
