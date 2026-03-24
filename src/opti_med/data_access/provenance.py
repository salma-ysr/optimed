"""Shared provenance helpers for analytical artifact builders."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def json_ready_value(value: object) -> object:
    """Convert pandas and datetime scalars into JSON-safe values."""
    if value is None:
        return None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    if isinstance(value, pd.Timestamp):
        return value.strftime(TIMESTAMP_FORMAT)
    if isinstance(value, datetime):
        return value.strftime(TIMESTAMP_FORMAT)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive branch
            pass
    return value


def source_provenance_payload(
    row: pd.Series,
    *,
    role_name: str,
    standardized_reference: dict[str, object] | None,
    source_metadata_columns: tuple[str, ...],
) -> dict[str, object]:
    """Build one provenance payload for one upstream source row."""
    row_source_metadata = {
        column_name: json_ready_value(row.get(column_name))
        for column_name in source_metadata_columns
    }
    return {
        "role": role_name,
        "row_source_metadata": row_source_metadata,
        "standardized_reference": standardized_reference,
    }


def dumps_json(payload: object) -> str:
    """Serialize JSON payloads in stable key order."""
    return json.dumps(payload, sort_keys=True)


def loads_json_or_none(value: object) -> object:
    """Deserialize one JSON string when present."""
    if value is None:
        return None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return json.loads(str(value))
