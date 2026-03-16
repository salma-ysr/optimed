"""Configurable lab mappings for MVP lab features."""

from __future__ import annotations

from opti_med.config import Settings


def serum_creatinine_itemids(settings: Settings) -> tuple[int, ...]:
    """Return the configured serum creatinine item IDs."""
    return settings.serum_creatinine_itemids
