"""Configurable lab mappings for MVP lab features."""

from __future__ import annotations

from opti_med.config import Settings


DEFAULT_POTASSIUM_ITEMIDS = (50971, 52610, 50822, 52452)
DEFAULT_SODIUM_ITEMIDS = (50983, 52623, 50824, 52455)


def serum_creatinine_itemids(settings: Settings) -> tuple[int, ...]:
    """Return the configured serum creatinine item IDs."""
    return settings.serum_creatinine_itemids


def potassium_itemids() -> tuple[int, ...]:
    """Return curated potassium item IDs limited to blood and whole-blood contexts."""
    return DEFAULT_POTASSIUM_ITEMIDS


def sodium_itemids() -> tuple[int, ...]:
    """Return curated sodium item IDs limited to blood and whole-blood contexts."""
    return DEFAULT_SODIUM_ITEMIDS
