"""Medication-query normalization helpers for ontology-backed lookups."""

from __future__ import annotations

import re

import pandas as pd

from opti_med.data_access.medication_consolidation import normalize_medication_name
from opti_med.medication_semantics.contracts import NormalizedMedicationName


RXNORM_QUERY_NORMALIZER_NAME = "rxnorm_query_normalizer"
RXNORM_QUERY_NORMALIZER_VERSION = "0.1"


class RxNormQueryNormalizer:
    """Build stable raw and normalized query strings for identity resolution."""

    normalizer_name = RXNORM_QUERY_NORMALIZER_NAME
    normalizer_version = RXNORM_QUERY_NORMALIZER_VERSION

    def normalize_name(
        self,
        raw_name: object,
        *,
        normalized_hint: object = None,
    ) -> NormalizedMedicationName:
        """Normalize a raw medication string into a stable queryable representation."""
        raw_text = coerce_query_text(raw_name)
        normalized_text = normalize_medication_name(normalized_hint)
        if normalized_text is None:
            normalized_text = normalize_medication_name(raw_text)
        return NormalizedMedicationName(
            raw_name=raw_text,
            normalized_text=normalized_text,
            normalizer_name=self.normalizer_name,
            normalizer_version=self.normalizer_version,
            normalization_status=(
                "placeholder_text_only" if normalized_text is not None else "not_standardized"
            ),
            provenance={
                "query_text_used": build_rxnorm_query_text(
                    raw_name,
                    normalized_hint=normalized_hint,
                ),
            },
        )


def build_rxnorm_query_text(
    raw_name: object,
    *,
    normalized_hint: object = None,
) -> str | None:
    """Build a conservative query string for RxNorm API calls."""
    raw_text = coerce_query_text(raw_name)
    if raw_text:
        return raw_text
    normalized_text = normalize_medication_name(normalized_hint)
    if normalized_text is None:
        return None
    return normalized_text


def coerce_query_text(raw_name: object) -> str | None:
    """Canonicalize raw medication text without collapsing it to a match key."""
    if raw_name is None:
        return None
    if raw_name is pd.NA:
        return None
    try:
        if pd.isna(raw_name):
            return None
    except TypeError:
        pass
    text = str(raw_name).strip()
    if not text or text.lower() in {"nan", "none", "___"}:
        return None
    text = text.lstrip("*")
    text = re.sub(r"\s+", " ", text)
    return text or None
