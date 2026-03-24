"""Shared raw-string labels for review-time medication semantics.

These values intentionally stay as plain strings because the current CSV
artifacts, API payloads, and feature tables persist them verbatim. Centralizing
them reduces ambiguity without changing runtime behavior.
"""

from __future__ import annotations

from typing import Final

MEDICATION_STATUS_ACTIVE_AT_REVIEW: Final = "active_at_review_time"
MEDICATION_STATUS_INACTIVE_BEFORE_REVIEW: Final = "inactive_before_review_time"
MEDICATION_STATUS_PRE_ADMISSION_ONLY: Final = "pre_admission_only"
MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW: Final = (
    "activity_uncertain_at_review_time"
)

MEDICATION_REVIEW_STATUS_VALUES: Final = (
    MEDICATION_STATUS_ACTIVE_AT_REVIEW,
    MEDICATION_STATUS_INACTIVE_BEFORE_REVIEW,
    MEDICATION_STATUS_PRE_ADMISSION_ONLY,
    MEDICATION_STATUS_ACTIVITY_UNCERTAIN_AT_REVIEW,
)

REVIEW_TIMESTAMP_SOURCE_MEDICATION_ADMINISTRATION: Final = (
    "medication_administration"
)
REVIEW_TIMESTAMP_SOURCE_MEDICATION_ORDER: Final = "medication_order"
REVIEW_TIMESTAMP_SOURCE_LAB: Final = "lab"
REVIEW_TIMESTAMP_SOURCE_VITALS: Final = "vitals"
REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_END: Final = "encounter_end"
REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_BOUNDARY: Final = "encounter_boundary"
REVIEW_TIMESTAMP_SOURCE_PREBUILT_SNAPSHOT: Final = "prebuilt_snapshot"

REVIEW_TIMESTAMP_SOURCE_VALUES: Final = (
    REVIEW_TIMESTAMP_SOURCE_MEDICATION_ADMINISTRATION,
    REVIEW_TIMESTAMP_SOURCE_MEDICATION_ORDER,
    REVIEW_TIMESTAMP_SOURCE_LAB,
    REVIEW_TIMESTAMP_SOURCE_VITALS,
    REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_END,
    REVIEW_TIMESTAMP_SOURCE_ENCOUNTER_BOUNDARY,
    REVIEW_TIMESTAMP_SOURCE_PREBUILT_SNAPSHOT,
)

PROVENANCE_UNAVAILABLE: Final = "unavailable"
MISSINGNESS_OBSERVED: Final = "observed"
