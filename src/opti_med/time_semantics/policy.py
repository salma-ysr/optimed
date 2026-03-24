"""Review-time policies and leakage validation helpers.

This module is intentionally independent from dataset loading and feature
building. It exists so future builders have one import path for review-time
semantics and temporal leakage guardrails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal


ReviewTimeAfterDischargeBehavior = Literal[
    "allow_with_manual_review",
    "error",
    "cap_to_discharge",
]


class ReviewTimeViolationError(ValueError):
    """Raised when a row violates explicit review-time leakage rules."""


class ReviewTimePolicyName(str, Enum):
    """Supported review-time policies for future analytical builders."""

    LATEST_AVAILABLE = "latest_available"
    BOUNDED_LATEST_AVAILABLE = "bounded_latest_available"
    DISCHARGE_CAPPED_LATEST_AVAILABLE = "discharge_capped_latest_available"


@dataclass(frozen=True, slots=True)
class ReviewTimePolicySpec:
    """Operational description of one review-time policy."""

    name: ReviewTimePolicyName
    purpose: str
    counts_as_review_time: str
    requires_all_feature_timestamps_lte_review_time: bool
    forbids_post_review_feature_evidence: bool
    requires_review_time_lte_discharge: bool
    review_time_after_discharge_behavior: ReviewTimeAfterDischargeBehavior


REPO_TIME_SEMANTICS_INVARIANTS: tuple[str, ...] = (
    "Future ML features must be available at or before review_timestamp.",
    "No feature timestamp may be after review_timestamp.",
    "No feature window may end after review_timestamp.",
    "When a review-time policy requires discharge protection, review_timestamp must be less than or equal to discharge_timestamp.",
    "Post-review or post-discharge facts must never be encoded as predictive features.",
    "Any cap, bound, or suppression applied to review time must be made explicit in provenance.",
)


REVIEW_TIME_POLICY_SPECS: dict[ReviewTimePolicyName, ReviewTimePolicySpec] = {
    ReviewTimePolicyName.LATEST_AVAILABLE: ReviewTimePolicySpec(
        name=ReviewTimePolicyName.LATEST_AVAILABLE,
        purpose=(
            "Retrospective encounter reconstruction using the latest admissible encounter-local evidence."
        ),
        counts_as_review_time=(
            "The latest encounter-local timestamp from the allowed evidence hierarchy."
        ),
        requires_all_feature_timestamps_lte_review_time=True,
        forbids_post_review_feature_evidence=True,
        requires_review_time_lte_discharge=False,
        review_time_after_discharge_behavior="allow_with_manual_review",
    ),
    ReviewTimePolicyName.BOUNDED_LATEST_AVAILABLE: ReviewTimePolicySpec(
        name=ReviewTimePolicyName.BOUNDED_LATEST_AVAILABLE,
        purpose=(
            "Predictive-safe latest-available review time with an explicit hard upper bound supplied by the builder."
        ),
        counts_as_review_time=(
            "The latest encounter-local timestamp that remains inside the caller-supplied safe bound."
        ),
        requires_all_feature_timestamps_lte_review_time=True,
        forbids_post_review_feature_evidence=True,
        requires_review_time_lte_discharge=True,
        review_time_after_discharge_behavior="error",
    ),
    ReviewTimePolicyName.DISCHARGE_CAPPED_LATEST_AVAILABLE: ReviewTimePolicySpec(
        name=ReviewTimePolicyName.DISCHARGE_CAPPED_LATEST_AVAILABLE,
        purpose=(
            "Latest-available review time that is forcibly capped at discharge when a later timestamp is proposed."
        ),
        counts_as_review_time=(
            "The latest encounter-local timestamp, capped to discharge_timestamp when needed."
        ),
        requires_all_feature_timestamps_lte_review_time=True,
        forbids_post_review_feature_evidence=True,
        requires_review_time_lte_discharge=True,
        review_time_after_discharge_behavior="cap_to_discharge",
    ),
}


def validate_review_time_policy(value: ReviewTimePolicyName | str) -> ReviewTimePolicyName:
    """Normalize a policy name into the supported enum."""
    if isinstance(value, ReviewTimePolicyName):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    try:
        return ReviewTimePolicyName(normalized)
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in ReviewTimePolicyName)
        raise ValueError(f"Unsupported review-time policy '{value}'. Expected one of: {allowed}.") from exc


def get_review_time_policy_spec(
    policy: ReviewTimePolicyName | str,
) -> ReviewTimePolicySpec:
    """Return the policy spec for one supported review-time policy."""
    normalized_policy = validate_review_time_policy(policy)
    return REVIEW_TIME_POLICY_SPECS[normalized_policy]


def assert_timestamp_not_after_review_time(
    timestamp: object,
    review_timestamp: object,
    *,
    field_name: str = "timestamp",
) -> None:
    """Assert that a candidate timestamp does not occur after review time."""
    candidate = _coerce_optional_timestamp(timestamp, field_name=field_name)
    review = _coerce_required_timestamp(review_timestamp, field_name="review_timestamp")
    if candidate is None:
        return
    if candidate > review:
        raise ReviewTimeViolationError(
            f"{field_name}={candidate.isoformat(sep=' ')} is after "
            f"review_timestamp={review.isoformat(sep=' ')}."
        )


def assert_review_time_not_after_discharge_when_policy_requires(
    review_timestamp: object,
    discharge_timestamp: object,
    policy: ReviewTimePolicyName | str,
) -> None:
    """Assert review time does not exceed discharge when the policy requires it."""
    policy_spec = get_review_time_policy_spec(policy)
    if not policy_spec.requires_review_time_lte_discharge:
        return

    review = _coerce_required_timestamp(review_timestamp, field_name="review_timestamp")
    discharge = _coerce_optional_timestamp(discharge_timestamp, field_name="discharge_timestamp")
    if discharge is None:
        return
    if review <= discharge:
        return

    behavior = policy_spec.review_time_after_discharge_behavior
    if behavior == "cap_to_discharge":
        raise ReviewTimeViolationError(
            "review_timestamp exceeds discharge_timestamp under "
            f"{policy_spec.name.value}; cap review time to discharge and record that cap in provenance."
        )
    raise ReviewTimeViolationError(
        "review_timestamp exceeds discharge_timestamp under "
        f"{policy_spec.name.value}; choose an earlier bounded review time."
    )


def assert_feature_window_valid(
    window_start: object,
    window_end: object,
    review_timestamp: object,
    *,
    policy: ReviewTimePolicyName | str = ReviewTimePolicyName.LATEST_AVAILABLE,
    window_name: str = "feature_window",
) -> None:
    """Assert a feature window is internally valid and does not leak past review time."""
    policy_spec = get_review_time_policy_spec(policy)
    review = _coerce_required_timestamp(review_timestamp, field_name="review_timestamp")
    start = _coerce_optional_timestamp(window_start, field_name=f"{window_name}_start")
    end = _coerce_optional_timestamp(window_end, field_name=f"{window_name}_end")

    if start is not None and end is not None and start > end:
        raise ReviewTimeViolationError(
            f"{window_name}_start={start.isoformat(sep=' ')} is after "
            f"{window_name}_end={end.isoformat(sep=' ')}."
        )

    if policy_spec.forbids_post_review_feature_evidence:
        assert_timestamp_not_after_review_time(
            start,
            review,
            field_name=f"{window_name}_start",
        )
        assert_timestamp_not_after_review_time(
            end,
            review,
            field_name=f"{window_name}_end",
        )


def _coerce_required_timestamp(value: object, *, field_name: str) -> datetime:
    timestamp = _coerce_optional_timestamp(value, field_name=field_name)
    if timestamp is None:
        raise ReviewTimeViolationError(f"{field_name} is required for review-time validation.")
    return timestamp


def _coerce_optional_timestamp(value: object, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ReviewTimeViolationError(
            f"{field_name}='{value}' is not a valid ISO-like timestamp."
        ) from exc
