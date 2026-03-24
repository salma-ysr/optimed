"""Tests for review-time leakage guardrails using synthetic in-memory values only."""

from __future__ import annotations

import unittest
from datetime import datetime

from opti_med.time_semantics import (
    REPO_TIME_SEMANTICS_INVARIANTS,
    ReviewTimePolicyName,
    ReviewTimeViolationError,
    assert_feature_window_valid,
    assert_review_time_not_after_discharge_when_policy_requires,
    assert_timestamp_not_after_review_time,
    get_review_time_policy_spec,
    validate_review_time_policy,
)


class TimeSemanticsPolicyTests(unittest.TestCase):
    def test_validate_review_time_policy_accepts_supported_string(self) -> None:
        policy = validate_review_time_policy("bounded_latest_available")
        self.assertEqual(policy, ReviewTimePolicyName.BOUNDED_LATEST_AVAILABLE)

    def test_get_review_time_policy_spec_exposes_discharge_capped_behavior(self) -> None:
        spec = get_review_time_policy_spec("discharge_capped_latest_available")
        self.assertTrue(spec.requires_review_time_lte_discharge)
        self.assertEqual(spec.review_time_after_discharge_behavior, "cap_to_discharge")

    def test_repo_invariants_explicitly_forbid_post_review_features(self) -> None:
        self.assertIn(
            "Future ML features must be available at or before review_timestamp.",
            REPO_TIME_SEMANTICS_INVARIANTS,
        )


class TimeSemanticsAssertionTests(unittest.TestCase):
    def test_assert_timestamp_not_after_review_time_allows_equal_timestamp(self) -> None:
        review_time = datetime.fromisoformat("2125-03-20 09:00:00")
        assert_timestamp_not_after_review_time(
            "2125-03-20 09:00:00",
            review_time,
            field_name="lab_charttime",
        )

    def test_assert_timestamp_not_after_review_time_raises_for_future_timestamp(self) -> None:
        with self.assertRaises(ReviewTimeViolationError):
            assert_timestamp_not_after_review_time(
                "2125-03-20 09:01:00",
                "2125-03-20 09:00:00",
                field_name="lab_charttime",
            )

    def test_latest_available_does_not_require_discharge_guard(self) -> None:
        assert_review_time_not_after_discharge_when_policy_requires(
            "2125-03-20 10:30:00",
            "2125-03-20 10:00:00",
            ReviewTimePolicyName.LATEST_AVAILABLE,
        )

    def test_bounded_latest_available_raises_when_review_time_exceeds_discharge(self) -> None:
        with self.assertRaises(ReviewTimeViolationError):
            assert_review_time_not_after_discharge_when_policy_requires(
                "2125-03-20 10:30:00",
                "2125-03-20 10:00:00",
                ReviewTimePolicyName.BOUNDED_LATEST_AVAILABLE,
            )

    def test_discharge_capped_policy_raises_until_builder_caps_review_time(self) -> None:
        with self.assertRaises(ReviewTimeViolationError):
            assert_review_time_not_after_discharge_when_policy_requires(
                "2125-03-20 10:30:00",
                "2125-03-20 10:00:00",
                ReviewTimePolicyName.DISCHARGE_CAPPED_LATEST_AVAILABLE,
            )

    def test_assert_feature_window_valid_accepts_window_that_ends_at_review_time(self) -> None:
        assert_feature_window_valid(
            "2125-03-19 08:00:00",
            "2125-03-20 09:00:00",
            "2125-03-20 09:00:00",
            policy="bounded_latest_available",
            window_name="creatinine_window",
        )

    def test_assert_feature_window_valid_raises_when_window_end_exceeds_review_time(self) -> None:
        with self.assertRaises(ReviewTimeViolationError):
            assert_feature_window_valid(
                "2125-03-19 08:00:00",
                "2125-03-20 09:30:00",
                "2125-03-20 09:00:00",
                policy="bounded_latest_available",
                window_name="creatinine_window",
            )

    def test_assert_feature_window_valid_raises_when_window_start_is_after_end(self) -> None:
        with self.assertRaises(ReviewTimeViolationError):
            assert_feature_window_valid(
                "2125-03-20 09:00:00",
                "2125-03-19 08:00:00",
                "2125-03-20 10:00:00",
                window_name="vitals_window",
            )


if __name__ == "__main__":
    unittest.main()
