"""Tests for canonical medication episode consolidation."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from opti_med.data_access.medication_consolidation import collapse_continuation_intervals
from opti_med.features.builder import build_medication_burden_features


class MedicationConsolidationTests(unittest.TestCase):
    def test_adjacent_same_medication_rows_collapse_into_one_episode(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "hadm_id": 11,
                    "drug": "Fentanyl Citrate",
                    "drug_normalized": "fentanyl citrate",
                    "starttime": "2135-01-06 00:00:00",
                    "stoptime": "2135-01-08 00:00:00",
                },
                {
                    "subject_id": 1,
                    "hadm_id": 11,
                    "drug": "Fentanyl Citrate",
                    "drug_normalized": "fentanyl citrate",
                    "starttime": "2135-01-08 00:00:00",
                    "stoptime": "2135-01-13 00:00:00",
                },
                {
                    "subject_id": 1,
                    "hadm_id": 11,
                    "drug": "Fentanyl Citrate",
                    "drug_normalized": "fentanyl citrate",
                    "starttime": "2135-01-13 00:00:00",
                    "stoptime": "2135-01-16 00:00:00",
                },
            ]
        )

        collapsed = collapse_continuation_intervals(
            frame,
            group_columns=["subject_id", "hadm_id", "drug_normalized"],
            start_column="starttime",
            stop_column="stoptime",
            segment_fields=["drug", "starttime", "stoptime"],
            episode_id_prefix="test-episode",
        )

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed.iloc[0]["starttime"], "2135-01-06 00:00:00")
        self.assertEqual(collapsed.iloc[0]["stoptime"], "2135-01-16 00:00:00")
        self.assertEqual(collapsed.iloc[0]["prescription_segment_count"], 3)
        segments = json.loads(collapsed.iloc[0]["prescription_segments_json"])
        self.assertEqual(len(segments), 3)

    def test_burden_uses_collapsed_episodes_and_peak_concurrency(self) -> None:
        cohort = pd.DataFrame(
            [
                {
                    "hadm_id": 11,
                    "medication_episode_id": "episode-1",
                    "starttime": "2135-01-06 00:00:00",
                    "stoptime": "2135-01-16 00:00:00",
                    "dischtime": "2135-01-20 00:00:00",
                },
                {
                    "hadm_id": 11,
                    "medication_episode_id": "episode-2",
                    "starttime": "2135-01-07 00:00:00",
                    "stoptime": "2135-01-10 00:00:00",
                    "dischtime": "2135-01-20 00:00:00",
                },
                {
                    "hadm_id": 11,
                    "medication_episode_id": "episode-3",
                    "starttime": "2135-01-17 00:00:00",
                    "stoptime": "2135-01-18 00:00:00",
                    "dischtime": "2135-01-20 00:00:00",
                },
            ]
        )

        class SettingsStub:
            polypharmacy_threshold = 2

        burden = build_medication_burden_features(cohort, SettingsStub())
        self.assertEqual(int(burden.iloc[0]["total_medication_count"]), 3)
        self.assertEqual(int(burden.iloc[0]["peak_concurrent_medication_count"]), 2)
        self.assertEqual(int(burden.iloc[0]["polypharmacy_flag"]), 1)


if __name__ == "__main__":
    unittest.main()
