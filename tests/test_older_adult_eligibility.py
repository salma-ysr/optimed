"""Tests for the persisted older-adult encounter eligibility branch."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opti_med.cohort import (
    OlderAdultEncounterEligibilityBuilder,
    limit_eligibility_to_subject_count,
    semi_join_to_eligible_encounters,
    summarize_older_adult_eligibility,
    write_older_adult_eligibility_qc_report,
)
from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import ENCOUNTER_INDEX_CONTRACT_VERSION


class OlderAdultEligibilityTests(unittest.TestCase):
    def test_builder_persists_65plus_encounter_artifact_without_mutating_encounter_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = _build_roots(Path(temp_dir))
            encounter_index_path = roots["standardized"] / "encounter_index.parquet"
            encounter_index = _build_encounter_index_dataframe()
            encounter_index.to_parquet(encounter_index_path, index=False)
            encounter_index_before = pd.read_parquet(encounter_index_path)

            settings = Settings(
                standardized_root=roots["standardized"],
                analytical_root=roots["analytical"],
            )
            builder = OlderAdultEncounterEligibilityBuilder(settings)

            eligibility = builder.build()
            result = builder.save(eligibility)
            saved = builder.load()

            self.assertEqual(result.output_path, settings.older_adult_eligibility_output_path)
            self.assertTrue(result.output_path.exists())
            self.assertEqual(
                set(saved["encounter_id"].tolist()),
                {"hadm:10", "hadm:20", "stay:300"},
            )
            self.assertEqual(int(saved["eligibility_flag"].sum()), 3)
            self.assertEqual(
                saved["age_group"].value_counts().to_dict(),
                {"65-74": 1, "75-84": 1, "85+": 1},
            )
            summary_lines = summarize_older_adult_eligibility(saved)
            self.assertIn("unique_subjects=3", summary_lines)
            self.assertIn("unique_encounters=3", summary_lines)
            self.assertIn(
                "age_group_distribution=65-74:1, 75-84:1, 85+:1",
                summary_lines,
            )

            encounter_index_after = pd.read_parquet(encounter_index_path)
            pd.testing.assert_frame_equal(encounter_index_after, encounter_index_before)

    def test_semi_join_helper_filters_downstream_rows_by_eligible_encounter_id(self) -> None:
        eligibility = OlderAdultEncounterEligibilityBuilder(
            Settings(
                standardized_root=Path("/tmp/unused-standardized"),
                analytical_root=Path("/tmp/unused-analytical"),
            )
        ).build(encounter_index=_build_encounter_index_dataframe())
        downstream = pd.DataFrame(
            [
                {"encounter_id": "hadm:10", "value": 1},
                {"encounter_id": "hadm:20", "value": 2},
                {"encounter_id": "stay:300", "value": 3},
                {"encounter_id": "hadm:40", "value": 4},
            ]
        )

        filtered = semi_join_to_eligible_encounters(downstream, eligibility)

        self.assertEqual(filtered["encounter_id"].tolist(), ["hadm:10", "hadm:20", "stay:300"])
        self.assertEqual(filtered["value"].tolist(), [1, 2, 3])

    def test_limit_eligibility_to_subject_count_is_deterministic_by_subject_id(self) -> None:
        eligibility = OlderAdultEncounterEligibilityBuilder(
            Settings(
                standardized_root=Path("/tmp/unused-standardized"),
                analytical_root=Path("/tmp/unused-analytical"),
            )
        ).build(encounter_index=_build_encounter_index_dataframe())

        limited = limit_eligibility_to_subject_count(eligibility, 2)

        self.assertEqual(limited["subject_id"].drop_duplicates().tolist(), [1, 2])
        self.assertEqual(limited["encounter_id"].tolist(), ["hadm:10", "hadm:20"])
        criteria = json.loads(limited.loc[0, "eligibility_criteria_json"])
        self.assertEqual(criteria["subject_limit"]["max_subjects"], 2)
        self.assertEqual(
            criteria["subject_limit"]["selection_rule"],
            "lowest_subject_id_ascending",
        )

    def test_qc_report_contains_requested_qa_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = _build_roots(Path(temp_dir))
            eligibility = OlderAdultEncounterEligibilityBuilder(
                Settings(
                    standardized_root=roots["standardized"],
                    analytical_root=roots["analytical"],
                )
            ).build(encounter_index=_build_encounter_index_dataframe())
            report_path = roots["workspace"] / "older_adult_eligibility_qc.md"

            write_older_adult_eligibility_qc_report(
                dataframe=eligibility,
                output_path=report_path,
            )

            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("- unique subjects: 3", report_text)
            self.assertIn("- unique encounters: 3", report_text)
            self.assertIn("- age_group=65-74: 1", report_text)
            self.assertIn("- age_group=75-84: 1", report_text)
            self.assertIn("- age_group=85+: 1", report_text)
            self.assertIn("- encounter_source=hospital_only: 1", report_text)
            self.assertIn("- encounter_source=ed_to_inpatient: 1", report_text)
            self.assertIn("- encounter_source=ed_only: 1", report_text)


def _build_roots(base_dir: Path) -> dict[str, Path]:
    standardized_root = base_dir / "standardized"
    analytical_root = base_dir / "analytical"
    standardized_root.mkdir(parents=True, exist_ok=True)
    analytical_root.mkdir(parents=True, exist_ok=True)
    return {
        "workspace": base_dir,
        "standardized": standardized_root,
        "analytical": analytical_root,
    }


def _build_encounter_index_dataframe() -> pd.DataFrame:
    build_run_id = "encounter-index-test"
    return pd.DataFrame(
        [
            {
                "subject_id": 1,
                "sex": "M",
                "age_proxy": 66,
                "age_group": "65-74",
                "encounter_id": "hadm:10",
                "hadm_id": 10,
                "stay_id": pd.NA,
                "encounter_source": "hospital_only",
                "linked_ed_stay_flag": 0,
                "linked_hospital_admission_flag": 1,
                "admission_type": "URGENT",
                "admittime": "2125-01-01 10:00:00",
                "dischtime": "2125-01-03 10:00:00",
                "hospital_length_of_stay_days": 2.0,
                "intime": pd.NA,
                "outtime": pd.NA,
                "ed_length_of_stay_hours": pd.NA,
                "ed_disposition": pd.NA,
                "arrival_transport": pd.NA,
                "encounter_start": "2125-01-01 10:00:00",
                "encounter_end": "2125-01-03 10:00:00",
                "source_tables_json": "[\"patients\", \"admissions\"]",
                "source_record_provenance_json": "{\"patients\": {\"role\": \"patients\"}, \"admissions\": {\"role\": \"admissions\"}, \"edstays\": null}",
                "encounter_index_build_run_id": build_run_id,
                "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
            },
            {
                "subject_id": 2,
                "sex": "F",
                "age_proxy": 82,
                "age_group": "75-84",
                "encounter_id": "hadm:20",
                "hadm_id": 20,
                "stay_id": 200,
                "encounter_source": "ed_to_inpatient",
                "linked_ed_stay_flag": 1,
                "linked_hospital_admission_flag": 1,
                "admission_type": "EMERGENCY",
                "admittime": "2125-01-04 09:00:00",
                "dischtime": "2125-01-05 18:00:00",
                "hospital_length_of_stay_days": 1.375,
                "intime": "2125-01-04 07:30:00",
                "outtime": "2125-01-04 08:45:00",
                "ed_length_of_stay_hours": 1.25,
                "ed_disposition": "ADMITTED",
                "arrival_transport": "AMBULANCE",
                "encounter_start": "2125-01-04 07:30:00",
                "encounter_end": "2125-01-05 18:00:00",
                "source_tables_json": "[\"patients\", \"admissions\", \"edstays\"]",
                "source_record_provenance_json": "{\"patients\": {\"role\": \"patients\"}, \"admissions\": {\"role\": \"admissions\"}, \"edstays\": {\"role\": \"edstays\"}}",
                "encounter_index_build_run_id": build_run_id,
                "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
            },
            {
                "subject_id": 3,
                "sex": "M",
                "age_proxy": 87,
                "age_group": "85+",
                "encounter_id": "stay:300",
                "hadm_id": pd.NA,
                "stay_id": 300,
                "encounter_source": "ed_only",
                "linked_ed_stay_flag": 1,
                "linked_hospital_admission_flag": 0,
                "admission_type": pd.NA,
                "admittime": pd.NA,
                "dischtime": pd.NA,
                "hospital_length_of_stay_days": pd.NA,
                "intime": "2125-01-06 11:00:00",
                "outtime": "2125-01-06 15:00:00",
                "ed_length_of_stay_hours": 4.0,
                "ed_disposition": "HOME",
                "arrival_transport": "WALK IN",
                "encounter_start": "2125-01-06 11:00:00",
                "encounter_end": "2125-01-06 15:00:00",
                "source_tables_json": "[\"patients\", \"edstays\"]",
                "source_record_provenance_json": "{\"patients\": {\"role\": \"patients\"}, \"admissions\": null, \"edstays\": {\"role\": \"edstays\"}}",
                "encounter_index_build_run_id": build_run_id,
                "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
            },
            {
                "subject_id": 4,
                "sex": "F",
                "age_proxy": 60,
                "age_group": "<65",
                "encounter_id": "hadm:40",
                "hadm_id": 40,
                "stay_id": pd.NA,
                "encounter_source": "hospital_only",
                "linked_ed_stay_flag": 0,
                "linked_hospital_admission_flag": 1,
                "admission_type": "ELECTIVE",
                "admittime": "2125-01-08 12:00:00",
                "dischtime": "2125-01-09 12:00:00",
                "hospital_length_of_stay_days": 1.0,
                "intime": pd.NA,
                "outtime": pd.NA,
                "ed_length_of_stay_hours": pd.NA,
                "ed_disposition": pd.NA,
                "arrival_transport": pd.NA,
                "encounter_start": "2125-01-08 12:00:00",
                "encounter_end": "2125-01-09 12:00:00",
                "source_tables_json": "[\"patients\", \"admissions\"]",
                "source_record_provenance_json": "{\"patients\": {\"role\": \"patients\"}, \"admissions\": {\"role\": \"admissions\"}, \"edstays\": null}",
                "encounter_index_build_run_id": build_run_id,
                "encounter_index_contract_version": ENCOUNTER_INDEX_CONTRACT_VERSION,
            },
        ]
    )
