"""Smoke tests for analytical artifacts built from standardized tables only."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.encounter_medication_state import (
    DEFAULT_ENCOUNTER_MEDICATION_STATE_OUTPUT_PATH,
    EncounterMedicationStateArtifactBuilder,
)
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION,
    ENCOUNTER_INDEX_CONTRACT_VERSION,
    MEDICATION_EVENTS_CONTRACT_VERSION,
)
from opti_med.data_access.encounters import EncounterIndexBuilder
from opti_med.data_access.medication_events import CanonicalMedicationEventBuilder
from opti_med.pipeline import (
    write_encounter_and_medication_events_qc_report,
    write_encounter_medication_state_qc_report,
)
from opti_med.pipeline.ingestion import FullDataIngestionPipeline


class StandardizedAnalyticalArtifactTests(unittest.TestCase):
    def test_encounter_and_medication_artifacts_build_from_standardized_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = _build_raw_tree_with_medication_sources(Path(temp_dir))
            ingestion_settings = _build_ingestion_settings(roots)
            FullDataIngestionPipeline(ingestion_settings).run()

            # Remove the raw trees to prove the analytical builders only depend on standardized Parquet.
            for path in [roots["clinical"], roots["ed"]]:
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()

            settings = _build_standardized_only_settings(roots)
            encounter_builder = EncounterIndexBuilder(settings)
            medication_builder = CanonicalMedicationEventBuilder(settings)
            state_builder = EncounterMedicationStateArtifactBuilder(settings)

            encounter_index = encounter_builder.build()
            encounter_result = encounter_builder.save(encounter_index)
            medication_events = medication_builder.build(encounter_index=encounter_index)
            medication_result = medication_builder.save(medication_events)
            encounter_medication_state = state_builder.build(
                encounter_index=encounter_index,
                medication_events=medication_events,
            )
            state_result = state_builder.save(
                encounter_medication_state,
                output_path=roots["workspace"] / DEFAULT_ENCOUNTER_MEDICATION_STATE_OUTPUT_PATH,
            )

            self.assertTrue(encounter_result.output_path.exists())
            self.assertTrue(medication_result.output_path.exists())
            self.assertTrue(state_result.output_path.exists())

            self.assertIn("source_tables_json", encounter_index.columns)
            self.assertIn("source_record_provenance_json", encounter_index.columns)
            self.assertEqual(
                set(encounter_index["encounter_index_contract_version"].unique().tolist()),
                {ENCOUNTER_INDEX_CONTRACT_VERSION},
            )
            self.assertEqual(encounter_index["encounter_id"].tolist(), ["hadm:10"])

            self.assertIn("event_source_category", medication_events.columns)
            self.assertIn("event_source_table", medication_events.columns)
            self.assertIn("order_enrichment_applied_flag", medication_events.columns)
            self.assertIn("medication_prestandardized_text", medication_events.columns)
            self.assertEqual(
                set(medication_events["medication_event_contract_version"].unique().tolist()),
                {MEDICATION_EVENTS_CONTRACT_VERSION},
            )
            self.assertEqual(
                set(medication_events["event_source_category"].unique().tolist()),
                {
                    "home_medication_reconciliation",
                    "ed_medication_event",
                    "hospital_medication_order",
                    "hospital_administration_event",
                },
            )
            self.assertEqual(
                medication_events.loc[
                    medication_events["medication_normalized"] == "furosemide",
                    "continued_from_home_inferred",
                ].max(),
                1,
            )
            self.assertEqual(
                medication_events.loc[
                    medication_events["medication_normalized"].astype(str).str.startswith("lisinopril"),
                    "newly_started_during_encounter_inferred",
                ].max(),
                1,
            )
            self.assertEqual(
                medication_events.loc[
                    medication_events["medication_event_type"] == "hospital_order",
                    "order_enrichment_applied_flag",
                ].max(),
                1,
            )
            self.assertTrue(
                medication_events["medication_prestandardized_text"].equals(
                    medication_events["medication_normalized"]
                )
            )
            self.assertEqual(
                set(
                    encounter_medication_state[
                        "encounter_medication_state_contract_version"
                    ].unique().tolist()
                ),
                {ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION},
            )
            self.assertEqual(
                set(encounter_medication_state["review_time_policy_name"].unique().tolist()),
                {"discharge_capped_latest_available"},
            )
            self.assertEqual(
                int(
                    (
                        pd.to_datetime(
                            encounter_medication_state["review_timestamp"],
                            errors="coerce",
                        )
                        > pd.to_datetime(
                            encounter_medication_state["discharge_boundary"],
                            errors="coerce",
                        )
                    ).sum()
                ),
                0,
            )
            self.assertEqual(
                int(
                    encounter_medication_state.duplicated(
                        subset=[
                            "subject_id",
                            "encounter_id",
                            "medication_standardized",
                            "review_timestamp",
                        ],
                        keep=False,
                    ).sum()
                ),
                0,
            )
            self.assertIn(
                "inactive_before_review_time",
                set(encounter_medication_state["medication_status_at_review"].tolist()),
            )

            encounter_saved = pd.read_parquet(encounter_result.output_path)
            medication_saved = pd.read_parquet(medication_result.output_path)
            state_saved = pd.read_parquet(state_result.output_path)
            self.assertEqual(len(encounter_saved), len(encounter_index))
            self.assertEqual(len(medication_saved), len(medication_events))
            self.assertEqual(len(state_saved), len(encounter_medication_state))

            qc_path = roots["workspace"] / "encounter_medication_qc.md"
            write_encounter_and_medication_events_qc_report(
                encounter_index=encounter_index,
                medication_events=medication_events,
                output_path=qc_path,
            )
            self.assertTrue(qc_path.exists())
            report_text = qc_path.read_text(encoding="utf-8")
            self.assertIn("## Row Counts", report_text)
            self.assertIn("## Source Mix", report_text)
            self.assertIn("home_medication_reconciliation", report_text)
            self.assertIn("hospital_administration_event", report_text)

            state_qc_path = roots["workspace"] / "encounter_medication_state_qc.md"
            write_encounter_medication_state_qc_report(
                encounter_medication_state=encounter_medication_state,
                output_path=state_qc_path,
            )
            self.assertTrue(state_qc_path.exists())
            state_report_text = state_qc_path.read_text(encoding="utf-8")
            self.assertIn("rows with review_timestamp > discharge_boundary", state_report_text)
            self.assertIn("duplicate primary-key rows", state_report_text)
            self.assertIn("medication_status_at_review=inactive_before_review_time", state_report_text)


def _build_ingestion_settings(roots: dict[str, Path]) -> Settings:
    standardized_root = roots["workspace"] / "standardized"
    return Settings(
        raw_clinical_data_root=roots["clinical"],
        raw_ed_data_root=roots["ed"],
        standardized_root=standardized_root,
        manifest_root=standardized_root / "manifests",
        file_extension=".csv",
        ingestion_behavior="overwrite",
        ingestion_chunk_size=2,
    )


def _build_standardized_only_settings(roots: dict[str, Path]) -> Settings:
    standardized_root = roots["workspace"] / "standardized"
    return Settings(
        raw_clinical_data_root=roots["workspace"] / "missing-clinical",
        raw_ed_data_root=roots["workspace"] / "missing-ed",
        standardized_root=standardized_root,
        manifest_root=standardized_root / "manifests",
        file_extension=".csv",
    )


def _build_raw_tree_with_medication_sources(base_dir: Path) -> dict[str, Path]:
    clinical_root = base_dir / "mimic-iv-3.1"
    ed_root = base_dir / "mimic-iv-ed"
    hosp_dir = clinical_root / "hosp"
    ed_dir = ed_root / "ed"
    hosp_dir.mkdir(parents=True, exist_ok=True)
    ed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"subject_id": 1, "gender": "M", "anchor_age": 77}]
    ).to_csv(hosp_dir / "patients.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "admittime": "2125-03-19 18:00:00",
                "dischtime": "2125-03-20 10:00:00",
                "admission_type": "URGENT",
            }
        ]
    ).to_csv(hosp_dir / "admissions.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "pharmacy_id": 500,
                "poe_id": 700,
                "starttime": "2125-03-19 19:00:00",
                "stoptime": "2125-03-20 08:00:00",
                "drug": "Furosemide 20 mg",
                "route": "",
                "dose_val_rx": "20",
                "dose_unit_rx": "mg",
                "doses_per_24_hrs": "2",
            },
            {
                "subject_id": 1,
                "hadm_id": 10,
                "pharmacy_id": 501,
                "poe_id": 701,
                "starttime": "2125-03-19 20:00:00",
                "stoptime": "2125-03-20 09:00:00",
                "drug": "Lisinopril 10 mg",
                "route": "PO",
                "dose_val_rx": "10",
                "dose_unit_rx": "mg",
                "doses_per_24_hrs": "1",
            },
        ]
    ).to_csv(hosp_dir / "prescriptions.csv", index=False)
    pd.DataFrame(
        [{"subject_id": 1, "hadm_id": 10, "icd_code": "I10", "icd_version": 10}]
    ).to_csv(hosp_dir / "diagnoses_icd.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "itemid": 50912,
                "charttime": "2125-03-20 06:00:00",
                "valuenum": 1.2,
            }
        ]
    ).to_csv(hosp_dir / "labevents.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "pharmacy_id": 500,
                "poe_id": 700,
                "medication": "Furosemide",
                "status": "active",
                "route": "PO",
                "frequency": "BID",
                "starttime": "2125-03-19 19:00:00",
                "stoptime": "2125-03-20 08:00:00",
            }
        ]
    ).to_csv(hosp_dir / "pharmacy.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "emar_id": 900,
                "emar_seq": 1,
                "poe_id": 700,
                "pharmacy_id": 500,
                "charttime": "2125-03-19 21:00:00",
                "medication": "Furosemide",
                "event_txt": "Administered",
                "scheduletime": "2125-03-19 21:00:00",
            }
        ]
    ).to_csv(hosp_dir / "emar.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "emar_id": 900,
                "emar_seq": 1,
                "pharmacy_id": 500,
                "dose_given": 20,
                "dose_given_unit": "mg",
                "route": "PO",
            }
        ]
    ).to_csv(hosp_dir / "emar_detail.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "stay_id": 100,
                "intime": "2125-03-19 12:36:00",
                "outtime": "2125-03-19 16:59:47",
                "gender": "M",
                "race": "WHITE",
                "arrival_transport": "WALK IN",
                "disposition": "ADMITTED",
            }
        ]
    ).to_csv(ed_dir / "edstays.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "stay_id": 100,
                "name": "Furosemide 20 mg",
                "charttime": "2125-03-19 13:00:00",
            }
        ]
    ).to_csv(ed_dir / "medrecon.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "stay_id": 100,
                "charttime": "2125-03-19 14:00:00",
                "name": "Ondansetron",
                "med_rn": "Ondansetron",
            }
        ]
    ).to_csv(ed_dir / "pyxis.csv", index=False)

    return {
        "workspace": base_dir,
        "clinical": clinical_root,
        "ed": ed_root,
    }


if __name__ == "__main__":
    unittest.main()
