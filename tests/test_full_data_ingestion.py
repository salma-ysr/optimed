"""Smoke tests for the full-data raw-to-standardized ingestion pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.pipeline import FullDataIngestionPipeline


class FullDataIngestionPipelineTests(unittest.TestCase):
    def test_pipeline_writes_standardized_parquet_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = _build_minimal_raw_tree(Path(temp_dir))
            settings = _build_settings(roots)

            manifest = FullDataIngestionPipeline(settings).run()

            patients_path = settings.standardized_root / "clinical" / "patients.parquet"
            admissions_path = settings.standardized_root / "clinical" / "admissions.parquet"
            triage_path = settings.standardized_root / "ed" / "triage.parquet"
            self.assertTrue(patients_path.exists())
            self.assertTrue(admissions_path.exists())
            self.assertTrue(triage_path.exists())
            self.assertTrue(manifest.manifest_path.exists())
            self.assertTrue(settings.standardized_manifest_path.exists())

            patients = pd.read_parquet(patients_path)
            admissions = pd.read_parquet(admissions_path)
            triage = pd.read_parquet(triage_path)
            self.assertEqual(patients["subject_id"].tolist(), [1, 2])
            self.assertEqual(str(patients["subject_id"].dtype), "Int64")
            self.assertIn("anchor_year", patients.columns)
            self.assertTrue(pd.api.types.is_datetime64_ns_dtype(admissions["admittime"]))
            self.assertEqual(triage["pain"].tolist(), [3.0])
            self.assertIn("pain__duplicate_2", triage.columns)
            self.assertEqual(triage["_source_dataset"].iloc[0], "ed")
            self.assertEqual(triage["_source_table"].iloc[0], "triage")
            self.assertIn("_ingestion_run_id", triage.columns)

            with settings.standardized_manifest_path.open("r", encoding="utf-8") as handle:
                latest_payload = json.load(handle)

            manifest_entries = {
                (entry["dataset_name"], entry["table_name"]): entry
                for entry in latest_payload["tables"]
            }
            patients_entry = manifest_entries[("clinical", "patients")]
            triage_entry = manifest_entries[("ed", "triage")]
            pharmacy_entry = manifest_entries[("clinical", "pharmacy")]
            self.assertEqual(patients_entry["row_count"], 2)
            self.assertEqual(triage_entry["duplicate_source_columns"], ["pain"])
            self.assertEqual(pharmacy_entry["status"], "missing_optional")

    def test_incremental_run_skips_unchanged_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = _build_minimal_raw_tree(Path(temp_dir))
            settings = _build_settings(roots, ingestion_behavior="overwrite")
            FullDataIngestionPipeline(settings).run()

            incremental_settings = _build_settings(roots, ingestion_behavior="incremental")
            manifest = FullDataIngestionPipeline(incremental_settings).run()
            entries = {
                (entry.dataset_name, entry.table_name): entry
                for entry in manifest.tables
            }
            self.assertEqual(entries[("clinical", "patients")].status, "skipped_incremental")
            self.assertEqual(entries[("clinical", "admissions")].status, "skipped_incremental")
            self.assertEqual(entries[("ed", "edstays")].status, "skipped_incremental")
            self.assertEqual(entries[("clinical", "pharmacy")].status, "missing_optional")

    def test_pipeline_fails_when_required_table_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = _build_minimal_raw_tree(Path(temp_dir))
            (roots["ed"] / "ed" / "edstays.csv").unlink()
            settings = _build_settings(roots)

            with self.assertRaises(DataLoadError):
                FullDataIngestionPipeline(settings).run()

    def test_pipeline_preserves_parent_field_ordinal_as_string(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            roots = _build_minimal_raw_tree(Path(temp_dir))
            settings = _build_settings(roots)

            manifest = FullDataIngestionPipeline(settings).run()

            emar_detail_path = settings.standardized_root / "clinical" / "emar_detail.parquet"
            self.assertTrue(emar_detail_path.exists())
            self.assertTrue(manifest.manifest_path.exists())

            emar_detail = pd.read_parquet(emar_detail_path)
            self.assertEqual(emar_detail["parent_field_ordinal"].iloc[0], "1.1")
            self.assertTrue(pd.isna(emar_detail["parent_field_ordinal"].iloc[1]))
            self.assertTrue(str(emar_detail["parent_field_ordinal"].dtype).startswith("string"))


def _build_settings(
    roots: dict[str, Path],
    *,
    ingestion_behavior: str = "overwrite",
) -> Settings:
    standardized_root = roots["workspace"] / "standardized"
    return Settings(
        raw_clinical_data_root=roots["clinical"],
        raw_ed_data_root=roots["ed"],
        standardized_root=standardized_root,
        manifest_root=standardized_root / "manifests",
        file_extension=".csv",
        ingestion_behavior=ingestion_behavior,
        ingestion_chunk_size=1,
    )


def _build_minimal_raw_tree(base_dir: Path) -> dict[str, Path]:
    clinical_root = base_dir / "mimic-iv-3.1"
    ed_root = base_dir / "mimic-iv-ed"
    hosp_dir = clinical_root / "hosp"
    ed_dir = ed_root / "ed"
    hosp_dir.mkdir(parents=True, exist_ok=True)
    ed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"subject_id": 1, "gender": "M", "anchor_age": 70},
            {"subject_id": 2, "gender": "F", "anchor_age": 82},
        ]
    ).to_csv(hosp_dir / "patients.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "admittime": "2125-03-19 18:00:00",
                "dischtime": "2125-03-20 10:00:00",
                "admission_type": "URGENT",
            },
            {
                "subject_id": 2,
                "hadm_id": 20,
                "admittime": "2125-04-01 08:00:00",
                "dischtime": "2125-04-05 12:00:00",
                "admission_type": "ELECTIVE",
            },
        ]
    ).to_csv(hosp_dir / "admissions.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "starttime": "2125-03-19 19:00:00",
                "stoptime": "2125-03-20 08:00:00",
                "drug": "Furosemide",
                "dose_val_rx": "20",
            },
            {
                "subject_id": 2,
                "hadm_id": 20,
                "starttime": "2125-04-01 09:00:00",
                "stoptime": "2125-04-04 09:00:00",
                "drug": "Metoprolol",
                "dose_val_rx": "25",
            },
        ]
    ).to_csv(hosp_dir / "prescriptions.csv", index=False)
    pd.DataFrame(
        [
            {"subject_id": 1, "hadm_id": 10, "icd_code": "I10", "icd_version": 10},
            {"subject_id": 2, "hadm_id": 20, "icd_code": "E11", "icd_version": 10},
        ]
    ).to_csv(hosp_dir / "diagnoses_icd.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "itemid": 50912,
                "charttime": "2125-03-20 06:00:00",
                "valuenum": 1.4,
            },
            {
                "subject_id": 2,
                "hadm_id": 20,
                "itemid": 50912,
                "charttime": "2125-04-02 07:00:00",
                "valuenum": 1.1,
            },
        ]
    ).to_csv(hosp_dir / "labevents.csv", index=False)
    pd.DataFrame(
        [
            {
                "subject_id": 1,
                "emar_id": 900,
                "emar_seq": 1,
                "parent_field_ordinal": "1.1",
            },
            {
                "subject_id": 2,
                "emar_id": 901,
                "emar_seq": 1,
                "parent_field_ordinal": "",
            },
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

    (ed_dir / "triage.csv").write_text(
        "\n".join(
            [
                "subject_id,stay_id,chiefcomplaint,acuity,pain,pain,temperature",
                "1,100,fall,2,3,3,37.1",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "workspace": base_dir,
        "clinical": clinical_root,
        "ed": ed_root,
    }


if __name__ == "__main__":
    unittest.main()
