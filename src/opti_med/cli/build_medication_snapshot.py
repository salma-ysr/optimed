"""Command-line entry point for building encounter-relative medication snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

from opti_med.config import Settings
from opti_med.data_access.medication_snapshot import (
    MedicationSnapshotBuilder,
    summarize_medication_snapshot,
    validate_snapshot_strategy,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build one encounter-relative medication snapshot row per patient-medication."
    )
    parser.add_argument(
        "--clinical-data-root",
        type=Path,
        default=None,
        help="Path to the MIMIC-IV clinical demo root directory.",
    )
    parser.add_argument(
        "--ed-data-root",
        type=Path,
        default=None,
        help="Path to the MIMIC-IV ED demo root directory.",
    )
    parser.add_argument(
        "--snapshot-strategy",
        default=None,
        help="Encounter-relative snapshot strategy: ed, hospital, or latest_available.",
    )
    parser.add_argument(
        "--file-extension",
        default=None,
        help="Table file extension, for example .csv.gz or .csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for the medication snapshot layer.",
    )
    return parser


def main() -> int:
    """Run the medication snapshot build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    snapshot_strategy = validate_snapshot_strategy(
        args.snapshot_strategy or base_settings.snapshot_strategy
    )
    settings = Settings(
        data_root=args.clinical_data_root or base_settings.data_root,
        external_data_root=base_settings.external_data_root,
        clinical_data_root=args.clinical_data_root or base_settings.clinical_data_root,
        ed_data_root=args.ed_data_root or base_settings.ed_data_root,
        hosp_dir_name=base_settings.hosp_dir_name,
        ed_dir_name=base_settings.ed_dir_name,
        file_extension=args.file_extension or base_settings.file_extension,
        interim_root=base_settings.interim_root,
        processed_root=base_settings.processed_root,
        final_root=base_settings.final_root,
        older_adult_age_threshold=base_settings.older_adult_age_threshold,
        polypharmacy_threshold=base_settings.polypharmacy_threshold,
        renal_risk_creatinine_threshold=base_settings.renal_risk_creatinine_threshold,
        serum_creatinine_itemids=base_settings.serum_creatinine_itemids,
        snapshot_strategy=snapshot_strategy,
    )

    builder = MedicationSnapshotBuilder(settings, snapshot_strategy=snapshot_strategy)
    dataframe = builder.build()
    result = builder.save(dataframe, output_path=args.output)

    print("Built encounter-relative medication snapshot successfully.")
    print(f"Clinical data root: {settings.clinical_data_root}")
    print(f"ED data root: {settings.ed_data_root}")
    print(f"Snapshot strategy: {snapshot_strategy}")
    print(f"Output: {result.output_path}")
    for summary in summarize_medication_snapshot(result.dataframe):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
