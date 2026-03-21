"""Command-line entry point for building the canonical medication event layer."""

from __future__ import annotations

import argparse
from pathlib import Path

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.medication_events import (
    CanonicalMedicationEventBuilder,
    summarize_medication_events,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build the canonical medication events table across home, ED, and hospital sources."
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
        "--file-extension",
        default=None,
        help="Table file extension, for example .csv.gz or .csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for the medication event layer.",
    )
    return parser


def main() -> int:
    """Run the canonical medication event build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
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
    )

    builder = CanonicalMedicationEventBuilder(settings)
    try:
        dataframe = builder.build()
        result = builder.save(dataframe, output_path=args.output)
    except DataLoadError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built canonical medication events successfully.")
    print(f"Clinical data root: {settings.clinical_data_root}")
    print(f"ED data root: {settings.ed_data_root}")
    print(f"Output: {result.output_path}")
    for summary in summarize_medication_events(result.dataframe):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
