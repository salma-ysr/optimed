"""Command-line entry point for building the minimal medication cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from opti_med.cohort.builder import OlderAdultMedicationCohortBuilder, summarize_cohort
from opti_med.cohort.exceptions import CohortBuildError
from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build the older-adult patient-medication-admission cohort."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Path to the MIMIC-IV dataset root directory.",
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
        help="Output CSV path for the interim cohort file.",
    )
    parser.add_argument(
        "--age-threshold",
        type=int,
        default=None,
        help="Minimum anchor age to include in the older-adult cohort.",
    )
    return parser


def main() -> int:
    """Run the minimal cohort build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = Settings(
        data_root=args.data_root or base_settings.data_root,
        hosp_dir_name=base_settings.hosp_dir_name,
        file_extension=args.file_extension or base_settings.file_extension,
        interim_root=base_settings.interim_root,
        older_adult_age_threshold=(
            args.age_threshold or base_settings.older_adult_age_threshold
        ),
    )

    builder = OlderAdultMedicationCohortBuilder(settings)

    try:
        cohort = builder.build()
        result = builder.save(cohort, output_path=args.output)
    except (DataLoadError, CohortBuildError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built older-adult medication cohort successfully.")
    print(f"Data root: {settings.data_root}")
    print(f"Output: {result.output_path}")
    if result.dropped_duplicate_rows:
        print(
            f"- dropped_duplicate_rows={result.dropped_duplicate_rows:,}"
        )
    for summary in summarize_cohort(result.dataframe):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
