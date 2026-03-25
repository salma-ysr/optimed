"""Command-line entry point for building the persisted 65+ eligibility branch."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from opti_med.cohort import (
    OlderAdultEncounterEligibilityBuilder,
    limit_eligibility_to_subject_count,
    summarize_older_adult_eligibility,
    write_older_adult_eligibility_qc_report,
)
from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build the persisted 65+ encounter eligibility analytical artifact."
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=None,
        help="Root directory containing encounter_index and other standardized artifacts.",
    )
    parser.add_argument(
        "--analytical-root",
        type=Path,
        default=None,
        help="Root directory for persisted analytical artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path for eligible_encounters_65plus.",
    )
    parser.add_argument(
        "--qc-report",
        type=Path,
        default=None,
        help="Optional Markdown QA report path for the 65+ eligibility artifact.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional deterministic limit on unique eligible subjects, ordered by subject_id ascending.",
    )
    return parser


def main() -> int:
    """Run the persisted 65+ eligibility build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    builder = OlderAdultEncounterEligibilityBuilder(settings)

    try:
        eligibility = builder.build()
        eligibility = limit_eligibility_to_subject_count(
            eligibility,
            args.max_subjects,
        )
        result = builder.save(
            eligibility,
            output_path=args.output or settings.older_adult_eligibility_output_path,
        )
        qc_path = None
        if args.qc_report is not None:
            qc_path = write_older_adult_eligibility_qc_report(
                dataframe=result.dataframe,
                output_path=args.qc_report,
            )
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built persisted 65+ encounter eligibility successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Analytical root: {settings.analytical_root}")
    if args.max_subjects is not None:
        print(f"Max subjects: {args.max_subjects}")
    print(f"Output: {result.output_path}")
    if qc_path is not None:
        print(f"QC report: {qc_path}")
    for summary in summarize_older_adult_eligibility(result.dataframe):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
