"""Command-line entry point for building the scored MVP cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from opti_med.cohort.exceptions import CohortBuildError
from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.scoring.scorer import DeprescribingPriorityScorer, summarize_scored_cohort


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build the scored older-adult medication cohort for deprescribing priority."
    )
    parser.add_argument("--data-root", type=Path, help="Path to the MIMIC-IV dataset root directory.")
    parser.add_argument(
        "--file-extension",
        default=None,
        help="Table file extension, for example .csv.gz or .csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path for the scored cohort file.",
    )
    parser.add_argument(
        "--age-threshold",
        type=int,
        default=None,
        help="Minimum anchor age to include in the older-adult cohort.",
    )
    parser.add_argument(
        "--polypharmacy-threshold",
        type=int,
        default=None,
        help="Minimum number of distinct medications per admission for the polypharmacy flag.",
    )
    parser.add_argument(
        "--renal-risk-threshold",
        type=float,
        default=None,
        help="Creatinine max threshold used for the renal risk flag.",
    )
    return parser


def main() -> int:
    """Run the scored cohort build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = Settings(
        data_root=args.data_root or base_settings.data_root,
        external_data_root=base_settings.external_data_root,
        clinical_data_root=args.data_root or base_settings.clinical_data_root,
        ed_data_root=base_settings.ed_data_root,
        hosp_dir_name=base_settings.hosp_dir_name,
        ed_dir_name=base_settings.ed_dir_name,
        file_extension=args.file_extension or base_settings.file_extension,
        interim_root=base_settings.interim_root,
        processed_root=base_settings.processed_root,
        final_root=base_settings.final_root,
        older_adult_age_threshold=args.age_threshold or base_settings.older_adult_age_threshold,
        polypharmacy_threshold=args.polypharmacy_threshold or base_settings.polypharmacy_threshold,
        renal_risk_creatinine_threshold=(
            args.renal_risk_threshold or base_settings.renal_risk_creatinine_threshold
        ),
        serum_creatinine_itemids=base_settings.serum_creatinine_itemids,
    )

    scorer = DeprescribingPriorityScorer(settings)
    try:
        dataframe = scorer.build()
        result = scorer.save(dataframe, output_path=args.output)
    except (DataLoadError, CohortBuildError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built scored older-adult medication cohort successfully.")
    print(f"Clinical data root: {settings.clinical_data_root}")
    print(f"Output: {result.output_path}")
    if result.dropped_duplicate_rows:
        print(f"- dropped_duplicate_rows={result.dropped_duplicate_rows:,}")
    for summary in summarize_scored_cohort(result.dataframe):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
