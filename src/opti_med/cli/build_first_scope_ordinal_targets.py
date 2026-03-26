"""Build ordinal target preparation artifacts from clinician-reviewed labels."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.modeling.first_scope_ordinal_targets import (
    build_first_scope_ordinal_targets,
    write_first_scope_ordinal_target_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the first-scope ordinal target sidecar from the latest-active "
            "clinician review snapshot."
        )
    )
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--modeling-root", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--splits-path", type=Path, default=None)
    parser.add_argument("--clinician-review-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    return parser


def main() -> int:
    """Build the persisted ordinal target artifacts."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        label_root=args.label_root or base_settings.label_root,
        modeling_root=args.modeling_root or base_settings.modeling_root,
    )

    dataset_path = args.dataset_path or settings.first_scope_dataset_output_path
    splits_path = args.splits_path or settings.first_scope_splits_output_path
    clinician_review_path = (
        args.clinician_review_path or settings.clinician_review_snapshot_output_path
    )
    output_path = args.output_path or settings.first_scope_ordinal_targets_output_path
    summary_path = args.summary_path or settings.first_scope_ordinal_targets_summary_path
    report_output = args.report_output or settings.first_scope_ordinal_target_qc_report_path

    try:
        dataset = pd.read_parquet(dataset_path)
        splits = pd.read_parquet(splits_path)
        clinician_reviews = pd.read_parquet(clinician_review_path)
        result = build_first_scope_ordinal_targets(
            encounter_medication_dataset=dataset,
            latest_clinician_reviews=clinician_reviews,
            splits=splits,
        )
        artifact_paths = write_first_scope_ordinal_target_artifacts(
            result=result,
            output_path=output_path,
            summary_path=summary_path,
            report_path=report_output,
        )
    except (DataLoadError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built first-scope ordinal target preparation artifacts successfully.")
    print(f"Dataset input: {dataset_path}")
    print(f"Splits input: {splits_path}")
    print(f"Clinician review input: {clinician_review_path}")
    print(f"Prepared target rows: {result.summary['prepared_target_row_count']}")
    print(
        "Default training-included rows: "
        f"{result.summary['default_training_included_row_count']}"
    )
    print(f"Targets output: {artifact_paths['targets']}")
    print(f"Summary output: {artifact_paths['summary']}")
    print(f"Report output: {artifact_paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
