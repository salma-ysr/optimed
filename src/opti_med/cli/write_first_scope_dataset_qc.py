"""Write the Dataset v1 QC report from persisted dataset and split artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.modeling.first_scope_dataset import (
    validate_first_scope_dataset_artifact,
    validate_first_scope_split_artifact,
    write_first_scope_dataset_qc_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a Markdown QC report for the persisted Dataset v1 and "
            "subject-safe split artifacts for the 65+ first-scope subset."
        )
    )
    parser.add_argument("--modeling-root", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--splits-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    """Render and persist the Dataset v1 QC report."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        modeling_root=args.modeling_root or base_settings.modeling_root,
    )

    dataset_path = args.dataset_path or settings.first_scope_dataset_output_path
    splits_path = args.splits_path or settings.first_scope_splits_output_path
    output_path = args.output or settings.first_scope_dataset_qc_report_path

    try:
        dataset = pd.read_parquet(dataset_path)
        splits = pd.read_parquet(splits_path)
        validate_first_scope_dataset_artifact(dataset)
        validate_first_scope_split_artifact(splits)
        write_first_scope_dataset_qc_report(
            encounter_medication_dataset=dataset,
            splits=splits,
            output_path=output_path,
        )
    except (DataLoadError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Wrote Dataset v1 QC report successfully.")
    print(f"Dataset path: {dataset_path}")
    print(f"Splits path: {splits_path}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
