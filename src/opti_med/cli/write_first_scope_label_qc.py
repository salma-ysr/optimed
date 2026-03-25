"""Write the first-scope Labels v1 QC report from the persisted artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    validate_encounter_medication_labels_v1_first_scope_artifact,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.labels.first_scope import write_first_scope_label_qc_report


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a Markdown QC report for the persisted "
            "encounter_medication_labels_v1_65plus_first_scope artifact."
        )
    )
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    """Render and persist the first-scope Labels v1 QC report."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        label_root=args.label_root or base_settings.label_root,
    )

    labels_path = args.labels_path or settings.first_scope_labels_output_path
    output_path = args.output or settings.first_scope_label_qc_report_path

    try:
        labels = pd.read_parquet(labels_path)
        validate_encounter_medication_labels_v1_first_scope_artifact(labels)
        write_first_scope_label_qc_report(
            encounter_medication_labels=labels,
            output_path=output_path,
        )
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Wrote first-scope Labels v1 QC report successfully.")
    print(f"Labels path: {labels_path}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
