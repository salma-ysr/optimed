"""Build the canonical first-scope 65+ artifact and its QC report."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    validate_encounter_medication_burden_artifact,
    validate_encounter_medication_first_scope_artifact,
    validate_encounter_medication_semantics_artifact,
)
from opti_med.data_access.encounter_medication_semantics import (
    build_encounter_medication_first_scope,
    summarize_encounter_medication_first_scope,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.pipeline.qc import write_first_scope_semantics_and_burden_qc_report


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build encounter_medication_first_scope_65plus from the persisted "
            "65+ semantics and burden artifacts, then write one QC report."
        )
    )
    parser.add_argument(
        "--analytical-root",
        type=Path,
        default=None,
        help="Root directory containing persisted 65+ analytical artifacts.",
    )
    parser.add_argument(
        "--semantics-path",
        type=Path,
        default=None,
        help="Input Parquet path for encounter_medication_semantics_65plus.",
    )
    parser.add_argument(
        "--burden-path",
        type=Path,
        default=None,
        help="Input Parquet path for encounter_medication_burden_65plus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path for encounter_medication_first_scope_65plus.",
    )
    parser.add_argument(
        "--qc-output",
        type=Path,
        default=None,
        help="Output Markdown path for the first-scope 65+ QC report.",
    )
    return parser


def main() -> int:
    """Run the canonical first-scope 65+ build and QC workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    semantics_path = args.semantics_path or settings.encounter_medication_semantics_output_path
    burden_path = args.burden_path or settings.encounter_medication_burden_output_path
    output_path = args.output or settings.encounter_medication_first_scope_output_path
    qc_output_path = args.qc_output or settings.encounter_medication_first_scope_qc_report_path

    try:
        encounter_medication_semantics = pd.read_parquet(semantics_path)
        encounter_medication_burden = pd.read_parquet(burden_path)
        validate_encounter_medication_semantics_artifact(encounter_medication_semantics)
        validate_encounter_medication_burden_artifact(encounter_medication_burden)
        first_scope = build_encounter_medication_first_scope(
            encounter_medication_semantics=encounter_medication_semantics,
            encounter_medication_burden=encounter_medication_burden,
        )
        validate_encounter_medication_first_scope_artifact(first_scope)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first_scope.to_parquet(output_path, index=False)
        write_first_scope_semantics_and_burden_qc_report(
            encounter_medication_semantics=encounter_medication_semantics,
            encounter_medication_burden=encounter_medication_burden,
            encounter_medication_first_scope=first_scope,
            output_path=qc_output_path,
        )
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built encounter_medication_first_scope_65plus successfully.")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Semantics path: {semantics_path}")
    print(f"Burden path: {burden_path}")
    print(f"Output: {output_path}")
    print(f"QC output: {qc_output_path}")
    for summary in summarize_encounter_medication_first_scope(
        encounter_medication_first_scope=first_scope,
    ):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
