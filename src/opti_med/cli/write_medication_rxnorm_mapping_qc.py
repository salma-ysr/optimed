"""Write a QC report for the persisted 65+ RxNorm mapping artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    validate_encounter_medication_state_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.pipeline import write_medication_rxnorm_mapping_qc_report


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Write a QC report for medication_rxnorm_mapping_65plus."
    )
    parser.add_argument(
        "--analytical-root",
        type=Path,
        default=None,
        help="Root directory containing persisted 65+ analytical artifacts.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=None,
        help="Input Parquet path for medication_rxnorm_mapping_65plus.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help=(
            "Optional input Parquet path for encounter_medication_state_65plus. "
            "When provided, unresolved strings are ranked by state-row frequency."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Markdown path for the QC report.",
    )
    return parser


def main() -> int:
    """Write the RxNorm mapping QC report."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    mapping_path = args.mapping_path or settings.medication_rxnorm_mapping_output_path
    state_path = args.state_path or settings.encounter_medication_state_output_path

    try:
        medication_rxnorm_mapping = pd.read_parquet(mapping_path)
        validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
        encounter_medication_state = None
        if state_path.exists():
            encounter_medication_state = pd.read_parquet(state_path)
            validate_encounter_medication_state_artifact(encounter_medication_state)
        output_path = write_medication_rxnorm_mapping_qc_report(
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            encounter_medication_state=encounter_medication_state,
            output_path=args.output,
        )
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Wrote medication_rxnorm_mapping_65plus QC report successfully.")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Mapping path: {mapping_path}")
    if state_path.exists():
        print(f"State path: {state_path}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
