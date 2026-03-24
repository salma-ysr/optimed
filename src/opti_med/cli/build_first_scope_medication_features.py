"""Build first-pass encounter-medication class semantics and burden artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from opti_med.config import Settings
from opti_med.data_access.encounter_medication_semantics import (
    EncounterMedicationScopeArtifactsBuilder,
    summarize_first_scope_artifacts,
)
from opti_med.data_access.exceptions import DataLoadError


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build first-pass encounter-medication semantics and burden artifacts "
            "for the initial high-risk deprescribing scope."
        )
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=None,
        help="Root directory containing standardized source tables and upstream artifacts.",
    )
    parser.add_argument(
        "--analytical-root",
        type=Path,
        default=None,
        help="Root directory containing analytical parquet outputs.",
    )
    parser.add_argument(
        "--semantics-output",
        type=Path,
        default=None,
        help="Output parquet path for encounter_medication_semantics.",
    )
    parser.add_argument(
        "--burden-output",
        type=Path,
        default=None,
        help="Output parquet path for encounter_medication_burden.",
    )
    return parser


def main() -> int:
    """Run the first-pass class and burden build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    builder = EncounterMedicationScopeArtifactsBuilder(settings)

    try:
        semantics, burden = builder.build()
        result = builder.save(
            encounter_medication_semantics=semantics,
            encounter_medication_burden=burden,
            semantics_output_path=(
                args.semantics_output or settings.encounter_medication_semantics_output_path
            ),
            burden_output_path=(
                args.burden_output or settings.encounter_medication_burden_output_path
            ),
        )
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built first-pass encounter medication semantics and burden artifacts successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Semantics output: {result.semantics_output_path}")
    print(f"Burden output: {result.burden_output_path}")
    for summary in summarize_first_scope_artifacts(
        encounter_medication_semantics=result.encounter_medication_semantics,
        encounter_medication_burden=result.encounter_medication_burden,
    ):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
