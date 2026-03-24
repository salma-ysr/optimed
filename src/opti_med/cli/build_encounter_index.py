"""Command-line entry point for building the standardized encounter-index artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from opti_med.config import Settings
from opti_med.data_access.encounters import EncounterIndexBuilder, summarize_encounter_index
from opti_med.data_access.exceptions import DataLoadError


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build the standardized encounter_index analytical artifact."
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=None,
        help="Root directory containing standardized source tables and output artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path for encounter_index.",
    )
    return parser


def main() -> int:
    """Run the encounter-index build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
    )
    builder = EncounterIndexBuilder(settings)
    try:
        encounter_index = builder.build()
        result = builder.save(encounter_index, output_path=args.output)
    except DataLoadError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built standardized encounter_index successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Output: {result.output_path}")
    for summary in summarize_encounter_index(result.dataframe):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
