"""Join the persisted 65+ RxNorm mapping artifact back into the persisted state artifact."""

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
from opti_med.data_access.encounter_medication_state_rxnorm import (
    build_encounter_medication_state_rxnorm,
    summarize_encounter_medication_state_rxnorm,
)
from opti_med.data_access.exceptions import DataLoadError


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Join medication_rxnorm_mapping_65plus back into encounter_medication_state_65plus "
            "without rerunning the encounter-medication-state builder."
        )
    )
    parser.add_argument(
        "--analytical-root",
        type=Path,
        default=None,
        help="Root directory containing persisted 65+ analytical artifacts.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Input Parquet path for encounter_medication_state_65plus.",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=None,
        help="Input Parquet path for medication_rxnorm_mapping_65plus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path for encounter_medication_state_65plus_rxnorm.",
    )
    return parser


def main() -> int:
    """Run the decoupled 65+ RxNorm state join workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    state_path = args.state_path or settings.encounter_medication_state_output_path
    mapping_path = args.mapping_path or settings.medication_rxnorm_mapping_output_path
    output_path = args.output or settings.encounter_medication_state_rxnorm_output_path

    try:
        encounter_medication_state = pd.read_parquet(state_path)
        medication_rxnorm_mapping = pd.read_parquet(mapping_path)
        validate_encounter_medication_state_artifact(encounter_medication_state)
        validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
        enriched_state = build_encounter_medication_state_rxnorm(
            encounter_medication_state=encounter_medication_state,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enriched_state.to_parquet(output_path, index=False)
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built encounter_medication_state_65plus_rxnorm successfully.")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"State path: {state_path}")
    print(f"Mapping path: {mapping_path}")
    print(f"Output: {output_path}")
    for summary in summarize_encounter_medication_state_rxnorm(enriched_state):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
