"""Build the 65+ ontology-aware encounter-medication semantics artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    validate_encounter_medication_semantics_artifact,
    validate_encounter_medication_state_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.encounter_medication_semantics import (
    build_encounter_medication_semantics,
)
from opti_med.data_access.exceptions import DataLoadError


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build encounter_medication_semantics_65plus from the persisted "
            "encounter_medication_state_65plus_rxnorm artifact."
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
        help="Input Parquet path for encounter_medication_state_65plus_rxnorm.",
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
        help="Output Parquet path for encounter_medication_semantics_65plus.",
    )
    return parser


def main() -> int:
    """Run the 65+ encounter-medication semantics build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    state_path = args.state_path or settings.encounter_medication_state_rxnorm_output_path
    mapping_path = args.mapping_path or settings.medication_rxnorm_mapping_output_path
    output_path = args.output or settings.encounter_medication_semantics_output_path

    try:
        encounter_medication_state = pd.read_parquet(state_path)
        medication_rxnorm_mapping = pd.read_parquet(mapping_path)
        validate_encounter_medication_state_artifact(encounter_medication_state)
        validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
        semantics = build_encounter_medication_semantics(
            encounter_medication_state=encounter_medication_state,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
        )
        validate_encounter_medication_semantics_artifact(semantics)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        semantics.to_parquet(output_path, index=False)
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    active_semantics = semantics.loc[semantics["active_at_review_flag"] == 1].copy()
    class_counts = (
        active_semantics["medication_class_standardized"]
        .fillna("unresolved")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    print("Built encounter_medication_semantics_65plus successfully.")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"State path: {state_path}")
    print(f"Mapping path: {mapping_path}")
    print(f"Output: {output_path}")
    print(f"- rows={len(semantics):,}")
    print(f"- active_rows={len(active_semantics):,}")
    print(f"- class_coverage_active={class_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
