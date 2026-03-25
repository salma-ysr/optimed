"""Build the 65+ encounter-medication burden artifact from persisted semantics."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    validate_encounter_index_artifact,
    validate_encounter_medication_burden_artifact,
    validate_encounter_medication_semantics_artifact,
    validate_medication_events_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.encounter_medication_semantics import (
    build_encounter_medication_burden,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.standardized import StandardizedParquetRepository


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build encounter_medication_burden_65plus from the persisted "
            "encounter_medication_semantics_65plus artifact."
        )
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=None,
        help="Root directory containing standardized source tables and analytical artifacts.",
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
        "--mapping-path",
        type=Path,
        default=None,
        help="Input Parquet path for medication_rxnorm_mapping_65plus.",
    )
    parser.add_argument(
        "--medication-events-path",
        type=Path,
        default=None,
        help="Input Parquet path for the canonical medication_events artifact.",
    )
    parser.add_argument(
        "--encounter-index-path",
        type=Path,
        default=None,
        help="Input Parquet path for the canonical encounter_index artifact.",
    )
    parser.add_argument(
        "--labevents-path",
        type=Path,
        default=None,
        help="Optional standardized labevents Parquet path override.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path for encounter_medication_burden_65plus.",
    )
    return parser


def main() -> int:
    """Run the 65+ encounter-medication burden build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    repository = StandardizedParquetRepository(settings)
    semantics_path = args.semantics_path or settings.encounter_medication_semantics_output_path
    mapping_path = args.mapping_path or settings.medication_rxnorm_mapping_output_path
    medication_events_path = args.medication_events_path or settings.medication_events_output_path
    encounter_index_path = args.encounter_index_path or settings.encounter_index_output_path
    output_path = args.output or settings.encounter_medication_burden_output_path

    try:
        encounter_medication_semantics = pd.read_parquet(semantics_path)
        medication_rxnorm_mapping = pd.read_parquet(mapping_path)
        medication_events = pd.read_parquet(medication_events_path)
        encounter_index = pd.read_parquet(encounter_index_path)
        if args.labevents_path is not None:
            labevents = pd.read_parquet(args.labevents_path)
        else:
            loaded = repository.load_optional_source_table(
                "clinical",
                "labevents",
                columns=["hadm_id", "itemid", "charttime", "valuenum"],
            )
            labevents = loaded.dataframe if loaded else None

        validate_encounter_medication_semantics_artifact(encounter_medication_semantics)
        validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
        validate_medication_events_artifact(medication_events)
        validate_encounter_index_artifact(encounter_index)

        burden = build_encounter_medication_burden(
            encounter_medication_semantics=encounter_medication_semantics,
            medication_events=medication_events,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            encounter_index=encounter_index,
            labevents=labevents,
            settings=settings,
        )
        validate_encounter_medication_burden_artifact(burden)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        burden.to_parquet(output_path, index=False)
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    duplicate_rows = int((burden["same_class_duplicate_therapy_flag"] == 1).sum())
    print("Built encounter_medication_burden_65plus successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Semantics path: {semantics_path}")
    print(f"Mapping path: {mapping_path}")
    print(f"Medication events path: {medication_events_path}")
    print(f"Encounter index path: {encounter_index_path}")
    if args.labevents_path is not None:
        print(f"Labevents path: {args.labevents_path}")
    print(f"Output: {output_path}")
    print(f"- rows={len(burden):,}")
    print(f"- exact_duplicate_therapy_signal_rows={duplicate_rows:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
