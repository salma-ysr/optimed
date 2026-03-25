"""Command-line entry point for building the 65+ RxNorm mapping artifact from persisted state."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import validate_encounter_medication_state_artifact
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.medication_rxnorm_mapping import (
    LOOKUP_MODE_CACHE_FIRST,
    LOOKUP_MODE_CACHE_ONLY,
    MedicationRxNormMappingBuilder,
    extract_medication_rxnorm_queries_from_encounter_medication_state,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the medication_rxnorm_mapping_65plus cache artifact from the persisted "
            "encounter_medication_state_65plus artifact without rebuilding review-time state."
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
        "--cache-path",
        type=Path,
        default=None,
        help=(
            "Optional existing RxNorm mapping cache path. Defaults to the canonical "
            "medication_rxnorm_mapping_65plus artifact path."
        ),
    )
    parser.add_argument(
        "--rxnorm-lookup-mode",
        default=LOOKUP_MODE_CACHE_FIRST,
        choices=[LOOKUP_MODE_CACHE_FIRST, LOOKUP_MODE_CACHE_ONLY],
        help="RxNorm lookup mode: cache_first or cache_only.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path for medication_rxnorm_mapping_65plus.",
    )
    return parser


def main() -> int:
    """Run the 65+ RxNorm mapping build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    state_path = args.state_path or settings.encounter_medication_state_output_path
    output_path = args.output or settings.medication_rxnorm_mapping_output_path
    cache_path = args.cache_path or output_path

    try:
        encounter_medication_state = pd.read_parquet(state_path)
        validate_encounter_medication_state_artifact(encounter_medication_state)
        mapping_queries = extract_medication_rxnorm_queries_from_encounter_medication_state(
            encounter_medication_state
        )
        existing_cache = None
        if cache_path.exists():
            existing_cache = pd.read_parquet(cache_path)
        builder = MedicationRxNormMappingBuilder(
            settings,
            lookup_mode=args.rxnorm_lookup_mode,
        )
        mapping = builder.build(
            mapping_queries=mapping_queries,
            existing_cache=existing_cache,
        )
        result = builder.save(
            mapping,
            output_path=output_path,
        )
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built medication_rxnorm_mapping_65plus successfully.")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"State path: {state_path}")
    print(f"Cache path: {cache_path}")
    print(f"RxNorm lookup mode: {args.rxnorm_lookup_mode}")
    print(f"Output: {result.output_path}")
    print(f"- state rows={len(encounter_medication_state):,}")
    print(
        f"- distinct medication candidates={mapping_queries['medication_query_key'].nunique():,}"
    )
    print(f"- mapping rows={len(result.dataframe):,}")
    if not result.dataframe.empty:
        status_counts = result.dataframe["lookup_status"].value_counts(dropna=False).to_dict()
        print(f"- lookup_status_counts={status_counts}")
        api_called_counts = (
            result.dataframe["api_called_flag"].value_counts(dropna=False).sort_index().to_dict()
        )
        print(f"- api_called_flag_counts={api_called_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
