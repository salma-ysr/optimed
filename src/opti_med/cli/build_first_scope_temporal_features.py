"""Build the first-scope 65+ temporal physiology stage artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.features.first_scope_feature_store import (
    build_first_scope_temporal_features,
    validate_first_scope_temporal_features_artifact,
)
from opti_med.standardized import StandardizedParquetRepository


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build point-in-time-safe temporal physiology features for the persisted "
            "65+ first-scope encounter-medication subset."
        )
    )
    parser.add_argument("--standardized-root", type=Path, default=None)
    parser.add_argument("--analytical-root", type=Path, default=None)
    parser.add_argument("--feature-store-root", type=Path, default=None)
    parser.add_argument("--first-scope-path", type=Path, default=None)
    parser.add_argument("--labevents-path", type=Path, default=None)
    parser.add_argument("--edstays-path", type=Path, default=None)
    parser.add_argument("--triage-path", type=Path, default=None)
    parser.add_argument("--vitalsign-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    """Run the first-scope temporal stage build."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
        analytical_root=args.analytical_root or base_settings.analytical_root,
        feature_store_root=args.feature_store_root or base_settings.feature_store_root,
    )
    repository = StandardizedParquetRepository(settings)

    first_scope_path = args.first_scope_path or settings.encounter_medication_first_scope_output_path
    output_path = args.output or settings.first_scope_temporal_features_output_path

    try:
        first_scope = pd.read_parquet(first_scope_path)
        labevents = (
            pd.read_parquet(args.labevents_path)
            if args.labevents_path is not None
            else repository.load_source_table("clinical", "labevents").dataframe
        )
        edstays_loaded = (
            pd.read_parquet(args.edstays_path)
            if args.edstays_path is not None
            else repository.load_optional_source_table("ed", "edstays")
        )
        edstays = edstays_loaded if isinstance(edstays_loaded, pd.DataFrame) else (
            edstays_loaded.dataframe if edstays_loaded is not None else None
        )
        triage_loaded = (
            pd.read_parquet(args.triage_path)
            if args.triage_path is not None
            else repository.load_optional_source_table("ed", "triage")
        )
        triage = triage_loaded if isinstance(triage_loaded, pd.DataFrame) else (
            triage_loaded.dataframe if triage_loaded is not None else None
        )
        vitalsign_loaded = (
            pd.read_parquet(args.vitalsign_path)
            if args.vitalsign_path is not None
            else repository.load_optional_source_table("ed", "vitalsign")
        )
        vitalsign = vitalsign_loaded if isinstance(vitalsign_loaded, pd.DataFrame) else (
            vitalsign_loaded.dataframe if vitalsign_loaded is not None else None
        )

        temporal = build_first_scope_temporal_features(
            encounter_medication_first_scope=first_scope,
            labevents=labevents,
            serum_creatinine_item_ids=settings.serum_creatinine_itemids,
            edstays=edstays,
            triage=triage,
            vitalsign=vitalsign,
        )
        validate_first_scope_temporal_features_artifact(temporal)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporal.to_parquet(output_path, index=False)
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built first_scope_temporal_features_65plus successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Feature store root: {settings.feature_store_root}")
    print(f"First-scope path: {first_scope_path}")
    print(f"Output: {output_path}")
    print(f"- rows={len(temporal):,}")
    print(
        "- creatinine_available_rows="
        f"{int(temporal['creatinine_latest_time'].notna().sum()):,}"
    )
    print(
        "- vital_window_rows="
        f"{int(temporal['vital_window_end'].notna().sum()):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
