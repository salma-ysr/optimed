"""Write the first-scope Feature Store v1 leakage QC report."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.features.first_scope_feature_store import (
    build_first_scope_feature_store_leakage_qc_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Write a Markdown leakage QC report for the persisted 65+ "
            "first-scope Feature Store v1 artifact."
        )
    )
    parser.add_argument("--analytical-root", type=Path, default=None)
    parser.add_argument("--feature-store-root", type=Path, default=None)
    parser.add_argument("--first-scope-path", type=Path, default=None)
    parser.add_argument("--patient-context-path", type=Path, default=None)
    parser.add_argument("--temporal-path", type=Path, default=None)
    parser.add_argument("--feature-store-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    """Render and persist the leakage QC report."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        analytical_root=args.analytical_root or base_settings.analytical_root,
        feature_store_root=args.feature_store_root or base_settings.feature_store_root,
    )

    first_scope_path = args.first_scope_path or settings.encounter_medication_first_scope_output_path
    patient_context_path = (
        args.patient_context_path or settings.first_scope_patient_context_features_output_path
    )
    temporal_path = args.temporal_path or settings.first_scope_temporal_features_output_path
    feature_store_path = args.feature_store_path or settings.first_scope_feature_store_output_path
    output_path = args.output or settings.first_scope_feature_store_leakage_qc_report_path

    try:
        first_scope = pd.read_parquet(first_scope_path)
        patient_context = pd.read_parquet(patient_context_path)
        temporal = pd.read_parquet(temporal_path)
        feature_store = pd.read_parquet(feature_store_path)
        report = build_first_scope_feature_store_leakage_qc_report(
            encounter_medication_first_scope=first_scope,
            patient_context_features=patient_context,
            temporal_features=temporal,
            feature_store=feature_store,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Wrote first-scope Feature Store v1 leakage QC report successfully.")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
