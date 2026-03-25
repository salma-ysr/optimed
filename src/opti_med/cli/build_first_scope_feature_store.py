"""Build Feature Store v1 for the 65+ first-scope subset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.features.first_scope_feature_store import (
    STATE_PROVENANCE_COLUMNS,
    build_first_scope_feature_store,
    validate_first_scope_feature_store_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Join the persisted first-scope base artifact with staged patient-context "
            "and temporal artifacts to produce Feature Store v1 and its metadata JSON."
        )
    )
    parser.add_argument("--analytical-root", type=Path, default=None)
    parser.add_argument("--feature-store-root", type=Path, default=None)
    parser.add_argument("--first-scope-path", type=Path, default=None)
    parser.add_argument("--patient-context-path", type=Path, default=None)
    parser.add_argument("--temporal-path", type=Path, default=None)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser


def main() -> int:
    """Run the final first-scope feature-store v1 join."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        analytical_root=args.analytical_root or base_settings.analytical_root,
        feature_store_root=args.feature_store_root or base_settings.feature_store_root,
    )

    state_default_path = (
        settings.encounter_medication_state_rxnorm_output_path
        if settings.encounter_medication_state_rxnorm_output_path.exists()
        else settings.encounter_medication_state_output_path
    )
    first_scope_path = args.first_scope_path or settings.encounter_medication_first_scope_output_path
    patient_context_path = (
        args.patient_context_path or settings.first_scope_patient_context_features_output_path
    )
    temporal_path = args.temporal_path or settings.first_scope_temporal_features_output_path
    state_path = args.state_path or state_default_path
    output_path = args.output or settings.first_scope_feature_store_output_path
    metadata_output_path = (
        args.metadata_output or settings.first_scope_feature_store_metadata_output_path
    )

    try:
        first_scope = pd.read_parquet(first_scope_path)
        patient_context = pd.read_parquet(patient_context_path)
        temporal = pd.read_parquet(temporal_path)
        state_subset = pd.read_parquet(state_path, columns=STATE_PROVENANCE_COLUMNS)
        result = build_first_scope_feature_store(
            encounter_medication_first_scope=first_scope,
            patient_context_features=patient_context,
            temporal_features=temporal,
            encounter_medication_state_provenance=state_subset,
        )
        validate_first_scope_feature_store_artifact(result.feature_store)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
        result.feature_store.to_parquet(output_path, index=False)
        metadata_payload = dict(result.metadata)
        metadata_payload["artifact_paths"] = {
            "first_scope_input": str(first_scope_path),
            "patient_context_stage": str(patient_context_path),
            "temporal_stage": str(temporal_path),
            "state_provenance_input": str(state_path),
            "feature_store_output": str(output_path),
            "metadata_output": str(metadata_output_path),
        }
        with metadata_output_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_payload, handle, indent=2, sort_keys=True)
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built encounter_medication_features_v1_65plus_first_scope successfully.")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Feature store root: {settings.feature_store_root}")
    print(f"First-scope path: {first_scope_path}")
    print(f"Patient-context path: {patient_context_path}")
    print(f"Temporal path: {temporal_path}")
    print(f"State path: {state_path}")
    print(f"Output: {output_path}")
    print(f"Metadata output: {metadata_output_path}")
    print(f"- rows={len(result.feature_store):,}")
    print(
        "- feature_available_rows="
        f"{int(result.feature_store['feature_available_as_of_review_time_flag'].sum()):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
