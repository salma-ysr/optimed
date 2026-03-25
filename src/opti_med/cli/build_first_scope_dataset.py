"""Build Dataset v1 for the 65+ first-scope subset."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    validate_encounter_medication_burden_artifact,
    validate_encounter_medication_first_scope_artifact,
    validate_encounter_medication_labels_v1_first_scope_artifact,
    validate_encounter_medication_semantics_artifact,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.features.first_scope_feature_store import (
    validate_first_scope_feature_store_artifact,
)
from opti_med.modeling.first_scope_dataset import (
    build_first_scope_dataset,
    summarize_first_scope_dataset,
    validate_first_scope_dataset_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Assemble encounter_medication_dataset_v1_65plus_first_scope by joining "
            "the persisted first-scope subset, semantics, burden, Feature Store v1, "
            "and Labels v1 artifacts while keeping benchmark-only columns explicit."
        )
    )
    parser.add_argument("--analytical-root", type=Path, default=None)
    parser.add_argument("--feature-store-root", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--modeling-root", type=Path, default=None)
    parser.add_argument("--first-scope-path", type=Path, default=None)
    parser.add_argument("--semantics-path", type=Path, default=None)
    parser.add_argument("--burden-path", type=Path, default=None)
    parser.add_argument("--feature-store-path", type=Path, default=None)
    parser.add_argument("--labels-path", type=Path, default=None)
    parser.add_argument("--benchmark-scores-path", type=Path, default=None)
    parser.add_argument("--polypharmacy-threshold", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--feature-list-output", type=Path, default=None)
    return parser


def main() -> int:
    """Run the Dataset v1 assembly workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        analytical_root=args.analytical_root or base_settings.analytical_root,
        feature_store_root=args.feature_store_root or base_settings.feature_store_root,
        label_root=args.label_root or base_settings.label_root,
        modeling_root=args.modeling_root or base_settings.modeling_root,
    )

    first_scope_path = args.first_scope_path or settings.encounter_medication_first_scope_output_path
    semantics_path = args.semantics_path or settings.encounter_medication_semantics_output_path
    burden_path = args.burden_path or settings.encounter_medication_burden_output_path
    feature_store_path = args.feature_store_path or settings.first_scope_feature_store_output_path
    labels_path = args.labels_path or settings.first_scope_labels_output_path
    output_path = args.output or settings.first_scope_dataset_output_path
    feature_list_output_path = (
        args.feature_list_output or settings.first_scope_feature_list_output_path
    )

    try:
        first_scope = pd.read_parquet(first_scope_path)
        semantics = pd.read_parquet(semantics_path)
        burden = pd.read_parquet(burden_path)
        feature_store = pd.read_parquet(feature_store_path)
        labels = pd.read_parquet(labels_path)
        benchmark_scores = (
            pd.read_parquet(args.benchmark_scores_path)
            if args.benchmark_scores_path is not None
            else None
        )

        validate_encounter_medication_first_scope_artifact(first_scope)
        validate_encounter_medication_semantics_artifact(semantics)
        validate_encounter_medication_burden_artifact(burden)
        validate_first_scope_feature_store_artifact(feature_store)
        validate_encounter_medication_labels_v1_first_scope_artifact(labels)

        result = build_first_scope_dataset(
            encounter_medication_first_scope=first_scope,
            encounter_medication_semantics=semantics,
            encounter_medication_burden=burden,
            first_scope_feature_store=feature_store,
            encounter_medication_labels=labels,
            polypharmacy_threshold=(
                args.polypharmacy_threshold
                if args.polypharmacy_threshold is not None
                else settings.polypharmacy_threshold
            ),
            benchmark_scores=benchmark_scores,
        )
        validate_first_scope_dataset_artifact(result.dataset)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_list_output_path.parent.mkdir(parents=True, exist_ok=True)
        result.dataset.to_parquet(output_path, index=False)

        feature_list_payload = dict(result.feature_list)
        feature_list_payload["artifact_paths"] = {
            "first_scope_input": str(first_scope_path),
            "semantics_input": str(semantics_path),
            "burden_input": str(burden_path),
            "feature_store_input": str(feature_store_path),
            "labels_input": str(labels_path),
            "benchmark_scores_input": (
                str(args.benchmark_scores_path)
                if args.benchmark_scores_path is not None
                else None
            ),
            "dataset_output": str(output_path),
            "feature_list_output": str(feature_list_output_path),
        }
        with feature_list_output_path.open("w", encoding="utf-8") as handle:
            json.dump(feature_list_payload, handle, indent=2, sort_keys=True)
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built encounter_medication_dataset_v1_65plus_first_scope successfully.")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Feature store root: {settings.feature_store_root}")
    print(f"Label root: {settings.label_root}")
    print(f"Modeling root: {settings.modeling_root}")
    print(f"First-scope path: {first_scope_path}")
    print(f"Semantics path: {semantics_path}")
    print(f"Burden path: {burden_path}")
    print(f"Feature-store path: {feature_store_path}")
    print(f"Labels path: {labels_path}")
    if args.benchmark_scores_path is not None:
        print(f"Benchmark scores path: {args.benchmark_scores_path}")
    print(f"Output: {output_path}")
    print(f"Feature-list output: {feature_list_output_path}")
    for summary in summarize_first_scope_dataset(
        encounter_medication_dataset=result.dataset,
    ):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
