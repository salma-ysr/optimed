"""Build the full Phase 2.5 modeling hand-off artifacts for the 65+ first-scope branch."""

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
    CANONICAL_TRAINING_FILTER,
    build_first_scope_dataset,
    build_first_scope_splits,
    calculate_first_scope_dataset_qc_metrics,
    summarize_first_scope_dataset,
    validate_first_scope_dataset_artifact,
    validate_first_scope_split_artifact,
    write_first_scope_dataset_qc_report,
)
from opti_med.evaluation.contracts import TrainValidationTestSplitConfig


DEFAULT_SPLIT_RANDOM_SEED = 20260324
DEFAULT_STRATIFICATION_LABEL = "label__primary_action_binary"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build Dataset v1, subject-safe splits, and the dataset QC report for the "
            "canonical 65+ first-scope modeling branch."
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
    parser.add_argument("--dataset-output", type=Path, default=None)
    parser.add_argument("--feature-list-output", type=Path, default=None)
    parser.add_argument("--splits-output", type=Path, default=None)
    parser.add_argument("--qc-output", type=Path, default=None)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SPLIT_RANDOM_SEED)
    parser.add_argument(
        "--stratification-label-name",
        type=str,
        default=DEFAULT_STRATIFICATION_LABEL,
    )
    return parser


def main() -> int:
    """Run the full Phase 2.5 modeling hand-off workflow."""
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
    dataset_output_path = args.dataset_output or settings.first_scope_dataset_output_path
    feature_list_output_path = (
        args.feature_list_output or settings.first_scope_feature_list_output_path
    )
    splits_output_path = args.splits_output or settings.first_scope_splits_output_path
    qc_output_path = args.qc_output or settings.first_scope_dataset_qc_report_path
    stratification_label_name = (
        args.stratification_label_name.strip() or None
        if args.stratification_label_name is not None
        else None
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

        dataset_result = build_first_scope_dataset(
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
        validate_first_scope_dataset_artifact(dataset_result.dataset)

        splits = build_first_scope_splits(
            encounter_medication_dataset=dataset_result.dataset,
            split_config=TrainValidationTestSplitConfig(
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                random_seed=args.random_seed,
                stratification_label_name=stratification_label_name,
            ),
        )
        validate_first_scope_split_artifact(splits)

        dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_list_output_path.parent.mkdir(parents=True, exist_ok=True)
        splits_output_path.parent.mkdir(parents=True, exist_ok=True)
        qc_output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset_result.dataset.to_parquet(dataset_output_path, index=False)
        splits.to_parquet(splits_output_path, index=False)
        write_first_scope_dataset_qc_report(
            encounter_medication_dataset=dataset_result.dataset,
            splits=splits,
            output_path=qc_output_path,
        )

        feature_list_payload = dict(dataset_result.feature_list)
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
            "dataset_output": str(dataset_output_path),
            "feature_list_output": str(feature_list_output_path),
            "splits_output": str(splits_output_path),
            "dataset_qc_output": str(qc_output_path),
        }
        feature_list_payload["split_defaults"] = {
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "test_fraction": args.test_fraction,
            "random_seed": args.random_seed,
            "stratification_label_name": stratification_label_name or "unstratified",
        }
        feature_list_payload["training_filter"] = dict(CANONICAL_TRAINING_FILTER)
        feature_list_payload["qc_summary"] = calculate_first_scope_dataset_qc_metrics(
            encounter_medication_dataset=dataset_result.dataset,
            splits=splits,
        )
        with feature_list_output_path.open("w", encoding="utf-8") as handle:
            json.dump(feature_list_payload, handle, indent=2, sort_keys=True)
    except (DataLoadError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built the Phase 2.5 first-scope modeling hand-off successfully.")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Feature store root: {settings.feature_store_root}")
    print(f"Label root: {settings.label_root}")
    print(f"Modeling root: {settings.modeling_root}")
    print(f"Dataset output: {dataset_output_path}")
    print(f"Splits output: {splits_output_path}")
    print(f"Feature-list output: {feature_list_output_path}")
    print(f"QC output: {qc_output_path}")
    print(
        "Canonical training filter: "
        "meta__dataset_row_eligible_for_training_flag == 1"
    )
    for summary in summarize_first_scope_dataset(
        encounter_medication_dataset=dataset_result.dataset,
    ):
        print(f"- {summary}")
    for partition_name, count in (
        splits["split_partition"]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
        .items()
    ):
        print(f"- {partition_name}: rows={count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
