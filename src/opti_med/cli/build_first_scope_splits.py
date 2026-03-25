"""Build subject-safe train/validation/test splits for Dataset v1."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.evaluation.contracts import TrainValidationTestSplitConfig
from opti_med.modeling.first_scope_dataset import (
    build_first_scope_splits,
    validate_first_scope_dataset_artifact,
    validate_first_scope_split_artifact,
)


DEFAULT_SPLIT_RANDOM_SEED = 20260324
DEFAULT_STRATIFICATION_LABEL = "label__primary_action_binary"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate subject-safe train/validation/test splits for "
            "encounter_medication_dataset_v1_65plus_first_scope using the "
            "evaluation split contracts."
        )
    )
    parser.add_argument("--modeling-root", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
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
    """Run the Dataset v1 split-generation workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        modeling_root=args.modeling_root or base_settings.modeling_root,
    )

    dataset_path = args.dataset_path or settings.first_scope_dataset_output_path
    output_path = args.output or settings.first_scope_splits_output_path
    stratification_label_name = (
        args.stratification_label_name.strip() or None
        if args.stratification_label_name is not None
        else None
    )

    try:
        dataset = pd.read_parquet(dataset_path)
        validate_first_scope_dataset_artifact(dataset)

        splits = build_first_scope_splits(
            encounter_medication_dataset=dataset,
            split_config=TrainValidationTestSplitConfig(
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
                random_seed=args.random_seed,
                stratification_label_name=stratification_label_name,
            ),
        )
        validate_first_scope_split_artifact(splits)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        splits.to_parquet(output_path, index=False)
    except (DataLoadError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built splits_v1_65plus_first_scope successfully.")
    print(f"Modeling root: {settings.modeling_root}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"train_fraction={args.train_fraction}")
    print(f"validation_fraction={args.validation_fraction}")
    print(f"test_fraction={args.test_fraction}")
    print(f"random_seed={args.random_seed}")
    print(f"stratification_label_name={stratification_label_name or 'unstratified'}")
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
