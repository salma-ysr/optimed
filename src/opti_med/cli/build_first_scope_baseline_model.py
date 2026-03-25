"""Build the baseline modeling and Phase 4 level-aware comparison artifacts for the 65+ first-scope branch."""

from __future__ import annotations

import argparse
from pathlib import Path

from opti_med.data_access.exceptions import DataLoadError
from opti_med.modeling.first_scope_baseline import (
    build_first_scope_baseline_supervised_data,
    load_first_scope_baseline_inputs,
    run_first_scope_baseline,
    write_first_scope_baseline_artifacts,
)


DEFAULT_MODELING_ROOT = Path("data/modeling")
DEFAULT_OUTPUT_ROOT = Path("data/model_outputs/first_scope_baseline_v1_65plus_first_scope")
DEFAULT_REPORT_OUTPUT = Path(
    "docs/ml_pivot/19a_phase4_level_aware_comparison_65plus_first_scope.md"
)
DEFAULT_RANDOM_SEED = 20260325


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the baseline modeling scaffold plus Phase 4 level-aware "
            "comparison artifacts on top of the persisted Phase 2.5 dataset and split artifacts."
        )
    )
    parser.add_argument("--modeling-root", type=Path, default=DEFAULT_MODELING_ROOT)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--splits-path", type=Path, default=None)
    parser.add_argument("--feature-list-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def main() -> int:
    """Run the canonical baseline modeling and Phase 4 comparison workflow."""
    args = build_parser().parse_args()
    dataset_path = (
        args.dataset_path
        or args.modeling_root / "encounter_medication_dataset_v1_65plus_first_scope.parquet"
    )
    splits_path = (
        args.splits_path
        or args.modeling_root / "splits_v1_65plus_first_scope.parquet"
    )
    feature_list_path = (
        args.feature_list_path
        or args.modeling_root / "feature_list_v1_65plus_first_scope.json"
    )

    try:
        dataset, splits, feature_list_payload = load_first_scope_baseline_inputs(
            dataset_path=dataset_path,
            splits_path=splits_path,
            feature_list_path=feature_list_path,
        )
        supervised_data = build_first_scope_baseline_supervised_data(
            encounter_medication_dataset=dataset,
            splits=splits,
            feature_list_payload=feature_list_payload,
        )
        result = run_first_scope_baseline(
            supervised_data=supervised_data,
            dataset_path=dataset_path,
            splits_path=splits_path,
            feature_list_path=feature_list_path,
            random_seed=args.random_seed,
        )
        artifact_paths = write_first_scope_baseline_artifacts(
            result=result,
            output_root=args.output_root,
            report_output=args.report_output,
        )
    except (DataLoadError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built the first-scope baseline and Phase 4 level-aware comparison artifacts successfully.")
    print(f"Dataset input: {dataset_path}")
    print(f"Splits input: {splits_path}")
    print(f"Feature-list input: {feature_list_path}")
    print(f"Run id: {result.run_id}")
    print(f"Predictions output: {artifact_paths['predictions']}")
    print(f"Level-aware comparison details output: {artifact_paths['level_comparison_details']}")
    print(f"Metrics output: {artifact_paths['metrics']}")
    print(f"Feature manifest output: {artifact_paths['feature_manifest']}")
    print(f"Report output: {artifact_paths['report']}")
    if "report_copy" in artifact_paths:
        print(f"Docs report copy: {artifact_paths['report_copy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
