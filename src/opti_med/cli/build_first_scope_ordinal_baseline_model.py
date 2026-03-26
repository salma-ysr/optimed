"""Run the minimal ordinal supervised experiment on the prepared sidecar."""

from __future__ import annotations

import argparse
from pathlib import Path

from opti_med.data_access.exceptions import DataLoadError
from opti_med.modeling.first_scope_ordinal_baseline import (
    build_first_scope_ordinal_supervised_data,
    run_first_scope_ordinal_baseline,
    write_first_scope_ordinal_baseline_artifacts,
)
import pandas as pd


DEFAULT_MODELING_ROOT = Path("data/modeling")
DEFAULT_OUTPUT_ROOT = Path("data/model_outputs/first_scope_ordinal_baseline_v1_65plus_first_scope")
DEFAULT_REPORT_OUTPUT = Path(
    "docs/ml_pivot/21_phase8_minimal_ordinal_supervised_experiment_65plus_first_scope.md"
)
DEFAULT_RANDOM_SEED = 20260325


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal multiclass ordinal supervised experiment on the prepared "
            "clinician-review sidecar."
        )
    )
    parser.add_argument("--modeling-root", type=Path, default=DEFAULT_MODELING_ROOT)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--ordinal-targets-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def main() -> int:
    """Run the minimal ordinal supervised experiment."""
    args = build_parser().parse_args()
    dataset_path = (
        args.dataset_path
        or args.modeling_root / "encounter_medication_dataset_v1_65plus_first_scope.parquet"
    )
    ordinal_targets_path = (
        args.ordinal_targets_path
        or args.modeling_root / "encounter_medication_ordinal_targets_v1_65plus_first_scope.parquet"
    )
    try:
        dataset = pd.read_parquet(dataset_path)
        ordinal_targets = pd.read_parquet(ordinal_targets_path)
        supervised_data = build_first_scope_ordinal_supervised_data(
            encounter_medication_dataset=dataset,
            ordinal_targets=ordinal_targets,
        )
        result = run_first_scope_ordinal_baseline(
            supervised_data=supervised_data,
            dataset_path=dataset_path,
            ordinal_targets_path=ordinal_targets_path,
            random_seed=args.random_seed,
        )
        artifact_paths = write_first_scope_ordinal_baseline_artifacts(
            result=result,
            output_root=args.output_root,
            report_output=args.report_output,
        )
    except (DataLoadError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built the minimal first-scope ordinal supervised experiment successfully.")
    print(f"Dataset input: {dataset_path}")
    print(f"Ordinal targets input: {ordinal_targets_path}")
    print(f"Run id: {result.run_id}")
    print(f"Predictions output: {artifact_paths['predictions']}")
    print(f"Metrics output: {artifact_paths['metrics']}")
    print(f"Feature manifest output: {artifact_paths['feature_manifest']}")
    print(f"Report output: {artifact_paths['report']}")
    if "report_copy" in artifact_paths:
        print(f"Docs report copy: {artifact_paths['report_copy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
