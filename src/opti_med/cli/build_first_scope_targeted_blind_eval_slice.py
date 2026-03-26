"""Materialize the targeted blind clinician evaluation slice from the real ordinal model."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.api.clinician_reviews import ClinicianReviewRepository
from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.modeling.first_scope_targeted_blind_eval import (
    build_first_scope_targeted_blind_eval,
    write_targeted_blind_eval_artifacts,
)


DEFAULT_RANDOM_SEED = 20260325


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the targeted blind clinician evaluation slice using the real ordinal model."
    )
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--modeling-root", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--ordinal-targets-path", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        label_root=args.label_root or base_settings.label_root,
        modeling_root=args.modeling_root or base_settings.modeling_root,
    )
    repository = ClinicianReviewRepository(settings)
    dataset_path = args.dataset_path or settings.first_scope_dataset_output_path
    ordinal_targets_path = args.ordinal_targets_path or settings.first_scope_ordinal_targets_output_path
    try:
        dataset = pd.read_parquet(dataset_path)
        ordinal_targets = pd.read_parquet(ordinal_targets_path)
        reviewable = repository.load_reviewable_universe()
        latest = repository.load_latest_reviews()
        result = build_first_scope_targeted_blind_eval(
            encounter_medication_dataset=dataset,
            reviewable_universe=reviewable,
            latest_clinician_reviews=latest,
            ordinal_targets=ordinal_targets,
            random_seed=args.random_seed,
        )
        artifact_paths = write_targeted_blind_eval_artifacts(
            result=result,
            scored_universe_output_path=settings.first_scope_ordinal_scored_universe_output_path,
            slice_output_path=settings.targeted_blind_eval_slice_output_path,
            summary_output_path=settings.targeted_blind_eval_summary_path,
            report_output_path=settings.targeted_blind_eval_qc_report_path,
        )
    except (DataLoadError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built the targeted blind clinician evaluation slice successfully.")
    print(f"Dataset input: {dataset_path}")
    print(f"Ordinal targets input: {ordinal_targets_path}")
    print(f"Scored universe output: {artifact_paths['scored_universe']}")
    print(f"Blind eval slice output: {artifact_paths['blind_eval_slice']}")
    print(f"Summary output: {artifact_paths['summary']}")
    print(f"Report output: {artifact_paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
