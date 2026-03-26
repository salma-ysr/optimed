"""Build the post-session comparison between blind clinician labels and the ordinal model."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.modeling.first_scope_post_blind_eval_comparison import (
    build_first_scope_post_blind_eval_comparison,
    write_first_scope_post_blind_eval_comparison_artifacts,
)


DEFAULT_OUTPUT_ROOT = Path(
    "data/model_outputs/first_scope_post_blind_eval_comparison_v1_65plus_first_scope"
)
DEFAULT_REPORT_OUTPUT = Path(
    "docs/ml_pivot/23_phase8_post_blind_model_comparison_65plus_first_scope.md"
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare the final blind-session clinician labels against the current ordinal model "
            "predictions on the targeted blind evaluation slice."
        )
    )
    parser.add_argument("--blind-slice-path", type=Path, default=None)
    parser.add_argument("--latest-reviews-path", type=Path, default=None)
    parser.add_argument("--scored-universe-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser


def main() -> int:
    """Build the persisted post-blind comparison artifacts."""
    args = build_parser().parse_args()
    settings = Settings.from_env()
    blind_slice_path = args.blind_slice_path or settings.targeted_blind_eval_slice_output_path
    latest_reviews_path = args.latest_reviews_path or settings.clinician_review_snapshot_output_path
    scored_universe_path = (
        args.scored_universe_path or settings.first_scope_ordinal_scored_universe_output_path
    )

    try:
        blind_slice = pd.read_parquet(blind_slice_path)
        latest_reviews = pd.read_parquet(latest_reviews_path)
        scored_universe = pd.read_parquet(scored_universe_path)
        result = build_first_scope_post_blind_eval_comparison(
            blind_eval_slice=blind_slice,
            latest_clinician_reviews=latest_reviews,
            scored_universe=scored_universe,
        )
        artifact_paths = write_first_scope_post_blind_eval_comparison_artifacts(
            result=result,
            output_root=args.output_root,
            report_output=args.report_output,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built the post-blind clinician vs model comparison successfully.")
    print(f"Blind slice input: {blind_slice_path}")
    print(f"Latest reviews input: {latest_reviews_path}")
    print(f"Scored universe input: {scored_universe_path}")
    print(f"Rowwise comparison output: {artifact_paths['rowwise_comparison']}")
    print(f"Summary output: {artifact_paths['summary']}")
    print(f"Report output: {artifact_paths['report']}")
    if "report_copy" in artifact_paths:
        print(f"Docs report copy: {artifact_paths['report_copy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
