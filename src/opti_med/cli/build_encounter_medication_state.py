"""Command-line entry point for building the encounter-medication-state artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.cohort import limit_eligibility_to_subject_count
from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import validate_older_adult_eligibility_artifact
from opti_med.data_access.encounter_medication_state import (
    DEFAULT_ENCOUNTER_MEDICATION_STATE_POLICY,
    LOOKUP_MODE_DISABLED,
    EncounterMedicationStateArtifactBuilder,
    summarize_encounter_medication_state,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.medication_rxnorm_mapping import MedicationRxNormMappingBuilder
from opti_med.pipeline import write_encounter_medication_state_qc_report
from opti_med.time_semantics import validate_review_time_policy


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Build the 65+ encounter-medication-state analytical artifact."
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=None,
        help="Root directory containing standardized source tables and upstream artifacts.",
    )
    parser.add_argument(
        "--analytical-root",
        type=Path,
        default=None,
        help="Root directory containing older-adult analytical artifacts.",
    )
    parser.add_argument(
        "--review-time-policy",
        default=None,
        help=(
            "Review-time policy to apply. Defaults to discharge_capped_latest_available."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path for encounter_medication_state_65plus.",
    )
    parser.add_argument(
        "--rxnorm-lookup-mode",
        default=LOOKUP_MODE_DISABLED,
        help="RxNorm lookup mode: disabled, cache_first, or cache_only.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=None,
        help="Output Parquet path for the medication_rxnorm_mapping cache artifact.",
    )
    parser.add_argument(
        "--qc-report",
        type=Path,
        default=None,
        help="Optional Markdown QC report path for the encounter-medication-state artifact.",
    )
    parser.add_argument(
        "--eligibility-path",
        type=Path,
        default=None,
        help="Optional Parquet path for the eligible encounter artifact to consume.",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional deterministic limit on unique eligible subjects, ordered by subject_id ascending.",
    )
    return parser


def main() -> int:
    """Run the encounter-medication-state build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
        analytical_root=args.analytical_root or base_settings.analytical_root,
    )
    review_time_policy = (
        validate_review_time_policy(args.review_time_policy)
        if args.review_time_policy
        else DEFAULT_ENCOUNTER_MEDICATION_STATE_POLICY
    )
    builder = EncounterMedicationStateArtifactBuilder(
        settings,
        review_time_policy=review_time_policy,
        semantic_lookup_mode=args.rxnorm_lookup_mode,
    )
    eligible_encounters = None
    if args.eligibility_path is not None or args.max_subjects is not None:
        eligibility_path = args.eligibility_path or settings.older_adult_eligibility_output_path
        eligible_encounters = pd.read_parquet(eligibility_path)
        validate_older_adult_eligibility_artifact(eligible_encounters)
        eligible_encounters = limit_eligibility_to_subject_count(
            eligible_encounters,
            args.max_subjects,
        )

    try:
        encounter_medication_state = builder.build(
            eligible_encounters=eligible_encounters,
        )
        result = builder.save(
            encounter_medication_state,
            output_path=args.output or settings.encounter_medication_state_output_path,
        )
        mapping_result = None
        if builder.last_medication_rxnorm_mapping is not None:
            mapping_result = MedicationRxNormMappingBuilder(
                settings,
                semantic_mapper=builder.semantic_mapper,
                lookup_mode=args.rxnorm_lookup_mode,
            ).save(
                builder.last_medication_rxnorm_mapping,
                output_path=args.mapping_output or settings.medication_rxnorm_mapping_output_path,
            )
        qc_path = None
        if args.qc_report is not None:
            qc_path = write_encounter_medication_state_qc_report(
                encounter_medication_state=result.dataframe,
                output_path=args.qc_report,
            )
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built encounter_medication_state_65plus successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Review-time policy: {review_time_policy.value}")
    print(f"RxNorm lookup mode: {args.rxnorm_lookup_mode}")
    if args.eligibility_path is not None:
        print(f"Eligibility path: {args.eligibility_path}")
    if args.max_subjects is not None:
        print(f"Max subjects: {args.max_subjects}")
    print(f"Output: {result.output_path}")
    if mapping_result is not None:
        print(f"Medication RxNorm mapping output: {mapping_result.output_path}")
    if qc_path is not None:
        print(f"QC report: {qc_path}")
    for summary in summarize_encounter_medication_state(result.dataframe):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
