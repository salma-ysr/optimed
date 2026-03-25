"""Command-line entry point for building encounter and medication analytical artifacts."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from opti_med.config import Settings
from opti_med.data_access.encounters import EncounterIndexBuilder, summarize_encounter_index
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.medication_events import (
    CanonicalMedicationEventBuilder,
    DEFAULT_STREAMING_BATCH_SIZE,
    DEFAULT_STREAMING_SUBJECT_PARTITION_COUNT,
    summarize_medication_events,
    summarize_medication_events_metrics,
)
from opti_med.pipeline import write_encounter_and_medication_events_qc_report
from opti_med.pipeline import write_encounter_and_medication_events_qc_report_from_metrics


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build the standardized encounter_index and medication_events analytical artifacts."
        )
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        default=None,
        help="Root directory containing standardized source tables and output artifacts.",
    )
    parser.add_argument(
        "--encounter-output",
        type=Path,
        default=None,
        help="Output Parquet path for encounter_index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path for medication_events.",
    )
    parser.add_argument(
        "--qc-report",
        type=Path,
        default=None,
        help="Markdown QC report path for the encounter and medication events artifacts.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Force the subject-partitioned streaming build path for medication_events.",
    )
    parser.add_argument(
        "--subject-partitions",
        type=int,
        default=None,
        help="Number of subject partitions to use for streaming medication_events builds.",
    )
    parser.add_argument(
        "--partition-batch-size",
        type=int,
        default=None,
        help="Parquet batch size to use while partitioning large standardized source tables.",
    )
    return parser


def main() -> int:
    """Run the encounter-index and medication-events build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
        encounter_medication_qc_report_path=(
            args.qc_report or base_settings.encounter_medication_qc_report_path
        ),
    )

    encounter_builder = EncounterIndexBuilder(settings)
    medication_builder = CanonicalMedicationEventBuilder(settings)
    try:
        encounter_index = encounter_builder.build()
        encounter_result = encounter_builder.save(
            encounter_index,
            output_path=args.encounter_output,
        )
        use_streaming = args.streaming or medication_builder.should_use_streaming()
        if use_streaming:
            medication_result = medication_builder.build_and_save_streaming(
                encounter_index=encounter_index,
                output_path=args.output,
                subject_partition_count=(
                    args.subject_partitions or DEFAULT_STREAMING_SUBJECT_PARTITION_COUNT
                ),
                partition_batch_size=args.partition_batch_size or DEFAULT_STREAMING_BATCH_SIZE,
            )
            qc_path = write_encounter_and_medication_events_qc_report_from_metrics(
                encounter_index=encounter_result.dataframe,
                medication_events_metrics=medication_result.metrics,
                output_path=settings.encounter_medication_qc_report_path,
            )
            medication_summaries = summarize_medication_events_metrics(medication_result.metrics)
        else:
            medication_events = medication_builder.build(encounter_index=encounter_index)
            medication_result = medication_builder.save(
                medication_events,
                output_path=args.output,
            )
            qc_path = write_encounter_and_medication_events_qc_report(
                encounter_index=encounter_result.dataframe,
                medication_events=medication_result.dataframe,
                output_path=settings.encounter_medication_qc_report_path,
            )
            medication_summaries = summarize_medication_events(medication_result.dataframe)
    except DataLoadError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built standardized encounter and medication analytical artifacts successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Encounter output: {encounter_result.output_path}")
    print(f"Medication events output: {medication_result.output_path}")
    print(f"QC report: {qc_path}")
    for summary in summarize_encounter_index(encounter_result.dataframe):
        print(f"- encounter_index {summary}")
    for summary in medication_summaries:
        print(f"- medication_events {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
