"""Build Labels v1 for the 65+ first-scope subset and write its QC report."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    validate_encounter_index_artifact,
    validate_encounter_medication_first_scope_artifact,
    validate_encounter_medication_labels_v1_first_scope_artifact,
    validate_encounter_medication_state_artifact,
    validate_medication_events_artifact,
    validate_medication_rxnorm_mapping_artifact,
)
from opti_med.data_access.exceptions import DataLoadError
from opti_med.labels.first_scope import (
    build_first_scope_labels,
    summarize_first_scope_labels,
    write_first_scope_label_qc_report,
)
from opti_med.standardized import StandardizedParquetRepository


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build encounter_medication_labels_v1_65plus_first_scope from the persisted "
            "first-scope artifact, RxNorm-enriched state artifact, canonical medication "
            "events, encounter boundaries, and feasible post-review harm signals."
        )
    )
    parser.add_argument("--standardized-root", type=Path, default=None)
    parser.add_argument("--analytical-root", type=Path, default=None)
    parser.add_argument("--label-root", type=Path, default=None)
    parser.add_argument("--first-scope-path", type=Path, default=None)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--mapping-path", type=Path, default=None)
    parser.add_argument("--medication-events-path", type=Path, default=None)
    parser.add_argument("--encounter-index-path", type=Path, default=None)
    parser.add_argument("--labevents-path", type=Path, default=None)
    parser.add_argument("--triage-path", type=Path, default=None)
    parser.add_argument("--vitalsign-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--qc-output", type=Path, default=None)
    return parser


def main() -> int:
    """Run the first-scope Labels v1 build workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
        analytical_root=args.analytical_root or base_settings.analytical_root,
        label_root=args.label_root or base_settings.label_root,
    )
    repository = StandardizedParquetRepository(settings)

    state_default_path = (
        settings.encounter_medication_state_rxnorm_output_path
        if settings.encounter_medication_state_rxnorm_output_path.exists()
        else settings.encounter_medication_state_output_path
    )
    first_scope_path = args.first_scope_path or settings.encounter_medication_first_scope_output_path
    state_path = args.state_path or state_default_path
    mapping_path = args.mapping_path or settings.medication_rxnorm_mapping_output_path
    medication_events_path = args.medication_events_path or settings.medication_events_output_path
    encounter_index_path = args.encounter_index_path or settings.encounter_index_output_path
    output_path = args.output or settings.first_scope_labels_output_path
    qc_output_path = args.qc_output or settings.first_scope_label_qc_report_path

    try:
        first_scope = pd.read_parquet(first_scope_path)
        encounter_medication_state = pd.read_parquet(state_path)
        medication_rxnorm_mapping = pd.read_parquet(mapping_path)
        medication_events = pd.read_parquet(medication_events_path)
        encounter_index = pd.read_parquet(encounter_index_path)
        if args.labevents_path is not None:
            labevents = pd.read_parquet(args.labevents_path)
        else:
            loaded = repository.load_optional_source_table(
                "clinical",
                "labevents",
                columns=["hadm_id", "itemid", "charttime", "valuenum"],
            )
            labevents = loaded.dataframe if loaded else None
        if args.triage_path is not None:
            triage = pd.read_parquet(args.triage_path)
        else:
            loaded = repository.load_optional_source_table(
                "ed",
                "triage",
                columns=["subject_id", "stay_id", "charttime", "sbp", "dbp", "heartrate", "resprate", "o2sat", "pain"],
            )
            triage = loaded.dataframe if loaded else None
        if args.vitalsign_path is not None:
            vitalsign = pd.read_parquet(args.vitalsign_path)
        else:
            loaded = repository.load_optional_source_table(
                "ed",
                "vitalsign",
                columns=["subject_id", "stay_id", "charttime", "sbp", "dbp", "heartrate", "resprate", "o2sat", "pain"],
            )
            vitalsign = loaded.dataframe if loaded else None

        validate_encounter_medication_first_scope_artifact(first_scope)
        validate_encounter_medication_state_artifact(encounter_medication_state)
        validate_medication_rxnorm_mapping_artifact(medication_rxnorm_mapping)
        validate_medication_events_artifact(medication_events)
        validate_encounter_index_artifact(encounter_index)

        labels = build_first_scope_labels(
            encounter_medication_first_scope=first_scope,
            encounter_medication_state=encounter_medication_state,
            medication_events=medication_events,
            medication_rxnorm_mapping=medication_rxnorm_mapping,
            encounter_index=encounter_index,
            labevents=labevents,
            triage=triage,
            vitalsign=vitalsign,
            serum_creatinine_item_ids=settings.serum_creatinine_itemids,
        )
        validate_encounter_medication_labels_v1_first_scope_artifact(labels)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(output_path, index=False)
        write_first_scope_label_qc_report(
            encounter_medication_labels=labels,
            output_path=qc_output_path,
        )
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built encounter_medication_labels_v1_65plus_first_scope successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Label root: {settings.label_root}")
    print(f"First-scope path: {first_scope_path}")
    print(f"State path: {state_path}")
    print(f"Mapping path: {mapping_path}")
    print(f"Medication events path: {medication_events_path}")
    print(f"Encounter index path: {encounter_index_path}")
    print(f"Output: {output_path}")
    print(f"QC output: {qc_output_path}")
    for summary in summarize_first_scope_labels(encounter_medication_labels=labels):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
