"""Build the first-scope 65+ patient-context stage artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.features.first_scope_feature_store import (
    build_first_scope_patient_context_features,
    validate_first_scope_patient_context_features_artifact,
)
from opti_med.standardized import StandardizedParquetRepository


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build point-in-time-safe patient-context features for the persisted "
            "65+ first-scope encounter-medication subset."
        )
    )
    parser.add_argument("--standardized-root", type=Path, default=None)
    parser.add_argument("--analytical-root", type=Path, default=None)
    parser.add_argument("--feature-store-root", type=Path, default=None)
    parser.add_argument("--first-scope-path", type=Path, default=None)
    parser.add_argument("--patients-path", type=Path, default=None)
    parser.add_argument("--admissions-path", type=Path, default=None)
    parser.add_argument("--diagnoses-path", type=Path, default=None)
    parser.add_argument("--labevents-path", type=Path, default=None)
    parser.add_argument("--omr-path", type=Path, default=None)
    parser.add_argument("--edstays-path", type=Path, default=None)
    parser.add_argument("--ed-diagnosis-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    """Run the first-scope patient-context stage build."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = replace(
        base_settings,
        standardized_root=args.standardized_root or base_settings.standardized_root,
        analytical_root=args.analytical_root or base_settings.analytical_root,
        feature_store_root=args.feature_store_root or base_settings.feature_store_root,
    )
    repository = StandardizedParquetRepository(settings)

    first_scope_path = args.first_scope_path or settings.encounter_medication_first_scope_output_path
    output_path = args.output or settings.first_scope_patient_context_features_output_path

    try:
        first_scope = pd.read_parquet(first_scope_path)
        patients = (
            pd.read_parquet(args.patients_path)
            if args.patients_path is not None
            else repository.load_source_table("clinical", "patients").dataframe
        )
        admissions = (
            pd.read_parquet(args.admissions_path)
            if args.admissions_path is not None
            else repository.load_source_table("clinical", "admissions").dataframe
        )
        diagnoses_icd = (
            pd.read_parquet(args.diagnoses_path)
            if args.diagnoses_path is not None
            else repository.load_source_table("clinical", "diagnoses_icd").dataframe
        )
        labevents = (
            pd.read_parquet(args.labevents_path)
            if args.labevents_path is not None
            else repository.load_source_table("clinical", "labevents").dataframe
        )
        omr_loaded = (
            pd.read_parquet(args.omr_path)
            if args.omr_path is not None
            else repository.load_optional_source_table("clinical", "omr")
        )
        omr = omr_loaded if isinstance(omr_loaded, pd.DataFrame) else (
            omr_loaded.dataframe if omr_loaded is not None else None
        )
        edstays_loaded = (
            pd.read_parquet(args.edstays_path)
            if args.edstays_path is not None
            else repository.load_optional_source_table("ed", "edstays")
        )
        edstays = edstays_loaded if isinstance(edstays_loaded, pd.DataFrame) else (
            edstays_loaded.dataframe if edstays_loaded is not None else None
        )
        ed_diagnosis_loaded = (
            pd.read_parquet(args.ed_diagnosis_path)
            if args.ed_diagnosis_path is not None
            else repository.load_optional_source_table("ed", "diagnosis")
        )
        ed_diagnosis = (
            ed_diagnosis_loaded
            if isinstance(ed_diagnosis_loaded, pd.DataFrame)
            else (ed_diagnosis_loaded.dataframe if ed_diagnosis_loaded is not None else None)
        )

        patient_context = build_first_scope_patient_context_features(
            encounter_medication_first_scope=first_scope,
            patients=patients,
            admissions=admissions,
            diagnoses_icd=diagnoses_icd,
            labevents=labevents,
            omr=omr,
            edstays=edstays,
            ed_diagnosis=ed_diagnosis,
            serum_creatinine_item_ids=settings.serum_creatinine_itemids,
        )
        validate_first_scope_patient_context_features_artifact(patient_context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        patient_context.to_parquet(output_path, index=False)
    except (DataLoadError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Built first_scope_patient_context_features_65plus successfully.")
    print(f"Standardized root: {settings.standardized_root}")
    print(f"Analytical root: {settings.analytical_root}")
    print(f"Feature store root: {settings.feature_store_root}")
    print(f"First-scope path: {first_scope_path}")
    print(f"Output: {output_path}")
    print(f"- rows={len(patient_context):,}")
    print(
        "- egfr_available_rows="
        f"{int(patient_context['egfr_ml_min_1_73m2'].notna().sum()):,}"
    )
    print(
        "- cockcroft_gault_available_rows="
        f"{int(patient_context['cockcroft_gault_ml_min'].notna().sum()):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
