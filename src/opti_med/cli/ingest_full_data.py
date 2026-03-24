"""Command-line entry point for full-data raw discovery and standardization."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.pipeline import FullDataIngestionPipeline, summarize_ingestion_manifest


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for full-data ingestion."""
    parser = argparse.ArgumentParser(
        description=(
            "Discover, validate, and standardize the full MIMIC-IV clinical and ED raw tables."
        )
    )
    parser.add_argument(
        "--clinical-root",
        type=Path,
        help="Path to the full raw MIMIC-IV clinical dataset root.",
    )
    parser.add_argument(
        "--ed-root",
        type=Path,
        help="Path to the full raw MIMIC-IV-ED dataset root.",
    )
    parser.add_argument(
        "--standardized-root",
        type=Path,
        help="Directory where standardized Parquet tables should be written.",
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        help="Directory where ingestion manifests should be written.",
    )
    parser.add_argument(
        "--file-extension",
        default=None,
        help="Preferred source extension hint, for example .csv.gz or .csv.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="CSV chunk size for standardization reads.",
    )
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite standardized tables even when a matching manifest already exists.",
    )
    behavior.add_argument(
        "--incremental",
        action="store_true",
        help="Reuse unchanged standardized tables when the raw source signature matches the latest manifest.",
    )
    return parser


def main() -> int:
    """Run the full-data ingestion workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    standardized_root = args.standardized_root or base_settings.standardized_root
    manifest_root = args.manifest_root or (standardized_root / "manifests")
    ingestion_behavior = base_settings.ingestion_behavior
    if args.overwrite:
        ingestion_behavior = "overwrite"
    elif args.incremental:
        ingestion_behavior = "incremental"

    settings = replace(
        base_settings,
        raw_clinical_data_root=args.clinical_root or base_settings.raw_clinical_data_root,
        raw_ed_data_root=args.ed_root or base_settings.raw_ed_data_root,
        standardized_root=standardized_root,
        manifest_root=manifest_root,
        file_extension=args.file_extension or base_settings.file_extension,
        ingestion_behavior=ingestion_behavior,
        ingestion_chunk_size=args.chunk_size or base_settings.ingestion_chunk_size,
    )

    try:
        manifest = FullDataIngestionPipeline(settings).run()
    except DataLoadError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Standardized full-data ingestion completed.")
    print(f"Manifest: {manifest.manifest_path}")
    for line in summarize_ingestion_manifest(manifest):
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
