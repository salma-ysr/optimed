"""Command-line entry point for loading core MIMIC-IV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

from opti_med.config import Settings
from opti_med.data_access.exceptions import DataLoadError
from opti_med.data_access.loaders import MimicCoreLoader, summarize_tables


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Load and validate the core MIMIC-IV hospital tables."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Path to the MIMIC-IV dataset root directory.",
    )
    parser.add_argument(
        "--file-extension",
        default=None,
        help="Table file extension, for example .csv.gz or .csv.",
    )
    return parser


def main() -> int:
    """Run the core-table loading workflow."""
    args = build_parser().parse_args()
    base_settings = Settings.from_env()
    settings = Settings(
        data_root=args.data_root or base_settings.data_root,
        hosp_dir_name=base_settings.hosp_dir_name,
        file_extension=args.file_extension or base_settings.file_extension,
    )

    loader = MimicCoreLoader(settings)

    try:
        tables = loader.load_all()
    except DataLoadError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("Loaded core MIMIC-IV tables successfully.")
    print(f"Data root: {settings.data_root}")
    for summary in summarize_tables(tables):
        print(f"- {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
