"""Configuration helpers for OPTI-MED data loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("data/external/mimic-iv-clinical-database-demo-2.2")


@dataclass(frozen=True)
class Settings:
    """Application settings for locating MIMIC-IV source data."""

    data_root: Path = DEFAULT_DATA_ROOT
    hosp_dir_name: str = "hosp"
    file_extension: str = ".csv.gz"

    @property
    def hosp_root(self) -> Path:
        """Return the configured hospital tables directory."""
        return self.data_root / self.hosp_dir_name

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        data_root = Path(os.getenv("OPTI_MED_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
        hosp_dir_name = os.getenv("OPTI_MED_HOSP_DIR", "hosp")
        file_extension = os.getenv("OPTI_MED_FILE_EXTENSION", ".csv.gz")
        return cls(
            data_root=data_root,
            hosp_dir_name=hosp_dir_name,
            file_extension=file_extension,
        )
