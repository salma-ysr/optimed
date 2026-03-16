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
    interim_root: Path = Path("data/interim")
    older_adult_age_threshold: int = 65

    @property
    def hosp_root(self) -> Path:
        """Return the configured hospital tables directory."""
        return self.data_root / self.hosp_dir_name

    @property
    def cohort_output_path(self) -> Path:
        """Return the default interim cohort output path."""
        return self.interim_root / "older_adult_medication_cohort.csv"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        data_root = Path(os.getenv("OPTI_MED_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
        hosp_dir_name = os.getenv("OPTI_MED_HOSP_DIR", "hosp")
        file_extension = os.getenv("OPTI_MED_FILE_EXTENSION", ".csv.gz")
        interim_root = Path(os.getenv("OPTI_MED_INTERIM_ROOT", "data/interim"))
        older_adult_age_threshold = int(
            os.getenv("OPTI_MED_OLDER_ADULT_AGE_THRESHOLD", "65")
        )
        return cls(
            data_root=data_root,
            hosp_dir_name=hosp_dir_name,
            file_extension=file_extension,
            interim_root=interim_root,
            older_adult_age_threshold=older_adult_age_threshold,
        )
