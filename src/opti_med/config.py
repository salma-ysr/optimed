"""Configuration helpers for OPTI-MED data loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTERNAL_DATA_ROOT = Path("data/external")
DEFAULT_CLINICAL_DATA_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "mimic-iv-clinical-database-demo-2.2"
DEFAULT_ED_DATA_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "mimic-iv-ed-demo-2.2"
DEFAULT_SERUM_CREATININE_ITEMIDS = (50912, 51081, 51977, 52546)
DEFAULT_SNAPSHOT_STRATEGY = "latest_available"


@dataclass(frozen=True)
class Settings:
    """Application settings for locating MIMIC-IV source data."""

    data_root: Path = DEFAULT_CLINICAL_DATA_ROOT
    external_data_root: Path = DEFAULT_EXTERNAL_DATA_ROOT
    clinical_data_root: Path = DEFAULT_CLINICAL_DATA_ROOT
    ed_data_root: Path = DEFAULT_ED_DATA_ROOT
    hosp_dir_name: str = "hosp"
    ed_dir_name: str = "ed"
    file_extension: str = ".csv.gz"
    interim_root: Path = Path("data/interim")
    processed_root: Path = Path("data/processed")
    final_root: Path = Path("data/final")
    older_adult_age_threshold: int = 65
    polypharmacy_threshold: int = 5
    renal_risk_creatinine_threshold: float = 1.5
    serum_creatinine_itemids: tuple[int, ...] = DEFAULT_SERUM_CREATININE_ITEMIDS
    snapshot_strategy: str = DEFAULT_SNAPSHOT_STRATEGY

    @property
    def hosp_root(self) -> Path:
        """Return the configured hospital tables directory for the clinical demo."""
        return self.clinical_data_root / self.hosp_dir_name

    @property
    def ed_root(self) -> Path:
        """Return the configured emergency department tables directory."""
        return self.ed_data_root / self.ed_dir_name

    @property
    def cohort_output_path(self) -> Path:
        """Return the default interim cohort output path."""
        return self.interim_root / "older_adult_medication_cohort.csv"

    @property
    def encounter_index_output_path(self) -> Path:
        """Return the default interim encounter index path."""
        return self.interim_root / "encounter_index.csv"

    @property
    def medication_events_output_path(self) -> Path:
        """Return the default interim medication events path."""
        return self.interim_root / "medication_events.csv"

    @property
    def medication_snapshot_output_path(self) -> Path:
        """Return the default interim medication snapshot path."""
        return self.interim_root / "medication_snapshot.csv"

    @property
    def processed_output_path(self) -> Path:
        """Return the default processed cohort output path."""
        return self.processed_root / "older_adult_medication_features.csv"

    @property
    def scored_output_path(self) -> Path:
        """Return the default scored cohort output path."""
        return self.final_root / "older_adult_medication_scores.csv"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables."""
        external_data_root = Path(
            os.getenv("OPTI_MED_EXTERNAL_DATA_ROOT", str(DEFAULT_EXTERNAL_DATA_ROOT))
        )
        data_root = Path(os.getenv("OPTI_MED_DATA_ROOT", str(DEFAULT_CLINICAL_DATA_ROOT)))
        clinical_data_root = Path(
            os.getenv("OPTI_MED_CLINICAL_DATA_ROOT", str(data_root))
        )
        ed_data_root = Path(
            os.getenv("OPTI_MED_ED_DATA_ROOT", str(DEFAULT_ED_DATA_ROOT))
        )
        hosp_dir_name = os.getenv("OPTI_MED_HOSP_DIR", "hosp")
        ed_dir_name = os.getenv("OPTI_MED_ED_DIR", "ed")
        file_extension = os.getenv("OPTI_MED_FILE_EXTENSION", ".csv.gz")
        interim_root = Path(os.getenv("OPTI_MED_INTERIM_ROOT", "data/interim"))
        processed_root = Path(os.getenv("OPTI_MED_PROCESSED_ROOT", "data/processed"))
        final_root = Path(os.getenv("OPTI_MED_FINAL_ROOT", "data/final"))
        older_adult_age_threshold = int(
            os.getenv("OPTI_MED_OLDER_ADULT_AGE_THRESHOLD", "65")
        )
        polypharmacy_threshold = int(os.getenv("OPTI_MED_POLYPHARMACY_THRESHOLD", "5"))
        renal_risk_creatinine_threshold = float(
            os.getenv("OPTI_MED_RENAL_RISK_CREATININE_THRESHOLD", "1.5")
        )
        serum_creatinine_itemids_raw = os.getenv("OPTI_MED_SERUM_CREATININE_ITEMIDS")
        snapshot_strategy = os.getenv(
            "OPTI_MED_SNAPSHOT_STRATEGY",
            DEFAULT_SNAPSHOT_STRATEGY,
        )
        serum_creatinine_itemids = DEFAULT_SERUM_CREATININE_ITEMIDS
        if serum_creatinine_itemids_raw:
            serum_creatinine_itemids = tuple(
                int(itemid.strip())
                for itemid in serum_creatinine_itemids_raw.split(",")
                if itemid.strip()
            )
        return cls(
            data_root=data_root,
            external_data_root=external_data_root,
            clinical_data_root=clinical_data_root,
            ed_data_root=ed_data_root,
            hosp_dir_name=hosp_dir_name,
            ed_dir_name=ed_dir_name,
            file_extension=file_extension,
            interim_root=interim_root,
            processed_root=processed_root,
            final_root=final_root,
            older_adult_age_threshold=older_adult_age_threshold,
            polypharmacy_threshold=polypharmacy_threshold,
            renal_risk_creatinine_threshold=renal_risk_creatinine_threshold,
            serum_creatinine_itemids=serum_creatinine_itemids,
            snapshot_strategy=snapshot_strategy,
        )
