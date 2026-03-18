"""CSV-backed repository for scored OPTI-MED outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from opti_med.api.schemas import ScoredOutputSummary
from opti_med.config import Settings


@dataclass
class ScoredDataRepository:
    """Load and cache scored output data from disk."""

    settings: Settings
    _cached_df: pd.DataFrame | None = None
    _cached_mtime_ns: int | None = None

    @property
    def output_path(self) -> Path:
        """Return the configured scored output path."""
        return self.settings.scored_output_path

    def exists(self) -> bool:
        """Return whether the scored output file exists."""
        return self.output_path.exists()

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """Load the scored output dataframe from disk."""
        path = self.output_path
        if not path.exists():
            raise FileNotFoundError(
                f"Scored output not found at '{path}'. Run the scoring pipeline first or call the refresh endpoint."
            )

        current_mtime_ns = path.stat().st_mtime_ns
        if (
            not force_reload
            and self._cached_df is not None
            and self._cached_mtime_ns == current_mtime_ns
        ):
            return self._cached_df.copy()

        dataframe = pd.read_csv(path)
        dataframe = dataframe.where(pd.notna(dataframe), None)
        self._cached_df = dataframe
        self._cached_mtime_ns = current_mtime_ns
        return dataframe.copy()

    def summary(self) -> ScoredOutputSummary:
        """Build a summary of the scored output file."""
        dataframe = self.load()
        path = self.output_path
        last_modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        return ScoredOutputSummary(
            path=str(path),
            row_count=len(dataframe),
            column_count=dataframe.shape[1],
            unique_subjects=int(dataframe["subject_id"].nunique()),
            unique_admissions=int(dataframe["hadm_id"].nunique()),
            last_modified=last_modified,
        )
