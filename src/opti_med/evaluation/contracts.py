"""Evaluation and experiment contracts for future ML work.

This module is intentionally model-agnostic. It defines split, metric, and
baseline contracts before any model training or dataset access begins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from opti_med.contracts.pipeline import (
    ANALYTICAL_GRAIN_DESCRIPTION,
    ANALYTICAL_GRAIN_PRIMARY_KEY,
)


ANALYTICAL_EVALUATION_GRAIN_DESCRIPTION = ANALYTICAL_GRAIN_DESCRIPTION
ANALYTICAL_EVALUATION_PRIMARY_KEY = ANALYTICAL_GRAIN_PRIMARY_KEY


class SplitPartition(str, Enum):
    """Allowed dataset partitions for future experiments."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class RankingMetric(str, Enum):
    """Ranking metrics for future prioritization evaluation."""

    NDCG_AT_K = "ndcg_at_k"
    MAP_AT_K = "map_at_k"
    MRR = "mrr"
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"


class CalibrationMetric(str, Enum):
    """Calibration metrics for probabilistic outputs."""

    BRIER_SCORE = "brier_score"
    EXPECTED_CALIBRATION_ERROR = "expected_calibration_error"
    CALIBRATION_SLOPE = "calibration_slope"
    CALIBRATION_INTERCEPT = "calibration_intercept"


class BaselineName(str, Enum):
    """Explicit baselines that future models must compare against."""

    CURRENT_RULE_SCORE = "current_rule_score"
    MEDICATION_CLASS_ONLY_BASELINE = "medication_class_only_baseline"
    POLYPHARMACY_ONLY_BASELINE = "polypharmacy_only_baseline"


BASELINE_NAMES: tuple[str, ...] = tuple(baseline.value for baseline in BaselineName)
DEFAULT_BASELINE_NAMES: tuple[BaselineName, ...] = (
    BaselineName.CURRENT_RULE_SCORE,
    BaselineName.MEDICATION_CLASS_ONLY_BASELINE,
    BaselineName.POLYPHARMACY_ONLY_BASELINE,
)


@dataclass(frozen=True, slots=True)
class SubjectLeakagePreventionConfig:
    """Contract for subject-level leakage prevention in split generation."""

    subject_id_field: str = "subject_id"
    require_subject_level_grouping: bool = True
    forbid_subject_overlap_across_partitions: bool = True
    leakage_error_message: str = (
        "A subject may not appear in more than one of train/validation/test."
    )


@dataclass(frozen=True, slots=True)
class TrainValidationTestSplitConfig:
    """Patient-level split contract for future experiments."""

    train_fraction: float
    validation_fraction: float
    test_fraction: float
    partition_names: tuple[SplitPartition, ...] = (
        SplitPartition.TRAIN,
        SplitPartition.VALIDATION,
        SplitPartition.TEST,
    )
    subject_leakage_prevention: SubjectLeakagePreventionConfig = (
        SubjectLeakagePreventionConfig()
    )
    random_seed: int | None = None
    stratification_label_name: str | None = None


@dataclass(frozen=True, slots=True)
class RankingEvaluationConfig:
    """Config contract for ranking evaluation."""

    primary_metric: RankingMetric
    metrics: tuple[RankingMetric, ...]
    top_k_values: tuple[int, ...] = (5, 10, 20)
    evaluation_unit: str = "subject_id"
    include_baselines: tuple[BaselineName, ...] = DEFAULT_BASELINE_NAMES


@dataclass(frozen=True, slots=True)
class CalibrationEvaluationConfig:
    """Config contract for calibration evaluation."""

    primary_metric: CalibrationMetric
    metrics: tuple[CalibrationMetric, ...]
    n_bins: int = 10
    include_baselines: tuple[BaselineName, ...] = DEFAULT_BASELINE_NAMES


@dataclass(frozen=True, slots=True)
class BaselineComparisonConfig:
    """Explicit list of baselines to report in every experiment."""

    baselines: tuple[BaselineName, ...] = DEFAULT_BASELINE_NAMES
    require_current_rule_score: bool = True


@dataclass(frozen=True, slots=True)
class SampledClinicianReviewExportConfig:
    """Config contract for sampled clinician-review exports."""

    sample_size: int
    subject_id_field: str = "subject_id"
    encounter_id_field: str = "encounter_id"
    review_timestamp_field: str = "review_timestamp"
    medication_field: str = "medication_standardized"
    include_baselines: tuple[BaselineName, ...] = DEFAULT_BASELINE_NAMES
    include_ground_truth_labels_when_available: bool = True


class SplitGenerator(ABC):
    """Placeholder interface for future subject-level split generators."""

    @abstractmethod
    def generate_subject_partitions(
        self,
        subject_ids: Iterable[int],
        config: TrainValidationTestSplitConfig,
    ) -> Mapping[SplitPartition, set[int]]:
        """Generate train/validation/test subject partitions."""
        raise NotImplementedError


class RankingEvaluator(ABC):
    """Placeholder interface for future ranking evaluators."""

    @abstractmethod
    def evaluate(
        self,
        config: RankingEvaluationConfig,
    ) -> Mapping[str, float]:
        """Evaluate ranking outputs against the configured metrics."""
        raise NotImplementedError


class CalibrationEvaluator(ABC):
    """Placeholder interface for future calibration evaluators."""

    @abstractmethod
    def evaluate(
        self,
        config: CalibrationEvaluationConfig,
    ) -> Mapping[str, float]:
        """Evaluate probabilistic outputs against calibration metrics."""
        raise NotImplementedError


class ClinicianReviewExporter(ABC):
    """Placeholder interface for sampled clinician-review export generation."""

    @abstractmethod
    def export_sample(
        self,
        config: SampledClinicianReviewExportConfig,
    ) -> object:
        """Build a model-agnostic sampled export for clinician review."""
        raise NotImplementedError


def validate_split_config(config: TrainValidationTestSplitConfig) -> None:
    """Validate a train/validation/test split contract."""
    total = config.train_fraction + config.validation_fraction + config.test_fraction
    if round(total, 8) != 1.0:
        raise ValueError(
            "Split fractions must sum to 1.0 exactly within rounding tolerance."
        )
    if min(config.train_fraction, config.validation_fraction, config.test_fraction) <= 0:
        raise ValueError("All split fractions must be strictly positive.")
    if config.partition_names != (
        SplitPartition.TRAIN,
        SplitPartition.VALIDATION,
        SplitPartition.TEST,
    ):
        raise ValueError(
            "Partition names must remain the stable train/validation/test tuple."
        )
    if not config.subject_leakage_prevention.require_subject_level_grouping:
        raise ValueError("Subject-level grouping is required for experiment splits.")
    if not config.subject_leakage_prevention.forbid_subject_overlap_across_partitions:
        raise ValueError("Subject overlap prevention may not be disabled.")


def build_partition_subject_index(
    train_subject_ids: Iterable[int],
    validation_subject_ids: Iterable[int],
    test_subject_ids: Iterable[int],
) -> dict[SplitPartition, set[int]]:
    """Build a canonical subject index for synthetic split integrity checks."""
    return {
        SplitPartition.TRAIN: set(train_subject_ids),
        SplitPartition.VALIDATION: set(validation_subject_ids),
        SplitPartition.TEST: set(test_subject_ids),
    }


def assert_no_subject_leakage(
    partition_subject_ids: Mapping[SplitPartition, set[int]],
) -> None:
    """Assert that no subject appears in more than one partition."""
    train_subjects = set(partition_subject_ids.get(SplitPartition.TRAIN, set()))
    validation_subjects = set(
        partition_subject_ids.get(SplitPartition.VALIDATION, set())
    )
    test_subjects = set(partition_subject_ids.get(SplitPartition.TEST, set()))

    overlap_pairs = {
        "train_validation": train_subjects & validation_subjects,
        "train_test": train_subjects & test_subjects,
        "validation_test": validation_subjects & test_subjects,
    }
    leaked_subjects = sorted(
        subject_id
        for overlap in overlap_pairs.values()
        for subject_id in overlap
    )
    if leaked_subjects:
        raise ValueError(
            "Subject leakage detected across partitions for subject_ids="
            f"{leaked_subjects}."
        )
