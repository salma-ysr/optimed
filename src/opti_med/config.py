"""Configuration helpers for OPTI-MED data loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_EXTERNAL_DATA_ROOT = Path("data/external")
DEFAULT_CLINICAL_DATA_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "mimic-iv-clinical-database-demo-2.2"
DEFAULT_ED_DATA_ROOT = DEFAULT_EXTERNAL_DATA_ROOT / "mimic-iv-ed-demo-2.2"
DEFAULT_STANDARDIZED_ROOT = Path("data/standardized")
DEFAULT_ANALYTICAL_ROOT = Path("data/analytical")
DEFAULT_FEATURE_STORE_ROOT = Path("data/feature_store")
DEFAULT_LABEL_ROOT = Path("data/labels")
DEFAULT_MODELING_ROOT = Path("data/modeling")
DEFAULT_MANIFEST_ROOT = DEFAULT_STANDARDIZED_ROOT / "manifests"
DEFAULT_STANDARDIZED_FILE_EXTENSION = ".parquet"
DEFAULT_ENCOUNTER_QC_REPORT_PATH = Path("docs/ml_pivot/11_encounter_and_medication_events_qc.md")
DEFAULT_FIRST_SCOPE_QC_REPORT_PATH = Path(
    "docs/ml_pivot/14_first_scope_semantics_and_burden_65plus_qc.md"
)
DEFAULT_FEATURE_STORE_LEAKAGE_QC_REPORT_PATH = Path(
    "docs/ml_pivot/15a_feature_store_leakage_qc_65plus_first_scope.md"
)
DEFAULT_FIRST_SCOPE_LABEL_QC_REPORT_PATH = Path(
    "docs/ml_pivot/16a_labels_v1_65plus_first_scope_qc.md"
)
DEFAULT_FIRST_SCOPE_DATASET_QC_REPORT_PATH = Path(
    "docs/ml_pivot/17a_dataset_v1_65plus_first_scope_qc.md"
)
DEFAULT_FIRST_SCOPE_ORDINAL_TARGET_QC_REPORT_PATH = Path(
    "docs/ml_pivot/20_phase7_ordinal_target_preparation_65plus_first_scope.md"
)
DEFAULT_PHASE5_CLINICIAN_REVIEW_QC_REPORT_PATH = Path(
    "docs/ml_pivot/18_phase5_clinician_review_workflow_qc.md"
)
DEFAULT_PHASE6_CLINICIAN_REVIEW_DENSIFICATION_QC_REPORT_PATH = Path(
    "docs/ml_pivot/19_phase6_clinician_review_densification_qc.md"
)
DEFAULT_TARGETED_BLIND_EVAL_QC_REPORT_PATH = Path(
    "docs/ml_pivot/22_targeted_blind_clinician_evaluation_slice_65plus_first_scope.md"
)
DEFAULT_INGESTION_BEHAVIOR = "incremental"
DEFAULT_INGESTION_CHUNK_SIZE = 100_000
DEFAULT_SERUM_CREATININE_ITEMIDS = (50912, 51081, 51977, 52546)
DEFAULT_SNAPSHOT_STRATEGY = "latest_available"
DEFAULT_CLINICIAN_REVIEWER_ID = "pharmacist_demo_local"


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
    raw_clinical_data_root: Path | None = None
    raw_ed_data_root: Path | None = None
    standardized_root: Path = DEFAULT_STANDARDIZED_ROOT
    analytical_root: Path = DEFAULT_ANALYTICAL_ROOT
    feature_store_root: Path = DEFAULT_FEATURE_STORE_ROOT
    label_root: Path = DEFAULT_LABEL_ROOT
    modeling_root: Path = DEFAULT_MODELING_ROOT
    manifest_root: Path = DEFAULT_MANIFEST_ROOT
    standardized_file_extension: str = DEFAULT_STANDARDIZED_FILE_EXTENSION
    encounter_medication_qc_report_path: Path = DEFAULT_ENCOUNTER_QC_REPORT_PATH
    clinician_review_qc_report_output_path: Path = DEFAULT_PHASE5_CLINICIAN_REVIEW_QC_REPORT_PATH
    clinician_review_phase6_qc_report_output_path: Path = (
        DEFAULT_PHASE6_CLINICIAN_REVIEW_DENSIFICATION_QC_REPORT_PATH
    )
    first_scope_ordinal_target_qc_report_output_path: Path = (
        DEFAULT_FIRST_SCOPE_ORDINAL_TARGET_QC_REPORT_PATH
    )
    targeted_blind_eval_qc_report_output_path: Path = (
        DEFAULT_TARGETED_BLIND_EVAL_QC_REPORT_PATH
    )
    ingestion_behavior: str = DEFAULT_INGESTION_BEHAVIOR
    ingestion_chunk_size: int = DEFAULT_INGESTION_CHUNK_SIZE
    older_adult_age_threshold: int = 65
    polypharmacy_threshold: int = 5
    renal_risk_creatinine_threshold: float = 1.5
    serum_creatinine_itemids: tuple[int, ...] = DEFAULT_SERUM_CREATININE_ITEMIDS
    snapshot_strategy: str = DEFAULT_SNAPSHOT_STRATEGY
    default_clinician_reviewer_id: str = DEFAULT_CLINICIAN_REVIEWER_ID

    @property
    def hosp_root(self) -> Path:
        """Return the configured hospital tables directory for the clinical demo."""
        return self.clinical_data_root / self.hosp_dir_name

    @property
    def ed_root(self) -> Path:
        """Return the configured emergency department tables directory."""
        return self.ed_data_root / self.ed_dir_name

    @property
    def configured_raw_clinical_data_root(self) -> Path:
        """Return the configured raw clinical root for full-data ingestion."""
        return self.raw_clinical_data_root or self.clinical_data_root

    @property
    def configured_raw_ed_data_root(self) -> Path:
        """Return the configured raw ED root for full-data ingestion."""
        return self.raw_ed_data_root or self.ed_data_root

    @property
    def standardized_clinical_root(self) -> Path:
        """Return the standardized clinical artifact directory."""
        return self.standardized_root / "clinical"

    @property
    def standardized_ed_root(self) -> Path:
        """Return the standardized ED artifact directory."""
        return self.standardized_root / "ed"

    @property
    def standardized_manifest_path(self) -> Path:
        """Return the latest standardized ingestion manifest path."""
        return self.manifest_root / "latest.json"

    @property
    def encounter_medication_state_output_path(self) -> Path:
        """Return the default persisted encounter-medication-state artifact path."""
        return self.analytical_root / "encounter_medication_state_65plus.parquet"

    @property
    def older_adult_eligibility_output_path(self) -> Path:
        """Return the default persisted 65+ encounter-eligibility artifact path."""
        return self.analytical_root / "eligible_encounters_65plus.parquet"

    @property
    def medication_rxnorm_mapping_output_path(self) -> Path:
        """Return the default persisted RxNorm medication mapping artifact path."""
        return self.analytical_root / "medication_rxnorm_mapping_65plus.parquet"

    @property
    def encounter_medication_state_rxnorm_output_path(self) -> Path:
        """Return the default persisted RxNorm-enriched 65+ state artifact path."""
        return self.analytical_root / "encounter_medication_state_65plus_rxnorm.parquet"

    @property
    def encounter_medication_semantics_output_path(self) -> Path:
        """Return the default persisted encounter-medication-semantics artifact path."""
        return self.analytical_root / "encounter_medication_semantics_65plus.parquet"

    @property
    def encounter_medication_burden_output_path(self) -> Path:
        """Return the default persisted encounter-medication-burden artifact path."""
        return self.analytical_root / "encounter_medication_burden_65plus.parquet"

    @property
    def encounter_medication_first_scope_output_path(self) -> Path:
        """Return the default persisted first-scope 65+ encounter-medication artifact path."""
        return self.analytical_root / "encounter_medication_first_scope_65plus.parquet"

    @property
    def encounter_medication_first_scope_qc_report_path(self) -> Path:
        """Return the default QC report path for the first-scope 65+ artifacts."""
        return DEFAULT_FIRST_SCOPE_QC_REPORT_PATH

    @property
    def cohort_output_path(self) -> Path:
        """Return the default interim cohort output path."""
        return self.interim_root / "older_adult_medication_cohort.csv"

    @property
    def first_scope_patient_context_features_output_path(self) -> Path:
        """Return the default persisted first-scope patient-context stage path."""
        return self.feature_store_root / "first_scope_patient_context_features_65plus.parquet"

    @property
    def first_scope_temporal_features_output_path(self) -> Path:
        """Return the default persisted first-scope temporal stage path."""
        return self.feature_store_root / "first_scope_temporal_features_65plus.parquet"

    @property
    def first_scope_feature_store_output_path(self) -> Path:
        """Return the default persisted first-scope feature-store v1 path."""
        return self.feature_store_root / "encounter_medication_features_v1_65plus_first_scope.parquet"

    @property
    def first_scope_feature_store_metadata_output_path(self) -> Path:
        """Return the default persisted feature dictionary / metadata artifact path."""
        return (
            self.feature_store_root
            / "encounter_medication_features_v1_65plus_first_scope_metadata.json"
        )

    @property
    def first_scope_feature_store_leakage_qc_report_path(self) -> Path:
        """Return the default leakage QC report path for first-scope feature-store v1."""
        return DEFAULT_FEATURE_STORE_LEAKAGE_QC_REPORT_PATH

    @property
    def first_scope_labels_output_path(self) -> Path:
        """Return the default persisted first-scope Labels v1 artifact path."""
        return self.label_root / "encounter_medication_labels_v1_65plus_first_scope.parquet"

    @property
    def first_scope_label_qc_report_path(self) -> Path:
        """Return the default QC report path for first-scope Labels v1."""
        return DEFAULT_FIRST_SCOPE_LABEL_QC_REPORT_PATH

    @property
    def first_scope_dataset_output_path(self) -> Path:
        """Return the default persisted Dataset v1 path for the 65+ first-scope subset."""
        return self.modeling_root / "encounter_medication_dataset_v1_65plus_first_scope.parquet"

    @property
    def first_scope_splits_output_path(self) -> Path:
        """Return the default persisted split assignment path for Dataset v1."""
        return self.modeling_root / "splits_v1_65plus_first_scope.parquet"

    @property
    def first_scope_feature_list_output_path(self) -> Path:
        """Return the default persisted feature-list JSON path for Dataset v1."""
        return self.modeling_root / "feature_list_v1_65plus_first_scope.json"

    @property
    def first_scope_dataset_qc_report_path(self) -> Path:
        """Return the default QC report path for Dataset v1 and its splits."""
        return DEFAULT_FIRST_SCOPE_DATASET_QC_REPORT_PATH

    @property
    def clinician_review_event_log_path(self) -> Path:
        """Return the append-only Phase 5 clinician review event log path."""
        return self.label_root / "clinician_review_events_v1_phase5.jsonl"

    @property
    def clinician_review_snapshot_output_path(self) -> Path:
        """Return the latest-active Phase 5 clinician review snapshot path."""
        return self.label_root / "clinician_review_labels_v1_phase5.parquet"

    @property
    def clinician_review_qc_summary_path(self) -> Path:
        """Return the machine-readable Phase 5 clinician review QC summary path."""
        return self.label_root / "clinician_review_workflow_qc_v1_phase5.json"

    @property
    def clinician_review_qc_report_path(self) -> Path:
        """Return the markdown QC report path for the Phase 5 workflow."""
        return self.clinician_review_qc_report_output_path

    @property
    def clinician_review_phase6_queue_output_path(self) -> Path:
        """Return the Phase 6 deterministic review queue artifact path."""
        return self.label_root / "clinician_review_queue_v1_phase6.parquet"

    @property
    def clinician_review_phase6_universe_output_path(self) -> Path:
        """Return the Phase 6 reviewable-universe artifact path."""
        return self.label_root / "clinician_review_universe_v1_phase6.parquet"

    @property
    def clinician_review_phase6_qc_summary_path(self) -> Path:
        """Return the machine-readable Phase 6 densification QC summary path."""
        return self.label_root / "clinician_review_densification_qc_v1_phase6.json"

    @property
    def clinician_review_phase6_qc_report_path(self) -> Path:
        """Return the markdown QC report path for the Phase 6 densification workflow."""
        return self.clinician_review_phase6_qc_report_output_path

    @property
    def first_scope_ordinal_targets_output_path(self) -> Path:
        """Return the persisted ordinal-target sidecar artifact path."""
        return self.modeling_root / "encounter_medication_ordinal_targets_v1_65plus_first_scope.parquet"

    @property
    def first_scope_ordinal_targets_summary_path(self) -> Path:
        """Return the machine-readable ordinal-target QC summary path."""
        return self.modeling_root / "encounter_medication_ordinal_targets_v1_65plus_first_scope_summary.json"

    @property
    def first_scope_ordinal_target_qc_report_path(self) -> Path:
        """Return the markdown QC report path for ordinal target preparation."""
        return self.first_scope_ordinal_target_qc_report_output_path

    @property
    def first_scope_ordinal_scored_universe_output_path(self) -> Path:
        """Return the persisted full-universe ordinal model prediction artifact path."""
        return self.modeling_root / "encounter_medication_ordinal_scored_universe_v1_65plus_first_scope.parquet"

    @property
    def targeted_blind_eval_slice_output_path(self) -> Path:
        """Return the targeted blind clinician evaluation slice artifact path."""
        return self.label_root / "clinician_review_targeted_blind_eval_slice_v1_65plus_first_scope.parquet"

    @property
    def targeted_blind_eval_summary_path(self) -> Path:
        """Return the machine-readable targeted blind evaluation summary path."""
        return self.label_root / "clinician_review_targeted_blind_eval_slice_v1_65plus_first_scope_summary.json"

    @property
    def targeted_blind_eval_qc_report_path(self) -> Path:
        """Return the markdown QC report path for the targeted blind evaluation slice."""
        return self.targeted_blind_eval_qc_report_output_path

    @property
    def encounter_index_output_path(self) -> Path:
        """Return the default persisted encounter-index artifact path."""
        return self.standardized_root / "encounter_index.parquet"

    @property
    def medication_events_output_path(self) -> Path:
        """Return the default persisted medication-events artifact path."""
        return self.standardized_root / "medication_events.parquet"

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
        raw_clinical_root_raw = os.getenv("OPTI_MED_RAW_CLINICAL_DATA_ROOT")
        raw_ed_root_raw = os.getenv("OPTI_MED_RAW_ED_DATA_ROOT")
        standardized_root = Path(
            os.getenv("OPTI_MED_STANDARDIZED_ROOT", str(DEFAULT_STANDARDIZED_ROOT))
        )
        analytical_root = Path(
            os.getenv("OPTI_MED_ANALYTICAL_ROOT", str(DEFAULT_ANALYTICAL_ROOT))
        )
        feature_store_root = Path(
            os.getenv("OPTI_MED_FEATURE_STORE_ROOT", str(DEFAULT_FEATURE_STORE_ROOT))
        )
        label_root = Path(os.getenv("OPTI_MED_LABEL_ROOT", str(DEFAULT_LABEL_ROOT)))
        modeling_root = Path(os.getenv("OPTI_MED_MODELING_ROOT", str(DEFAULT_MODELING_ROOT)))
        manifest_root = Path(
            os.getenv("OPTI_MED_MANIFEST_ROOT", str(DEFAULT_MANIFEST_ROOT))
        )
        standardized_file_extension = os.getenv(
            "OPTI_MED_STANDARDIZED_FILE_EXTENSION",
            DEFAULT_STANDARDIZED_FILE_EXTENSION,
        )
        encounter_medication_qc_report_path = Path(
            os.getenv(
                "OPTI_MED_ENCOUNTER_MEDICATION_QC_REPORT_PATH",
                str(DEFAULT_ENCOUNTER_QC_REPORT_PATH),
            )
        )
        clinician_review_qc_report_output_path = Path(
            os.getenv(
                "OPTI_MED_CLINICIAN_REVIEW_QC_REPORT_PATH",
                str(DEFAULT_PHASE5_CLINICIAN_REVIEW_QC_REPORT_PATH),
            )
        )
        clinician_review_phase6_qc_report_output_path = Path(
            os.getenv(
                "OPTI_MED_CLINICIAN_REVIEW_PHASE6_QC_REPORT_PATH",
                str(DEFAULT_PHASE6_CLINICIAN_REVIEW_DENSIFICATION_QC_REPORT_PATH),
            )
        )
        first_scope_ordinal_target_qc_report_output_path = Path(
            os.getenv(
                "OPTI_MED_FIRST_SCOPE_ORDINAL_TARGET_QC_REPORT_PATH",
                str(DEFAULT_FIRST_SCOPE_ORDINAL_TARGET_QC_REPORT_PATH),
            )
        )
        targeted_blind_eval_qc_report_output_path = Path(
            os.getenv(
                "OPTI_MED_TARGETED_BLIND_EVAL_QC_REPORT_PATH",
                str(DEFAULT_TARGETED_BLIND_EVAL_QC_REPORT_PATH),
            )
        )
        ingestion_behavior = os.getenv(
            "OPTI_MED_INGESTION_BEHAVIOR",
            DEFAULT_INGESTION_BEHAVIOR,
        )
        ingestion_chunk_size = int(
            os.getenv(
                "OPTI_MED_INGESTION_CHUNK_SIZE",
                str(DEFAULT_INGESTION_CHUNK_SIZE),
            )
        )
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
        default_clinician_reviewer_id = os.getenv(
            "OPTI_MED_DEFAULT_CLINICIAN_REVIEWER_ID",
            DEFAULT_CLINICIAN_REVIEWER_ID,
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
            raw_clinical_data_root=Path(raw_clinical_root_raw) if raw_clinical_root_raw else None,
            raw_ed_data_root=Path(raw_ed_root_raw) if raw_ed_root_raw else None,
            standardized_root=standardized_root,
            analytical_root=analytical_root,
            feature_store_root=feature_store_root,
            label_root=label_root,
            modeling_root=modeling_root,
            manifest_root=manifest_root,
            standardized_file_extension=standardized_file_extension,
            encounter_medication_qc_report_path=encounter_medication_qc_report_path,
            clinician_review_qc_report_output_path=clinician_review_qc_report_output_path,
            clinician_review_phase6_qc_report_output_path=(
                clinician_review_phase6_qc_report_output_path
            ),
            first_scope_ordinal_target_qc_report_output_path=(
                first_scope_ordinal_target_qc_report_output_path
            ),
            targeted_blind_eval_qc_report_output_path=(
                targeted_blind_eval_qc_report_output_path
            ),
            ingestion_behavior=ingestion_behavior,
            ingestion_chunk_size=ingestion_chunk_size,
            older_adult_age_threshold=older_adult_age_threshold,
            polypharmacy_threshold=polypharmacy_threshold,
            renal_risk_creatinine_threshold=renal_risk_creatinine_threshold,
            serum_creatinine_itemids=serum_creatinine_itemids,
            snapshot_strategy=snapshot_strategy,
            default_clinician_reviewer_id=default_clinician_reviewer_id,
        )
