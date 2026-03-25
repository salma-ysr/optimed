"""Feature extraction and feature-store contracts for OPTI-MED."""

from opti_med.features.contracts import (
    FEATURE_GROUP_SCHEMAS,
    FEATURE_STORE_DESIGN_PRINCIPLES,
    FEATURE_STORE_GRAIN_DESCRIPTION,
    FEATURE_STORE_PRIMARY_KEY,
    FeatureDerivationType,
    FeatureFamily,
    FeatureGroupSchema,
    FeatureSpec,
    validate_feature_group_schema,
    validate_feature_registry,
)
from opti_med.features.first_scope_feature_store import (
    FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION,
    FIRST_SCOPE_PATIENT_CONTEXT_CONTRACT_VERSION,
    FIRST_SCOPE_TEMPORAL_CONTRACT_VERSION,
    build_first_scope_feature_store,
    build_first_scope_feature_store_leakage_qc_report,
    build_first_scope_patient_context_features,
    build_first_scope_temporal_features,
    validate_first_scope_feature_store_artifact,
    validate_first_scope_patient_context_features_artifact,
    validate_first_scope_temporal_features_artifact,
)

__all__ = [
    "FEATURE_GROUP_SCHEMAS",
    "FEATURE_STORE_DESIGN_PRINCIPLES",
    "FEATURE_STORE_GRAIN_DESCRIPTION",
    "FEATURE_STORE_PRIMARY_KEY",
    "FeatureDerivationType",
    "FeatureFamily",
    "FeatureGroupSchema",
    "FeatureSpec",
    "validate_feature_group_schema",
    "validate_feature_registry",
    "FIRST_SCOPE_PATIENT_CONTEXT_CONTRACT_VERSION",
    "FIRST_SCOPE_TEMPORAL_CONTRACT_VERSION",
    "FIRST_SCOPE_FEATURE_STORE_CONTRACT_VERSION",
    "build_first_scope_patient_context_features",
    "build_first_scope_temporal_features",
    "build_first_scope_feature_store",
    "build_first_scope_feature_store_leakage_qc_report",
    "validate_first_scope_patient_context_features_artifact",
    "validate_first_scope_temporal_features_artifact",
    "validate_first_scope_feature_store_artifact",
]
