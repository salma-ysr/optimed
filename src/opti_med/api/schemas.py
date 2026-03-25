"""Response schemas for the OPTI-MED API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok"]
    scored_output_available: bool
    scored_output_path: str


class ScoredRow(BaseModel):
    """A single scored medication exposure row."""

    subject_id: int
    hadm_id: int
    sex: str
    age_proxy: int
    age_group: str
    admission_type: str
    admittime: str
    dischtime: str
    length_of_stay_days: float
    drug: str
    drug_normalized: str | None = None
    starttime: str
    stoptime: str | None = None
    medication_episode_id: str | None = None
    prescription_segment_count: int | None = None
    prescription_segments_json: list[dict[str, object]] | None = None
    current_medication_count: int | None = None
    current_peak_concurrent_medication_count: int | None = None
    current_polypharmacy_flag: int | None = None
    historical_medication_count: int | None = None
    historical_polypharmacy_flag: int | None = None
    total_medication_count: int
    peak_concurrent_medication_count: int | None = None
    polypharmacy_flag: int
    benzodiazepine_flag: int
    opioid_flag: int
    anticholinergic_flag: int
    ppi_flag: int
    antipsychotic_flag: int
    ckd_flag: int
    dementia_flag: int
    delirium_flag: int
    heart_failure_flag: int
    diabetes_flag: int
    creatinine_first: float | None = None
    creatinine_max: float | None = None
    creatinine_mean: float | None = None
    renal_risk_flag: int
    deprescribing_priority_score: int
    deprescribing_priority_label: str
    deprescribing_priority_summary_alert: str
    deprescribing_priority_explanation: str
    deprescribing_priority_bucket_scores_json: dict[str, int]
    deprescribing_priority_reasons_json: list[str]
    deprescribing_priority_evidence_json: dict[str, object]


class ScoredRowsResponse(BaseModel):
    """Paginated scored row response."""

    total_rows: int
    limit: int
    offset: int
    rows: list[ScoredRow]


class AdmissionSummary(BaseModel):
    """Admission-level summary used on the main dashboard."""

    subject_id: int
    hadm_id: int
    sex: str
    age_proxy: int
    age_group: str
    admission_type: str
    admittime: str
    dischtime: str
    length_of_stay_days: float
    current_medication_count: int | None = None
    current_polypharmacy_flag: int | None = None
    historical_medication_count: int | None = None
    total_medication_count: int
    peak_concurrent_medication_count: int | None = None
    polypharmacy_flag: int
    creatinine_max: float | None = None
    renal_risk_flag: int
    overall_priority_score: int
    overall_priority_label: str
    flagged_medication_count: int
    overall_priority_drivers: list[str]


class AdmissionSummariesResponse(BaseModel):
    """Paginated admission summary response."""

    total_admissions: int
    limit: int
    offset: int
    rows: list[AdmissionSummary]


class MedicationRowSummary(BaseModel):
    """Medication-level summary used on the admission details page."""

    drug: str
    drug_normalized: str | None = None
    medication_classes: list[str]
    starttime: str
    stoptime: str | None = None
    medication_episode_id: str | None = None
    prescription_segment_count: int | None = None
    prescription_segments_json: list[dict[str, object]] | None = None
    deprescribing_priority_score: int
    deprescribing_priority_label: str
    deprescribing_priority_summary_alert: str
    deprescribing_priority_explanation: str
    deprescribing_priority_bucket_scores_json: dict[str, int]
    deprescribing_priority_reasons_json: list[str]
    deprescribing_priority_evidence_json: dict[str, object]
    benzodiazepine_flag: int
    opioid_flag: int
    anticholinergic_flag: int
    ppi_flag: int
    antipsychotic_flag: int


class AdmissionDetailResponse(BaseModel):
    """Admission-level response for the details page."""

    subject_id: int
    hadm_id: int
    sex: str
    age_proxy: int
    age_group: str
    admission_type: str
    admittime: str
    dischtime: str
    length_of_stay_days: float
    current_medication_count: int | None = None
    current_polypharmacy_flag: int | None = None
    historical_medication_count: int | None = None
    total_medication_count: int
    peak_concurrent_medication_count: int | None = None
    polypharmacy_flag: int
    ckd_flag: int
    dementia_flag: int
    delirium_flag: int
    heart_failure_flag: int
    diabetes_flag: int
    creatinine_first: float | None = None
    creatinine_max: float | None = None
    creatinine_mean: float | None = None
    renal_risk_flag: int
    highest_priority_score: int
    highest_priority_label: str
    flagged_medication_count: int
    score_explanations: list[str]
    overall_priority_drivers: list[str]
    flagged_medications: list[MedicationRowSummary]
    medications: list[MedicationRowSummary]


class ScoredOutputSummary(BaseModel):
    """Summary of the latest saved scored output."""

    path: str
    row_count: int
    column_count: int
    unique_subjects: int
    unique_admissions: int
    last_modified: str = Field(description="UTC ISO timestamp for the scored output file.")


class RefreshScoresResponse(BaseModel):
    """Response returned after rebuilding the scored output."""

    message: str
    output: ScoredOutputSummary


class PatientSummary(BaseModel):
    """Patient-level rollup for patient-centered list views."""

    subject_id: int
    sex: str
    age_proxy: int
    age_group: str
    current_medication_count: int | None = None
    historical_medication_count: int | None = None
    encounter_count: int
    medication_count: int
    flagged_medication_count: int
    highest_priority_score: int
    highest_priority_label: str
    top_problem_flashes: list[str]
    discharge_imminent_review_flash: str | None = None
    discharge_review_flash: str | None = None
    imminent_review_flash: str | None = None
    latest_hadm_id: int | None = None
    latest_admission_type: str | None = None
    latest_dischtime: str | None = None


class PatientSummariesResponse(BaseModel):
    """Paginated patient summary response."""

    total_patients: int
    limit: int
    offset: int
    rows: list[PatientSummary]


class PatientEncounterSummary(BaseModel):
    """Encounter-level summary nested under a patient dossier."""

    subject_id: int
    hadm_id: int
    admission_type: str
    admittime: str
    dischtime: str
    length_of_stay_days: float
    overall_priority_score: int
    overall_priority_label: str
    current_medication_count: int | None = None
    current_polypharmacy_flag: int | None = None
    historical_medication_count: int | None = None
    total_medication_count: int
    peak_concurrent_medication_count: int | None = None
    flagged_medication_count: int
    overall_priority_drivers: list[str]


class DossierEncounterSelection(BaseModel):
    """Selected encounter and review-time metadata for the patient dossier."""

    encounter_id: str
    hadm_id: int | None = None
    stay_id: int | None = None
    encounter_source: str
    selection_mode: str
    requested_hadm_id: int | None = None
    review_timestamp: str | None = None
    review_timestamp_source: str | None = None


class PatientEncountersResponse(BaseModel):
    """Patient encounter list response."""

    subject_id: int
    total_encounters: int
    rows: list[PatientEncounterSummary]


class PatientContextObject(BaseModel):
    """Explicit left-column patient context object for future patient-first UI work."""

    sex: str
    age_proxy: int
    age_group: str
    current_medication_count: int | None = None
    historical_medication_count: int | None = None
    encounter_count: int
    medication_count: int
    peak_concurrent_medication_count: int | None = None
    flagged_medication_count: int
    polypharmacy_present: bool
    renal_risk_present: bool
    ckd_present: bool
    dementia_present: bool
    delirium_present: bool
    heart_failure_present: bool
    diabetes_present: bool
    latest_creatinine_max: float | None = None
    latest_egfr_ml_min_1_73m2: float | None = None
    latest_weight_kg: float | None = None
    latest_bmi: float | None = None
    frailty_present: bool | None = None
    hepatic_signal_summary: str | None = None
    baseline_potassium_summary: str | None = None
    baseline_sodium_summary: str | None = None
    vigilance_summary: str | None = None
    rass_summary: str | None = None
    pain_summary: str | None = None
    hemodynamic_stability_summary: str | None = None
    level_of_care: str | None = None
    dysphagia_present: bool | None = None
    feeding_route_summary: str | None = None
    administration_constraints_summary: str | None = None
    latest_weight_provenance: str | None = None
    latest_bmi_provenance: str | None = None
    latest_renal_provenance: str | None = None


class ProblemFlash(BaseModel):
    """Short clinician-readable top problem flash for a patient dossier."""

    key: str
    label: str
    severity: str
    reason: str


class ClinicianReviewRecord(BaseModel):
    """Latest clinician review state for one analytical-grain medication row."""

    subject_id: int
    encounter_id: str
    hadm_id: int | None = None
    stay_id: int | None = None
    medication_standardized: str
    medication_normalized: str | None = None
    review_timestamp: str
    modeling__row_id: str | None = None
    review_submission_id: str
    review_version: int
    review_artifact_version: str
    review_submission_timestamp: str
    reviewer_id: str
    label__clinician_priority_level: Literal["low", "medium", "high"]
    label__clinician_priority_score: float | None = None
    label__clinician_priority_score_level: Literal["low", "medium", "high"] | None = None
    label__clinician_review_status: Literal[
        "reviewed",
        "uncertain",
        "insufficient_context",
        "skip",
    ]
    label__clinician_reason_tags: list[str] = Field(default_factory=list)
    label__clinician_note: str | None = None
    label__clinician_reviewed_flag: int
    label__clinician_suggested_action: Literal[
        "keep",
        "monitor",
        "reconsider",
        "deprescribe_candidate",
        "needs_more_info",
    ] | None = None
    review_provenance_json: dict[str, object] | None = None


class ClinicianReviewSubmissionRequest(BaseModel):
    """Clinician review submission payload posted from the dossier UI."""

    subject_id: int
    encounter_id: str
    hadm_id: int | None = None
    stay_id: int | None = None
    medication_standardized: str
    review_timestamp: str
    modeling__row_id: str | None = None
    reviewer_id: str | None = None
    label__clinician_priority_level: Literal["low", "medium", "high"]
    label__clinician_priority_score: float | None = Field(default=None, ge=0, le=10)
    label__clinician_review_status: Literal[
        "reviewed",
        "uncertain",
        "insufficient_context",
        "skip",
    ]
    label__clinician_reason_tags: list[str] = Field(default_factory=list)
    label__clinician_note: str | None = None
    label__clinician_suggested_action: Literal[
        "keep",
        "monitor",
        "reconsider",
        "deprescribe_candidate",
        "needs_more_info",
    ] | None = None


class ReviewQueueSummary(BaseModel):
    """Simple review-queue counts for one patient dossier."""

    reviewable_rows: int = 0
    reviewed_rows: int = 0
    unreviewed_rows: int = 0
    priority_rows: int = 0
    disagreement_candidate_rows: int = 0


class ClinicianReviewWorkflowReport(BaseModel):
    """Compact structured report for the Phase 5 review workflow."""

    contract_version: str
    generated_at: str
    phase_scope_statement: str
    artifact_paths: dict[str, str]
    reviewable_row_count: int
    clinician_reviewed_row_count: int
    required_level_populated_count: int
    numeric_score_populated_count: int
    reason_tag_row_coverage_count: int
    priority_level_frequencies: dict[str, int]
    reason_tag_frequencies: dict[str, int]
    review_status_frequencies: dict[str, int]
    traceability_validation: dict[str, int]


class ClinicianReviewSubmissionResponse(BaseModel):
    """Response returned after saving a clinician review."""

    review: ClinicianReviewRecord
    workflow_report: ClinicianReviewWorkflowReport


class PatientMedicationCard(BaseModel):
    """Ranked patient-centered medication card."""

    subject_id: int
    encounter_id: str | None = None
    hadm_id: int
    stay_id: int | None = None
    admission_type: str
    admittime: str
    dischtime: str
    length_of_stay_days: float
    drug: str
    drug_normalized: str | None = None
    medication_classes: list[str]
    starttime: str
    stoptime: str | None = None
    medication_episode_id: str | None = None
    prescription_segment_count: int | None = None
    prescription_segments_json: list[dict[str, object]] | None = None
    deprescribing_priority_score: int
    deprescribing_priority_label: str
    deprescribing_priority_summary_alert: str
    deprescribing_priority_explanation: str
    deprescribing_priority_bucket_scores_json: dict[str, int]
    deprescribing_priority_reasons_json: list[str]
    deprescribing_priority_evidence_json: dict[str, object]
    benzodiazepine_flag: int
    opioid_flag: int
    anticholinergic_flag: int
    ppi_flag: int
    antipsychotic_flag: int
    status: str | None = None
    medication_status: str | None = None
    last_active_time: str | None = None
    review_timestamp: str | None = None
    review_timestamp_source: str | None = None
    priority_score_source: str | None = None
    medication_standardized: str | None = None
    modeling__row_id: str | None = None
    first_scope_supported_class_flag: int | None = None
    benchmark__current_rule_score: float | None = None
    benchmark__current_rule_score_level: Literal["low", "medium", "high"] | None = None
    benchmark__current_rule_available_flag: int | None = None
    benchmark__medication_class_only_medication_class_standardized: str | None = None
    reviewable_flag: bool = False
    review_queue_priority: str | None = None
    review_queue_reasons: list[str] = Field(default_factory=list)
    clinician_review: ClinicianReviewRecord | None = None
    # Reserved for future model output. These stay null until a separate ML layer is wired in.
    ml_priority_score: float | None = None
    ml_priority_rank_within_encounter: int | None = None
    ml_top_drivers: list[str] | None = None
    ml_driver_raw_values: dict[str, object] | None = None
    guidance_summary: str | None = None


class PatientMedicationsResponse(BaseModel):
    """Patient medication card list response."""

    subject_id: int
    total_medications: int
    rows: list[PatientMedicationCard]


class PatientDetailResponse(BaseModel):
    """Patient-centered dossier response."""

    patient_summary: PatientSummary
    encounter_summaries: list[PatientEncounterSummary]
    selected_encounter: DossierEncounterSelection | None = None
    review_timestamp: str | None = None
    current_medication_count: int = 0
    current_flagged_medication_count: int = 0
    historical_medication_count: int = 0
    left_column_context: PatientContextObject
    current_medications: list[PatientMedicationCard] = Field(default_factory=list)
    previous_medication_history: list[PatientMedicationCard] = Field(default_factory=list)
    ranked_medication_cards: list[PatientMedicationCard]
    historical_medication_cards: list[PatientMedicationCard] = Field(default_factory=list)
    top_problem_flashes: list[ProblemFlash]
    flagged_medication_count: int
    medication_card_count: int
    review_queue_summary: ReviewQueueSummary | None = None
    # These right-rail sections are explicit placeholders today. The current backend does not
    # derive them from saved artifacts or model output yet.
    time_to_benefit_summary: str | None = None
    time_to_benefit_note: str | None = None
    taper_protocol_steps: list[str] | None = None
    taper_protocol_summary: str | None = None
    peer_validation_references: list[str] | None = None
    peer_validation_summary: str | None = None
    patient_perceived_symptoms: list[str] | None = None
    patient_symptom_summary: str | None = None
