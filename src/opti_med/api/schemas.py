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
    encounter_count: int
    medication_count: int
    flagged_medication_count: int
    highest_priority_score: int
    highest_priority_label: str
    top_problem_flashes: list[str]


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
    total_medication_count: int
    peak_concurrent_medication_count: int | None = None
    flagged_medication_count: int
    overall_priority_drivers: list[str]


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


class ProblemFlash(BaseModel):
    """Short clinician-readable top problem flash for a patient dossier."""

    key: str
    label: str
    severity: str
    reason: str


class PatientMedicationCard(BaseModel):
    """Ranked patient-centered medication card."""

    subject_id: int
    hadm_id: int
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


class PatientMedicationsResponse(BaseModel):
    """Patient medication card list response."""

    subject_id: int
    total_medications: int
    rows: list[PatientMedicationCard]


class PatientDetailResponse(BaseModel):
    """Patient-centered dossier response."""

    patient_summary: PatientSummary
    encounter_summaries: list[PatientEncounterSummary]
    left_column_context: PatientContextObject
    ranked_medication_cards: list[PatientMedicationCard]
    top_problem_flashes: list[ProblemFlash]
    flagged_medication_count: int
    medication_card_count: int
