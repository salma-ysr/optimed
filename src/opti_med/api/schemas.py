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
    starttime: str
    stoptime: str | None = None
    total_medication_count: int
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
    deprescribing_priority_explanation: str


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
    medication_classes: list[str]
    starttime: str
    stoptime: str | None = None
    deprescribing_priority_score: int
    deprescribing_priority_label: str
    deprescribing_priority_explanation: str
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
