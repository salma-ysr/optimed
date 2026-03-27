export type RiskLabel = "low" | "medium" | "high";
export type ClinicianReviewStatus = "reviewed" | "uncertain" | "insufficient_context" | "skip";
export type ClinicianSuggestedAction =
  | "keep"
  | "monitor"
  | "reconsider"
  | "deprescribe_candidate"
  | "needs_more_info";

export interface ClinicianReviewRecord {
  subject_id: number;
  encounter_id: string;
  hadm_id?: number | null;
  stay_id?: number | null;
  medication_standardized: string;
  medication_normalized?: string | null;
  review_timestamp: string;
  modeling__row_id?: string | null;
  review_submission_id: string;
  review_version: number;
  review_artifact_version: string;
  review_submission_timestamp: string;
  reviewer_id: string;
  label__clinician_priority_level: RiskLabel;
  label__clinician_priority_score?: number | null;
  label__clinician_priority_score_level?: RiskLabel | null;
  label__clinician_review_status: ClinicianReviewStatus;
  label__clinician_reason_tags: string[];
  label__clinician_note?: string | null;
  label__clinician_reviewed_flag: number;
  label__clinician_suggested_action?: ClinicianSuggestedAction | null;
  review_provenance_json?: Record<string, unknown> | null;
}

export interface ReviewQueueSummary {
  reviewable_rows: number;
  reviewed_rows: number;
  unreviewed_rows: number;
  priority_rows: number;
  disagreement_candidate_rows: number;
}

export interface ClinicianReviewWorkflowReport {
  contract_version: string;
  generated_at: string;
  phase_scope_statement: string;
  artifact_paths: Record<string, string>;
  queue_generation_logic: string[];
  reviewable_row_count: number;
  clinician_reviewed_row_count: number;
  remaining_unreviewed_row_count: number;
  clinician_review_coverage_rate: number;
  unlabeled_reviewable_rate: number;
  required_level_populated_count: number;
  numeric_score_populated_count: number;
  reason_tag_row_coverage_count: number;
  reviewable_distinct_subject_count: number;
  reviewable_distinct_encounter_count: number;
  clinician_reviewed_distinct_subject_count: number;
  clinician_reviewed_distinct_encounter_count: number;
  clinician_reviewed_medication_class_count: number;
  reviewable_rows_with_modeling_row_id_count: number;
  reviewable_rows_with_modeling_row_id_rate: number;
  benchmark_available_reviewable_row_count: number;
  benchmark_available_reviewable_row_rate: number;
  priority_level_frequencies: Record<string, number>;
  reason_tag_frequencies: Record<string, number>;
  review_status_frequencies: Record<string, number>;
  reviewable_rows_by_medication_class: Record<string, number>;
  clinician_reviewed_rows_by_medication_class: Record<string, number>;
  clinician_reviewed_rows_by_patient: Record<string, number>;
  queue_priority_reason_frequencies: Record<string, number>;
  label_balance_assessment: Record<string, unknown>;
  milestone_status: Record<string, Record<string, unknown>>;
  traceability_validation: Record<string, number>;
  top_queue_preview: Array<Record<string, unknown>>;
}

export interface ClinicianReviewSubmissionRequest {
  subject_id: number;
  encounter_id: string;
  hadm_id?: number | null;
  stay_id?: number | null;
  medication_standardized: string;
  review_timestamp: string;
  modeling__row_id?: string | null;
  reviewer_id?: string | null;
  review_submission_source?: string | null;
  label__clinician_priority_level: RiskLabel;
  label__clinician_priority_score?: number | null;
  label__clinician_review_status: ClinicianReviewStatus;
  label__clinician_reason_tags: string[];
  label__clinician_note?: string | null;
  label__clinician_suggested_action?: ClinicianSuggestedAction | null;
}

export interface ClinicianReviewSubmissionResponse {
  review: ClinicianReviewRecord;
  workflow_report: ClinicianReviewWorkflowReport;
}

export interface ClinicianReviewQueueEntry {
  subject_id: number;
  encounter_id: string;
  hadm_id?: number | null;
  stay_id?: number | null;
  review_timestamp: string;
  medication_standardized: string;
  medication_normalized?: string | null;
  medication_class_standardized?: string | null;
  first_scope_supported_class_flag?: number | null;
  medication_status_at_review?: string | null;
  active_at_review_flag?: number | null;
  dose_value?: string | null;
  dose_unit?: string | null;
  route?: string | null;
  frequency?: string | null;
  modeling__row_id?: string | null;
  benchmark__current_rule_score?: number | null;
  benchmark__current_rule_score_level?: RiskLabel | null;
  benchmark__current_rule_available_flag?: number | null;
  label__primary_action_label?: string | null;
  label__unknown_or_insufficient_evidence_flag?: number | null;
  meta__dataset_row_eligible_for_training_flag?: number | null;
  reviewable_flag: boolean;
  current_clinician_reviewed_flag: number;
  queue_rank: number;
  queue_priority_score: number;
  queue_priority_band: string;
  queue_priority_reasons: string[];
  needs_review_justification: string;
  class_reviewed_count: number;
  class_reviewable_count: number;
  subject_reviewed_count: number;
  subject_reviewable_count: number;
  encounter_reviewed_count: number;
  encounter_reviewable_count: number;
  clinician_review?: ClinicianReviewRecord | null;
}

export interface ClinicianReviewQueueResponse {
  generated_at: string;
  total_queue_rows: number;
  filtered_queue_rows: number;
  filters_applied: Record<string, unknown>;
  workflow_report: ClinicianReviewWorkflowReport;
  rows: ClinicianReviewQueueEntry[];
}

export interface ScoredRow {
  subject_id: number;
  hadm_id: number;
  sex: string;
  age_proxy: number;
  age_group: string;
  admission_type: string;
  admittime: string;
  dischtime: string;
  length_of_stay_days: number;
  drug: string;
  starttime: string;
  stoptime: string | null;
  current_medication_count?: number | null;
  current_peak_concurrent_medication_count?: number | null;
  current_polypharmacy_flag?: number | null;
  historical_medication_count?: number | null;
  historical_polypharmacy_flag?: number | null;
  total_medication_count: number;
  polypharmacy_flag: number;
  benzodiazepine_flag: number;
  opioid_flag: number;
  anticholinergic_flag: number;
  ppi_flag: number;
  antipsychotic_flag: number;
  ckd_flag: number;
  dementia_flag: number;
  delirium_flag: number;
  heart_failure_flag: number;
  diabetes_flag: number;
  creatinine_first: number | null;
  creatinine_max: number | null;
  creatinine_mean: number | null;
  renal_risk_flag: number;
  deprescribing_priority_score: number;
  deprescribing_priority_label: RiskLabel;
  deprescribing_priority_summary_alert: string;
  deprescribing_priority_explanation: string;
  deprescribing_priority_bucket_scores_json: Record<string, number>;
  deprescribing_priority_reasons_json: string[];
  deprescribing_priority_evidence_json: Record<string, unknown>;
}

export interface ScoredRowsResponse {
  total_rows: number;
  limit: number;
  offset: number;
  rows: ScoredRow[];
}

export interface AdmissionSummary {
  subject_id: number;
  hadm_id: number;
  sex: string;
  age_proxy: number;
  age_group: string;
  admission_type: string;
  admittime: string;
  dischtime: string;
  length_of_stay_days: number;
  current_medication_count?: number | null;
  current_polypharmacy_flag?: number | null;
  historical_medication_count?: number | null;
  total_medication_count: number;
  polypharmacy_flag: number;
  creatinine_max: number | null;
  renal_risk_flag: number;
  overall_priority_score: number;
  overall_priority_label: RiskLabel;
  flagged_medication_count: number;
  overall_priority_drivers: string[];
}

export interface AdmissionSummariesResponse {
  total_admissions: number;
  limit: number;
  offset: number;
  rows: AdmissionSummary[];
}

export interface PatientSummary {
  subject_id: number;
  sex?: string | null;
  age_proxy?: number | null;
  age_group?: string | null;
  current_medication_count?: number | null;
  historical_medication_count?: number | null;
  encounter_count?: number | null;
  medication_count?: number | null;
  flagged_medication_count: number;
  highest_priority_score: number;
  highest_priority_label: RiskLabel;
  highest_priority_confidence?: number | null;
  dominant_medication_class?: string | null;
  top_problem_flashes?: string[] | null;
  discharge_imminent_review_flash?: string | null;
  discharge_review_flash?: string | null;
  imminent_review_flash?: string | null;
  latest_hadm_id?: number | null;
  latest_admission_type?: string | null;
  latest_dischtime?: string | null;
}

export interface PatientSummariesResponse {
  total_patients: number;
  limit: number;
  offset: number;
  rows: PatientSummary[];
}

export interface MedicationRowSummary {
  drug: string;
  medication_classes: string[];
  dose_value?: string | null;
  dose_unit?: string | null;
  route?: string | null;
  frequency?: string | null;
  starttime: string;
  stoptime: string | null;
  deprescribing_priority_score: number;
  deprescribing_priority_label: RiskLabel;
  deprescribing_priority_summary_alert: string;
  deprescribing_priority_explanation: string;
  deprescribing_priority_bucket_scores_json: Record<string, number>;
  deprescribing_priority_reasons_json: string[];
  deprescribing_priority_evidence_json: Record<string, unknown>;
  benzodiazepine_flag: number;
  opioid_flag: number;
  anticholinergic_flag: number;
  ppi_flag: number;
  antipsychotic_flag: number;
}

export interface AdmissionDetailResponse {
  subject_id: number;
  hadm_id: number;
  sex: string;
  age_proxy: number;
  age_group: string;
  admission_type: string;
  admittime: string;
  dischtime: string;
  length_of_stay_days: number;
  current_medication_count?: number | null;
  current_polypharmacy_flag?: number | null;
  historical_medication_count?: number | null;
  total_medication_count: number;
  polypharmacy_flag: number;
  ckd_flag: number;
  dementia_flag: number;
  delirium_flag: number;
  heart_failure_flag: number;
  diabetes_flag: number;
  creatinine_first: number | null;
  creatinine_max: number | null;
  creatinine_mean: number | null;
  renal_risk_flag: number;
  highest_priority_score: number;
  highest_priority_label: RiskLabel;
  flagged_medication_count: number;
  score_explanations: string[];
  overall_priority_drivers: string[];
  flagged_medications: MedicationRowSummary[];
  medications: MedicationRowSummary[];
}

export interface ScoredOutputSummary {
  path: string;
  row_count: number;
  column_count: number;
  unique_subjects: number;
  unique_admissions: number;
  last_modified: string;
}

export interface ProblemFlash {
  key: string;
  label: string;
  severity: RiskLabel;
  reason: string;
}

export interface PatientEncounterSummary {
  subject_id: number;
  hadm_id: number;
  admission_type: string;
  admittime: string;
  dischtime: string;
  length_of_stay_days: number;
  overall_priority_score: number;
  overall_priority_label: RiskLabel;
  overall_priority_confidence?: number | null;
  current_medication_count?: number | null;
  current_polypharmacy_flag?: number | null;
  historical_medication_count?: number | null;
  total_medication_count: number;
  flagged_medication_count: number;
  overall_priority_drivers: string[];
}

export interface DossierEncounterSelection {
  encounter_id: string;
  hadm_id?: number | null;
  stay_id?: number | null;
  encounter_source: string;
  selection_mode: string;
  requested_hadm_id?: number | null;
  review_timestamp?: string | null;
  review_timestamp_source?: string | null;
}

export interface PatientContextObject {
  sex?: string | null;
  age_proxy?: number | null;
  age_group?: string | null;
  current_medication_count?: number | null;
  historical_medication_count?: number | null;
  encounter_count?: number | null;
  medication_count?: number | null;
  flagged_medication_count?: number | null;
  polypharmacy_present?: boolean | null;
  renal_risk_present?: boolean | null;
  ckd_present?: boolean | null;
  dementia_present?: boolean | null;
  delirium_present?: boolean | null;
  heart_failure_present?: boolean | null;
  diabetes_present?: boolean | null;
  latest_creatinine_max?: number | null;
  latest_egfr_ml_min_1_73m2?: number | null;
  latest_weight_kg?: number | null;
  latest_bmi?: number | null;
  frailty_present?: boolean | null;
  hepatic_signal_summary?: string | null;
  baseline_potassium_summary?: string | null;
  baseline_sodium_summary?: string | null;
  vigilance_summary?: string | null;
  rass_summary?: string | null;
  pain_summary?: string | null;
  hemodynamic_stability_summary?: string | null;
  level_of_care?: string | null;
  dysphagia_present?: boolean | null;
  feeding_route_summary?: string | null;
  administration_constraints_summary?: string | null;
  latest_weight_provenance?: string | null;
  latest_bmi_provenance?: string | null;
  latest_renal_provenance?: string | null;
}

export interface PatientMedicationCard extends MedicationRowSummary {
  subject_id: number;
  encounter_id?: string | null;
  hadm_id: number;
  stay_id?: number | null;
  admission_type: string;
  admittime: string;
  dischtime: string;
  length_of_stay_days: number;
  status?: string | null;
  medication_status?: string | null;
  last_active_time?: string | null;
  review_timestamp?: string | null;
  review_timestamp_source?: string | null;
  priority_score_source?: string | null;
  medication_standardized?: string | null;
  rxnorm_rxcui?: string | null;
  rxnorm_term_type?: string | null;
  ingredient_standardized?: string | null;
  modeling__row_id?: string | null;
  first_scope_supported_class_flag?: number | null;
  benchmark__current_rule_score?: number | null;
  benchmark__current_rule_score_level?: RiskLabel | null;
  benchmark__current_rule_available_flag?: number | null;
  benchmark__medication_class_only_medication_class_standardized?: string | null;
  pharmacist_review_alignment?: "agreed" | "off_by_one" | "corrected" | null;
  pharmacist_review_alignment_label?: string | null;
  ml_priority_label?: RiskLabel | null;
  ml_priority_confidence?: number | null;
  ml_probability_low?: number | null;
  ml_probability_medium?: number | null;
  ml_probability_high?: number | null;
  reviewable_flag?: boolean;
  review_queue_priority?: string | null;
  review_queue_reasons?: string[] | null;
  clinician_review?: ClinicianReviewRecord | null;
  ml_priority_score?: number | null;
  ml_priority_rank_within_encounter?: number | null;
  ml_top_drivers?: string[] | null;
  ml_driver_raw_values?: Record<string, unknown> | null;
  guidance_summary?: string | null;
}

export interface PatientDetailResponse {
  patient_summary: PatientSummary;
  encounter_summaries: PatientEncounterSummary[];
  selected_encounter?: DossierEncounterSelection | null;
  review_timestamp?: string | null;
  current_medication_count?: number | null;
  current_flagged_medication_count?: number | null;
  historical_medication_count?: number | null;
  left_column_context: PatientContextObject;
  current_medications?: PatientMedicationCard[] | null;
  previous_medication_history?: PatientMedicationCard[] | null;
  ranked_medication_cards: PatientMedicationCard[];
  historical_medication_cards?: PatientMedicationCard[] | null;
  top_problem_flashes: ProblemFlash[];
  flagged_medication_count: number;
  medication_card_count: number;
  review_queue_summary?: ReviewQueueSummary | null;
  // Explicit placeholder-only sections in the current dossier contract.
  time_to_benefit_summary?: string | null;
  time_to_benefit_note?: string | null;
  taper_protocol_steps?: string[] | null;
  taper_protocol_summary?: string | null;
  peer_validation_references?: string[] | null;
  peer_validation_summary?: string | null;
  patient_perceived_symptoms?: string[] | null;
  patient_symptom_summary?: string | null;
}
