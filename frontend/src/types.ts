export type RiskLabel = "low" | "medium" | "high";

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
  current_medication_count?: number | null;
  current_polypharmacy_flag?: number | null;
  historical_medication_count?: number | null;
  total_medication_count: number;
  flagged_medication_count: number;
  overall_priority_drivers: string[];
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
  hadm_id: number;
  admission_type: string;
  admittime: string;
  dischtime: string;
  length_of_stay_days: number;
  status?: string | null;
  medication_status?: string | null;
  last_active_time?: string | null;
  review_timestamp?: string | null;
  review_timestamp_source?: string | null;
}

export interface PatientDetailResponse {
  patient_summary: PatientSummary;
  encounter_summaries: PatientEncounterSummary[];
  selected_encounter?: {
    encounter_id: string;
    hadm_id?: number | null;
    stay_id?: number | null;
    encounter_source: string;
    review_timestamp?: string | null;
    review_timestamp_source?: string | null;
  } | null;
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
  time_to_benefit_summary?: string | null;
  time_to_benefit_note?: string | null;
  taper_protocol_steps?: string[] | null;
  taper_protocol_summary?: string | null;
  peer_validation_references?: string[] | null;
  peer_validation_summary?: string | null;
  patient_perceived_symptoms?: string[] | null;
  patient_symptom_summary?: string | null;
}
