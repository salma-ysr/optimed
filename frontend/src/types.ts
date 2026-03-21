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
