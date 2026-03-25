import type {
  AdmissionSummariesResponse,
  ClinicianReviewSubmissionRequest,
  ClinicianReviewSubmissionResponse,
  ClinicianReviewWorkflowReport,
  PatientDetailResponse,
  PatientSummariesResponse,
  ScoredOutputSummary,
} from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
    },
    ...init,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as T;
}

export function getAdmissionSummaries(
  limit = 100,
  offset = 0,
): Promise<AdmissionSummariesResponse> {
  return apiRequest<AdmissionSummariesResponse>(`/admissions?limit=${limit}&offset=${offset}`);
}

export function getPatientSummaries(params?: {
  limit?: number;
  offset?: number;
  riskLabel?: "low" | "medium" | "high";
}): Promise<PatientSummariesResponse> {
  const query = new URLSearchParams();
  query.set("limit", String(params?.limit ?? 100));
  query.set("offset", String(params?.offset ?? 0));
  if (params?.riskLabel) {
    query.set("risk_label", params.riskLabel);
  }
  return apiRequest<PatientSummariesResponse>(`/patients?${query.toString()}`);
}

export function getPatientDetail(
  subjectId: string,
  options?: { hadmId?: number | string | null },
): Promise<PatientDetailResponse> {
  const query = new URLSearchParams();
  if (options?.hadmId != null && options.hadmId !== "") {
    query.set("hadm_id", String(options.hadmId));
  }
  const queryString = query.toString();
  const suffix = queryString ? `?${queryString}` : "";
  return apiRequest<PatientDetailResponse>(`/patients/${subjectId}${suffix}`);
}

export function getLatestScoredOutput(): Promise<ScoredOutputSummary> {
  return apiRequest<ScoredOutputSummary>("/scores/latest");
}

export function submitClinicianReview(
  payload: ClinicianReviewSubmissionRequest,
): Promise<ClinicianReviewSubmissionResponse> {
  return apiRequest<ClinicianReviewSubmissionResponse>("/clinician-reviews", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getClinicianReviewWorkflowReport(): Promise<ClinicianReviewWorkflowReport> {
  return apiRequest<ClinicianReviewWorkflowReport>("/clinician-reviews/report");
}
