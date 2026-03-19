import type {
  AdmissionSummariesResponse,
  AdmissionDetailResponse,
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

export function getAdmissionDetail(params: {
  subjectId: string;
  hadmId: string;
}): Promise<AdmissionDetailResponse> {
  const query = new URLSearchParams({
    subject_id: params.subjectId,
    hadm_id: params.hadmId,
  });
  return apiRequest<AdmissionDetailResponse>(`/scores/admission?${query.toString()}`);
}

export function getLatestScoredOutput(): Promise<ScoredOutputSummary> {
  return apiRequest<ScoredOutputSummary>("/scores/latest");
}
