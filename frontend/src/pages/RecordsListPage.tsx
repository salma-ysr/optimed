import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  getAdmissionSummaries,
  getLatestScoredOutput,
  refreshScores,
} from "../api/client";
import type { AdmissionSummary, ScoredOutputSummary } from "../types";

function scoreTone(label: AdmissionSummary["overall_priority_label"]) {
  if (label === "high") {
    return "chip chip-high";
  }
  if (label === "medium") {
    return "chip chip-medium";
  }
  return "chip chip-low";
}

export function RecordsListPage() {
  const [rows, setRows] = useState<AdmissionSummary[]>([]);
  const [summary, setSummary] = useState<ScoredOutputSummary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void loadPageData();
  }, []);

  async function loadPageData() {
    setIsLoading(true);
    setError(null);
    try {
      const [rowsResponse, latestSummary] = await Promise.all([
        getAdmissionSummaries(120, 0),
        getLatestScoredOutput(),
      ]);
      setRows(rowsResponse.rows);
      setSummary(latestSummary);
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Failed to load admission summaries from the backend.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  async function handleRefreshScores() {
    setIsRefreshing(true);
    setError(null);
    try {
      await refreshScores();
      await loadPageData();
    } catch (caughtError) {
      setError(
        caughtError instanceof Error
          ? caughtError.message
          : "Failed to refresh scored output.",
      );
    } finally {
      setIsRefreshing(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">OPTI-MED MVP</p>
          <h1>Medication Risk Review Dashboard</h1>
          <p className="hero-copy">
            Review each admission as a whole and identify which medications may deserve
            reconsideration for older adults during the hospital stay.
          </p>
        </div>
        <div className="hero-actions">
          <button
            className="primary-button"
            onClick={() => {
              void handleRefreshScores();
            }}
            disabled={isRefreshing}
          >
            {isRefreshing ? "Refreshing..." : "Refresh Scores"}
          </button>
        </div>
      </section>

      {summary ? (
        <section className="summary-grid">
          <article className="summary-card">
            <span className="summary-label">Admissions reviewed</span>
            <strong>{summary.unique_admissions.toLocaleString()}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Patients</span>
            <strong>{summary.unique_subjects.toLocaleString()}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Medication rows scored</span>
            <strong>{summary.row_count.toLocaleString()}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Last updated</span>
            <strong>{new Date(summary.last_modified).toLocaleString()}</strong>
          </article>
        </section>
      ) : null}

      {isLoading ? (
        <section className="state-panel">Loading admissions for clinical review...</section>
      ) : null}

      {!isLoading && error ? (
        <section className="state-panel error-panel">
          <strong>Unable to load admissions.</strong>
          <p>{error}</p>
        </section>
      ) : null}

      {!isLoading && !error ? (
        <section className="records-grid">
          {rows.map((row) => {
            const detailsPath = `/records/${row.subject_id}/${row.hadm_id}`;

            return (
              <Link key={`${row.subject_id}-${row.hadm_id}`} to={detailsPath} className="record-card">
                <div className="record-topline">
                  <div>
                    <p className="record-title">
                      Patient {row.subject_id} / Admission {row.hadm_id}
                    </p>
                    <p className="record-subtitle">
                      {row.sex} • Age {row.age_proxy} ({row.age_group}) • {row.admission_type}
                    </p>
                  </div>
                  <span className={scoreTone(row.overall_priority_label)}>
                    {row.overall_priority_label} • {row.overall_priority_score}
                  </span>
                </div>

                <div className="record-body">
                  <div className="metric-grid">
                    <div className="metric-item">
                      <span className="metric-label">Creatinine max</span>
                      <strong>{row.creatinine_max ?? "N/A"}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Overall priority</span>
                      <strong>{row.overall_priority_score}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Medication count</span>
                      <strong>{row.total_medication_count}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Flagged medications</span>
                      <strong>{row.flagged_medication_count}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Renal review</span>
                      <strong>{row.renal_risk_flag ? "Flagged" : "No"}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Length of stay</span>
                      <strong>{row.length_of_stay_days.toFixed(1)} days</strong>
                    </div>
                  </div>

                  <div className="metric-group admission-driver-block">
                    <span className="metric-label">Overall priority drivers</span>
                    <strong>
                      {row.overall_priority_drivers.length > 0
                        ? row.overall_priority_drivers.join(" • ")
                        : "No major driver listed"}
                    </strong>
                  </div>
                </div>

                <div className="chip-row">
                  {row.polypharmacy_flag ? <span className="chip chip-neutral">Polypharmacy</span> : null}
                  {row.renal_risk_flag ? <span className="chip chip-neutral">Renal risk</span> : null}
                  {row.flagged_medication_count > 0 ? (
                    <span className="chip chip-neutral">
                      {row.flagged_medication_count} flagged medication
                      {row.flagged_medication_count === 1 ? "" : "s"}
                    </span>
                  ) : null}
                </div>
              </Link>
            );
          })}
        </section>
      ) : null}
    </main>
  );
}
