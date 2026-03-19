import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getAdmissionSummaries, getLatestScoredOutput } from "../api/client";
import type { AdmissionSummary, ScoredOutputSummary } from "../types";
import { translateDriverList, translateRiskLabel } from "../uiText";

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
      const detail =
        caughtError instanceof Error && caughtError.message
          ? ` Détail: ${caughtError.message}`
          : "";
      setError(`Impossible de charger les résumés d’hospitalisation depuis le backend.${detail}`);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">OPTI-MED MVP</p>
          <h1>Tableau de revue du risque médicamenteux</h1>
          <p className="hero-copy">
            Examinez chaque hospitalisation dans son ensemble et identifiez les
            médicaments qui méritent une réévaluation chez les patients âgés pendant
            le séjour hospitalier.
          </p>
        </div>
      </section>

      {summary ? (
        <section className="summary-grid">
          <article className="summary-card">
            <span className="summary-label">Hospitalisations revues</span>
            <strong>{summary.unique_admissions.toLocaleString()}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Patients</span>
            <strong>{summary.unique_subjects.toLocaleString()}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Lignes médicamenteuses scorées</span>
            <strong>{summary.row_count.toLocaleString()}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Dernière mise à jour</span>
            <strong>{new Date(summary.last_modified).toLocaleString()}</strong>
          </article>
        </section>
      ) : null}

      {isLoading ? (
        <section className="state-panel">Chargement des hospitalisations à revoir...</section>
      ) : null}

      {!isLoading && error ? (
        <section className="state-panel error-panel">
          <strong>Impossible de charger les hospitalisations.</strong>
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
                      {row.sex} • Âge {row.age_proxy} ({row.age_group}) • {row.admission_type}
                    </p>
                  </div>
                  <span className={scoreTone(row.overall_priority_label)}>
                    {translateRiskLabel(row.overall_priority_label)} • {row.overall_priority_score}
                  </span>
                </div>

                <div className="record-body">
                  <div className="metric-grid">
                    <div className="metric-item">
                      <span className="metric-label">Créatinine max</span>
                      <strong>{row.creatinine_max ?? "N/D"}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Priorité globale</span>
                      <strong>{row.overall_priority_score}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Nombre de médicaments</span>
                      <strong>{row.total_medication_count}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Médicaments signalés</span>
                      <strong>{row.flagged_medication_count}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Revue rénale</span>
                      <strong>{row.renal_risk_flag ? "Signalé" : "Non"}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Durée de séjour</span>
                      <strong>{row.length_of_stay_days.toFixed(1)} jours</strong>
                    </div>
                  </div>

                  <div className="metric-group admission-driver-block">
                    <span className="metric-label">Facteurs principaux de priorité</span>
                    <strong>
                      {row.overall_priority_drivers.length > 0
                        ? translateDriverList(row.overall_priority_drivers).join(" • ")
                        : "Aucun facteur majeur affiché"}
                    </strong>
                  </div>
                </div>

                <div className="chip-row">
                  {row.polypharmacy_flag ? <span className="chip chip-neutral">Polypharmacie</span> : null}
                  {row.renal_risk_flag ? <span className="chip chip-neutral">Risque rénal</span> : null}
                  {row.flagged_medication_count > 0 ? (
                    <span className="chip chip-neutral">
                      {row.flagged_medication_count} médicament
                      {row.flagged_medication_count === 1 ? " signalé" : "s signalés"}
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
