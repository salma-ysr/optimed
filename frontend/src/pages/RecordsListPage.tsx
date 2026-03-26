import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getLatestScoredOutput, getPatientSummaries } from "../api/client";
import type { PatientSummary, RiskLabel, ScoredOutputSummary } from "../types";
import { translateProblemFlash, translateRiskLabel } from "../uiText";

type ProblemFilter = "all" | "renal_toxicity" | "oversedation" | "fall_risk" | "confusion";
type RiskFilter = "all" | RiskLabel;
type SortMode = "risk_desc" | "flagged_desc" | "subject_asc";

const problemFilterOptions: Array<{ value: ProblemFilter; label: string }> = [
  { value: "all", label: "Tous les patients" },
  { value: "renal_toxicity", label: "Toxicité rénale" },
  { value: "oversedation", label: "Sursédation" },
  { value: "fall_risk", label: "Risque de chute" },
  { value: "confusion", label: "Confusion / risque cognitif" },
];

const riskFilterOptions: Array<{ value: RiskFilter; label: string }> = [
  { value: "all", label: "Tous niveaux" },
  { value: "high", label: "Élevé" },
  { value: "medium", label: "Modéré" },
  { value: "low", label: "Faible" },
];

function scoreTone(label: RiskLabel) {
  if (label === "high") {
    return "chip chip-high";
  }
  if (label === "medium") {
    return "chip chip-medium";
  }
  return "chip chip-low";
}

function normalizedText(value: string) {
  return value.trim().toLowerCase();
}

function getProblemLabels(row: PatientSummary) {
  return (row.top_problem_flashes ?? []).filter(Boolean);
}

function matchesProblemFilter(row: PatientSummary, filter: ProblemFilter) {
  if (filter === "all") {
    return true;
  }

  const labels = getProblemLabels(row).map((item) => normalizedText(item));
  if (filter === "renal_toxicity") {
    return labels.some((item) => item.includes("renal"));
  }
  if (filter === "oversedation") {
    return labels.some((item) => item.includes("sedation"));
  }
  if (filter === "fall_risk") {
    return labels.some((item) => item.includes("fall"));
  }
  return labels.some(
    (item) =>
      item.includes("confusion") ||
      item.includes("cognitive") ||
      item.includes("delir") ||
      item.includes("dement"),
  );
}

function ageAndSexLine(row: PatientSummary) {
  const parts: string[] = [];
  if (row.sex) {
    parts.push(row.sex);
  }
  if (row.age_proxy != null) {
    const ageLabel = row.age_group ? `Âge ${row.age_proxy} (${row.age_group})` : `Âge ${row.age_proxy}`;
    parts.push(ageLabel);
  }
  return parts.join(" • ");
}

function patientIdentifier(row: PatientSummary) {
  return `Dossier ${row.subject_id}`;
}

function flashDiagnostic(row: PatientSummary) {
  const topProblem = getProblemLabels(row)[0];
  if (topProblem) {
    return `Alerte dominante : ${translateProblemFlash(topProblem)}.`;
  }
  if (row.highest_priority_label === "high") {
    return "Révision médicamenteuse prioritaire à court terme.";
  }
  if (row.highest_priority_label === "medium") {
    return "Révision clinique utile lors du prochain passage d’équipe.";
  }
  return "Profil plus stable, sans alerte dominante structurée.";
}

function summaryChips(row: PatientSummary) {
  return getProblemLabels(row).slice(0, 3);
}

function dischargeReviewFlash(row: PatientSummary) {
  return (
    row.discharge_imminent_review_flash ??
    row.discharge_review_flash ??
    row.imminent_review_flash ??
    null
  );
}

export function RecordsListPage() {
  const [rows, setRows] = useState<PatientSummary[]>([]);
  const [summary, setSummary] = useState<ScoredOutputSummary | null>(null);
  const [problemFilter, setProblemFilter] = useState<ProblemFilter>("all");
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [sortMode, setSortMode] = useState<SortMode>("risk_desc");
  const [searchText, setSearchText] = useState("");
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
        getPatientSummaries({ limit: 200, offset: 0 }),
        getLatestScoredOutput(),
      ]);
      setRows(rowsResponse.rows);
      setSummary(latestSummary);
    } catch (caughtError) {
      const detail =
        caughtError instanceof Error && caughtError.message
          ? ` Détail: ${caughtError.message}`
          : "";
      setError(`Impossible de charger les dossiers patients depuis le backend.${detail}`);
    } finally {
      setIsLoading(false);
    }
  }

  const visibleRows = useMemo(() => {
    const normalizedSearch = normalizedText(searchText);
    return rows
      .filter((row) => (riskFilter === "all" ? true : row.highest_priority_label === riskFilter))
      .filter((row) => matchesProblemFilter(row, problemFilter))
      .filter((row) =>
        normalizedSearch.length === 0
          ? true
          : patientIdentifier(row).toLowerCase().includes(normalizedSearch) ||
            String(row.subject_id).includes(normalizedSearch),
      )
      .sort((left, right) => {
        if (sortMode === "flagged_desc") {
          if (right.flagged_medication_count !== left.flagged_medication_count) {
            return right.flagged_medication_count - left.flagged_medication_count;
          }
          if (right.highest_priority_score !== left.highest_priority_score) {
            return right.highest_priority_score - left.highest_priority_score;
          }
          return left.subject_id - right.subject_id;
        }

        if (sortMode === "subject_asc") {
          return left.subject_id - right.subject_id;
        }

        if (right.highest_priority_score !== left.highest_priority_score) {
          return right.highest_priority_score - left.highest_priority_score;
        }
        if (right.flagged_medication_count !== left.flagged_medication_count) {
          return right.flagged_medication_count - left.flagged_medication_count;
        }
        return left.subject_id - right.subject_id;
      });
  }, [problemFilter, riskFilter, rows, searchText, sortMode]);

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">OPTI-MED MVP</p>
          <h1>Tableau de bord patient du risque médicamenteux</h1>
          <p className="hero-copy">
            Priorisez les dossiers patients à revoir, repérez les signaux dominants et
            ouvrez chaque dossier clinique sans centrer l’interface sur l’admission.
          </p>
          <div className="medication-list-controls">
            <Link to="/review-queue" className="list-control-button hero-link-button">
              Ouvrir la file de revue Phase 6
            </Link>
            <Link to="/blind-eval-queue" className="list-control-button hero-link-button">
              Ouvrir la session blindée finale
            </Link>
          </div>
        </div>
      </section>

      {summary ? (
        <section className="summary-grid">
          <article className="summary-card">
            <span className="summary-label">Patients suivis</span>
            <strong>{summary.unique_subjects.toLocaleString()}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Séjours de support</span>
            <strong>{summary.unique_admissions.toLocaleString()}</strong>
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

      <section className="filters-panel">
        <div className="filters-group">
          {problemFilterOptions.map((option) => (
            <button
              key={option.value}
              type="button"
              className={option.value === problemFilter ? "chip chip-active" : "chip chip-neutral"}
              onClick={() => setProblemFilter(option.value)}
            >
              {option.label}
            </button>
          ))}
        </div>
        <div className="filters-toolbar">
          <div className="filters-group">
            {riskFilterOptions.map((option) => (
              <button
                key={option.value}
                type="button"
                className={option.value === riskFilter ? "chip chip-active" : "chip chip-neutral"}
                onClick={() => setRiskFilter(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
          <div className="controls-row">
            <label className="sort-field">
              <span className="summary-label">Trier par</span>
              <select value={sortMode} onChange={(event) => setSortMode(event.target.value as SortMode)}>
                <option value="risk_desc">Risque le plus élevé</option>
                <option value="flagged_desc">Médicaments signalés</option>
                <option value="subject_asc">Numéro de dossier</option>
              </select>
            </label>
            <label className="search-field">
              <span className="summary-label">Rechercher un dossier</span>
              <input
                type="search"
                value={searchText}
                onChange={(event) => setSearchText(event.target.value)}
                placeholder="ID patient ou numéro de dossier"
              />
            </label>
          </div>
        </div>
      </section>

      {isLoading ? (
        <section className="state-panel">Chargement des dossiers patients à revoir...</section>
      ) : null}

      {!isLoading && error ? (
        <section className="state-panel error-panel">
          <strong>Impossible de charger les dossiers patients.</strong>
          <p>{error}</p>
        </section>
      ) : null}

      {!isLoading && !error && visibleRows.length === 0 ? (
        <section className="state-panel">
          Aucun patient ne correspond aux filtres actuels.
        </section>
      ) : null}

      {!isLoading && !error && visibleRows.length > 0 ? (
        <section className="records-grid">
          {visibleRows.map((row) => {
            const detailsPath = `/patients/${row.subject_id}`;
            const supportingEncounter =
              row.latest_hadm_id != null || row.latest_admission_type || row.latest_dischtime;
            const dischargeFlash = dischargeReviewFlash(row);

            return (
              <Link key={row.subject_id} to={detailsPath} className="record-card">
                <div className="record-topline">
                  <div>
                    <p className="record-title">{patientIdentifier(row)}</p>
                    {ageAndSexLine(row) ? <p className="record-subtitle">{ageAndSexLine(row)}</p> : null}
                    {supportingEncounter ? (
                      <p className="record-subtitle">
                        {row.latest_hadm_id != null ? `Séjour ${row.latest_hadm_id}` : null}
                        {row.latest_admission_type ? ` • ${row.latest_admission_type}` : null}
                        {row.latest_dischtime ? ` • Sortie ${row.latest_dischtime}` : null}
                      </p>
                    ) : null}
                  </div>
                  <span className={scoreTone(row.highest_priority_label)}>
                    {translateRiskLabel(row.highest_priority_label)} • {row.highest_priority_score}
                  </span>
                </div>

                <div className="record-body">
                  <p className="flash-diagnostic">{flashDiagnostic(row)}</p>

                  <div className="metric-grid">
                    <div className="metric-item">
                      <span className="metric-label">Médicaments signalés</span>
                      <strong>{row.flagged_medication_count}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Médicaments actuels</span>
                      <strong>{row.current_medication_count ?? row.medication_count ?? "N/D"}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Séjours suivis</span>
                      <strong>{row.encounter_count ?? "N/D"}</strong>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Niveau le plus élevé</span>
                      <strong>{translateRiskLabel(row.highest_priority_label)}</strong>
                    </div>
                  </div>
                </div>

                <div className="chip-row">
                  {dischargeFlash ? <span className="chip chip-medium">{dischargeFlash}</span> : null}
                  {summaryChips(row).map((item) => (
                    <span key={item} className="chip chip-neutral">
                      {translateProblemFlash(item)}
                    </span>
                  ))}
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
