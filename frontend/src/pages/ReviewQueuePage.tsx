import { type FormEvent, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getClinicianReviewQueue, submitClinicianReview } from "../api/client";
import type {
  ClinicianReviewQueueEntry,
  ClinicianReviewQueueResponse,
  ClinicianReviewStatus,
  ClinicianSuggestedAction,
  ClinicianReviewSubmissionRequest,
  RiskLabel,
} from "../types";
import { translateMedicationClass, translateRiskLabel } from "../uiText";

const REVIEWER_ID_STORAGE_KEY = "opti_med_phase5_reviewer_id";
const REVIEW_QUEUE_STATUS_STORAGE_KEY = "opti_med_phase6_queue_status";
const REVIEW_QUEUE_ACTION_STORAGE_KEY = "opti_med_phase6_queue_action";
const REVIEW_QUEUE_REASON_TAGS_STORAGE_KEY = "opti_med_phase6_queue_reason_tags";

const REVIEW_STATUS_OPTIONS: Array<{ value: "all" | ClinicianReviewStatus | "unreviewed"; label: string }> = [
  { value: "all", label: "Tous statuts" },
  { value: "unreviewed", label: "Non revus" },
  { value: "reviewed", label: "Revu" },
  { value: "uncertain", label: "Incertain" },
  { value: "insufficient_context", label: "Contexte insuffisant" },
  { value: "skip", label: "Passé" },
];
const SAVEABLE_STATUS_OPTIONS: Array<{ value: ClinicianReviewStatus; label: string }> = [
  { value: "reviewed", label: "Revu" },
  { value: "uncertain", label: "Incertain" },
  { value: "insufficient_context", label: "Contexte insuffisant" },
  { value: "skip", label: "Passer" },
];
const SUGGESTED_ACTION_OPTIONS: Array<{ value: ClinicianSuggestedAction; label: string }> = [
  { value: "keep", label: "Garder" },
  { value: "monitor", label: "Surveiller" },
  { value: "reconsider", label: "Réévaluer" },
  { value: "deprescribe_candidate", label: "Déprescription possible" },
  { value: "needs_more_info", label: "Infos requises" },
];
const REASON_TAG_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "polypharmacy", label: "Polypharmacie" },
  { value: "duplication", label: "Duplication" },
  { value: "renal_risk", label: "Risque rénal" },
  { value: "fall_risk", label: "Risque de chute" },
  { value: "anticholinergic_burden", label: "Charge anticholinergique" },
  { value: "interaction_risk", label: "Risque d’interaction" },
  { value: "questionable_indication", label: "Indication discutable" },
  { value: "monitoring_needed", label: "Surveillance" },
  { value: "tapering_candidate", label: "Sevrage" },
  { value: "insufficient_context", label: "Contexte insuffisant" },
  { value: "other", label: "Autre" },
];

function reviewerIdInitialValue() {
  if (typeof window === "undefined") {
    return "pharmacist_demo_local";
  }
  return window.localStorage.getItem(REVIEWER_ID_STORAGE_KEY) ?? "pharmacist_demo_local";
}

function defaultReviewStatusValue() {
  if (typeof window === "undefined") {
    return "reviewed" as ClinicianReviewStatus;
  }
  const stored = window.localStorage.getItem(REVIEW_QUEUE_STATUS_STORAGE_KEY);
  return SAVEABLE_STATUS_OPTIONS.some((option) => option.value === stored)
    ? (stored as ClinicianReviewStatus)
    : "reviewed";
}

function defaultSuggestedActionValue() {
  if (typeof window === "undefined") {
    return "";
  }
  const stored = window.localStorage.getItem(REVIEW_QUEUE_ACTION_STORAGE_KEY) ?? "";
  return SUGGESTED_ACTION_OPTIONS.some((option) => option.value === stored) ? stored : "";
}

function defaultReasonTagsValue() {
  if (typeof window === "undefined") {
    return [] as string[];
  }
  const stored = window.localStorage.getItem(REVIEW_QUEUE_REASON_TAGS_STORAGE_KEY);
  if (!stored) {
    return [];
  }
  try {
    const parsed = JSON.parse(stored);
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((value) =>
      REASON_TAG_OPTIONS.some((option) => option.value === String(value)),
    ) as string[];
  } catch {
    return [];
  }
}

function riskTone(label: RiskLabel) {
  if (label === "high") {
    return "chip chip-high";
  }
  if (label === "medium") {
    return "chip chip-medium";
  }
  return "chip chip-low";
}

function queueBandLabel(value: string) {
  const labels: Record<string, string> = {
    disagreement_candidate: "Désaccord",
    high_signal_unreviewed: "Signal fort",
    coverage_expansion_unreviewed: "Couverture",
    review_backlog: "Backlog",
    reviewed_disagreement_followup: "Revoir désaccord",
    reviewed_backlog: "Déjà revu",
  };
  return labels[value] ?? value;
}

function reasonLabel(value: string) {
  const labels: Record<string, string> = {
    unreviewed_row: "Sans revue",
    already_reviewed: "Déjà revu",
    clinician_rule_disagreement: "Clinicien vs règle",
    benchmark_signal_low: "Signal règle faible",
    benchmark_signal_medium: "Signal règle modéré",
    benchmark_signal_high: "Signal règle élevé",
    benchmark_sparse_but_reviewable: "Benchmark sparse mais revuable",
    primary_action_ambiguous: "Action primaire ambiguë",
    training_excluded_but_reviewable: "Hors entraînement mais utile",
    expand_medication_class_coverage: "Étendre la couverture de classe",
    expand_patient_coverage: "Étendre la couverture patient",
    expand_encounter_coverage: "Étendre la couverture séjour",
    supported_first_scope_class: "Classe de première portée",
  };
  return labels[value] ?? value;
}

function classLabel(value?: string | null) {
  if (!value) {
    return "Classe N/D";
  }
  return translateMedicationClass(value);
}

function routeDoseFrequencyLine(row: ClinicianReviewQueueEntry) {
  const parts = [
    row.dose_value && row.dose_unit ? `${row.dose_value} ${row.dose_unit}` : row.dose_value,
    row.route,
    row.frequency,
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(" • ") : "Posologie détaillée indisponible";
}

function savedReviewSummary(row: ClinicianReviewQueueEntry) {
  if (!row.clinician_review) {
    return "Aucune revue enregistrée";
  }
  return [
    `Niveau ${translateRiskLabel(row.clinician_review.label__clinician_priority_level)}`,
    row.clinician_review.label__clinician_review_status,
    `v${row.clinician_review.review_version}`,
  ].join(" • ");
}

function dossierHref(row: ClinicianReviewQueueEntry) {
  if (row.hadm_id != null) {
    return `/patients/${row.subject_id}?hadm_id=${row.hadm_id}`;
  }
  return `/patients/${row.subject_id}`;
}

function queueRowKey(row: ClinicianReviewQueueEntry) {
  return [row.subject_id, row.encounter_id, row.medication_standardized, row.review_timestamp].join("|");
}

export function ReviewQueuePage() {
  const [queueResponse, setQueueResponse] = useState<ClinicianReviewQueueResponse | null>(null);
  const [activeIndex, setActiveIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reviewerId, setReviewerId] = useState<string>(reviewerIdInitialValue);
  const [unreviewedOnly, setUnreviewedOnly] = useState(true);
  const [medicationClass, setMedicationClass] = useState("all");
  const [reviewStatus, setReviewStatus] = useState<"all" | ClinicianReviewStatus | "unreviewed">("all");
  const [reasonTagPresence, setReasonTagPresence] = useState<"all" | "has_reason_tags" | "no_reason_tags">("all");
  const [subjectInput, setSubjectInput] = useState("");
  const [priorityLevel, setPriorityLevel] = useState<RiskLabel>("medium");
  const [queueReviewStatus, setQueueReviewStatus] = useState<ClinicianReviewStatus>(defaultReviewStatusValue);
  const [suggestedAction, setSuggestedAction] = useState<ClinicianSuggestedAction | "">(
    defaultSuggestedActionValue() as ClinicianSuggestedAction | "",
  );
  const [reasonTags, setReasonTags] = useState<string[]>(defaultReasonTagsValue);
  const [priorityScore, setPriorityScore] = useState("");
  const [note, setNote] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(REVIEWER_ID_STORAGE_KEY, reviewerId);
  }, [reviewerId]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(REVIEW_QUEUE_STATUS_STORAGE_KEY, queueReviewStatus);
  }, [queueReviewStatus]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(REVIEW_QUEUE_ACTION_STORAGE_KEY, suggestedAction);
  }, [suggestedAction]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(REVIEW_QUEUE_REASON_TAGS_STORAGE_KEY, JSON.stringify(reasonTags));
  }, [reasonTags]);

  useEffect(() => {
    void loadQueue();
  }, [unreviewedOnly, medicationClass, reviewStatus, reasonTagPresence, subjectInput]);

  const activeRow = queueResponse?.rows[activeIndex] ?? null;

  useEffect(() => {
    if (!activeRow) {
      return;
    }
    setPriorityLevel(
      activeRow.clinician_review?.label__clinician_priority_level ??
        activeRow.benchmark__current_rule_score_level ??
        "medium",
    );
    setQueueReviewStatus(activeRow.clinician_review?.label__clinician_review_status ?? defaultReviewStatusValue());
    setSuggestedAction(
      activeRow.clinician_review?.label__clinician_suggested_action ??
        (defaultSuggestedActionValue() as ClinicianSuggestedAction | ""),
    );
    setReasonTags(activeRow.clinician_review?.label__clinician_reason_tags ?? defaultReasonTagsValue());
    setPriorityScore(
      activeRow.clinician_review?.label__clinician_priority_score != null
        ? String(activeRow.clinician_review.label__clinician_priority_score)
        : "",
    );
    setNote(activeRow.clinician_review?.label__clinician_note ?? "");
    setSaveError(null);
    setSaveMessage(null);
  }, [activeRow?.queue_rank]);

  async function loadQueue(options?: { keepIndex?: boolean; preferredRowKey?: string | null }) {
    setIsLoading(true);
    setError(null);
    try {
      const subjectId =
        subjectInput.trim().length > 0 && /^\d+$/.test(subjectInput.trim())
          ? Number(subjectInput.trim())
          : null;
      const response = await getClinicianReviewQueue({
        limit: 200,
        offset: 0,
        unreviewedOnly,
        medicationClass: medicationClass === "all" ? undefined : medicationClass,
        reviewStatus,
        subjectId,
        reasonTagPresence,
      });
      setQueueResponse(response);
      setActiveIndex((current) => {
        const preferredRowKey = options?.preferredRowKey;
        if (preferredRowKey) {
          const preferredIndex = response.rows.findIndex((row) => queueRowKey(row) === preferredRowKey);
          if (preferredIndex >= 0) {
            return preferredIndex;
          }
        }
        if (!options?.keepIndex) {
          return 0;
        }
        return Math.min(current, Math.max(response.rows.length - 1, 0));
      });
    } catch (caughtError) {
      const detail = caughtError instanceof Error ? ` Détail: ${caughtError.message}` : "";
      setError(`Impossible de charger la file de revue.${detail}`);
    } finally {
      setIsLoading(false);
    }
  }

  function toggleReasonTag(tag: string) {
    setReasonTags((current) =>
      current.includes(tag) ? current.filter((item) => item !== tag) : [...current, tag],
    );
  }

  async function saveReview(level: RiskLabel, options?: { advanceAfterSave?: boolean }) {
    if (!activeRow) {
      return;
    }
    const payload: ClinicianReviewSubmissionRequest = {
      subject_id: activeRow.subject_id,
      encounter_id: activeRow.encounter_id,
      hadm_id: activeRow.hadm_id ?? null,
      stay_id: activeRow.stay_id ?? null,
      medication_standardized: activeRow.medication_standardized,
      review_timestamp: activeRow.review_timestamp,
      modeling__row_id: activeRow.modeling__row_id ?? null,
      reviewer_id: reviewerId.trim() || "pharmacist_demo_local",
      review_submission_source: "review_queue_ui_phase6",
      label__clinician_priority_level: level,
      label__clinician_priority_score: priorityScore.trim() ? Number(priorityScore) : null,
      label__clinician_review_status: queueReviewStatus,
      label__clinician_reason_tags: reasonTags,
      label__clinician_note: note.trim() || null,
      label__clinician_suggested_action: suggestedAction || null,
    };

    setIsSaving(true);
    setSaveError(null);
    setSaveMessage(null);
    try {
      const response = await submitClinicianReview(payload);
      setSaveMessage(
        `Revue enregistrée • ${response.review.reviewer_id} • ${response.review.review_submission_timestamp}`,
      );
      const currentRows = queueResponse?.rows ?? [];
      const preferredRowKey =
        options?.advanceAfterSave && currentRows.length > 0
          ? queueRowKey(currentRows[Math.min(activeIndex + 1, currentRows.length - 1)] ?? activeRow)
          : queueRowKey(activeRow);
      await loadQueue({ keepIndex: true, preferredRowKey });
    } catch (caughtError) {
      const detail = caughtError instanceof Error ? caughtError.message : "Erreur inattendue";
      setSaveError(`Enregistrement impossible. ${detail}`);
    } finally {
      setIsSaving(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    await saveReview(priorityLevel, { advanceAfterSave: false });
  }

  const medicationClassOptions = useMemo(() => {
    const counts = queueResponse?.workflow_report.reviewable_rows_by_medication_class ?? {};
    return Object.entries(counts).filter(([, count]) => count > 0);
  }, [queueResponse?.workflow_report.reviewable_rows_by_medication_class]);

  const milestoneRows = useMemo(
    () => Object.values(queueResponse?.workflow_report.milestone_status ?? {}),
    [queueResponse?.workflow_report.milestone_status],
  );

  return (
    <main className="page-shell">
      <div className="details-header">
        <Link to="/" className="back-link">
          ← Retour aux patients
        </Link>
      </div>

      <section className="hero-panel">
        <div>
          <p className="eyebrow">OPTI-MED Phase 6</p>
          <h1>File de revue clinicienne</h1>
          <p className="hero-copy">
            File déterministe pour densifier plus vite les labels cliniciens low / medium / high
            sur le grain analytique existant.
          </p>
        </div>
      </section>

      {queueResponse ? (
        <section className="summary-grid">
          <article className="summary-card">
            <span className="summary-label">Revuables</span>
            <strong>{queueResponse.workflow_report.reviewable_row_count}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Déjà revus</span>
            <strong>{queueResponse.workflow_report.clinician_reviewed_row_count}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">En attente</span>
            <strong>{queueResponse.workflow_report.remaining_unreviewed_row_count}</strong>
          </article>
          <article className="summary-card">
            <span className="summary-label">Couverture</span>
            <strong>{(queueResponse.workflow_report.clinician_review_coverage_rate * 100).toFixed(1)}%</strong>
          </article>
        </section>
      ) : null}

      <section className="filters-panel queue-filters-panel">
        <div className="controls-row queue-filter-row">
          <label className="search-field">
            <span className="summary-label">Patient</span>
            <input
              type="search"
              value={subjectInput}
              onChange={(event) => setSubjectInput(event.target.value)}
              placeholder="ID patient"
            />
          </label>
          <label className="sort-field">
            <span className="summary-label">Classe</span>
            <select value={medicationClass} onChange={(event) => setMedicationClass(event.target.value)}>
              <option value="all">Toutes</option>
              {medicationClassOptions.map(([value, count]) => (
                <option key={value} value={value}>
                  {classLabel(value)} ({count})
                </option>
              ))}
            </select>
          </label>
          <label className="sort-field">
            <span className="summary-label">Statut de revue</span>
            <select
              value={reviewStatus}
              onChange={(event) =>
                setReviewStatus(event.target.value as "all" | ClinicianReviewStatus | "unreviewed")
              }
            >
              {REVIEW_STATUS_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>
          <label className="sort-field">
            <span className="summary-label">Tags raison</span>
            <select
              value={reasonTagPresence}
              onChange={(event) =>
                setReasonTagPresence(event.target.value as "all" | "has_reason_tags" | "no_reason_tags")
              }
            >
              <option value="all">Tous</option>
              <option value="has_reason_tags">Avec tags</option>
              <option value="no_reason_tags">Sans tags</option>
            </select>
          </label>
          <label className="checkbox-chip queue-filter-toggle">
            <input
              type="checkbox"
              checked={unreviewedOnly}
              onChange={(event) => setUnreviewedOnly(event.target.checked)}
            />
            <span>Non revus seulement</span>
          </label>
        </div>
      </section>

      {isLoading ? <section className="state-panel">Chargement de la file de revue...</section> : null}
      {!isLoading && error ? (
        <section className="state-panel error-panel">
          <strong>Impossible de charger la file de revue.</strong>
          <p>{error}</p>
        </section>
      ) : null}

      {!isLoading && queueResponse ? (
        <>
          <section className="details-card queue-progress-card">
            <div className="queue-progress-header">
              <div>
                <h2>Progression de densification</h2>
                <p className="details-section-copy">{queueResponse.workflow_report.phase_scope_statement}</p>
              </div>
              <div className="chip-row">
                <span className="chip chip-neutral">
                  Sujets revus {queueResponse.workflow_report.clinician_reviewed_distinct_subject_count}/
                  {queueResponse.workflow_report.reviewable_distinct_subject_count}
                </span>
                <span className="chip chip-neutral">
                  Séjours revus {queueResponse.workflow_report.clinician_reviewed_distinct_encounter_count}/
                  {queueResponse.workflow_report.reviewable_distinct_encounter_count}
                </span>
                <span className="chip chip-neutral">
                  Classes revues {queueResponse.workflow_report.clinician_reviewed_medication_class_count}
                </span>
              </div>
            </div>
            <div className="queue-coverage-grid">
              {Object.entries(queueResponse.workflow_report.clinician_reviewed_rows_by_medication_class).map(
                ([className, reviewedCount]) => (
                  <article key={className} className="queue-coverage-item">
                    <strong>{classLabel(className)}</strong>
                    <span>
                      {reviewedCount} / {queueResponse.workflow_report.reviewable_rows_by_medication_class[className] ?? 0}
                    </span>
                  </article>
                ),
              )}
            </div>
            {milestoneRows.length > 0 ? (
              <div className="queue-milestone-grid">
                {milestoneRows.map((milestone) => (
                  <article key={String(milestone.label)} className="queue-milestone-item">
                    <strong>{String(milestone.label)}</strong>
                    <span>{String(milestone.target)}</span>
                    <span className={Boolean(milestone.met) ? "chip chip-low" : "chip chip-neutral"}>
                      {Boolean(milestone.met) ? "Atteint" : "En attente"}
                    </span>
                    <small>{String(milestone.current_value)}</small>
                  </article>
                ))}
              </div>
            ) : null}
          </section>

          {queueResponse.rows.length > 0 && activeRow ? (
            <section className="queue-layout">
              <article className="details-card queue-active-card">
                <div className="queue-active-header">
                  <div>
                    <p className="eyebrow">File ordonnée</p>
                    <h2>
                      Rang {activeRow.queue_rank} sur {queueResponse.filtered_queue_rows}
                    </h2>
                    <p className="record-subtitle">
                      Patient {activeRow.subject_id} • séjour {activeRow.hadm_id ?? activeRow.encounter_id}
                    </p>
                  </div>
                  <div className="chip-row">
                    <span className="chip chip-neutral">{queueBandLabel(activeRow.queue_priority_band)}</span>
                    {activeRow.clinician_review?.label__clinician_priority_level ? (
                      <span className={riskTone(activeRow.clinician_review.label__clinician_priority_level)}>
                        {translateRiskLabel(activeRow.clinician_review.label__clinician_priority_level)}
                      </span>
                    ) : null}
                  </div>
                </div>

                <div className="queue-navigation">
                  <button
                    type="button"
                    className="list-control-button"
                    disabled={activeIndex === 0}
                    onClick={() => setActiveIndex((current) => Math.max(0, current - 1))}
                  >
                    ← Précédent
                  </button>
                  <button
                    type="button"
                    className="list-control-button"
                    disabled={activeIndex >= queueResponse.rows.length - 1}
                    onClick={() =>
                      setActiveIndex((current) => Math.min(queueResponse.rows.length - 1, current + 1))
                    }
                  >
                    Suivant →
                  </button>
                  <Link to={dossierHref(activeRow)} className="list-control-button queue-open-dossier">
                    Ouvrir le dossier
                  </Link>
                </div>

                <section className="queue-row-summary">
                  <div>
                    <p className="queue-row-title">{activeRow.medication_standardized}</p>
                    <p className="medication-alert-meta">{classLabel(activeRow.medication_class_standardized)}</p>
                    <p className="details-section-copy">{routeDoseFrequencyLine(activeRow)}</p>
                  </div>
                  <div className="chip-row">
                    <span className="chip chip-neutral">Statut {activeRow.medication_status_at_review ?? "N/D"}</span>
                    <span className="chip chip-neutral">
                      Couverture classe {activeRow.class_reviewed_count}/{activeRow.class_reviewable_count}
                    </span>
                    <span className="chip chip-neutral">
                      Couverture patient {activeRow.subject_reviewed_count}/{activeRow.subject_reviewable_count}
                    </span>
                  </div>
                </section>

                <p className="details-section-copy">{activeRow.needs_review_justification}</p>

                <div className="chip-row review-reasons-row">
                  {activeRow.queue_priority_reasons.map((reason) => (
                    <span key={reason} className="chip chip-neutral">
                      {reasonLabel(reason)}
                    </span>
                  ))}
                </div>

                <div className="review-context-grid">
                  <span className="review-context-chip">
                    Benchmark courant:{" "}
                    {activeRow.benchmark__current_rule_available_flag
                      ? `${activeRow.benchmark__current_rule_score ?? "N/D"} / ${
                          activeRow.benchmark__current_rule_score_level
                            ? translateRiskLabel(activeRow.benchmark__current_rule_score_level)
                            : "N/D"
                        }`
                      : "indisponible"}
                  </span>
                  <span className="review-context-chip">
                    Action primaire: {activeRow.label__primary_action_label ?? "N/D"}
                  </span>
                  <span className="review-context-chip">
                    Ligne entraînable: {activeRow.meta__dataset_row_eligible_for_training_flag === 1 ? "oui" : "non"}
                  </span>
                </div>

                <div className="review-existing-summary">
                  <span className="chip chip-neutral">{savedReviewSummary(activeRow)}</span>
                  {activeRow.clinician_review?.review_submission_timestamp ? (
                    <p className="review-feedback review-feedback-success">
                      Dernière sauvegarde: {activeRow.clinician_review.review_submission_timestamp}
                    </p>
                  ) : (
                    <p className="details-section-copy">Aucune revue active enregistrée pour cette ligne.</p>
                  )}
                </div>

                <form className="review-form" onSubmit={handleSubmit}>
                  <div className="review-form-grid">
                    <label className="review-field">
                      <span>Identifiant relecteur</span>
                      <input
                        value={reviewerId}
                        onChange={(event) => setReviewerId(event.target.value)}
                        placeholder="pharmacist_demo_local"
                      />
                    </label>
                    <label className="review-field">
                      <span>Score numérique (optionnel)</span>
                      <input
                        type="number"
                        min={0}
                        max={10}
                        step={1}
                        value={priorityScore}
                        onChange={(event) => setPriorityScore(event.target.value)}
                        placeholder="Laisser vide"
                      />
                    </label>
                  </div>

                  <div className="review-field">
                    <span>Niveau sélectionné</span>
                    <div className="segmented-control">
                      {(["low", "medium", "high"] as RiskLabel[]).map((option) => (
                        <button
                          key={option}
                          type="button"
                          className={priorityLevel === option ? "segment-button segment-button-active" : "segment-button"}
                          onClick={() => setPriorityLevel(option)}
                        >
                          {translateRiskLabel(option)}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="review-form-grid">
                    <label className="review-field">
                      <span>Statut par défaut</span>
                      <select
                        value={queueReviewStatus}
                        onChange={(event) => setQueueReviewStatus(event.target.value as ClinicianReviewStatus)}
                      >
                        {SAVEABLE_STATUS_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="review-field">
                      <span>Disposition par défaut</span>
                      <select
                        value={suggestedAction}
                        onChange={(event) => setSuggestedAction(event.target.value as ClinicianSuggestedAction | "")}
                      >
                        <option value="">Aucune</option>
                        {SUGGESTED_ACTION_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <div className="review-field">
                    <span>Tags raison par défaut</span>
                    <div className="checkbox-chip-grid">
                      {REASON_TAG_OPTIONS.map((option) => (
                        <label key={option.value} className={reasonTags.includes(option.value) ? "checkbox-chip checkbox-chip-active" : "checkbox-chip"}>
                          <input
                            type="checkbox"
                            checked={reasonTags.includes(option.value)}
                            onChange={() => toggleReasonTag(option.value)}
                          />
                          <span>{option.label}</span>
                        </label>
                      ))}
                    </div>
                  </div>

                  <label className="review-field">
                    <span>Note libre</span>
                    <textarea
                      rows={3}
                      value={note}
                      onChange={(event) => setNote(event.target.value)}
                      placeholder="Point clinique ou justification rapide"
                    />
                  </label>

                  {saveError ? <p className="review-feedback review-feedback-error">{saveError}</p> : null}
                  {saveMessage ? <p className="review-feedback review-feedback-success">{saveMessage}</p> : null}

                  <div className="queue-quick-save-row">
                    {(["low", "medium", "high"] as RiskLabel[]).map((option) => (
                      <button
                        key={option}
                        type="button"
                        className="primary-button"
                        disabled={isSaving}
                        onClick={() => {
                          setPriorityLevel(option);
                          void saveReview(option, { advanceAfterSave: true });
                        }}
                      >
                        {isSaving ? "Enregistrement..." : `Sauver ${translateRiskLabel(option)} + suivant`}
                      </button>
                    ))}
                  </div>

                  <div className="review-form-actions">
                    <button type="submit" className="primary-button" disabled={isSaving}>
                      {isSaving ? "Enregistrement..." : "Enregistrer et rester sur cette ligne"}
                    </button>
                    <div className="review-traceability">
                      <span className="review-panel-subtitle">Trace analytique</span>
                      <code>{activeRow.modeling__row_id ?? "indisponible"}</code>
                    </div>
                  </div>
                </form>
              </article>

              <aside className="details-card queue-sidebar">
                <h2>Aperçu de file</h2>
                <p className="details-section-copy">
                  {queueResponse.filtered_queue_rows} ligne{queueResponse.filtered_queue_rows > 1 ? "s" : ""} après filtres.
                </p>
                <div className="queue-sidebar-list">
                  {queueResponse.rows.map((row, index) => (
                    <button
                      key={`${row.queue_rank}-${row.modeling__row_id ?? row.medication_standardized}`}
                      type="button"
                      className={index === activeIndex ? "queue-sidebar-item queue-sidebar-item-active" : "queue-sidebar-item"}
                      onClick={() => setActiveIndex(index)}
                    >
                      <strong>#{row.queue_rank} • {row.medication_standardized}</strong>
                      <span>Patient {row.subject_id} • {classLabel(row.medication_class_standardized)}</span>
                      <span>
                        {row.current_clinician_reviewed_flag === 1 ? "Déjà revu" : "À revoir"} •{" "}
                        {queueBandLabel(row.queue_priority_band)}
                      </span>
                    </button>
                  ))}
                </div>
              </aside>
            </section>
          ) : (
            <section className="state-panel">
              Aucun candidat ne correspond aux filtres actuels.
            </section>
          )}
        </>
      ) : null}
    </main>
  );
}
