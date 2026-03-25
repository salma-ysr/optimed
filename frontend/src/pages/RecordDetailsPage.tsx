import { type FormEvent, useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { getPatientDetail, submitClinicianReview } from "../api/client";
import type {
  ClinicianReviewRecord,
  ClinicianReviewStatus,
  ClinicianSuggestedAction,
  ClinicianReviewSubmissionRequest,
  PatientDetailResponse,
  PatientEncounterSummary,
  PatientMedicationCard,
  ProblemFlash,
  RiskLabel,
  ReviewQueueSummary,
} from "../types";
import {
  translateBucketLabel,
  translateDriverList,
  translateExplanation,
  translateMedicationClass,
  translateProblemFlash,
  translateRiskLabel,
} from "../uiText";

type ContextItem = {
  label: string;
  value?: string | number | null;
  hint?: string | null;
  hideWhenMissing?: boolean;
};

type ContextSectionProps = {
  title: string;
  items: ContextItem[];
};

type MedicationAlertCardProps = {
  medication: PatientMedicationCard;
  expanded: boolean;
  onToggle: () => void;
  reviewerId: string;
  onReviewerIdChange: (value: string) => void;
  onReviewSaved: () => Promise<void>;
  reviewContext: MedicationReviewContext;
};

type DecisionSupportPanelProps = {
  title: string;
  description?: string;
  items?: string[];
  summary?: string | null;
  emptyLabel: string;
};

type MedicationReviewContext = {
  burden: string;
  diagnoses: string | null;
  renal: string | null;
};

const REVIEWER_ID_STORAGE_KEY = "opti_med_phase5_reviewer_id";
const REASON_TAG_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "polypharmacy", label: "Polypharmacie" },
  { value: "duplication", label: "Duplication" },
  { value: "renal_risk", label: "Risque rénal" },
  { value: "fall_risk", label: "Risque de chute" },
  { value: "anticholinergic_burden", label: "Charge anticholinergique" },
  { value: "interaction_risk", label: "Risque d’interaction" },
  { value: "questionable_indication", label: "Indication discutable" },
  { value: "monitoring_needed", label: "Surveillance nécessaire" },
  { value: "tapering_candidate", label: "Candidat au sevrage" },
  { value: "insufficient_context", label: "Contexte insuffisant" },
  { value: "other", label: "Autre" },
];
const REVIEW_STATUS_OPTIONS: Array<{ value: ClinicianReviewStatus; label: string }> = [
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
  { value: "needs_more_info", label: "Informations requises" },
];

function riskTone(label: RiskLabel) {
  if (label === "high") {
    return "chip chip-high";
  }
  if (label === "medium") {
    return "chip chip-medium";
  }
  return "chip chip-low";
}

function medicationClassBadges(medication: PatientMedicationCard) {
  return medication.medication_classes.map(translateMedicationClass);
}

function bucketBreakdownText(medication: PatientMedicationCard) {
  return Object.entries(medication.deprescribing_priority_bucket_scores_json)
    .filter(([, value]) => value > 0)
    .map(([bucket, value]) => `${translateBucketLabel(bucket)}: ${value}`)
    .join(" • ");
}

function encounterSubtitle(encounter: PatientEncounterSummary) {
  const parts = [encounter.admission_type, `${encounter.length_of_stay_days.toFixed(1)} jours`].filter(Boolean);
  return parts.join(" • ");
}

function selectedEncounterSummary(record: PatientDetailResponse) {
  const selectedHadmId = record.selected_encounter?.hadm_id;
  if (selectedHadmId != null) {
    const matchedEncounter = record.encounter_summaries.find(
      (encounter) => encounter.hadm_id === selectedHadmId,
    );
    if (matchedEncounter) {
      return matchedEncounter;
    }
  }
  return record.encounter_summaries[0] ?? null;
}

function problemFlashReason(flash: ProblemFlash) {
  const translated = translateProblemFlash(flash.label);
  return `${translated} · ${flash.reason}`;
}

function dossierFlashDiagnostic(record: PatientDetailResponse) {
  const primaryFlash = record.top_problem_flashes[0];
  if (primaryFlash) {
    return `${translateProblemFlash(primaryFlash.label)} à prioriser dans la revue médicamenteuse.`;
  }
  if (record.patient_summary.highest_priority_label === "high") {
    return "Plusieurs expositions justifient une revue rapprochée.";
  }
  if (record.patient_summary.highest_priority_label === "medium") {
    return "Révision médicamenteuse utile au prochain point clinique.";
  }
  return "Profil sans signal majeur structuré pour le moment.";
}

function encounterContextLine(record: PatientDetailResponse) {
  const encounter = selectedEncounterSummary(record);
  if (!encounter) {
    return null;
  }
  const prefix =
    record.selected_encounter?.selection_mode === "requested_hadm_id"
      ? "Contexte sélectionné"
      : "Contexte récent";
  return `${prefix} : séjour ${encounter.hadm_id} • ${encounterSubtitle(encounter)}`;
}

function formatValue(value: string | number | null | undefined) {
  if (value == null || value === "") {
    return "Indisponible";
  }
  return String(value);
}

function yesNoUnavailable(value: boolean | null | undefined) {
  if (value == null) {
    return null;
  }
  return value ? "Présent" : "Absent";
}

function summarizeCognition(record: PatientDetailResponse) {
  if (record.left_column_context.dementia_present || record.left_column_context.delirium_present) {
    const parts: string[] = [];
    if (record.left_column_context.dementia_present) {
      parts.push("démence");
    }
    if (record.left_column_context.delirium_present) {
      parts.push("delirium");
    }
    return parts.join(" + ");
  }
  if (
    record.top_problem_flashes.some((flash) =>
      ["confusion", "cognitive_risk"].includes(flash.key) ||
      flash.label.toLowerCase().includes("confusion") ||
      flash.label.toLowerCase().includes("cognitive"),
    )
  ) {
    return "signal cognitif";
  }
  return null;
}

function summarizeFallRisk(record: PatientDetailResponse) {
  return record.top_problem_flashes.some((flash) => flash.key === "fall_risk")
    ? "Présent"
    : null;
}

function buildChronicRiskLabels(record: PatientDetailResponse) {
  const items: string[] = [];
  if (record.left_column_context.ckd_present) {
    items.push("IRC");
  }
  if (record.left_column_context.heart_failure_present) {
    items.push("insuffisance cardiaque");
  }
  if (record.left_column_context.diabetes_present) {
    items.push("diabète");
  }
  if (record.left_column_context.polypharmacy_present) {
    items.push("polypharmacie");
  }
  return items.length > 0 ? items.join(" • ") : null;
}

function ContextSection({ title, items }: ContextSectionProps) {
  const visibleItems = items.filter((item) =>
    item.hideWhenMissing ? item.value != null && item.value !== "" : true,
  );
  const hasMeaningfulValue = visibleItems.some((item) => item.value != null && item.value !== "");
  if (!hasMeaningfulValue && visibleItems.length === 0) {
    return null;
  }

  return (
    <section className="details-card clinical-context-card">
      <h2>{title}</h2>
      <dl className="detail-list clinical-context-list">
        {visibleItems.map((item) => (
          <div key={`${title}-${item.label}`}>
            <dt>{item.label}</dt>
            <dd>
              <span>{formatValue(item.value)}</span>
              {item.hint ? <small className="context-hint">{item.hint}</small> : null}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function scoreBadgeTone(label: RiskLabel) {
  if (label === "high") {
    return "med-score-badge med-score-badge-high";
  }
  if (label === "medium") {
    return "med-score-badge med-score-badge-medium";
  }
  return "med-score-badge med-score-badge-low";
}

function medicationCardTone(label: RiskLabel) {
  if (label === "high") {
    return "medication-alert-card medication-alert-card-high";
  }
  if (label === "medium") {
    return "medication-alert-card medication-alert-card-medium";
  }
  return "medication-alert-card medication-alert-card-low";
}

function routeDoseFrequencyLine(medication: PatientMedicationCard) {
  const evidence = medication.deprescribing_priority_evidence_json ?? {};
  const candidates = [
    evidence.dose,
    evidence.route,
    evidence.frequency,
    evidence.freq,
    evidence.schedule,
    evidence.sig,
  ]
    .filter((value) => value != null && value !== "")
    .map((value) => String(value));

  if (candidates.length > 0) {
    return candidates.join(" • ");
  }

  const classLine = medicationClassBadges(medication);
  if (classLine.length > 0) {
    return classLine.join(" • ");
  }

  return "Posologie détaillée indisponible";
}

function contextualReasonText(medication: PatientMedicationCard) {
  const reasons =
    medication.deprescribing_priority_reasons_json.length > 0
      ? medication.deprescribing_priority_reasons_json
      : medication.deprescribing_priority_explanation
          .split(";")
          .map((item) => item.trim())
          .filter(Boolean);

  if (reasons.length === 0) {
    return "Contexte structuré indisponible.";
  }

  return translateDriverList(reasons).join(" • ");
}

function alertSignalText(medication: PatientMedicationCard) {
  if (medication.deprescribing_priority_summary_alert.trim()) {
    return medication.deprescribing_priority_summary_alert.trim();
  }
  return translateExplanation(medication.deprescribing_priority_explanation);
}

function evidenceEntries(medication: PatientMedicationCard) {
  return Object.entries(medication.deprescribing_priority_evidence_json ?? {})
    .filter(([, value]) => {
      if (value == null) {
        return false;
      }
      if (Array.isArray(value)) {
        return value.length > 0;
      }
      if (typeof value === "object") {
        return Object.keys(value).length > 0;
      }
      return String(value).trim() !== "";
    })
    .slice(0, 6);
}

function evidenceLabel(key: string) {
  return key
    .replace(/_/g, " ")
    .replace(/\bml min 1 73m2\b/gi, "mL/min/1.73m2")
    .replace(/\begfr\b/gi, "eGFR")
    .replace(/\bhr\b/g, "HR")
    .replace(/\bbp\b/g, "BP")
    .replace(/\brass\b/gi, "RASS")
    .replace(/\b([a-z])/g, (match) => match.toUpperCase());
}

function evidenceValue(value: unknown) {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join(", ");
  }
  if (typeof value === "object" && value !== null) {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${evidenceLabel(key)}: ${String(item)}`)
      .join(" • ");
  }
  return String(value);
}

function medicationCardId(medication: PatientMedicationCard) {
  return `${medication.encounter_id ?? medication.hadm_id}-${medication.drug}-${medication.starttime}`;
}

function currentMedicationCards(record: PatientDetailResponse) {
  return record.current_medications ?? record.ranked_medication_cards;
}

function historicalMedicationCards(record: PatientDetailResponse) {
  return record.previous_medication_history ?? record.historical_medication_cards ?? [];
}

function reviewTimestampSummary(record: PatientDetailResponse) {
  const reviewTimestamp = record.review_timestamp ?? record.selected_encounter?.review_timestamp;
  if (!reviewTimestamp) {
    return null;
  }
  const source = record.selected_encounter?.review_timestamp_source;
  const sourceLabel =
    source === "medication_administration"
      ? "administrations"
      : source === "medication_order"
        ? "prescriptions"
        : source === "lab"
          ? "biologie"
          : source === "vitals"
            ? "constantes"
            : source === "encounter_end"
              ? "fin de séjour"
              : null;
  return sourceLabel
    ? `Revue au ${reviewTimestamp} • source ${sourceLabel}`
    : `Revue au ${reviewTimestamp}`;
}

function queuePriorityLabel(priority?: string | null) {
  if (priority === "disagreement_candidate") {
    return "Désaccord";
  }
  if (priority === "priority") {
    return "À revoir vite";
  }
  if (priority === "reviewed") {
    return "Déjà revu";
  }
  if (priority === "reviewable") {
    return "Revue possible";
  }
  return "Hors périmètre";
}

function queuePriorityTone(priority?: string | null) {
  if (priority === "disagreement_candidate") {
    return "chip chip-high";
  }
  if (priority === "priority") {
    return "chip chip-medium";
  }
  return "chip chip-neutral";
}

function queueReasonLabel(reason: string) {
  const labels: Record<string, string> = {
    lacks_clinician_review: "Sans revue clinicienne",
    already_reviewed: "Revue enregistrée",
    supported_medication_class: "Classe de première portée",
    constructed_label_ambiguous: "Label construit ambigu",
    rule_signal_present: "Signal règle présent",
    clinician_rule_disagreement: "Clinicien vs règle",
    not_in_phase5_review_scope: "Non aligné au grain Phase 5",
  };
  return labels[reason] ?? reason;
}

function reviewStatusLabel(status: ClinicianReviewStatus) {
  return REVIEW_STATUS_OPTIONS.find((option) => option.value === status)?.label ?? status;
}

function reasonTagLabel(tag: string) {
  return REASON_TAG_OPTIONS.find((option) => option.value === tag)?.label ?? tag;
}

function suggestedActionLabel(action?: ClinicianSuggestedAction | null) {
  if (!action) {
    return null;
  }
  return SUGGESTED_ACTION_OPTIONS.find((option) => option.value === action)?.label ?? action;
}

function savedReviewSummary(review?: ClinicianReviewRecord | null) {
  if (!review) {
    return null;
  }
  const parts = [
    `Niveau ${translateRiskLabel(review.label__clinician_priority_level)}`,
    reviewStatusLabel(review.label__clinician_review_status),
    `v${review.review_version}`,
  ];
  if (review.reviewer_id) {
    parts.push(review.reviewer_id);
  }
  return parts.join(" • ");
}

function buildMedicationReviewContext(record: PatientDetailResponse): MedicationReviewContext {
  const renal =
    record.left_column_context.latest_egfr_ml_min_1_73m2 != null
      ? `eGFR ${record.left_column_context.latest_egfr_ml_min_1_73m2}`
      : record.left_column_context.latest_creatinine_max != null
        ? `Créatinine max ${record.left_column_context.latest_creatinine_max}`
        : null;
  const burdenCount =
    record.current_medication_count ??
    record.left_column_context.current_medication_count ??
    currentMedicationCards(record).length;
  return {
    burden: `${burdenCount} médicaments actifs au temps de revue`,
    diagnoses: buildChronicRiskLabels(record),
    renal,
  };
}

function reviewerIdInitialValue() {
  if (typeof window === "undefined") {
    return "pharmacist_demo_local";
  }
  return window.localStorage.getItem(REVIEWER_ID_STORAGE_KEY) ?? "pharmacist_demo_local";
}

function PharmacistReviewPanel({
  medication,
  reviewerId,
  onReviewerIdChange,
  onReviewSaved,
  reviewContext,
}: Pick<MedicationAlertCardProps, "medication" | "reviewerId" | "onReviewerIdChange" | "onReviewSaved" | "reviewContext">) {
  const existingReview = medication.clinician_review;
  const [priorityLevel, setPriorityLevel] = useState<RiskLabel>(
    existingReview?.label__clinician_priority_level ?? medication.deprescribing_priority_label,
  );
  const [reviewStatus, setReviewStatus] = useState<ClinicianReviewStatus>(
    existingReview?.label__clinician_review_status ?? "reviewed",
  );
  const [priorityScore, setPriorityScore] = useState<string>(
    existingReview?.label__clinician_priority_score != null
      ? String(existingReview.label__clinician_priority_score)
      : "",
  );
  const [note, setNote] = useState<string>(existingReview?.label__clinician_note ?? "");
  const [suggestedAction, setSuggestedAction] = useState<ClinicianSuggestedAction | "">(
    existingReview?.label__clinician_suggested_action ?? "",
  );
  const [reasonTags, setReasonTags] = useState<string[]>(
    existingReview?.label__clinician_reason_tags ?? [],
  );
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    setPriorityLevel(existingReview?.label__clinician_priority_level ?? medication.deprescribing_priority_label);
    setReviewStatus(existingReview?.label__clinician_review_status ?? "reviewed");
    setPriorityScore(
      existingReview?.label__clinician_priority_score != null
        ? String(existingReview.label__clinician_priority_score)
        : "",
    );
    setNote(existingReview?.label__clinician_note ?? "");
    setSuggestedAction(existingReview?.label__clinician_suggested_action ?? "");
    setReasonTags(existingReview?.label__clinician_reason_tags ?? []);
    setSaveError(null);
  }, [existingReview, medication.deprescribing_priority_label]);

  function toggleReasonTag(tag: string) {
    setReasonTags((current) =>
      current.includes(tag) ? current.filter((item) => item !== tag) : [...current, tag],
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!medication.reviewable_flag || !medication.medication_standardized || !medication.review_timestamp) {
      return;
    }

    const payload: ClinicianReviewSubmissionRequest = {
      subject_id: medication.subject_id,
      encounter_id: medication.encounter_id ?? "",
      hadm_id: medication.hadm_id,
      stay_id: medication.stay_id ?? null,
      medication_standardized: medication.medication_standardized,
      review_timestamp: medication.review_timestamp,
      modeling__row_id: medication.modeling__row_id ?? null,
      reviewer_id: reviewerId.trim() || "pharmacist_demo_local",
      label__clinician_priority_level: priorityLevel,
      label__clinician_priority_score: priorityScore.trim() ? Number(priorityScore) : null,
      label__clinician_review_status: reviewStatus,
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
      await onReviewSaved();
    } catch (caughtError) {
      const detail = caughtError instanceof Error ? caughtError.message : "Erreur inattendue";
      setSaveError(`Enregistrement impossible. ${detail}`);
    } finally {
      setIsSaving(false);
    }
  }

  if (!medication.reviewable_flag || !medication.medication_standardized || !medication.review_timestamp) {
    return (
      <section className="review-panel review-panel-muted">
        <div className="review-panel-header">
          <strong>Revue clinicienne Phase 5</strong>
          <span className="chip chip-neutral">Non revuable</span>
        </div>
        <p className="details-section-copy">
          Cette ligne ne se rattache pas de façon suffisamment propre au grain analytique Phase 5
          pour créer un label clinicien traçable en aval.
        </p>
      </section>
    );
  }

  return (
    <section className="review-panel">
      <div className="review-panel-header">
        <div>
          <strong>Revue clinicienne Phase 5</strong>
          <p className="review-panel-subtitle">
            Grain: {medication.subject_id} • {medication.encounter_id} • {medication.medication_standardized}
          </p>
        </div>
        <span className={queuePriorityTone(medication.review_queue_priority)}>
          {queuePriorityLabel(medication.review_queue_priority)}
        </span>
      </div>

      <div className="review-context-grid">
        <span className="review-context-chip">Statut: {medication.medication_status ?? "Indisponible"}</span>
        <span className="review-context-chip">Charge: {reviewContext.burden}</span>
        {reviewContext.diagnoses ? (
          <span className="review-context-chip">Contexte: {reviewContext.diagnoses}</span>
        ) : null}
        {reviewContext.renal ? (
          <span className="review-context-chip">Rénal: {reviewContext.renal}</span>
        ) : null}
        <span className="review-context-chip">
          Règle dossier: {medication.deprescribing_priority_score} / {translateRiskLabel(medication.deprescribing_priority_label)}
        </span>
        {medication.benchmark__current_rule_available_flag ? (
          <span className="review-context-chip">
            Benchmark courant: {medication.benchmark__current_rule_score ?? "N/D"} /{" "}
            {medication.benchmark__current_rule_score_level
              ? translateRiskLabel(medication.benchmark__current_rule_score_level)
              : "N/D"}
          </span>
        ) : null}
      </div>

      {medication.review_queue_reasons && medication.review_queue_reasons.length > 0 ? (
        <div className="chip-row review-reasons-row">
          {medication.review_queue_reasons.map((reason) => (
            <span key={reason} className="chip chip-neutral">
              {queueReasonLabel(reason)}
            </span>
          ))}
        </div>
      ) : null}

      {existingReview ? (
        <div className="review-existing-summary">
          <span className="chip chip-neutral">{savedReviewSummary(existingReview)}</span>
          <p className="review-feedback review-feedback-success">
            Dernière sauvegarde: {existingReview.review_submission_timestamp}
          </p>
          {suggestedActionLabel(existingReview.label__clinician_suggested_action) ? (
            <p className="details-section-copy">
              Disposition: {suggestedActionLabel(existingReview.label__clinician_suggested_action)}
            </p>
          ) : null}
          {existingReview.label__clinician_reason_tags.length > 0 ? (
            <p className="details-section-copy">
              {existingReview.label__clinician_reason_tags.map(reasonTagLabel).join(" • ")}
            </p>
          ) : null}
          {existingReview.label__clinician_note ? (
            <p className="details-section-copy">{existingReview.label__clinician_note}</p>
          ) : null}
        </div>
      ) : null}

      <form className="review-form" onSubmit={handleSubmit}>
        <label className="review-field">
          <span>Identifiant relecteur</span>
          <input
            value={reviewerId}
            onChange={(event) => onReviewerIdChange(event.target.value)}
            placeholder="pharmacist_demo_local"
          />
        </label>

        <div className="review-field">
          <span>Niveau approuvé</span>
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

        <div className="review-field">
          <span>Statut</span>
          <div className="segmented-control segmented-control-wrap">
            {REVIEW_STATUS_OPTIONS.map((option) => (
              <button
                key={option.value}
                type="button"
                className={reviewStatus === option.value ? "segment-button segment-button-active" : "segment-button"}
                onClick={() => setReviewStatus(option.value)}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        <div className="review-form-grid">
          <label className="review-field">
            <span>Score numérique (0-10, optionnel)</span>
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

          <label className="review-field">
            <span>Disposition (optionnel)</span>
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
          <span>Tags raison</span>
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
            value={note}
            onChange={(event) => setNote(event.target.value)}
            rows={3}
            placeholder="Raison clinique ou point de vigilance"
          />
        </label>

        {saveError ? <p className="review-feedback review-feedback-error">{saveError}</p> : null}
        {saveMessage ? <p className="review-feedback review-feedback-success">{saveMessage}</p> : null}

        <div className="review-form-actions">
          <button type="submit" className="primary-button" disabled={isSaving}>
            {isSaving ? "Enregistrement..." : existingReview ? "Mettre à jour" : "Enregistrer"}
          </button>
          <div className="review-traceability">
            <span className="review-panel-subtitle">Trace analytique</span>
            <code>{medication.modeling__row_id ?? "indisponible"}</code>
          </div>
        </div>
      </form>
    </section>
  );
}

function MedicationAlertCard({
  medication,
  expanded,
  onToggle,
  reviewerId,
  onReviewerIdChange,
  onReviewSaved,
  reviewContext,
}: MedicationAlertCardProps) {
  const evidence = evidenceEntries(medication);
  const classBadges = medicationClassBadges(medication);
  const detailsId = `medication-details-${medicationCardId(medication)}`;

  return (
    <article className={medicationCardTone(medication.deprescribing_priority_label)}>
      <header className="medication-alert-header medication-alert-header-compact">
        <div className="medication-alert-title-block">
          <p className="medication-alert-name">{medication.drug}</p>
          <div className="medication-alert-collapsed-meta">
            <p className="medication-alert-meta">{routeDoseFrequencyLine(medication)}</p>
            {classBadges.length > 0 ? (
              <span className="chip chip-neutral">{classBadges[0]}</span>
            ) : null}
          </div>
        </div>
        <div className="medication-alert-score">
          <span className={scoreBadgeTone(medication.deprescribing_priority_label)}>
            IPD {medication.deprescribing_priority_score}
          </span>
          <span className={riskTone(medication.deprescribing_priority_label)}>
            {translateRiskLabel(medication.deprescribing_priority_label)}
          </span>
          <button
            type="button"
            className="medication-toggle"
            onClick={onToggle}
            aria-expanded={expanded}
            aria-controls={detailsId}
          >
            {expanded ? "Réduire" : "Développer"}
          </button>
        </div>
      </header>

      {!expanded ? (
        <p className="medication-alert-preview">{alertSignalText(medication)}</p>
      ) : (
        <div id={detailsId} className="medication-alert-details">
          <p className="record-subtitle">
            Séjour {medication.hadm_id} • {medication.admission_type}
            {medication.starttime ? ` • ${medication.starttime}` : ""}
            {medication.stoptime ? ` à ${medication.stoptime}` : ""}
          </p>

          <section className="medication-alert-section">
            <p className="medication-alert-section-title">Alerte signal</p>
            <p className="medication-alert-copy">{alertSignalText(medication)}</p>
          </section>

          <section className="medication-alert-section">
            <p className="medication-alert-section-title">Parce que</p>
            <p className="medication-alert-copy">{contextualReasonText(medication)}</p>
          </section>

          {evidence.length > 0 || bucketBreakdownText(medication) ? (
            <section className="medication-alert-section">
              <p className="medication-alert-section-title">Éléments contextuels</p>
              <div className="medication-evidence-list">
                {evidence.map(([key, value]) => (
                  <span key={key} className="medication-evidence-chip">
                    <strong>{evidenceLabel(key)}:</strong> {evidenceValue(value)}
                  </span>
                ))}
                {bucketBreakdownText(medication) ? (
                  <span className="medication-evidence-chip">
                    <strong>Score:</strong> {bucketBreakdownText(medication)}
                  </span>
                ) : null}
              </div>
            </section>
          ) : null}

          <PharmacistReviewPanel
            medication={medication}
            reviewerId={reviewerId}
            onReviewerIdChange={onReviewerIdChange}
            onReviewSaved={onReviewSaved}
            reviewContext={reviewContext}
          />
        </div>
      )}
    </article>
  );
}

function DecisionSupportPanel({
  title,
  description,
  items = [],
  summary,
  emptyLabel,
}: DecisionSupportPanelProps) {
  const hasContent = Boolean(summary) || items.length > 0;

  return (
    <section className="details-card decision-support-card">
      <h2>{title}</h2>
      {description ? <p className="details-section-copy">{description}</p> : null}
      {hasContent ? (
        <>
          {summary ? <p className="decision-support-summary">{summary}</p> : null}
          {items.length > 0 ? (
            <ul className="decision-support-list">
              {items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          ) : null}
        </>
      ) : (
        <div className="decision-support-empty">
          <span className="chip chip-neutral">Aide secondaire</span>
          <p>{emptyLabel}</p>
        </div>
      )}
    </section>
  );
}

export function RecordDetailsPage() {
  const { subjectId = "" } = useParams();
  const [searchParams] = useSearchParams();
  const selectedHadmIdParam = searchParams.get("hadm_id");
  const selectedHadmId =
    selectedHadmIdParam && /^\d+$/.test(selectedHadmIdParam) ? Number(selectedHadmIdParam) : null;

  const [record, setRecord] = useState<PatientDetailResponse | null>(null);
  const [expandedMedicationIds, setExpandedMedicationIds] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reviewerId, setReviewerId] = useState<string>(reviewerIdInitialValue);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(REVIEWER_ID_STORAGE_KEY, reviewerId);
    }
  }, [reviewerId]);

  async function loadRecord(options?: { resetExpanded?: boolean }) {
    setIsLoading(true);
    setError(null);
    try {
      const nextRecord = await getPatientDetail(subjectId, { hadmId: selectedHadmId });
      setRecord(nextRecord);
      if (options?.resetExpanded ?? true) {
        setExpandedMedicationIds([]);
      }
    } catch (caughtError) {
      const detail =
        caughtError instanceof Error && caughtError.message
          ? ` Détail: ${caughtError.message}`
          : "";
      setError(`Impossible de charger le dossier patient sélectionné.${detail}`);
    } finally {
      setIsLoading(false);
    }
  }

  useEffect(() => {
    void loadRecord({ resetExpanded: true });
  }, [selectedHadmId, subjectId]);

  function toggleMedication(cardId: string) {
    setExpandedMedicationIds((current) =>
      current.includes(cardId)
        ? current.filter((item) => item !== cardId)
        : [...current, cardId],
    );
  }

  function expandAllMedications() {
    if (!record) {
      return;
    }
    setExpandedMedicationIds(currentMedicationCards(record).map(medicationCardId));
  }

  function collapseAllMedications() {
    setExpandedMedicationIds([]);
  }

  return (
    <main className="page-shell">
      <div className="details-header">
        <Link to="/" className="back-link">
          ← Retour aux patients
        </Link>
      </div>

      {isLoading ? <section className="state-panel">Chargement du dossier patient...</section> : null}

      {!isLoading && error ? (
        <section className="state-panel error-panel">
          <strong>Impossible de charger le dossier patient.</strong>
          <p>{error}</p>
        </section>
      ) : null}

      {!isLoading && record ? (
        <section className="details-layout">
          {(() => {
            const currentMeds = currentMedicationCards(record);
            const historicalMeds = historicalMedicationCards(record);
            const currentMedicationCount =
              record.current_medication_count ??
              record.left_column_context.current_medication_count ??
              currentMeds.length;
            const currentFlaggedMedicationCount =
              record.current_flagged_medication_count ?? record.flagged_medication_count;
            const reviewSummary = reviewTimestampSummary(record);
            const reviewContext = buildMedicationReviewContext(record);
            const reviewQueueSummary: ReviewQueueSummary = record.review_queue_summary ?? {
              reviewable_rows: 0,
              reviewed_rows: 0,
              unreviewed_rows: 0,
              priority_rows: 0,
              disagreement_candidate_rows: 0,
            };

            return (
          <article className="details-main">
            <section className="details-hero">
              <div className="details-hero-copy">
                <p className="eyebrow">Dossier patient</p>
                <h1>Dossier {record.patient_summary.subject_id}</h1>
                <p className="hero-copy">
                  {record.left_column_context.sex ?? "Sexe N/D"}
                  {record.left_column_context.age_proxy != null
                    ? ` • Âge ${record.left_column_context.age_proxy}`
                    : ""}
                  {record.left_column_context.age_group
                    ? ` (${record.left_column_context.age_group})`
                    : ""}
                </p>
                <p className="dossier-summary">{dossierFlashDiagnostic(record)}</p>
                {encounterContextLine(record) ? (
                  <p className="record-subtitle">{encounterContextLine(record)}</p>
                ) : null}
              </div>

              <div className="dossier-hero-metrics">
                <div className="score-spotlight">
                  <span className="summary-label">Risque global</span>
                  <strong>{record.patient_summary.highest_priority_score}</strong>
                  <span className={riskTone(record.patient_summary.highest_priority_label)}>
                    {translateRiskLabel(record.patient_summary.highest_priority_label)}
                  </span>
                </div>

                <div className="details-stat-card">
                  <span className="summary-label">Médicaments actuels signalés</span>
                  <strong>{currentFlaggedMedicationCount}</strong>
                  <span className="score-spotlight-subcopy">
                    sur {currentMedicationCount} médicaments actuellement pris
                  </span>
                </div>

                <div className="details-stat-card">
                  <span className="summary-label">Flash dominant</span>
                  <strong>{record.top_problem_flashes.length}</strong>
                  <span className="score-spotlight-subcopy">
                    {record.top_problem_flashes.length > 0
                      ? translateProblemFlash(record.top_problem_flashes[0].label)
                      : "Aucun flash structuré"}
                  </span>
                </div>
              </div>
            </section>

            <div className="dossier-columns">
              <aside className="details-column details-column-left">
                <ContextSection
                  title="Identity & Morphology"
                  items={[
                    { label: "ID patient", value: record.patient_summary.subject_id },
                    {
                      label: "Âge",
                      value:
                        record.left_column_context.age_proxy != null
                          ? `${record.left_column_context.age_proxy}${
                              record.left_column_context.age_group
                                ? ` (${record.left_column_context.age_group})`
                                : ""
                            }`
                          : null,
                    },
                    { label: "Sexe", value: record.left_column_context.sex },
                    {
                      label: "Poids",
                      value:
                        record.left_column_context.latest_weight_kg != null
                          ? `${record.left_column_context.latest_weight_kg} kg`
                          : null,
                      hint: record.left_column_context.latest_weight_provenance ?? "Dernière valeur disponible",
                    },
                    {
                      label: "IMC",
                      value:
                        record.left_column_context.latest_bmi != null
                          ? `${record.left_column_context.latest_bmi}`
                          : null,
                      hint: record.left_column_context.latest_bmi_provenance ?? "Dernière valeur disponible",
                    },
                  ]}
                />

                <ContextSection
                  title="Clinical Profile / Geriatric Syndromes"
                  items={[
                    { label: "Cognition", value: summarizeCognition(record) },
                    { label: "Risque de chute", value: summarizeFallRisk(record) },
                    {
                      label: "Fragilité",
                      value: yesNoUnavailable(record.left_column_context.frailty_present),
                    },
                    {
                      label: "Risques chroniques",
                      value: buildChronicRiskLabels(record),
                    },
                  ]}
                />

                <ContextSection
                  title="Organic Reserve"
                  items={[
                    {
                      label: "Réserve rénale",
                      value:
                        record.left_column_context.latest_egfr_ml_min_1_73m2 != null
                          ? `eGFR ${record.left_column_context.latest_egfr_ml_min_1_73m2}`
                          : record.left_column_context.latest_creatinine_max != null
                            ? `Créatinine max ${record.left_column_context.latest_creatinine_max}`
                            : null,
                      hint: record.left_column_context.latest_renal_provenance ?? "Dernière valeur disponible",
                    },
                    {
                      label: "Signal hépatique",
                      value: record.left_column_context.hepatic_signal_summary,
                    },
                    {
                      label: "Potassium",
                      value: record.left_column_context.baseline_potassium_summary,
                      hint: record.left_column_context.baseline_potassium_summary ? "Valeur de base ou meilleure valeur disponible" : null,
                    },
                    {
                      label: "Sodium",
                      value: record.left_column_context.baseline_sodium_summary,
                      hint: record.left_column_context.baseline_sodium_summary ? "Valeur de base ou meilleure valeur disponible" : null,
                    },
                  ]}
                />

                <ContextSection
                  title="Live Clinical State"
                  items={[
                    {
                      label: "Vigilance / RASS",
                      value:
                        record.left_column_context.rass_summary ??
                        record.left_column_context.vigilance_summary,
                    },
                    {
                      label: "Douleur",
                      value: record.left_column_context.pain_summary,
                    },
                    {
                      label: "Stabilité hémodynamique",
                      value: record.left_column_context.hemodynamic_stability_summary,
                    },
                  ]}
                />

                <ContextSection
                  title="Goals / Care Philosophy"
                  items={[
                    {
                      label: "Niveau de soins",
                      value: record.left_column_context.level_of_care,
                    },
                    {
                      label: "Dysphagie",
                      value: yesNoUnavailable(record.left_column_context.dysphagia_present),
                    },
                    {
                      label: "Voie d’administration / alimentation",
                      value: record.left_column_context.feeding_route_summary,
                    },
                    {
                      label: "Contraintes d’administration",
                      value: record.left_column_context.administration_constraints_summary,
                    },
                  ]}
                />
              </aside>

              <section className="details-column details-column-center">
                <section className="details-card details-card-emphasis">
                  <div className="medication-list-header">
                    <div>
                      <h2>Médicaments actuellement pris</h2>
                      <p className="details-section-copy">
                        {currentFlaggedMedicationCount} signalé
                        {currentFlaggedMedicationCount === 1 ? "" : "s"} sur {currentMedicationCount} médicament
                        {currentMedicationCount === 1 ? "" : "s"} actif
                        {currentMedicationCount === 1 ? "" : "s"}.
                      </p>
                      {reviewSummary ? (
                        <p className="record-subtitle">{reviewSummary}</p>
                      ) : (
                        <p className="record-subtitle">
                          Revue centrée sur les médicaments actifs à l’instant clinique retenu.
                        </p>
                      )}
                      <p className="details-section-copy">
                        Tri décroissant par priorité actuelle de revue pharmaco-clinique.
                      </p>
                      <p className="details-section-copy">
                        Repères de tri heuristiques pour trouver plus vite les cas utiles à revoir.
                        Ils n’impliquent ni certitude de modèle ni validation clinique complète.
                      </p>
                      <div className="chip-row review-queue-summary-row">
                        <span className="chip chip-neutral">
                          Revuables {reviewQueueSummary.reviewable_rows}
                        </span>
                        <span className="chip chip-neutral">
                          Déjà revus {reviewQueueSummary.reviewed_rows}
                        </span>
                        <span className="chip chip-medium">
                          Prioritaires {reviewQueueSummary.priority_rows}
                        </span>
                        {reviewQueueSummary.disagreement_candidate_rows > 0 ? (
                          <span className="chip chip-high">
                            Désaccords {reviewQueueSummary.disagreement_candidate_rows}
                          </span>
                        ) : null}
                      </div>
                    </div>
                    <div className="medication-list-controls">
                      <button type="button" className="list-control-button" onClick={expandAllMedications}>
                        Tout développer
                      </button>
                      <button type="button" className="list-control-button" onClick={collapseAllMedications}>
                        Tout réduire
                      </button>
                    </div>
                  </div>
                  <div className="medication-scroll-panel">
                    <div className="medications-list medications-list-priority">
                      {currentMeds.length > 0 ? (
                        currentMeds.map((medication) => (
                          <MedicationAlertCard
                            key={medicationCardId(medication)}
                            medication={medication}
                            expanded={expandedMedicationIds.includes(medicationCardId(medication))}
                            onToggle={() => toggleMedication(medicationCardId(medication))}
                            reviewerId={reviewerId}
                            onReviewerIdChange={setReviewerId}
                            onReviewSaved={() => loadRecord({ resetExpanded: false })}
                            reviewContext={reviewContext}
                          />
                        ))
                      ) : (
                        <section className="state-panel">
                          Aucun médicament actif au temps de revue retenu.
                        </section>
                      )}
                    </div>
                  </div>
                </section>

                {historicalMeds.length > 0 ? (
                  <section className="details-card">
                    <details>
                      <summary>
                        Historique médicamenteux antérieur ({record.historical_medication_count ?? historicalMeds.length})
                      </summary>
                      <p className="details-section-copy">
                        Médicaments non actifs au temps de revue, conservés pour contexte clinique.
                      </p>
                      <div className="medications-list medications-list-priority">
                        {historicalMeds.map((medication) => (
                          <MedicationAlertCard
                            key={`history-${medicationCardId(medication)}`}
                            medication={medication}
                            expanded={expandedMedicationIds.includes(`history-${medicationCardId(medication)}`)}
                            onToggle={() => toggleMedication(`history-${medicationCardId(medication)}`)}
                            reviewerId={reviewerId}
                            onReviewerIdChange={setReviewerId}
                            onReviewSaved={() => loadRecord({ resetExpanded: false })}
                            reviewContext={reviewContext}
                          />
                        ))}
                      </div>
                    </details>
                  </section>
                ) : null}
              </section>

              <aside className="details-column details-column-right">
                <section className="details-card decision-support-card">
                  <h2>Rail d’aide à la décision</h2>
                  <div className="chip-row">
                    {record.top_problem_flashes.map((flash) => (
                      <span key={`${flash.key}-${flash.label}`} className={riskTone(flash.severity)}>
                        {translateProblemFlash(flash.label)}
                      </span>
                    ))}
                  </div>
                  <div className="explanation-panel">
                    <ul className="explanation-list">
                      {record.top_problem_flashes.map((flash) => (
                        <li key={flash.key}>{problemFlashReason(flash)}</li>
                      ))}
                    </ul>
                  </div>
                </section>

                <DecisionSupportPanel
                  title="Temps jusqu’au bénéfice"
                  description="Repère d’aide pour mettre en balance bénéfice attendu et risque immédiat."
                  summary={record.time_to_benefit_summary}
                  items={record.time_to_benefit_note ? [record.time_to_benefit_note] : []}
                  emptyLabel="Analyse non encore disponible."
                />

                <DecisionSupportPanel
                  title="Taper / protocole de déprescription"
                  description="Étapes de décroissance ou de sevrage lorsque la recommandation est structurée."
                  summary={record.taper_protocol_summary}
                  items={record.taper_protocol_steps ?? []}
                  emptyLabel="Aucun protocole structuré disponible pour l’instant."
                />

                <DecisionSupportPanel
                  title="Validation par les pairs / support de pratique"
                  description="Zone prévue pour références Beers, STOPP-START ou appuis de consensus."
                  summary={record.peer_validation_summary}
                  items={record.peer_validation_references ?? []}
                  emptyLabel="Support de pratique non encore câblé."
                />

                <DecisionSupportPanel
                  title="Symptômes perçus par le patient"
                  description="Signaux rapportés par le patient utiles pour contextualiser la révision thérapeutique."
                  summary={record.patient_symptom_summary}
                  items={record.patient_perceived_symptoms ?? []}
                  emptyLabel="Aucun symptôme patient structuré disponible."
                />

                <section className="details-card decision-support-card">
                  <h2>Contexte de séjour</h2>
                  <p className="details-section-copy">
                    Les séjours restent disponibles comme support clinique secondaire.
                  </p>
                  <div className="medications-list encounter-list">
                    {record.encounter_summaries.map((encounter) => (
                      <article key={encounter.hadm_id} className="medication-card">
                        <div className="record-topline">
                          <div>
                            <p className="record-title">Séjour {encounter.hadm_id}</p>
                            <p className="record-subtitle">{encounterSubtitle(encounter)}</p>
                          </div>
                          <span className={riskTone(encounter.overall_priority_label)}>
                            {translateRiskLabel(encounter.overall_priority_label)} •{" "}
                            {encounter.overall_priority_score}
                          </span>
                        </div>
                        <div className="metric-grid metric-grid-compact">
                          <div className="metric-item">
                            <span className="metric-label">Médicaments</span>
                            <strong>{encounter.current_medication_count ?? encounter.total_medication_count}</strong>
                          </div>
                          <div className="metric-item">
                            <span className="metric-label">Signalés</span>
                            <strong>{encounter.flagged_medication_count}</strong>
                          </div>
                        </div>
                        <div className="chip-row">
                          {translateDriverList(encounter.overall_priority_drivers).map((driver) => (
                            <span key={driver} className="chip chip-neutral">
                              {driver}
                            </span>
                          ))}
                        </div>
                      </article>
                    ))}
                  </div>
                </section>
              </aside>
            </div>
          </article>
            );
          })()}
        </section>
      ) : null}
    </main>
  );
}
