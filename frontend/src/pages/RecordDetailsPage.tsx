import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getPatientDetail } from "../api/client";
import type {
  PatientDetailResponse,
  PatientEncounterSummary,
  PatientMedicationCard,
  ProblemFlash,
  RiskLabel,
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
};

type DecisionSupportPanelProps = {
  title: string;
  description?: string;
  items?: string[];
  summary?: string | null;
  emptyLabel: string;
};

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

function encounterContextLine(encounters: PatientEncounterSummary[]) {
  const latestEncounter = encounters[0];
  if (!latestEncounter) {
    return null;
  }
  return `Contexte récent : séjour ${latestEncounter.hadm_id} • ${encounterSubtitle(latestEncounter)}`;
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
  return `${medication.hadm_id}-${medication.drug}-${medication.starttime}`;
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

function MedicationAlertCard({ medication, expanded, onToggle }: MedicationAlertCardProps) {
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

  const [record, setRecord] = useState<PatientDetailResponse | null>(null);
  const [expandedMedicationIds, setExpandedMedicationIds] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadRecord() {
      setIsLoading(true);
      setError(null);
      try {
        const nextRecord = await getPatientDetail(subjectId);
        setRecord(nextRecord);
        setExpandedMedicationIds([]);
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

    void loadRecord();
  }, [subjectId]);

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
                {encounterContextLine(record.encounter_summaries) ? (
                  <p className="record-subtitle">{encounterContextLine(record.encounter_summaries)}</p>
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
