import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getAdmissionDetail } from "../api/client";
import type { AdmissionDetailResponse, MedicationRowSummary, RiskLabel } from "../types";
import {
  translateDriverList,
  translateExplanation,
  translateMedicationClass,
  translateRiskLabel,
} from "../uiText";

function riskTone(label: RiskLabel) {
  if (label === "high") {
    return "chip chip-high";
  }
  if (label === "medium") {
    return "chip chip-medium";
  }
  return "chip chip-low";
}

function medicationClassBadges(medication: MedicationRowSummary) {
  return medication.medication_classes.map(translateMedicationClass);
}

export function RecordDetailsPage() {
  const { subjectId = "", hadmId = "" } = useParams();

  const [record, setRecord] = useState<AdmissionDetailResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadRecord() {
      setIsLoading(true);
      setError(null);
      try {
        const nextRecord = await getAdmissionDetail({ subjectId, hadmId });
        setRecord(nextRecord);
      } catch (caughtError) {
        const detail =
          caughtError instanceof Error && caughtError.message
            ? ` Détail: ${caughtError.message}`
            : "";
        setError(`Impossible de charger l’hospitalisation sélectionnée.${detail}`);
      } finally {
        setIsLoading(false);
      }
    }

    void loadRecord();
  }, [hadmId, subjectId]);

  return (
    <main className="page-shell">
      <div className="details-header">
        <Link to="/" className="back-link">
          ← Retour aux hospitalisations
        </Link>
      </div>

      {isLoading ? <section className="state-panel">Chargement de la revue d’hospitalisation...</section> : null}

      {!isLoading && error ? (
        <section className="state-panel error-panel">
          <strong>Impossible de charger la revue d’hospitalisation.</strong>
          <p>{error}</p>
        </section>
      ) : null}

      {!isLoading && record ? (
        <section className="details-layout">
          <article className="details-main">
            <section className="details-hero">
              <div>
                <p className="eyebrow">Revue d’hospitalisation</p>
                <h1>
                  Patient {record.subject_id} / Admission {record.hadm_id}
                </h1>
                <p className="hero-copy">
                  {record.sex} • Âge {record.age_proxy} ({record.age_group}) •{" "}
                  {record.admission_type}
                </p>
              </div>
              <div className="score-spotlight">
                <span className="summary-label">Priorité globale</span>
                <strong>{record.highest_priority_score}</strong>
                <span className={riskTone(record.highest_priority_label)}>
                  {translateRiskLabel(record.highest_priority_label)}
                </span>
                <span className="score-spotlight-subcopy">
                  {record.flagged_medication_count} médicament
                  {record.flagged_medication_count === 1 ? " signalé" : "s signalés"}
                </span>
              </div>
            </section>

            <div className="details-grid">
              <section className="details-card details-card-wide details-card-emphasis">
                <h2>Revue recommandée</h2>
                <p className="details-section-copy">
                  Médicaments les plus prioritaires à réévaluer en premier pour cette hospitalisation.
                </p>
                <div className="medications-list medications-list-priority">
                  {record.flagged_medications.map((medication) => (
                    <article
                      key={`${medication.drug}-${medication.starttime}-priority`}
                      className="medication-card medication-card-priority"
                    >
                      <div className="record-topline">
                        <div>
                          <p className="record-title">{medication.drug}</p>
                          <p className="record-subtitle">
                            {medication.medication_classes.length > 0
                              ? medication.medication_classes.map(translateMedicationClass).join(" • ")
                              : "Revue médicamenteuse"}
                          </p>
                        </div>
                        <span className={riskTone(medication.deprescribing_priority_label)}>
                          {translateRiskLabel(medication.deprescribing_priority_label)} •{" "}
                          {medication.deprescribing_priority_score}
                        </span>
                      </div>
                      <p className="medication-explanation">
                        {translateExplanation(medication.deprescribing_priority_explanation)}
                      </p>
                      <div className="chip-row">
                        {medicationClassBadges(medication).map((label) => (
                          <span key={label} className="chip chip-neutral">
                            {label}
                          </span>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              </section>

              <section className="details-card details-card-wide">
                <h2>Résumé du score de déprescription</h2>
                <div className="chip-row">
                  {translateDriverList(record.overall_priority_drivers).map((driver) => (
                    <span key={driver} className="chip chip-neutral">
                      {driver}
                    </span>
                  ))}
                </div>
                <div className="explanation-panel">
                  <p className="explanation-heading">Principales règles déclenchées</p>
                  <ul className="explanation-list">
                    {translateDriverList(record.score_explanations).map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </section>

              <section className="details-card">
                <h2>Vue d’ensemble du patient</h2>
                <dl className="detail-list">
                  <div><dt>ID patient</dt><dd>{record.subject_id}</dd></div>
                  <div><dt>ID admission</dt><dd>{record.hadm_id}</dd></div>
                  <div><dt>Sexe</dt><dd>{record.sex}</dd></div>
                  <div><dt>Âge</dt><dd>{record.age_proxy}</dd></div>
                  <div><dt>Groupe d’âge</dt><dd>{record.age_group}</dd></div>
                </dl>
              </section>

              <section className="details-card">
                <h2>Détails de l’hospitalisation</h2>
                <dl className="detail-list">
                  <div><dt>Type d’admission</dt><dd>{record.admission_type}</dd></div>
                  <div><dt>Date d’entrée</dt><dd>{record.admittime}</dd></div>
                  <div><dt>Date de sortie</dt><dd>{record.dischtime}</dd></div>
                  <div><dt>Durée de séjour</dt><dd>{record.length_of_stay_days.toFixed(1)} jours</dd></div>
                  <div><dt>Nombre de médicaments</dt><dd>{record.total_medication_count}</dd></div>
                  <div><dt>Polypharmacie</dt><dd>{record.polypharmacy_flag ? "Oui" : "Non"}</dd></div>
                </dl>
              </section>

              <section className="details-card">
                <h2>Valeurs biologiques</h2>
                <dl className="detail-list">
                  <div><dt>Créatinine initiale</dt><dd>{record.creatinine_first ?? "N/D"}</dd></div>
                  <div><dt>Créatinine max</dt><dd>{record.creatinine_max ?? "N/D"}</dd></div>
                  <div><dt>Créatinine moyenne</dt><dd>{record.creatinine_mean ?? "N/D"}</dd></div>
                  <div><dt>Risque rénal</dt><dd>{record.renal_risk_flag ? "Signalé" : "Non"}</dd></div>
                </dl>
              </section>

              <section className="details-card">
                <h2>Facteurs de risque cliniques</h2>
                <div className="chip-row">
                  {record.ckd_flag ? <span className="chip chip-neutral">IRC</span> : null}
                  {record.dementia_flag ? <span className="chip chip-neutral">Démence</span> : null}
                  {record.delirium_flag ? <span className="chip chip-neutral">Delirium</span> : null}
                  {record.heart_failure_flag ? <span className="chip chip-neutral">Insuffisance cardiaque</span> : null}
                  {record.diabetes_flag ? <span className="chip chip-neutral">Diabète</span> : null}
                  {record.renal_risk_flag ? <span className="chip chip-neutral">Risque rénal</span> : null}
                  {record.polypharmacy_flag ? <span className="chip chip-neutral">Polypharmacie</span> : null}
                </div>
              </section>

              <section className="details-card details-card-wide">
                <h2>Revue médicamenteuse</h2>
                <p className="details-section-copy">
                  Liste complète des médicaments de cette hospitalisation, ordonnée par priorité de révision.
                </p>
                <div className="medications-list">
                  {record.medications.map((medication) => (
                    <article
                      key={`${medication.drug}-${medication.starttime}`}
                      className="medication-card"
                    >
                      <div className="record-topline">
                        <div>
                          <p className="record-title">{medication.drug}</p>
                          <p className="record-subtitle">
                            {medication.starttime}
                            {medication.stoptime ? ` à ${medication.stoptime}` : ""}
                            {medication.medication_classes.length > 0
                              ? ` • ${medication.medication_classes.map(translateMedicationClass).join(" • ")}`
                              : ""}
                          </p>
                        </div>
                        <span className={riskTone(medication.deprescribing_priority_label)}>
                          {translateRiskLabel(medication.deprescribing_priority_label)} •{" "}
                          {medication.deprescribing_priority_score}
                        </span>
                      </div>
                      <p className="medication-explanation">
                        {translateExplanation(medication.deprescribing_priority_explanation)}
                      </p>
                      <div className="chip-row">
                        {medicationClassBadges(medication).map((label) => (
                          <span key={label} className="chip chip-neutral">
                            {label}
                          </span>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              </section>
            </div>
          </article>
        </section>
      ) : null}
    </main>
  );
}
