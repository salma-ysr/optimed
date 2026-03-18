import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getAdmissionDetail } from "../api/client";
import type { AdmissionDetailResponse, MedicationRowSummary, RiskLabel } from "../types";

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
  return medication.medication_classes;
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
        setError(
          caughtError instanceof Error
            ? caughtError.message
            : "Failed to load the selected admission.",
        );
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
          ← Back to admissions
        </Link>
      </div>

      {isLoading ? <section className="state-panel">Loading admission review...</section> : null}

      {!isLoading && error ? (
        <section className="state-panel error-panel">
          <strong>Unable to load the admission review.</strong>
          <p>{error}</p>
        </section>
      ) : null}

      {!isLoading && record ? (
        <section className="details-layout">
          <article className="details-main">
            <section className="details-hero">
              <div>
                <p className="eyebrow">Admission Review</p>
                <h1>
                  Patient {record.subject_id} / Admission {record.hadm_id}
                </h1>
                <p className="hero-copy">
                  {record.sex} • Age {record.age_proxy} ({record.age_group}) •{" "}
                  {record.admission_type}
                </p>
              </div>
              <div className="score-spotlight">
                <span className="summary-label">Overall priority</span>
                <strong>{record.highest_priority_score}</strong>
                <span className={riskTone(record.highest_priority_label)}>
                  {record.highest_priority_label}
                </span>
                <span className="score-spotlight-subcopy">
                  {record.flagged_medication_count} flagged medication
                  {record.flagged_medication_count === 1 ? "" : "s"}
                </span>
              </div>
            </section>

            <div className="details-grid">
              <section className="details-card details-card-wide details-card-emphasis">
                <h2>Recommended Review</h2>
                <p className="details-section-copy">
                  Highest-priority medications to reconsider first for this admission.
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
                              ? medication.medication_classes.join(" • ")
                              : "Medication review"}
                          </p>
                        </div>
                        <span className={riskTone(medication.deprescribing_priority_label)}>
                          {medication.deprescribing_priority_label} •{" "}
                          {medication.deprescribing_priority_score}
                        </span>
                      </div>
                      <p className="medication-explanation">
                        {medication.deprescribing_priority_explanation}
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
                <h2>Deprescribing Score Summary</h2>
                <div className="chip-row">
                  {record.overall_priority_drivers.map((driver) => (
                    <span key={driver} className="chip chip-neutral">
                      {driver}
                    </span>
                  ))}
                </div>
                <div className="explanation-panel">
                  <p className="explanation-heading">Main triggered rules</p>
                  <ul className="explanation-list">
                    {record.score_explanations.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </section>

              <section className="details-card">
                <h2>Patient Overview</h2>
                <dl className="detail-list">
                  <div><dt>Subject ID</dt><dd>{record.subject_id}</dd></div>
                  <div><dt>Admission ID</dt><dd>{record.hadm_id}</dd></div>
                  <div><dt>Sex</dt><dd>{record.sex}</dd></div>
                  <div><dt>Age</dt><dd>{record.age_proxy}</dd></div>
                  <div><dt>Age group</dt><dd>{record.age_group}</dd></div>
                </dl>
              </section>

              <section className="details-card">
                <h2>Hospitalization Details</h2>
                <dl className="detail-list">
                  <div><dt>Admission type</dt><dd>{record.admission_type}</dd></div>
                  <div><dt>Admit time</dt><dd>{record.admittime}</dd></div>
                  <div><dt>Discharge time</dt><dd>{record.dischtime}</dd></div>
                  <div><dt>Length of stay</dt><dd>{record.length_of_stay_days.toFixed(1)} days</dd></div>
                  <div><dt>Medication count</dt><dd>{record.total_medication_count}</dd></div>
                  <div><dt>Polypharmacy</dt><dd>{record.polypharmacy_flag ? "Yes" : "No"}</dd></div>
                </dl>
              </section>

              <section className="details-card">
                <h2>Biological Values</h2>
                <dl className="detail-list">
                  <div><dt>Creatinine first</dt><dd>{record.creatinine_first ?? "N/A"}</dd></div>
                  <div><dt>Creatinine max</dt><dd>{record.creatinine_max ?? "N/A"}</dd></div>
                  <div><dt>Creatinine mean</dt><dd>{record.creatinine_mean ?? "N/A"}</dd></div>
                  <div><dt>Renal risk</dt><dd>{record.renal_risk_flag ? "Flagged" : "No"}</dd></div>
                </dl>
              </section>

              <section className="details-card">
                <h2>Clinical Risk Factors</h2>
                <div className="chip-row">
                  {record.ckd_flag ? <span className="chip chip-neutral">CKD</span> : null}
                  {record.dementia_flag ? <span className="chip chip-neutral">Dementia</span> : null}
                  {record.delirium_flag ? <span className="chip chip-neutral">Delirium</span> : null}
                  {record.heart_failure_flag ? <span className="chip chip-neutral">Heart failure</span> : null}
                  {record.diabetes_flag ? <span className="chip chip-neutral">Diabetes</span> : null}
                  {record.renal_risk_flag ? <span className="chip chip-neutral">Renal risk</span> : null}
                  {record.polypharmacy_flag ? <span className="chip chip-neutral">Polypharmacy</span> : null}
                </div>
              </section>

              <section className="details-card details-card-wide">
                <h2>Medication Review</h2>
                <p className="details-section-copy">
                  Full medication list for this admission, ordered by highest review priority.
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
                            {medication.stoptime ? ` to ${medication.stoptime}` : ""}
                            {medication.medication_classes.length > 0
                              ? ` • ${medication.medication_classes.join(" • ")}`
                              : ""}
                          </p>
                        </div>
                        <span className={riskTone(medication.deprescribing_priority_label)}>
                          {medication.deprescribing_priority_label} •{" "}
                          {medication.deprescribing_priority_score}
                        </span>
                      </div>
                      <p className="medication-explanation">
                        {medication.deprescribing_priority_explanation}
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
