import type { RiskLabel } from "./types";

const explanationMap: Record<string, string> = {
  "very high medication burden (10+ distinct medications)":
    "charge médicamenteuse très élevée (10 médicaments distincts ou plus)",
  "very high current medication burden (10+ active medications)":
    "charge médicamenteuse actuelle très élevée (10 médicaments actifs ou plus)",
  "polypharmacy during admission": "polypharmacie pendant l’hospitalisation",
  "current polypharmacy at encounter review time":
    "polypharmacie actuelle au temps de revue de la rencontre",
  "renal risk based on creatinine": "risque rénal basé sur la créatinine",
  "benzodiazepine exposure": "exposition à une benzodiazépine",
  "opioid exposure": "exposition à un opioïde",
  "anticholinergic exposure": "exposition à un anticholinergique",
  "antipsychotic exposure": "exposition à un antipsychotique",
  "proton pump inhibitor exposure": "exposition à un inhibiteur de la pompe à protons",
  "chronic kidney disease diagnosis": "diagnostic d’insuffisance rénale chronique",
  "dementia diagnosis": "diagnostic de démence",
  "delirium diagnosis": "diagnostic de delirium",
  "heart failure diagnosis": "diagnostic d’insuffisance cardiaque",
  "diabetes diagnosis": "diagnostic de diabète",
  "reduced renal reserve or renal vulnerability": "réserve rénale réduite ou vulnérabilité rénale",
  "cognitive vulnerability from dementia or delirium context":
    "vulnérabilité cognitive liée à la démence ou au delirium",
  "cardiac or metabolic comorbidity may narrow medication tolerance":
    "le terrain cardiaque ou métabolique peut réduire la tolérance médicamenteuse",
  "advanced age or low reserve may reduce physiologic tolerance":
    "l’âge avancé ou une faible réserve peuvent réduire la tolérance physiologique",
  "creatinine is rising during the encounter": "la créatinine augmente pendant la rencontre",
  "electrolyte abnormality may increase medication-related instability":
    "une anomalie électrolytique peut accroître l’instabilité liée aux médicaments",
  "blood pressure or heart-rate instability is present":
    "une instabilité tensionnelle ou du rythme cardiaque est présente",
  "high pain burden may complicate deprescribing prioritization":
    "une douleur importante peut compliquer la priorisation de déprescription",
  "Rising creatinine or renal vulnerability is present.":
    "une créatinine en hausse ou une vulnérabilité rénale est présente.",
  "Sedating medication exposure is present.":
    "une exposition à des médicaments sédatifs est présente.",
  "Fall-prone medication exposure is present.":
    "une exposition à des médicaments favorisant les chutes est présente.",
  "no major structured IPD rule triggered": "aucune règle IPD structurée majeure déclenchée",
  "no major rule triggered": "aucune règle majeure déclenchée",
};

const bucketLabelMap: Record<string, string> = {
  base_medication_risk: "Risque médicamenteux de base",
  terrain_aggravating_context: "Terrain aggravant",
  dynamic_biologic_vital_evidence: "Signaux biologiques et vitaux",
};

const medicationClassMap: Record<string, string> = {
  Benzodiazepine: "Benzodiazépine",
  Opioid: "Opioïde",
  Anticholinergic: "Anticholinergique",
  Antipsychotic: "Antipsychotique",
  PPI: "IPP",
};

const problemFlashMap: Record<string, string> = {
  "Renal toxicity": "Toxicité rénale",
  Oversedation: "Sursédation",
  "Fall risk": "Risque de chute",
  Confusion: "Confusion",
  "Cognitive risk": "Risque cognitif",
  "Confusion / cognitive risk": "Confusion / risque cognitif",
};

export function translateRiskLabel(label: RiskLabel): string {
  if (label === "high") {
    return "Risque élevé";
  }
  if (label === "medium") {
    return "Risque modéré";
  }
  return "Risque faible";
}

export function translateExplanation(text: string): string {
  return text
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => explanationMap[part] ?? part)
    .join(" ; ");
}

export function translateDriverList(items: string[]): string[] {
  return items.map((item) => explanationMap[item] ?? item);
}

export function translateMedicationClass(label: string): string {
  return medicationClassMap[label] ?? label;
}

export function translateBucketLabel(label: string): string {
  return bucketLabelMap[label] ?? label;
}

export function translateProblemFlash(label: string): string {
  return problemFlashMap[label] ?? label;
}
