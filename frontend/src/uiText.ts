import type { RiskLabel } from "./types";

const explanationMap: Record<string, string> = {
  "very high medication burden (10+ distinct medications)":
    "charge médicamenteuse très élevée (10 médicaments distincts ou plus)",
  "polypharmacy during admission": "polypharmacie pendant l’hospitalisation",
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
  "no major rule triggered": "aucune règle majeure déclenchée",
};

const medicationClassMap: Record<string, string> = {
  Benzodiazepine: "Benzodiazépine",
  Opioid: "Opioïde",
  Anticholinergic: "Anticholinergique",
  Antipsychotic: "Antipsychotique",
  PPI: "IPP",
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
