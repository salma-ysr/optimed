"""Curated medication keyword mappings for MVP high-risk classes."""

HIGH_RISK_MEDICATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "benzodiazepine_flag": (
        "alprazolam",
        "clonazepam",
        "diazepam",
        "lorazepam",
        "midazolam",
        "temazepam",
    ),
    "opioid_flag": (
        "codeine",
        "fentanyl",
        "hydrocodone",
        "hydromorphone",
        "methadone",
        "morphine",
        "oxycodone",
        "tramadol",
    ),
    "anticholinergic_flag": (
        "benztropine",
        "diphenhydramine",
        "hydroxyzine",
        "oxybutynin",
        "promethazine",
        "scopolamine",
        "tolterodine",
    ),
    "ppi_flag": (
        "dexlansoprazole",
        "esomeprazole",
        "lansoprazole",
        "omeprazole",
        "pantoprazole",
        "rabeprazole",
    ),
    "antipsychotic_flag": (
        "aripiprazole",
        "chlorpromazine",
        "clozapine",
        "haloperidol",
        "olanzapine",
        "quetiapine",
        "risperidone",
        "ziprasidone",
    ),
}
