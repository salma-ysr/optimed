"""FastAPI app exposing the saved OPTI-MED scored output."""

from __future__ import annotations

from functools import lru_cache
from datetime import date, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from opti_med.api.clinician_reviews import (
    ClinicianReviewRepository,
    build_review_queue_hints,
    summarize_review_queue,
)
from opti_med.api.repository import ScoredDataRepository
from opti_med.api.schemas import (
    AdmissionDetailResponse,
    AdmissionSummariesResponse,
    AdmissionSummary,
    ClinicianReviewSubmissionRequest,
    ClinicianReviewSubmissionResponse,
    ClinicianReviewWorkflowReport,
    DossierEncounterSelection,
    HealthResponse,
    MedicationRowSummary,
    PatientContextObject,
    PatientDetailResponse,
    PatientEncounterSummary,
    PatientEncountersResponse,
    PatientMedicationCard,
    PatientMedicationsResponse,
    PatientSummariesResponse,
    PatientSummary,
    ProblemFlash,
    RefreshScoresResponse,
    ScoredOutputSummary,
    ScoredRow,
    ScoredRowsResponse,
)
from opti_med.config import Settings
from opti_med.data_access.encounters import EncounterIndexBuilder
from opti_med.data_access.medication_events import CanonicalMedicationEventBuilder
from opti_med.data_access.medication_snapshot import build_review_rows_for_encounter
from opti_med.data_access.medication_snapshot import select_review_timestamp_metadata_for_encounter
from opti_med.data_access.provenance import loads_json_or_none
from opti_med.medication_semantics import resolve_supported_scope_class_labels
from opti_med.scoring.scorer import DeprescribingPriorityScorer
from opti_med.time_semantics.constants import (
    MEDICATION_STATUS_ACTIVE_AT_REVIEW,
    REVIEW_TIMESTAMP_SOURCE_PREBUILT_SNAPSHOT,
)


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="OPTI-MED MVP API",
        version="0.1.0",
        description="Lightweight local API for browsing scored OPTI-MED medication rows.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        repository = get_repository()
        return HealthResponse(
            status="ok",
            scored_output_available=repository.exists(),
            scored_output_path=str(repository.output_path),
        )

    @app.get("/scores", response_model=ScoredRowsResponse)
    def list_scored_rows(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        subject_id: int | None = None,
        hadm_id: int | None = None,
        risk_label: str | None = Query(default=None, pattern="^(low|medium|high)$"),
    ) -> ScoredRowsResponse:
        dataframe = _load_scored_dataframe()
        filtered = dataframe
        if subject_id is not None:
            filtered = filtered.loc[filtered["subject_id"] == subject_id]
        if hadm_id is not None:
            filtered = filtered.loc[filtered["hadm_id"] == hadm_id]
        if risk_label is not None:
            filtered = filtered.loc[filtered["deprescribing_priority_label"] == risk_label]

        filtered = filtered.reset_index(drop=True)
        page = filtered.iloc[offset : offset + limit]
        return ScoredRowsResponse(
            total_rows=len(filtered),
            limit=limit,
            offset=offset,
            rows=[ScoredRow(**_clean_record(record)) for record in page.to_dict(orient="records")],
        )

    @app.get("/admissions", response_model=AdmissionSummariesResponse)
    def list_admission_summaries(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        risk_label: str | None = Query(default=None, pattern="^(low|medium|high)$"),
    ) -> AdmissionSummariesResponse:
        dataframe = _load_scored_dataframe()
        summaries = _build_admission_summaries(dataframe)
        if risk_label is not None:
            summaries = summaries.loc[summaries["overall_priority_label"] == risk_label].copy()

        summaries = summaries.sort_values(
            ["overall_priority_score", "flagged_medication_count", "admittime"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        page = summaries.iloc[offset : offset + limit]
        return AdmissionSummariesResponse(
            total_admissions=len(summaries),
            limit=limit,
            offset=offset,
            rows=[AdmissionSummary(**_clean_record(record)) for record in page.to_dict(orient="records")],
        )

    @app.get("/patients", response_model=PatientSummariesResponse)
    def list_patients(
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        risk_label: str | None = Query(default=None, pattern="^(low|medium|high)$"),
    ) -> PatientSummariesResponse:
        dataframe = _load_scored_dataframe()
        patient_summaries = _build_patient_summaries(dataframe)
        if risk_label is not None:
            patient_summaries = patient_summaries.loc[
                patient_summaries["highest_priority_label"] == risk_label
            ].copy()

        patient_summaries = patient_summaries.sort_values(
            ["highest_priority_score", "flagged_medication_count", "subject_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        page = patient_summaries.iloc[offset : offset + limit]
        return PatientSummariesResponse(
            total_patients=len(patient_summaries),
            limit=limit,
            offset=offset,
            rows=[PatientSummary(**_clean_record(record)) for record in page.to_dict(orient="records")],
        )

    @app.get("/scores/row", response_model=ScoredRow)
    def get_scored_row(
        subject_id: int,
        hadm_id: int,
        drug: str,
        starttime: str,
    ) -> ScoredRow:
        dataframe = _load_scored_dataframe()
        matched = dataframe.loc[
            (dataframe["subject_id"] == subject_id)
            & (dataframe["hadm_id"] == hadm_id)
            & (dataframe["drug"] == drug)
            & (dataframe["starttime"] == starttime)
        ]

        if matched.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Scored row not found for the provided subject_id, hadm_id, drug, and starttime."
                ),
            )
        if len(matched) > 1:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Multiple scored rows matched the provided identifiers. Use /scores to inspect the admission rows."
                ),
            )

        return ScoredRow(**_clean_record(matched.iloc[0].to_dict()))

    @app.get("/scores/admission", response_model=AdmissionDetailResponse)
    def get_admission_detail(subject_id: int, hadm_id: int) -> AdmissionDetailResponse:
        dataframe = _load_scored_dataframe()
        matched = dataframe.loc[
            (dataframe["subject_id"] == subject_id) & (dataframe["hadm_id"] == hadm_id)
        ].copy()

        if matched.empty:
            raise HTTPException(
                status_code=404,
                detail="No scored admission found for the provided subject_id and hadm_id.",
            )

        matched = matched.sort_values(
            ["deprescribing_priority_score", "drug", "starttime"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        top_row = matched.iloc[0].to_dict()
        medications = [
            _build_medication_summary(record)
            for record in matched.to_dict(orient="records")
        ]
        flagged_medications = [
            medication
            for medication in medications
            if medication.deprescribing_priority_label in {"medium", "high"}
        ]
        score_explanations = _collect_explanations(matched)
        overall_priority_drivers = _collect_explanations(
            matched.loc[
                matched["deprescribing_priority_score"]
                == matched["deprescribing_priority_score"].max()
            ]
        )

        return AdmissionDetailResponse(
            subject_id=int(top_row["subject_id"]),
            hadm_id=int(top_row["hadm_id"]),
            sex=str(top_row["sex"]),
            age_proxy=int(top_row["age_proxy"]),
            age_group=str(top_row["age_group"]),
            admission_type=str(top_row["admission_type"]),
            admittime=str(top_row["admittime"]),
            dischtime=str(top_row["dischtime"]),
            length_of_stay_days=float(top_row["length_of_stay_days"]),
            current_medication_count=int(top_row.get("current_medication_count") or top_row["total_medication_count"]),
            current_polypharmacy_flag=int(top_row.get("current_polypharmacy_flag") or top_row["polypharmacy_flag"]),
            historical_medication_count=(
                int(top_row["historical_medication_count"])
                if pd.notna(top_row.get("historical_medication_count"))
                else None
            ),
            total_medication_count=int(top_row["total_medication_count"]),
            peak_concurrent_medication_count=int(top_row.get("peak_concurrent_medication_count") or 0),
            polypharmacy_flag=int(top_row["polypharmacy_flag"]),
            ckd_flag=int(top_row["ckd_flag"]),
            dementia_flag=int(top_row["dementia_flag"]),
            delirium_flag=int(top_row["delirium_flag"]),
            heart_failure_flag=int(top_row["heart_failure_flag"]),
            diabetes_flag=int(top_row["diabetes_flag"]),
            creatinine_first=top_row.get("creatinine_first"),
            creatinine_max=top_row.get("creatinine_max"),
            creatinine_mean=top_row.get("creatinine_mean"),
            renal_risk_flag=int(top_row["renal_risk_flag"]),
            highest_priority_score=int(top_row["deprescribing_priority_score"]),
            highest_priority_label=str(top_row["deprescribing_priority_label"]),
            flagged_medication_count=len(flagged_medications),
            score_explanations=score_explanations,
            overall_priority_drivers=overall_priority_drivers,
            flagged_medications=flagged_medications,
            medications=medications,
        )

    @app.get("/patients/{subject_id}", response_model=PatientDetailResponse)
    def get_patient_detail(subject_id: int, hadm_id: int | None = None) -> PatientDetailResponse:
        dataframe = _load_scored_dataframe()
        patient_rows = dataframe.loc[dataframe["subject_id"] == subject_id].copy()
        if patient_rows.empty:
            raise HTTPException(status_code=404, detail="No scored patient found for the provided subject_id.")

        selected_encounter = _select_dossier_encounter(subject_id=subject_id, hadm_id=hadm_id)
        encounter_rows = (
            patient_rows.loc[patient_rows["hadm_id"] == selected_encounter["hadm_id"]].copy()
            if selected_encounter.get("hadm_id") is not None
            else pd.DataFrame(columns=patient_rows.columns)
        )
        current_cards, history_cards, flagged_current_count = _build_dossier_medication_views(
            subject_id=subject_id,
            selected_encounter=selected_encounter,
            scored_rows=encounter_rows,
        )
        context_rows = encounter_rows if not encounter_rows.empty else patient_rows
        patient_summary = _build_patient_summary(patient_rows)
        encounter_summaries = _build_patient_encounter_summaries(patient_rows)
        dossier_selection = _build_dossier_encounter_selection(
            selected_encounter,
            requested_hadm_id=hadm_id,
        )
        return PatientDetailResponse(
            patient_summary=patient_summary,
            encounter_summaries=encounter_summaries,
            selected_encounter=dossier_selection,
            review_timestamp=dossier_selection.review_timestamp,
            current_medication_count=len(current_cards),
            current_flagged_medication_count=flagged_current_count,
            historical_medication_count=len(history_cards),
            left_column_context=_build_patient_context_object(context_rows),
            current_medications=current_cards,
            previous_medication_history=history_cards,
            ranked_medication_cards=current_cards,
            historical_medication_cards=history_cards,
            top_problem_flashes=_build_problem_flashes(context_rows),
            flagged_medication_count=flagged_current_count,
            medication_card_count=len(current_cards),
            review_queue_summary=summarize_review_queue(
                [
                    card.model_dump()
                    for card in [*current_cards, *history_cards]
                ]
            ),
        )

    @app.get("/patients/{subject_id}/medications", response_model=PatientMedicationsResponse)
    def get_patient_medications(
        subject_id: int,
        hadm_id: int | None = None,
    ) -> PatientMedicationsResponse:
        dataframe = _load_scored_dataframe()
        patient_rows = dataframe.loc[dataframe["subject_id"] == subject_id].copy()
        if patient_rows.empty:
            raise HTTPException(status_code=404, detail="No scored patient found for the provided subject_id.")
        selected_encounter = _select_dossier_encounter(subject_id=subject_id, hadm_id=hadm_id)
        encounter_rows = (
            patient_rows.loc[patient_rows["hadm_id"] == selected_encounter["hadm_id"]].copy()
            if selected_encounter.get("hadm_id") is not None
            else pd.DataFrame(columns=patient_rows.columns)
        )
        cards, _, _ = _build_dossier_medication_views(
            subject_id=subject_id,
            selected_encounter=selected_encounter,
            scored_rows=encounter_rows,
        )
        return PatientMedicationsResponse(
            subject_id=subject_id,
            total_medications=len(cards),
            rows=cards,
        )

    @app.get("/patients/{subject_id}/encounters", response_model=PatientEncountersResponse)
    def get_patient_encounters(subject_id: int) -> PatientEncountersResponse:
        dataframe = _load_scored_dataframe()
        patient_rows = dataframe.loc[dataframe["subject_id"] == subject_id].copy()
        if patient_rows.empty:
            raise HTTPException(status_code=404, detail="No scored patient found for the provided subject_id.")
        rows = _build_patient_encounter_summaries(patient_rows)
        return PatientEncountersResponse(
            subject_id=subject_id,
            total_encounters=len(rows),
            rows=rows,
        )

    @app.post("/clinician-reviews", response_model=ClinicianReviewSubmissionResponse)
    def submit_clinician_review(
        payload: ClinicianReviewSubmissionRequest,
    ) -> ClinicianReviewSubmissionResponse:
        repository = get_clinician_review_repository()
        try:
            review, workflow_report = repository.save_review(submission=payload.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ClinicianReviewSubmissionResponse(
            review=review,
            workflow_report=ClinicianReviewWorkflowReport(**workflow_report),
        )

    @app.get("/clinician-reviews/report", response_model=ClinicianReviewWorkflowReport)
    def get_clinician_review_report() -> ClinicianReviewWorkflowReport:
        repository = get_clinician_review_repository()
        report = repository.write_qc_artifacts()
        return ClinicianReviewWorkflowReport(**report)

    @app.get("/scores/latest", response_model=ScoredOutputSummary)
    def get_latest_scored_output() -> ScoredOutputSummary:
        repository = get_repository()
        try:
            return repository.summary()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/scores/refresh", response_model=RefreshScoresResponse)
    def refresh_scored_output() -> RefreshScoresResponse:
        settings = get_settings()
        scorer = DeprescribingPriorityScorer(settings)
        dataframe = scorer.build()
        scorer.save(dataframe)
        repository = get_repository()
        repository.load(force_reload=True)
        return RefreshScoresResponse(
            message="Scored output rebuilt successfully.",
            output=repository.summary(),
        )

    return app


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings.from_env()


@lru_cache
def get_repository() -> ScoredDataRepository:
    """Return the scored output repository."""
    return ScoredDataRepository(get_settings())


@lru_cache
def get_clinician_review_repository() -> ClinicianReviewRepository:
    """Return the Phase 5 clinician review repository."""
    return ClinicianReviewRepository(get_settings())


def _load_scored_dataframe() -> pd.DataFrame:
    repository = get_repository()
    try:
        return repository.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@lru_cache
def _load_encounter_index_dataframe() -> pd.DataFrame:
    settings = get_settings()
    path = settings.encounter_index_output_path
    if path.exists():
        dataframe = pd.read_parquet(path)
        return dataframe.where(pd.notna(dataframe), None)
    return EncounterIndexBuilder(settings).build()


@lru_cache
def _load_medication_events_dataframe() -> pd.DataFrame:
    settings = get_settings()
    path = settings.medication_events_output_path
    if path.exists():
        dataframe = pd.read_parquet(path)
        return dataframe.where(pd.notna(dataframe), None)
    return CanonicalMedicationEventBuilder(settings).build()


@lru_cache
def _load_medication_snapshot_dataframe() -> pd.DataFrame:
    settings = get_settings()
    path = settings.medication_snapshot_output_path
    if path.exists():
        dataframe = pd.read_csv(path)
        return dataframe.where(pd.notna(dataframe), None)
    return pd.DataFrame()


@lru_cache
def _load_medication_rxnorm_mapping_dataframe() -> pd.DataFrame:
    settings = get_settings()
    path = settings.medication_rxnorm_mapping_output_path
    if path.exists():
        dataframe = pd.read_parquet(path)
        if "class_labels_json" in dataframe.columns:
            dataframe["class_labels_json"] = dataframe["class_labels_json"].map(loads_json_or_none)
        return dataframe.where(pd.notna(dataframe), None)
    return pd.DataFrame()


def _clean_record(record: dict) -> dict:
    return {
        key: (_normalize_value(value))
        for key, value in record.items()
    }


def _normalize_value(value: object) -> object:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return None
    if pd.isna(value):
        return None
    return value


def _build_admission_summaries(dataframe: pd.DataFrame) -> pd.DataFrame:
    summaries: list[dict] = []
    grouped = dataframe.groupby(["subject_id", "hadm_id"], sort=False)
    for (_, _), group in grouped:
        group = group.sort_values(
            ["deprescribing_priority_score", "drug", "starttime"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        top_row = group.iloc[0].to_dict()
        flagged_mask = group["deprescribing_priority_label"].isin(["medium", "high"])
        summaries.append(
            {
                "subject_id": int(top_row["subject_id"]),
                "hadm_id": int(top_row["hadm_id"]),
                "sex": str(top_row["sex"]),
                "age_proxy": int(top_row["age_proxy"]),
                "age_group": str(top_row["age_group"]),
                "admission_type": str(top_row["admission_type"]),
                "admittime": str(top_row["admittime"]),
                "dischtime": str(top_row["dischtime"]),
                "length_of_stay_days": float(top_row["length_of_stay_days"]),
                "current_medication_count": int(
                    top_row.get("current_medication_count") or top_row["total_medication_count"]
                ),
                "current_polypharmacy_flag": int(
                    top_row.get("current_polypharmacy_flag") or top_row["polypharmacy_flag"]
                ),
                "historical_medication_count": (
                    int(top_row["historical_medication_count"])
                    if pd.notna(top_row.get("historical_medication_count"))
                    else None
                ),
                "total_medication_count": int(top_row["total_medication_count"]),
                "peak_concurrent_medication_count": int(top_row.get("peak_concurrent_medication_count") or 0),
                "polypharmacy_flag": int(top_row["polypharmacy_flag"]),
                "creatinine_max": top_row.get("creatinine_max"),
                "renal_risk_flag": int(top_row["renal_risk_flag"]),
                "overall_priority_score": int(top_row["deprescribing_priority_score"]),
                "overall_priority_label": str(top_row["deprescribing_priority_label"]),
                "flagged_medication_count": int(flagged_mask.sum()),
                "overall_priority_drivers": _collect_explanations(
                    group.loc[
                        group["deprescribing_priority_score"]
                        == group["deprescribing_priority_score"].max()
                    ]
                ),
            }
        )
    return pd.DataFrame(summaries)


def _collect_explanations(dataframe: pd.DataFrame) -> list[str]:
    explanations: list[str] = []
    if "deprescribing_priority_reasons_json" in dataframe.columns:
        for reason_list in dataframe["deprescribing_priority_reasons_json"].dropna().tolist():
            for item in reason_list:
                if item not in explanations:
                    explanations.append(str(item))
    else:
        for explanation_text in dataframe["deprescribing_priority_explanation"].dropna().tolist():
            for item in [part.strip() for part in str(explanation_text).split(";") if part.strip()]:
                if item not in explanations:
                    explanations.append(item)
    return explanations[:4]


def _medication_classes_from_record(record: dict) -> list[str]:
    classes: list[str] = []
    if int(record.get("benzodiazepine_flag", 0)) == 1:
        classes.append("Benzodiazepine")
    if int(record.get("opioid_flag", 0)) == 1:
        classes.append("Opioid")
    if int(record.get("anticholinergic_flag", 0)) == 1:
        classes.append("Anticholinergic")
    if int(record.get("ppi_flag", 0)) == 1:
        classes.append("PPI")
    if int(record.get("antipsychotic_flag", 0)) == 1:
        classes.append("Antipsychotic")
    return classes


def _title_case_medication_class_label(label: str) -> str:
    normalized = str(label).strip().lower()
    mapping = {
        "benzodiazepine": "Benzodiazepine",
        "opioid": "Opioid",
        "anticholinergic": "Anticholinergic",
        "antipsychotic": "Antipsychotic",
        "ppi": "PPI",
    }
    return mapping.get(normalized, normalized.replace("_", " ").title())


def _mapping_class_labels(record: dict | None) -> list[str]:
    if record is None:
        return []
    parsed = record.get("class_labels_json")
    labels: list[str] = []
    if isinstance(parsed, list):
        for item in parsed:
            text = str(item).strip()
            if text:
                labels.append(_title_case_medication_class_label(text))
    if labels:
        return labels
    for candidate in [
        record.get("ingredient_standardized"),
        record.get("medication_standardized"),
        record.get("medication_query_key"),
    ]:
        for resolved in resolve_supported_scope_class_labels(candidate):
            title_cased = _title_case_medication_class_label(resolved)
            if title_cased not in labels:
                labels.append(title_cased)
    return labels


def _find_medication_rxnorm_mapping_record(*candidates: object) -> dict | None:
    mapping = _load_medication_rxnorm_mapping_dataframe()
    if mapping.empty:
        return None
    normalized_candidates = [
        _normalized_lookup_text(candidate)
        for candidate in candidates
    ]
    normalized_candidates = [candidate for candidate in normalized_candidates if candidate]
    if not normalized_candidates:
        return None
    for candidate in normalized_candidates:
        matches = mapping.loc[
            mapping["medication_query_key"].astype(str).str.lower() == candidate
        ].copy()
        if matches.empty:
            matches = mapping.loc[
                mapping["medication_normalized"].astype(str).str.lower() == candidate
            ].copy()
        if matches.empty:
            matches = mapping.loc[
                mapping["medication_standardized"].astype(str).str.lower() == candidate
            ].copy()
        if matches.empty:
            matches = mapping.loc[
                mapping["ingredient_standardized"].astype(str).str.lower() == candidate
            ].copy()
        if not matches.empty:
            sort_columns = [
                column
                for column in ["lookup_timestamp", "lookup_status"]
                if column in matches.columns
            ]
            if sort_columns:
                matches = matches.sort_values(
                    sort_columns,
                    ascending=[False, True][: len(sort_columns)],
                    na_position="last",
                ).reset_index(drop=True)
            else:
                matches = matches.reset_index(drop=True)
            return matches.iloc[0].to_dict()
    return None


def _resolved_medication_classes(
    *,
    score_record: dict | None,
    review_row: dict | None = None,
    reviewable_row: dict | None = None,
) -> list[str]:
    ordered: list[str] = []
    for label in _medication_classes_from_record(score_record or {}):
        if label not in ordered:
            ordered.append(label)
    if reviewable_row is not None:
        reviewable_class = _normalized_lookup_text(reviewable_row.get("medication_class_standardized"))
        if reviewable_class and reviewable_class != "unresolved":
            title_cased = _title_case_medication_class_label(reviewable_class)
            if title_cased not in ordered:
                ordered.append(title_cased)
    mapping_record = _find_medication_rxnorm_mapping_record(
        (review_row or {}).get("medication_normalized"),
        (review_row or {}).get("medication_name"),
        (review_row or {}).get("raw_medication_name"),
        (score_record or {}).get("drug_normalized"),
        (score_record or {}).get("drug"),
    )
    for label in _mapping_class_labels(mapping_record):
        if label not in ordered:
            ordered.append(label)
    return ordered


def _normalized_lookup_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _build_medication_summary(record: dict) -> MedicationRowSummary:
    summary_alert = record.get("deprescribing_priority_summary_alert") or record.get(
        "deprescribing_priority_explanation",
        "",
    )
    return MedicationRowSummary(
        drug=str(record["drug"]),
        drug_normalized=record.get("drug_normalized"),
        medication_classes=_medication_classes_from_record(record),
        starttime=str(record["starttime"]),
        stoptime=record.get("stoptime"),
        medication_episode_id=record.get("medication_episode_id"),
        prescription_segment_count=record.get("prescription_segment_count"),
        prescription_segments_json=list(record.get("prescription_segments_json") or []),
        deprescribing_priority_score=int(record["deprescribing_priority_score"]),
        deprescribing_priority_label=str(record["deprescribing_priority_label"]),
        deprescribing_priority_summary_alert=str(summary_alert),
        deprescribing_priority_explanation=str(record["deprescribing_priority_explanation"]),
        deprescribing_priority_bucket_scores_json=dict(
            record.get("deprescribing_priority_bucket_scores_json") or {}
        ),
        deprescribing_priority_reasons_json=list(
            record.get("deprescribing_priority_reasons_json") or []
        ),
        deprescribing_priority_evidence_json=dict(
            record.get("deprescribing_priority_evidence_json") or {}
        ),
        benzodiazepine_flag=int(record["benzodiazepine_flag"]),
        opioid_flag=int(record["opioid_flag"]),
        anticholinergic_flag=int(record["anticholinergic_flag"]),
        ppi_flag=int(record["ppi_flag"]),
        antipsychotic_flag=int(record["antipsychotic_flag"]),
    )


def _build_patient_summaries(dataframe: pd.DataFrame) -> pd.DataFrame:
    summaries = [_patient_summary_dict(group) for _, group in dataframe.groupby("subject_id", sort=False)]
    return pd.DataFrame(summaries)


def _build_patient_summary(patient_rows: pd.DataFrame) -> PatientSummary:
    return PatientSummary(**_clean_record(_patient_summary_dict(patient_rows)))


def _patient_summary_dict(patient_rows: pd.DataFrame) -> dict:
    sorted_rows = patient_rows.sort_values(
        ["deprescribing_priority_score", "hadm_id", "drug", "starttime"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)
    top_row = sorted_rows.iloc[0].to_dict()
    latest_rows = (
        patient_rows.assign(
            _latest_dischtime=pd.to_datetime(patient_rows["dischtime"], errors="coerce"),
            _latest_admittime=pd.to_datetime(patient_rows["admittime"], errors="coerce"),
        )
        .sort_values(["_latest_dischtime", "_latest_admittime"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )
    latest_row = latest_rows.iloc[0].to_dict()
    flagged_count = int(
        patient_rows["deprescribing_priority_label"].isin(["medium", "high"]).sum()
    )
    return {
        "subject_id": int(top_row["subject_id"]),
        "sex": str(top_row["sex"]),
        "age_proxy": int(top_row["age_proxy"]),
        "age_group": str(top_row["age_group"]),
        "encounter_count": int(patient_rows["hadm_id"].nunique()),
        "current_medication_count": len(patient_rows),
        "historical_medication_count": (
            int(patient_rows["historical_medication_count"].max())
            if "historical_medication_count" in patient_rows.columns
            and pd.notna(patient_rows["historical_medication_count"].max())
            else None
        ),
        "medication_count": len(patient_rows),
        "flagged_medication_count": flagged_count,
        "highest_priority_score": int(top_row["deprescribing_priority_score"]),
        "highest_priority_label": str(top_row["deprescribing_priority_label"]),
        "top_problem_flashes": [flash.label for flash in _build_problem_flashes(patient_rows)],
        "latest_hadm_id": (
            int(latest_row["hadm_id"])
            if latest_row.get("hadm_id") is not None and pd.notna(latest_row.get("hadm_id"))
            else None
        ),
        "latest_admission_type": (
            str(latest_row["admission_type"])
            if latest_row.get("admission_type") is not None and pd.notna(latest_row.get("admission_type"))
            else None
        ),
        "latest_dischtime": (
            str(latest_row["dischtime"])
            if latest_row.get("dischtime") is not None and pd.notna(latest_row.get("dischtime"))
            else None
        ),
    }


def _build_patient_encounter_summaries(patient_rows: pd.DataFrame) -> list[PatientEncounterSummary]:
    summaries = _build_admission_summaries(patient_rows)
    summaries = summaries.sort_values(
        ["overall_priority_score", "flagged_medication_count", "admittime"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return [
        PatientEncounterSummary(**_clean_record(record))
        for record in summaries.to_dict(orient="records")
    ]


def _build_patient_context_object(patient_rows: pd.DataFrame) -> PatientContextObject:
    sorted_rows = patient_rows.sort_values(["dischtime", "admittime"], ascending=[False, False]).reset_index(drop=True)
    latest_row = sorted_rows.iloc[0].to_dict()
    return PatientContextObject(
        sex=str(latest_row["sex"]),
        age_proxy=int(latest_row["age_proxy"]),
        age_group=str(latest_row["age_group"]),
        current_medication_count=int(
            latest_row.get("current_medication_count") or len(patient_rows)
        ),
        historical_medication_count=(
            int(latest_row["historical_medication_count"])
            if pd.notna(latest_row.get("historical_medication_count"))
            else None
        ),
        encounter_count=int(patient_rows["hadm_id"].nunique()),
        medication_count=len(patient_rows),
        peak_concurrent_medication_count=int(
            patient_rows[
                "current_peak_concurrent_medication_count"
                if "current_peak_concurrent_medication_count" in patient_rows.columns
                else "peak_concurrent_medication_count"
            ].max()
        )
        if (
            "current_peak_concurrent_medication_count" in patient_rows.columns
            or "peak_concurrent_medication_count" in patient_rows.columns
        )
        else None,
        flagged_medication_count=int(
            patient_rows["deprescribing_priority_label"].isin(["medium", "high"]).sum()
        ),
        polypharmacy_present=bool(
            patient_rows[
                "current_polypharmacy_flag"
                if "current_polypharmacy_flag" in patient_rows.columns
                else "polypharmacy_flag"
            ].max()
        ),
        renal_risk_present=bool(patient_rows["renal_risk_flag"].max()),
        ckd_present=bool(patient_rows["ckd_flag"].max()),
        dementia_present=bool(patient_rows["dementia_flag"].max()),
        delirium_present=bool(patient_rows["delirium_flag"].max()),
        heart_failure_present=bool(patient_rows["heart_failure_flag"].max()),
        diabetes_present=bool(patient_rows["diabetes_flag"].max()),
        latest_creatinine_max=latest_row.get("creatinine_max"),
        latest_egfr_ml_min_1_73m2=latest_row.get("egfr_ml_min_1_73m2"),
        latest_weight_kg=latest_row.get("weight_kg"),
        latest_bmi=latest_row.get("bmi"),
    )


def _build_patient_medication_cards(patient_rows: pd.DataFrame) -> list[PatientMedicationCard]:
    ranked_rows = patient_rows.sort_values(
        ["deprescribing_priority_score", "hadm_id", "drug", "starttime"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return [
        PatientMedicationCard(
            subject_id=int(record["subject_id"]),
            hadm_id=int(record["hadm_id"]),
            admission_type=str(record["admission_type"]),
            admittime=str(record["admittime"]),
            dischtime=str(record["dischtime"]),
            length_of_stay_days=float(record["length_of_stay_days"]),
            drug=str(record["drug"]),
            drug_normalized=record.get("drug_normalized"),
            medication_classes=_medication_classes_from_record(record),
            starttime=str(record["starttime"]),
            stoptime=record.get("stoptime"),
            medication_episode_id=record.get("medication_episode_id"),
            prescription_segment_count=record.get("prescription_segment_count"),
            prescription_segments_json=list(record.get("prescription_segments_json") or []),
            deprescribing_priority_score=int(record["deprescribing_priority_score"]),
            deprescribing_priority_label=str(record["deprescribing_priority_label"]),
            deprescribing_priority_summary_alert=str(
                record.get("deprescribing_priority_summary_alert")
                or record.get("deprescribing_priority_explanation", "")
            ),
            deprescribing_priority_explanation=str(record["deprescribing_priority_explanation"]),
            deprescribing_priority_bucket_scores_json=dict(
                record.get("deprescribing_priority_bucket_scores_json") or {}
            ),
            deprescribing_priority_reasons_json=list(
                record.get("deprescribing_priority_reasons_json")
                or _fallback_reasons(record.get("deprescribing_priority_explanation"))
            ),
            deprescribing_priority_evidence_json=dict(
                record.get("deprescribing_priority_evidence_json") or {}
            ),
            benzodiazepine_flag=int(record["benzodiazepine_flag"]),
            opioid_flag=int(record["opioid_flag"]),
            anticholinergic_flag=int(record["anticholinergic_flag"]),
            ppi_flag=int(record["ppi_flag"]),
            antipsychotic_flag=int(record["antipsychotic_flag"]),
        )
        for record in ranked_rows.to_dict(orient="records")
    ]


def _select_dossier_encounter(subject_id: int, hadm_id: int | None) -> dict:
    encounter_index = _load_encounter_index_dataframe()
    patient_encounters = encounter_index.loc[encounter_index["subject_id"] == subject_id].copy()
    if patient_encounters.empty:
        raise HTTPException(status_code=404, detail="No encounter found for the provided subject_id.")

    if hadm_id is not None:
        matched = patient_encounters.loc[patient_encounters["hadm_id"] == hadm_id].copy()
        if matched.empty:
            raise HTTPException(
                status_code=404,
                detail="No encounter found for the provided subject_id and hadm_id.",
            )
        patient_encounters = matched

    patient_encounters["encounter_sort_time"] = pd.to_datetime(
        patient_encounters["encounter_end"],
        errors="coerce",
    )
    patient_encounters["encounter_sort_time"] = patient_encounters["encounter_sort_time"].fillna(
        pd.to_datetime(patient_encounters["encounter_start"], errors="coerce")
    )
    patient_encounters = patient_encounters.sort_values(
        ["encounter_sort_time", "hadm_id", "stay_id"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return patient_encounters.iloc[0].drop(labels=["encounter_sort_time"]).to_dict()


def _build_dossier_encounter_selection(
    selected_encounter: dict,
    *,
    requested_hadm_id: int | None,
) -> DossierEncounterSelection:
    snapshot_rows = _load_medication_snapshot_dataframe()
    encounter_snapshot = snapshot_rows.loc[
        snapshot_rows["encounter_id"] == selected_encounter["encounter_id"]
    ].copy()
    review_timestamp = None
    review_timestamp_source = None
    if not encounter_snapshot.empty:
        top_row = encounter_snapshot.iloc[0].to_dict()
        review_timestamp = top_row.get("review_timestamp")
        review_timestamp_source = top_row.get("review_timestamp_source")
    else:
        review_value, review_source = select_review_timestamp_metadata_for_encounter(
            selected_encounter,
            strategy=get_settings().snapshot_strategy,
            medication_events=_load_medication_events_dataframe().loc[
                _load_medication_events_dataframe()["encounter_id"] == selected_encounter["encounter_id"]
            ].copy(),
        )
        if pd.notna(review_value):
            review_timestamp = pd.to_datetime(review_value, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
            review_timestamp_source = review_source
    return DossierEncounterSelection(
        encounter_id=str(selected_encounter["encounter_id"]),
        hadm_id=(
            int(selected_encounter["hadm_id"])
            if selected_encounter.get("hadm_id") is not None and pd.notna(selected_encounter.get("hadm_id"))
            else None
        ),
        stay_id=(
            int(selected_encounter["stay_id"])
            if selected_encounter.get("stay_id") is not None and pd.notna(selected_encounter.get("stay_id"))
            else None
        ),
        encounter_source=str(selected_encounter["encounter_source"]),
        selection_mode=(
            "requested_hadm_id" if requested_hadm_id is not None else "latest_available_encounter"
        ),
        requested_hadm_id=requested_hadm_id,
        review_timestamp=review_timestamp,
        review_timestamp_source=review_timestamp_source,
    )


def _build_dossier_medication_views(
    *,
    subject_id: int,
    selected_encounter: dict,
    scored_rows: pd.DataFrame,
) -> tuple[list[PatientMedicationCard], list[PatientMedicationCard], int]:
    encounter_review = _load_selected_encounter_medication_review(selected_encounter)
    if encounter_review.empty:
        return [], [], 0

    current_review = encounter_review.loc[
        encounter_review["medication_status"] == MEDICATION_STATUS_ACTIVE_AT_REVIEW
    ].copy()
    history_review = encounter_review.loc[
        encounter_review["medication_status"] != MEDICATION_STATUS_ACTIVE_AT_REVIEW
    ].copy()

    current_cards = _build_review_medication_cards(current_review, scored_rows, selected_encounter)
    history_cards = _build_review_medication_cards(history_review, scored_rows, selected_encounter)
    flagged_current_count = int(
        sum(card.deprescribing_priority_label in {"medium", "high"} for card in current_cards)
    )
    return current_cards, history_cards, flagged_current_count


def _load_selected_encounter_medication_review(selected_encounter: dict) -> pd.DataFrame:
    # Compute review status only for the selected encounter. This keeps the dossier fast and
    # preserves encounter-relative "current meds" semantics without rebuilding the full raw layer.
    medication_events = _load_medication_events_dataframe()
    encounter_events = medication_events.loc[
        medication_events["encounter_id"] == selected_encounter["encounter_id"]
    ].copy()
    if encounter_events.empty:
        return pd.DataFrame()

    snapshot_rows = _load_medication_snapshot_dataframe()
    encounter_snapshot = snapshot_rows.loc[
        snapshot_rows["encounter_id"] == selected_encounter["encounter_id"]
    ].copy()
    if not encounter_snapshot.empty:
        first_row = encounter_snapshot.iloc[0].to_dict()
        review_timestamp = first_row.get("review_timestamp") or first_row.get("snapshot_time")
        review_timestamp_source = str(
            first_row.get("review_timestamp_source") or REVIEW_TIMESTAMP_SOURCE_PREBUILT_SNAPSHOT
        )
    else:
        review_value, review_source = select_review_timestamp_metadata_for_encounter(
            selected_encounter,
            strategy=get_settings().snapshot_strategy,
            medication_events=encounter_events,
        )
        if pd.isna(review_value):
            return pd.DataFrame()
        review_timestamp = pd.to_datetime(review_value, errors="coerce").strftime("%Y-%m-%d %H:%M:%S")
        review_timestamp_source = review_source

    return build_review_rows_for_encounter(
        encounter_events=encounter_events,
        encounter=selected_encounter,
        review_timestamp=review_timestamp,
        review_timestamp_source=review_timestamp_source,
        snapshot_strategy=get_settings().snapshot_strategy,
    )


def _build_review_medication_cards(
    review_rows: pd.DataFrame,
    scored_rows: pd.DataFrame,
    selected_encounter: dict,
) -> list[PatientMedicationCard]:
    if review_rows.empty:
        return []

    cards: list[PatientMedicationCard] = []
    review_repository = get_clinician_review_repository()
    latest_review_lookup = _latest_clinician_review_lookup(review_repository.load_latest_reviews())
    ranked_review = review_rows.sort_values(
        ["review_timestamp", "medication_normalized"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    for review_row in ranked_review.to_dict(orient="records"):
        # TODO(ml-pivot): These dossier cards are keyed by encounter-relative review rows, but the
        # displayed rule score still comes from the saved admission-first scored output via string
        # matching. Future model output must attach separately to a dedicated review-time row.
        score_record = _best_scored_match_for_review(review_row, scored_rows)
        cards.append(
            _build_review_medication_card(
                review_row,
                selected_encounter,
                score_record,
                review_repository=review_repository,
                latest_review_lookup=latest_review_lookup,
            )
        )

    cards.sort(
        key=lambda card: (
            -card.deprescribing_priority_score,
            card.drug.lower(),
            card.starttime,
        ),
    )
    return cards


def _best_scored_match_for_review(review_row: dict, scored_rows: pd.DataFrame) -> dict | None:
    if scored_rows.empty:
        return None
    matches = scored_rows.loc[
        scored_rows["drug_normalized"] == review_row.get("medication_normalized")
    ].copy()
    if matches.empty:
        matches = scored_rows.loc[
            scored_rows["drug"].astype(str).str.lower()
            == str(review_row.get("medication_name") or "").strip().lower()
        ].copy()
    if matches.empty:
        return None
    matches = matches.sort_values(
        ["deprescribing_priority_score", "starttime"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    return matches.iloc[0].to_dict()


def _build_review_medication_card(
    review_row: dict,
    selected_encounter: dict,
    score_record: dict | None,
    *,
    review_repository: ClinicianReviewRepository,
    latest_review_lookup: dict[tuple[int, str, str, str], dict],
) -> PatientMedicationCard:
    reviewable_row = review_repository.resolve_reviewable_row(
        subject_id=int(review_row["subject_id"]),
        encounter_id=str(selected_encounter["encounter_id"]),
        review_timestamp=review_row.get("review_timestamp"),
        medication_candidates=[
            review_row.get("medication_normalized"),
            review_row.get("medication_name"),
            review_row.get("raw_medication_name"),
            (score_record or {}).get("drug_normalized"),
            (score_record or {}).get("drug"),
        ],
        selected_event_id=review_row.get("selected_event_id"),
    )
    base_record = {
        "subject_id": int(review_row["subject_id"]),
        "encounter_id": str(selected_encounter["encounter_id"]),
        "hadm_id": (
            int(review_row["hadm_id"])
            if review_row.get("hadm_id") is not None and pd.notna(review_row.get("hadm_id"))
            else int(selected_encounter["hadm_id"])
            if selected_encounter.get("hadm_id") is not None and pd.notna(selected_encounter.get("hadm_id"))
            else 0
        ),
        "stay_id": (
            int(selected_encounter["stay_id"])
            if selected_encounter.get("stay_id") is not None and pd.notna(selected_encounter.get("stay_id"))
            else None
        ),
        "admission_type": str(selected_encounter.get("admission_type") or "unknown"),
        "admittime": str(selected_encounter.get("admittime") or ""),
        "dischtime": str(selected_encounter.get("dischtime") or ""),
        "length_of_stay_days": float(selected_encounter.get("hospital_length_of_stay_days") or 0.0),
        "drug": str(review_row.get("medication_name") or review_row.get("raw_medication_name") or review_row["medication_normalized"]),
        "drug_normalized": review_row.get("medication_normalized"),
        "medication_classes": _resolved_medication_classes(
            score_record=score_record,
            review_row=review_row,
            reviewable_row=reviewable_row,
        ),
        "starttime": str((score_record or {}).get("starttime") or review_row.get("review_timestamp") or review_row.get("snapshot_time") or ""),
        "stoptime": (score_record or {}).get("stoptime"),
        "medication_episode_id": (score_record or {}).get("medication_episode_id"),
        "prescription_segment_count": (score_record or {}).get("prescription_segment_count"),
        "prescription_segments_json": list((score_record or {}).get("prescription_segments_json") or []),
        "deprescribing_priority_score": int((score_record or {}).get("deprescribing_priority_score") or 0),
        "deprescribing_priority_label": str((score_record or {}).get("deprescribing_priority_label") or "low"),
        "deprescribing_priority_summary_alert": str(
            (score_record or {}).get("deprescribing_priority_summary_alert")
            or "No structured current-encounter score available for this medication."
        ),
        "deprescribing_priority_explanation": str(
            (score_record or {}).get("deprescribing_priority_explanation")
            or "Medication classified from encounter-relative review data without a scored cohort row."
        ),
        "deprescribing_priority_bucket_scores_json": dict(
            (score_record or {}).get("deprescribing_priority_bucket_scores_json") or {}
        ),
        "deprescribing_priority_reasons_json": list(
            (score_record or {}).get("deprescribing_priority_reasons_json") or []
        ),
        "deprescribing_priority_evidence_json": dict(
            (score_record or {}).get("deprescribing_priority_evidence_json") or {}
        ),
        "benzodiazepine_flag": int((score_record or {}).get("benzodiazepine_flag") or 0),
        "opioid_flag": int((score_record or {}).get("opioid_flag") or 0),
        "anticholinergic_flag": int((score_record or {}).get("anticholinergic_flag") or 0),
        "ppi_flag": int((score_record or {}).get("ppi_flag") or 0),
        "antipsychotic_flag": int((score_record or {}).get("antipsychotic_flag") or 0),
        "status": review_row.get("medication_status"),
        "medication_status": review_row.get("medication_status"),
        "last_active_time": review_row.get("last_active_time"),
        "review_timestamp": review_row.get("review_timestamp"),
        "review_timestamp_source": review_row.get("review_timestamp_source"),
        "priority_score_source": (
            "admission_first_scored_output_match"
            if score_record is not None
            else "no_scored_match_for_review_medication"
        ),
    }
    clinician_review = None
    if reviewable_row is not None:
        review_lookup_key = _review_key_tuple(
            subject_id=int(reviewable_row["subject_id"]),
            encounter_id=str(reviewable_row["encounter_id"]),
            medication_standardized=str(reviewable_row["medication_standardized"]),
            review_timestamp=str(reviewable_row["review_timestamp"]),
        )
        clinician_review = latest_review_lookup.get(review_lookup_key)
        queue_priority, queue_reasons = build_review_queue_hints(
            reviewable_row=reviewable_row,
            clinician_review=clinician_review,
            displayed_rule_level=str(base_record["deprescribing_priority_label"]),
        )
        base_record.update(
            {
                "medication_standardized": reviewable_row.get("medication_standardized"),
                "modeling__row_id": reviewable_row.get("modeling__row_id"),
                "first_scope_supported_class_flag": _safe_int(
                    reviewable_row.get("first_scope_supported_class_flag")
                ),
                "benchmark__current_rule_score": reviewable_row.get(
                    "benchmark__current_rule_score"
                ),
                "benchmark__current_rule_score_level": reviewable_row.get(
                    "benchmark__current_rule_score_level"
                ),
                "benchmark__current_rule_available_flag": _safe_int(
                    reviewable_row.get("benchmark__current_rule_available_flag")
                ),
                "benchmark__medication_class_only_medication_class_standardized": reviewable_row.get(
                    "benchmark__medication_class_only_medication_class_standardized"
                ),
                "reviewable_flag": True,
                "review_queue_priority": queue_priority,
                "review_queue_reasons": queue_reasons,
                "clinician_review": clinician_review,
            }
        )
    else:
        base_record.update(
            {
                "reviewable_flag": False,
                "review_queue_priority": "out_of_scope",
                "review_queue_reasons": ["not_in_phase5_review_scope"],
                "clinician_review": None,
            }
        )
    return PatientMedicationCard(**_clean_record(base_record))


def _latest_clinician_review_lookup(dataframe: pd.DataFrame) -> dict[tuple[int, str, str, str], dict]:
    if dataframe.empty:
        return {}
    lookup: dict[tuple[int, str, str, str], dict] = {}
    for record in dataframe.to_dict(orient="records"):
        try:
            key = _review_key_tuple(
                subject_id=int(record["subject_id"]),
                encounter_id=str(record["encounter_id"]),
                medication_standardized=str(record["medication_standardized"]),
                review_timestamp=str(record["review_timestamp"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        lookup[key] = _clean_record(record)
    return lookup


def _review_key_tuple(
    *,
    subject_id: int,
    encounter_id: str,
    medication_standardized: str,
    review_timestamp: str,
) -> tuple[int, str, str, str]:
    return (
        int(subject_id),
        str(encounter_id),
        str(medication_standardized),
        str(review_timestamp),
    )


def _safe_int(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def _fallback_reasons(explanation_text: object) -> list[str]:
    if explanation_text is None:
        return []
    return [part.strip() for part in str(explanation_text).split(";") if part.strip()]


def _build_problem_flashes(patient_rows: pd.DataFrame) -> list[ProblemFlash]:
    flashes: list[ProblemFlash] = []
    if bool(patient_rows["renal_risk_flag"].max()) or (
        "creatinine_trend_direction" in patient_rows.columns
        and (patient_rows["creatinine_trend_direction"] == "rising").any()
    ):
        flashes.append(
            ProblemFlash(
                key="renal_toxicity",
                label="Renal toxicity",
                severity="high" if bool(patient_rows["renal_risk_flag"].max()) else "medium",
                reason="Rising creatinine or renal vulnerability is present.",
            )
        )
    if bool(patient_rows["opioid_flag"].max()) or bool(patient_rows["benzodiazepine_flag"].max()):
        flashes.append(
            ProblemFlash(
                key="oversedation",
                label="Oversedation",
                severity="high" if bool(patient_rows["benzodiazepine_flag"].max()) else "medium",
                reason="Sedating medication exposure is present.",
            )
        )
    if bool(patient_rows["benzodiazepine_flag"].max()) or bool(patient_rows["anticholinergic_flag"].max()):
        flashes.append(
            ProblemFlash(
                key="fall_risk",
                label="Fall risk",
                severity="medium",
                reason="Fall-prone medication exposure is present.",
            )
        )
    return flashes[:3]


app = create_app()
