"""FastAPI app exposing the saved OPTI-MED scored output."""

from __future__ import annotations

from functools import lru_cache
from datetime import date, datetime
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from opti_med.api.clinician_reviews import (
    ClinicianReviewRepository,
    build_review_queue_hints,
    overlay_latest_reviews_on_queue_rows,
    summarize_review_queue,
)
from opti_med.api.repository import ScoredDataRepository
from opti_med.api.schemas import (
    AdmissionDetailResponse,
    AdmissionSummariesResponse,
    AdmissionSummary,
    ClinicianReviewQueueEntry,
    ClinicianReviewQueueResponse,
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
from opti_med.modeling.first_scope_targeted_blind_eval import (
    build_first_scope_targeted_blind_eval,
    filter_targeted_blind_eval_slice,
    write_targeted_blind_eval_artifacts,
)
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
        limit: int = Query(default=20, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        risk_label: str | None = Query(default=None, pattern="^(low|medium|high)$"),
    ) -> PatientSummariesResponse:
        patient_summaries = _build_review_time_patient_summaries()
        if risk_label is not None:
            patient_summaries = patient_summaries.loc[
                patient_summaries["highest_priority_label"] == risk_label
            ].copy()
        patient_summaries = patient_summaries.reset_index(drop=True)
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
        dataframe = _load_review_time_patient_universe()
        patient_rows = dataframe.loc[dataframe["subject_id"] == subject_id].copy()
        if patient_rows.empty:
            raise HTTPException(status_code=404, detail="No patient found for the provided subject_id.")

        selected_encounter = _select_dossier_encounter(subject_id=subject_id, hadm_id=hadm_id)
        encounter_rows = (
            patient_rows.loc[patient_rows["encounter_id"] == selected_encounter["encounter_id"]].copy()
            if selected_encounter.get("encounter_id") is not None
            else pd.DataFrame(columns=patient_rows.columns)
        )
        current_cards, history_cards, flagged_current_count = _build_dossier_medication_views(
            review_rows=encounter_rows,
            selected_encounter=selected_encounter,
        )
        legacy_context_rows = _load_legacy_context_rows(
            subject_id=subject_id,
            hadm_id=selected_encounter.get("hadm_id"),
        )
        patient_summary = _build_review_time_patient_summary(subject_id=subject_id)
        encounter_summaries = _build_review_time_patient_encounter_summaries(subject_id=subject_id)
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
            left_column_context=(
                _build_patient_context_object(legacy_context_rows)
                if not legacy_context_rows.empty
                else _build_review_time_context_object(
                    patient_rows=patient_rows,
                    encounter_rows=encounter_rows,
                    current_cards=current_cards,
                    history_cards=history_cards,
                )
            ),
            current_medications=current_cards,
            previous_medication_history=history_cards,
            ranked_medication_cards=current_cards,
            historical_medication_cards=history_cards,
            top_problem_flashes=(
                _build_problem_flashes(legacy_context_rows)
                if not legacy_context_rows.empty
                else _build_review_time_problem_flashes(encounter_rows)
            ),
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
        dataframe = _load_review_time_patient_universe()
        patient_rows = dataframe.loc[dataframe["subject_id"] == subject_id].copy()
        if patient_rows.empty:
            raise HTTPException(status_code=404, detail="No patient found for the provided subject_id.")
        selected_encounter = _select_dossier_encounter(subject_id=subject_id, hadm_id=hadm_id)
        encounter_rows = (
            patient_rows.loc[patient_rows["encounter_id"] == selected_encounter["encounter_id"]].copy()
            if selected_encounter.get("encounter_id") is not None
            else pd.DataFrame(columns=patient_rows.columns)
        )
        cards, _, _ = _build_dossier_medication_views(
            review_rows=encounter_rows,
            selected_encounter=selected_encounter,
        )
        return PatientMedicationsResponse(
            subject_id=subject_id,
            total_medications=len(cards),
            rows=cards,
        )

    @app.get("/patients/{subject_id}/encounters", response_model=PatientEncountersResponse)
    def get_patient_encounters(subject_id: int) -> PatientEncountersResponse:
        dataframe = _load_review_time_patient_universe()
        patient_rows = dataframe.loc[dataframe["subject_id"] == subject_id].copy()
        if patient_rows.empty:
            raise HTTPException(status_code=404, detail="No patient found for the provided subject_id.")
        rows = _build_review_time_patient_encounter_summaries(subject_id=subject_id)
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

    @app.get("/clinician-reviews/queue", response_model=ClinicianReviewQueueResponse)
    def get_clinician_review_queue(
        limit: int = Query(default=200, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        unreviewed_only: bool = False,
        medication_class: str | None = None,
        review_status: str = Query(
            default="all",
            pattern="^(all|unreviewed|reviewed|uncertain|insufficient_context|skip)$",
        ),
        subject_id: int | None = None,
        reason_tag_presence: str = Query(
            default="all",
            pattern="^(all|has_reason_tags|no_reason_tags)$",
        ),
    ) -> ClinicianReviewQueueResponse:
        repository = get_clinician_review_repository()
        reviewable = repository.load_reviewable_universe()
        latest = repository.load_latest_reviews()
        queue = repository.load_review_queue(
            latest_reviews=latest,
            reviewable_universe=reviewable,
        )
        report = repository.write_qc_artifacts(
            latest_reviews=latest,
            reviewable_universe=reviewable,
            review_queue=queue,
        )
        filtered = repository.filter_review_queue(
            queue,
            unreviewed_only=unreviewed_only,
            medication_class=medication_class,
            review_status=review_status,
            subject_id=subject_id,
            reason_tag_presence=reason_tag_presence,
        )
        page = filtered.iloc[offset : offset + limit].copy()
        rows = [
            ClinicianReviewQueueEntry(**_build_clinician_review_queue_row(record))
            for record in page.to_dict(orient="records")
        ]
        return ClinicianReviewQueueResponse(
            generated_at=str(report["generated_at"]),
            total_queue_rows=int(len(queue)),
            filtered_queue_rows=int(len(filtered)),
            filters_applied={
                "unreviewed_only": bool(unreviewed_only),
                "medication_class": medication_class,
                "review_status": review_status,
                "subject_id": subject_id,
                "reason_tag_presence": reason_tag_presence,
                "limit": limit,
                "offset": offset,
            },
            workflow_report=ClinicianReviewWorkflowReport(**report),
            rows=rows,
        )

    @app.get("/clinician-reviews/blind-eval-slice", response_model=ClinicianReviewQueueResponse)
    def get_targeted_blind_eval_slice(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        unreviewed_only: bool = False,
        medication_class: str | None = None,
        review_status: str = Query(
            default="all",
            pattern="^(all|unreviewed|reviewed|uncertain|insufficient_context|skip)$",
        ),
        subject_id: int | None = None,
        priority_band: str = Query(
            default="all",
            pattern="^(all|model_rule_disagreement|uncertain_existing_review|medium_high_boundary|high_signal_unreviewed_fallback|mixed_signal_candidate)$",
        ),
    ) -> ClinicianReviewQueueResponse:
        repository = get_clinician_review_repository()
        settings = get_settings()
        try:
            dataset = pd.read_parquet(settings.first_scope_dataset_output_path)
            ordinal_targets = pd.read_parquet(settings.first_scope_ordinal_targets_output_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Targeted blind evaluation slice requires the persisted Dataset v1 and ordinal target artifacts."
                ),
            ) from exc

        reviewable = repository.load_reviewable_universe()
        latest = repository.load_latest_reviews()
        queue = repository.load_review_queue(
            latest_reviews=latest,
            reviewable_universe=reviewable,
        )
        report = repository.write_qc_artifacts(
            latest_reviews=latest,
            reviewable_universe=reviewable,
            review_queue=queue,
        )
        try:
            if settings.targeted_blind_eval_slice_output_path.exists():
                frozen_slice = pd.read_parquet(settings.targeted_blind_eval_slice_output_path)
            else:
                blind_eval_result = build_first_scope_targeted_blind_eval(
                    encounter_medication_dataset=dataset,
                    reviewable_universe=reviewable,
                    latest_clinician_reviews=latest,
                    ordinal_targets=ordinal_targets,
                    random_seed=20260325,
                )
                write_targeted_blind_eval_artifacts(
                    result=blind_eval_result,
                    scored_universe_output_path=settings.first_scope_ordinal_scored_universe_output_path,
                    slice_output_path=settings.targeted_blind_eval_slice_output_path,
                    summary_output_path=settings.targeted_blind_eval_summary_path,
                    report_output_path=settings.targeted_blind_eval_qc_report_path,
                )
                frozen_slice = blind_eval_result.blind_eval_slice
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        blind_eval_slice = overlay_latest_reviews_on_queue_rows(
            queue_rows=frozen_slice,
            latest_reviews=latest,
        )

        filtered = filter_targeted_blind_eval_slice(
            blind_eval_slice,
            unreviewed_only=unreviewed_only,
            medication_class=medication_class,
            review_status=review_status,
            subject_id=subject_id,
            priority_band=priority_band,
        )
        page = filtered.iloc[offset : offset + limit].copy()
        rows = [
            ClinicianReviewQueueEntry(**_build_clinician_review_queue_row(record))
            for record in page.to_dict(orient="records")
        ]
        return ClinicianReviewQueueResponse(
            generated_at=str(report["generated_at"]),
            total_queue_rows=int(len(blind_eval_slice)),
            filtered_queue_rows=int(len(filtered)),
            filters_applied={
                "blind_mode": True,
                "unreviewed_only": bool(unreviewed_only),
                "medication_class": medication_class,
                "review_status": review_status,
                "subject_id": subject_id,
                "priority_band": priority_band,
                "limit": limit,
                "offset": offset,
            },
            workflow_report=ClinicianReviewWorkflowReport(**report),
            rows=rows,
        )

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
def _load_latest_clinician_review_dataframe() -> pd.DataFrame:
    dataframe = get_clinician_review_repository().load_latest_reviews().copy()
    return dataframe.where(pd.notna(dataframe), None)


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


@lru_cache
def _load_review_time_state_dataframe() -> pd.DataFrame:
    settings = get_settings()
    path = settings.encounter_medication_state_rxnorm_output_path
    if not path.exists():
        return pd.DataFrame()
    dataframe = pd.read_parquet(path).copy()
    for column_name in ("review_timestamp",):
        if column_name in dataframe.columns:
            dataframe[column_name] = dataframe[column_name].map(_normalize_value)
    return dataframe.where(pd.notna(dataframe), None)


@lru_cache
def _load_ordinal_scored_universe_dataframe() -> pd.DataFrame:
    settings = get_settings()
    path = settings.first_scope_ordinal_scored_universe_output_path
    if not path.exists():
        return pd.DataFrame()
    dataframe = pd.read_parquet(path).copy()
    if "review_timestamp" in dataframe.columns:
        dataframe["review_timestamp"] = dataframe["review_timestamp"].map(_normalize_value)
    return dataframe.where(pd.notna(dataframe), None)


@lru_cache
def _load_review_time_patient_universe() -> pd.DataFrame:
    state = _load_review_time_state_dataframe().copy()
    if state.empty:
        return state

    ml = _load_ordinal_scored_universe_dataframe().copy()
    if not ml.empty:
        join_columns = [
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
        ]
        ml = ml.loc[
            :,
            join_columns
            + [
                "modeling__row_id",
                "benchmark__current_rule_score",
                "benchmark__current_rule_score_level",
                "prediction__ordinal_priority_level",
                "prediction__probability_low",
                "prediction__probability_medium",
                "prediction__probability_high",
            ],
        ].copy()
        state = state.merge(ml, how="left", on=join_columns)
    else:
        state["modeling__row_id"] = None
        state["benchmark__current_rule_score"] = None
        state["benchmark__current_rule_score_level"] = None
        state["prediction__ordinal_priority_level"] = None
        state["prediction__probability_low"] = None
        state["prediction__probability_medium"] = None
        state["prediction__probability_high"] = None

    encounter_index = _load_encounter_index_dataframe().copy()
    encounter_columns = [
        "subject_id",
        "encounter_id",
        "sex",
        "admission_type",
        "admittime",
        "dischtime",
        "hospital_length_of_stay_days",
        "encounter_source",
        "encounter_start",
        "encounter_end",
    ]
    encounter_subset = encounter_index.loc[:, encounter_columns].drop_duplicates(
        subset=["subject_id", "encounter_id"],
        keep="last",
    )
    state = state.merge(
        encounter_subset,
        how="left",
        on=["subject_id", "encounter_id"],
        suffixes=("", "_encounter"),
    )
    state["ml_priority_label"] = state["prediction__ordinal_priority_level"]
    probability_columns = [
        "prediction__probability_low",
        "prediction__probability_medium",
        "prediction__probability_high",
    ]
    state["ml_priority_confidence"] = (
        state.loc[:, probability_columns]
        .apply(pd.to_numeric, errors="coerce")
        .max(axis=1)
    )
    state["priority_rank"] = state["ml_priority_label"].map(_priority_rank)
    state["priority_rank"] = state["priority_rank"].fillna(-1).astype(int)
    state["priority_confidence_percent"] = (
        state["ml_priority_confidence"].fillna(0).mul(100).round().astype(int)
    )
    return state.where(pd.notna(state), None)


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
    if type(value).__name__ == "ndarray":
        return value.tolist()
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


def _normalize_string_list(value: object) -> list[str]:
    normalized = _normalize_value(value)
    if normalized is None:
        return []
    if isinstance(normalized, list):
        items = normalized
    elif isinstance(normalized, tuple):
        items = list(normalized)
    else:
        items = [normalized]
    cleaned: list[str] = []
    for item in items:
        normalized_item = _normalize_value(item)
        if normalized_item is None:
            continue
        text = str(normalized_item).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _priority_rank(value: object) -> int:
    return {
        "high": 3,
        "medium": 2,
        "low": 1,
    }.get(_normalize_value(value), 0)


def _load_legacy_context_rows(subject_id: int, hadm_id: object) -> pd.DataFrame:
    try:
        legacy = _load_scored_dataframe()
    except HTTPException:
        return pd.DataFrame()
    rows = legacy.loc[legacy["subject_id"] == subject_id].copy()
    if hadm_id is not None and pd.notna(hadm_id):
        rows = rows.loc[rows["hadm_id"] == int(hadm_id)].copy()
    return rows


def _latest_encounters_for_review_subjects(subject_ids: list[int] | None = None) -> pd.DataFrame:
    encounter_index = _load_encounter_index_dataframe().copy()
    if subject_ids is not None:
        encounter_index = encounter_index.loc[encounter_index["subject_id"].isin(subject_ids)].copy()
    if encounter_index.empty:
        return encounter_index
    encounter_index["encounter_sort_time"] = pd.to_datetime(
        encounter_index["encounter_end"],
        errors="coerce",
    )
    encounter_index["encounter_sort_time"] = encounter_index["encounter_sort_time"].fillna(
        pd.to_datetime(encounter_index["encounter_start"], errors="coerce")
    )
    encounter_index = encounter_index.sort_values(
        ["encounter_sort_time", "hadm_id", "stay_id"],
        ascending=[False, False, False],
        na_position="last",
    )
    latest = encounter_index.drop_duplicates(subset=["subject_id"], keep="first")

    latest_reviews = _load_latest_clinician_review_dataframe()
    if latest_reviews.empty:
        return latest

    reviewed = latest_reviews.copy()
    reviewed["review_sort_time"] = pd.to_datetime(reviewed["review_timestamp"], errors="coerce")
    reviewed = reviewed.sort_values(
        ["review_sort_time", "review_submission_timestamp"],
        ascending=[False, False],
        na_position="last",
    ).drop_duplicates(subset=["subject_id"], keep="first")
    reviewed_encounters = encounter_index.merge(
        reviewed.loc[:, ["subject_id", "encounter_id"]],
        how="inner",
        on=["subject_id", "encounter_id"],
    )
    if reviewed_encounters.empty:
        return latest

    reviewed_lookup = {
        int(row["subject_id"]): row
        for row in reviewed_encounters.to_dict(orient="records")
    }
    merged_rows: list[dict[str, object]] = []
    for row in latest.to_dict(orient="records"):
        merged_rows.append(reviewed_lookup.get(int(row["subject_id"]), row))
    return pd.DataFrame(merged_rows)


def _build_review_time_problem_flashes(rows: pd.DataFrame) -> list[ProblemFlash]:
    if rows.empty:
        return []
    classes = set(rows["medication_class_standardized"].dropna().astype(str).tolist())
    flashes: list[ProblemFlash] = []
    if {"opioid", "benzodiazepine"} & classes:
        flashes.append(
            ProblemFlash(
                key="oversedation",
                label="Oversedation",
                severity="high" if "benzodiazepine" in classes else "medium",
                reason="Sedating first-scope medication exposure is present at review time.",
            )
        )
    if {"benzodiazepine", "anticholinergic"} & classes:
        flashes.append(
            ProblemFlash(
                key="fall_risk",
                label="Fall risk",
                severity="medium",
                reason="Fall-prone medication exposure is present at review time.",
            )
        )
    if {"anticholinergic", "antipsychotic", "benzodiazepine"} & classes:
        flashes.append(
            ProblemFlash(
                key="confusion",
                label="Confusion",
                severity="medium",
                reason="Cognitive-risk medication exposure is present at review time.",
            )
        )
    return flashes[:3]


def _dominant_medication_class(rows: pd.DataFrame) -> str | None:
    if rows.empty or "medication_class_standardized" not in rows.columns:
        return None
    class_rows = rows.loc[
        rows["medication_class_standardized"].notna()
        & (rows["medication_class_standardized"] != "unresolved")
    ].copy()
    if class_rows.empty:
        return None
    counts = class_rows["medication_class_standardized"].value_counts()
    return str(counts.index[0]) if not counts.empty else None


def _review_time_priority_tuple(rows: pd.DataFrame) -> tuple[str, int, float]:
    if rows.empty:
        return ("low", 0, 0.0)
    ranked = rows.sort_values(
        ["priority_rank", "ml_priority_confidence", "medication_standardized"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    top_row = ranked.iloc[0]
    label = str(top_row.get("ml_priority_label") or "low")
    raw_confidence = top_row.get("ml_priority_confidence")
    confidence = (
        float(raw_confidence)
        if raw_confidence is not None and not pd.isna(raw_confidence)
        else 0.0
    )
    score = int(round(confidence * 100))
    return (label, score, confidence)


def _build_review_time_patient_summary_rows() -> list[dict[str, object]]:
    universe = _load_review_time_patient_universe().copy()
    if universe.empty:
        return []
    latest_reviews = _load_latest_clinician_review_dataframe()
    if not latest_reviews.empty:
        reviewed_subject_ids = latest_reviews["subject_id"].dropna().astype(int).unique().tolist()
        universe = universe.loc[universe["subject_id"].isin(reviewed_subject_ids)].copy()
    if universe.empty:
        return []

    latest_encounters = _latest_encounters_for_review_subjects(
        sorted(universe["subject_id"].dropna().astype(int).unique().tolist())
    )
    latest_lookup = {
        int(row["subject_id"]): row
        for row in latest_encounters.to_dict(orient="records")
    }
    summary_rows: list[dict[str, object]] = []
    for subject_id, patient_rows in universe.groupby("subject_id", sort=False):
        latest_encounter = latest_lookup.get(int(subject_id))
        latest_rows = (
            patient_rows.loc[patient_rows["encounter_id"] == latest_encounter["encounter_id"]].copy()
            if latest_encounter is not None
            else patient_rows.copy()
        )
        active_rows = latest_rows.loc[latest_rows["active_at_review_flag"] == 1].copy()
        ranking_rows = active_rows if not active_rows.empty else latest_rows
        highest_label, highest_score, highest_confidence = _review_time_priority_tuple(ranking_rows)
        legacy_rows = _load_legacy_context_rows(
            subject_id=int(subject_id),
            hadm_id=(latest_encounter or {}).get("hadm_id"),
        )
        flashes = (
            _build_problem_flashes(legacy_rows)
            if not legacy_rows.empty
            else _build_review_time_problem_flashes(ranking_rows)
        )
        summary_rows.append(
            {
                "subject_id": int(subject_id),
                "sex": str((latest_encounter or {}).get("sex") or latest_rows.iloc[0].get("sex") or ""),
                "age_proxy": int((latest_rows["age_proxy"].dropna().iloc[0]) if latest_rows["age_proxy"].notna().any() else 0),
                "age_group": str((latest_rows["age_group"].dropna().iloc[0]) if latest_rows["age_group"].notna().any() else ""),
                "encounter_count": int(patient_rows["encounter_id"].nunique(dropna=True)),
                "current_medication_count": int(len(active_rows)),
                "historical_medication_count": int(len(latest_rows.loc[latest_rows["active_at_review_flag"] != 1])),
                "medication_count": int(len(latest_rows)),
                "flagged_medication_count": int(
                    ranking_rows["ml_priority_label"].isin(["medium", "high"]).sum()
                ),
                "highest_priority_score": highest_score,
                "highest_priority_label": highest_label,
                "highest_priority_confidence": highest_confidence if highest_confidence > 0 else None,
                "dominant_medication_class": _dominant_medication_class(ranking_rows),
                "top_problem_flashes": [flash.label for flash in flashes],
                "latest_hadm_id": (
                    int(latest_encounter["hadm_id"])
                    if latest_encounter is not None
                    and latest_encounter.get("hadm_id") is not None
                    and pd.notna(latest_encounter.get("hadm_id"))
                    else None
                ),
                "latest_admission_type": (
                    str(latest_encounter.get("admission_type"))
                    if latest_encounter is not None and latest_encounter.get("admission_type") is not None
                    else None
                ),
                "latest_dischtime": (
                    str(latest_encounter.get("dischtime"))
                    if latest_encounter is not None and latest_encounter.get("dischtime") is not None
                    else None
                ),
            }
        )
    return summary_rows


def _build_balanced_patient_order(summary_frame: pd.DataFrame) -> list[int]:
    if summary_frame.empty:
        return []
    queues: dict[str, list[dict[str, object]]] = {}
    for label in ("high", "medium", "low"):
        subset = summary_frame.loc[summary_frame["highest_priority_label"] == label].copy()
        subset = subset.sort_values(
            ["highest_priority_confidence", "flagged_medication_count", "subject_id"],
            ascending=[False, False, True],
            na_position="last",
        )
        queues[label] = subset.to_dict(orient="records")

    exposure_counts: dict[str | None, int] = {}
    ordered_subject_ids: list[int] = []
    while any(queues.values()):
        for label in ("high", "medium", "low"):
            candidates = queues[label]
            if not candidates:
                continue
            chosen_index = min(
                range(len(candidates)),
                key=lambda index: (
                    exposure_counts.get(candidates[index].get("dominant_medication_class"), 0),
                    -int(candidates[index].get("flagged_medication_count") or 0),
                    -int(candidates[index].get("highest_priority_score") or 0),
                    int(candidates[index]["subject_id"]),
                ),
            )
            chosen = candidates.pop(chosen_index)
            dominant_class = chosen.get("dominant_medication_class")
            exposure_counts[dominant_class] = exposure_counts.get(dominant_class, 0) + 1
            ordered_subject_ids.append(int(chosen["subject_id"]))
    return ordered_subject_ids


def _build_review_time_patient_summaries() -> pd.DataFrame:
    summary_frame = pd.DataFrame(_build_review_time_patient_summary_rows())
    if summary_frame.empty:
        return summary_frame
    ordered_subject_ids = _build_balanced_patient_order(summary_frame)
    order_lookup = {subject_id: index for index, subject_id in enumerate(ordered_subject_ids)}
    summary_frame["_balanced_order"] = summary_frame["subject_id"].map(order_lookup)
    summary_frame = summary_frame.sort_values("_balanced_order", na_position="last").reset_index(drop=True)
    return summary_frame.drop(columns="_balanced_order")


def _build_review_time_patient_summary(*, subject_id: int) -> PatientSummary:
    summaries = _build_review_time_patient_summaries()
    matched = summaries.loc[summaries["subject_id"] == subject_id].copy()
    if matched.empty:
        raise HTTPException(status_code=404, detail="No patient found for the provided subject_id.")
    return PatientSummary(**_clean_record(matched.iloc[0].to_dict()))


def _build_review_time_patient_encounter_summaries(*, subject_id: int) -> list[PatientEncounterSummary]:
    universe = _load_review_time_patient_universe().copy()
    patient_rows = universe.loc[universe["subject_id"] == subject_id].copy()
    if patient_rows.empty:
        return []
    summaries: list[PatientEncounterSummary] = []
    for encounter_id, encounter_rows in patient_rows.groupby("encounter_id", sort=False):
        active_rows = encounter_rows.loc[encounter_rows["active_at_review_flag"] == 1].copy()
        ranking_rows = active_rows if not active_rows.empty else encounter_rows
        label, score, confidence = _review_time_priority_tuple(ranking_rows)
        top_row = encounter_rows.iloc[0].to_dict()
        summaries.append(
            PatientEncounterSummary(
                subject_id=int(subject_id),
                hadm_id=_safe_int(top_row.get("hadm_id")),
                admission_type=str(top_row.get("admission_type") or "unknown"),
                admittime=str(top_row.get("admittime") or ""),
                dischtime=str(top_row.get("dischtime") or ""),
                length_of_stay_days=float(top_row.get("hospital_length_of_stay_days") or 0.0),
                overall_priority_score=score,
                overall_priority_label=label,
                overall_priority_confidence=confidence if confidence > 0 else None,
                current_medication_count=int(len(active_rows)),
                current_polypharmacy_flag=int(len(active_rows) >= get_settings().polypharmacy_threshold),
                historical_medication_count=int(len(encounter_rows.loc[encounter_rows["active_at_review_flag"] != 1])),
                total_medication_count=int(len(encounter_rows)),
                peak_concurrent_medication_count=None,
                flagged_medication_count=int(
                    ranking_rows["ml_priority_label"].isin(["medium", "high"]).sum()
                ),
                overall_priority_drivers=_review_time_priority_drivers(ranking_rows),
            )
        )
    summaries.sort(
        key=lambda item: (
            -_priority_rank(item.overall_priority_label),
            -(item.overall_priority_confidence or 0.0),
            -item.flagged_medication_count,
            item.hadm_id,
        )
    )
    return summaries


def _build_review_time_context_object(
    *,
    patient_rows: pd.DataFrame,
    encounter_rows: pd.DataFrame,
    current_cards: list[PatientMedicationCard],
    history_cards: list[PatientMedicationCard],
) -> PatientContextObject:
    latest_row = encounter_rows.iloc[0].to_dict() if not encounter_rows.empty else patient_rows.iloc[0].to_dict()
    current_count = len(current_cards)
    return PatientContextObject(
        sex=str(latest_row.get("sex") or ""),
        age_proxy=_safe_int(latest_row.get("age_proxy")),
        age_group=str(latest_row.get("age_group") or ""),
        current_medication_count=current_count,
        historical_medication_count=len(history_cards),
        encounter_count=int(patient_rows["encounter_id"].nunique(dropna=True)),
        medication_count=int(len(encounter_rows) if not encounter_rows.empty else len(patient_rows)),
        peak_concurrent_medication_count=None,
        flagged_medication_count=int(
            sum(card.ml_priority_label in {"medium", "high"} for card in current_cards)
        ),
        polypharmacy_present=current_count >= get_settings().polypharmacy_threshold,
        renal_risk_present=False,
        ckd_present=False,
        dementia_present=False,
        delirium_present=False,
        heart_failure_present=False,
        diabetes_present=False,
    )



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


def _review_time_priority_drivers(rows: pd.DataFrame) -> list[str]:
    drivers: list[str] = []
    dominant_class = _dominant_medication_class(rows)
    if dominant_class is not None:
        drivers.append(f"classe {dominant_class}")
    active_count = int((rows.get("active_at_review_flag") == 1).sum()) if "active_at_review_flag" in rows.columns else 0
    if active_count > 0:
        drivers.append(f"{active_count} médicament(s) actif(s) au temps de revue")
    for label in rows.get("ml_priority_label", pd.Series(dtype="object")).dropna().astype(str).unique().tolist():
        if label not in {"low", "medium", "high"}:
            continue
        drivers.append(f"signal ML {label}")
    return drivers[:4]


def _review_row_medication_classes(review_row: dict, reviewable_row: dict | None) -> list[str]:
    ordered: list[str] = []
    review_class = _normalized_lookup_text(review_row.get("medication_class_standardized"))
    if review_class and review_class != "unresolved":
        ordered.append(_title_case_medication_class_label(review_class))
    if reviewable_row is not None:
        reviewable_class = _normalized_lookup_text(reviewable_row.get("medication_class_standardized"))
        if reviewable_class and reviewable_class != "unresolved":
            title_cased = _title_case_medication_class_label(reviewable_class)
            if title_cased not in ordered:
                ordered.append(title_cased)
    return ordered


def _priority_distance(left: object, right: object) -> int | None:
    left_rank = _priority_rank(left)
    right_rank = _priority_rank(right)
    if left_rank == 0 or right_rank == 0:
        return None
    return abs(left_rank - right_rank)


def _pharmacist_review_alignment(review_row: dict, clinician_review: dict | None) -> tuple[str | None, str | None]:
    if clinician_review is None:
        return (None, None)
    distance = _priority_distance(
        review_row.get("ml_priority_label"),
        clinician_review.get("label__clinician_priority_level"),
    )
    if distance is None:
        return (None, None)
    if distance == 0:
        return ("agreed", "Accord pharmacien")
    if distance == 1:
        return ("off_by_one", "Écart 1 niveau")
    return ("corrected", "Corrigé par pharmacien")


def _ml_confidence_percent(review_row: dict) -> int:
    confidence = review_row.get("ml_priority_confidence")
    if confidence is None or pd.isna(confidence):
        return 0
    return int(round(float(confidence) * 100))


def _review_time_summary_alert(review_row: dict) -> str:
    ml_label = review_row.get("ml_priority_label")
    if ml_label:
        return f"ML pilote {ml_label} • confiance {_ml_confidence_percent(review_row)}%"
    standardized = review_row.get("medication_standardized") or review_row.get("ingredient_standardized")
    if standardized:
        return f"Information RxNorm minimale • {standardized}"
    return "Information minimale disponible au temps de revue."


def _review_time_explanation(review_row: dict) -> str:
    parts: list[str] = []
    ml_label = review_row.get("ml_priority_label")
    if ml_label:
        parts.append(f"Priorité ML pilote {ml_label}")
        parts.append(f"confiance {_ml_confidence_percent(review_row)}%")
    else:
        parts.append("Hors première portée ML")
    review_class = review_row.get("medication_class_standardized")
    if review_class and review_class != "unresolved":
        parts.append(f"classe {review_class}")
    status = review_row.get("medication_status_at_review")
    if status:
        parts.append(f"statut {status}")
    ingredient = review_row.get("ingredient_standardized")
    if ingredient:
        parts.append(f"ingrédient {ingredient}")
    return " ; ".join(parts)


def _review_time_evidence(review_row: dict) -> dict[str, object]:
    evidence: dict[str, object] = {}
    if review_row.get("route") is not None:
        evidence["route"] = review_row.get("route")
    if review_row.get("frequency") is not None:
        evidence["frequency"] = review_row.get("frequency")
    if review_row.get("medication_status_at_review") is not None:
        evidence["statut_revue"] = review_row.get("medication_status_at_review")
    if review_row.get("ingredient_standardized") is not None:
        evidence["ingredient_rxnorm"] = review_row.get("ingredient_standardized")
    if review_row.get("rxnorm_term_type") is not None:
        evidence["type_rxnorm"] = review_row.get("rxnorm_term_type")
    if review_row.get("ml_priority_label") is not None:
        evidence["confiance_ml"] = f"{_ml_confidence_percent(review_row)}%"
    return evidence


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
    else:
        preferred = _latest_encounters_for_review_subjects([subject_id])
        if not preferred.empty:
            preferred_encounter_id = preferred.iloc[0].get("encounter_id")
            matched = patient_encounters.loc[
                patient_encounters["encounter_id"] == preferred_encounter_id
            ].copy()
            if not matched.empty:
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
    review_rows: pd.DataFrame,
    selected_encounter: dict,
) -> tuple[list[PatientMedicationCard], list[PatientMedicationCard], int]:
    if review_rows.empty:
        return [], [], 0

    current_review = review_rows.loc[
        review_rows["active_at_review_flag"] == 1
    ].copy()
    history_review = review_rows.loc[
        review_rows["active_at_review_flag"] != 1
    ].copy()

    current_cards = _build_review_medication_cards(current_review, selected_encounter)
    history_cards = _build_review_medication_cards(history_review, selected_encounter)
    flagged_current_count = int(
        sum(card.ml_priority_label in {"medium", "high"} for card in current_cards)
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
    selected_encounter: dict,
) -> list[PatientMedicationCard]:
    if review_rows.empty:
        return []

    cards: list[PatientMedicationCard] = []
    review_repository = get_clinician_review_repository()
    latest_review_lookup = _latest_clinician_review_lookup(review_repository.load_latest_reviews())
    ranked_review = review_rows.sort_values(
        ["priority_rank", "ml_priority_confidence", "medication_normalized"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    for review_row in ranked_review.to_dict(orient="records"):
        cards.append(
            _build_review_medication_card(
                review_row,
                selected_encounter,
                review_repository=review_repository,
                latest_review_lookup=latest_review_lookup,
            )
        )

    cards.sort(
        key=lambda card: (
            -_priority_rank(card.ml_priority_label),
            -(card.ml_priority_confidence or 0.0),
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
    *,
    review_repository: ClinicianReviewRepository,
    latest_review_lookup: dict[tuple[int, str, str, str], dict],
) -> PatientMedicationCard:
    reviewable_row = review_repository.resolve_reviewable_row(
        subject_id=int(review_row["subject_id"]),
        encounter_id=str(selected_encounter["encounter_id"]),
        review_timestamp=review_row.get("review_timestamp"),
        medication_candidates=[
            review_row.get("medication_standardized"),
            review_row.get("medication_normalized"),
            review_row.get("medication_raw"),
        ],
        selected_event_id=review_row.get("selected_medication_event_id"),
    )
    medication_classes = _review_row_medication_classes(review_row, reviewable_row)
    ml_label = _normalize_value(review_row.get("ml_priority_label"))
    ml_confidence = (
        float(review_row.get("ml_priority_confidence"))
        if review_row.get("ml_priority_confidence") is not None
        and not pd.isna(review_row.get("ml_priority_confidence"))
        else None
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
        "drug": str(
            review_row.get("medication_raw")
            or review_row.get("medication_standardized")
            or review_row.get("medication_normalized")
            or "Medication"
        ),
        "drug_normalized": review_row.get("medication_normalized"),
        "medication_classes": medication_classes,
        "starttime": str(review_row.get("review_timestamp") or ""),
        "stoptime": None,
        "medication_episode_id": review_row.get("selected_medication_event_id"),
        "prescription_segment_count": None,
        "prescription_segments_json": [],
        "deprescribing_priority_score": _ml_confidence_percent(review_row),
        "deprescribing_priority_label": str(ml_label or "low"),
        "deprescribing_priority_summary_alert": _review_time_summary_alert(review_row),
        "deprescribing_priority_explanation": _review_time_explanation(review_row),
        "deprescribing_priority_bucket_scores_json": {},
        "deprescribing_priority_reasons_json": _review_time_priority_drivers(pd.DataFrame([review_row])),
        "deprescribing_priority_evidence_json": _review_time_evidence(review_row),
        "benzodiazepine_flag": int("Benzodiazepine" in medication_classes),
        "opioid_flag": int("Opioid" in medication_classes),
        "anticholinergic_flag": int("Anticholinergic" in medication_classes),
        "ppi_flag": int("PPI" in medication_classes),
        "antipsychotic_flag": int("Antipsychotic" in medication_classes),
        "status": review_row.get("status"),
        "medication_status": review_row.get("medication_status_at_review"),
        "last_active_time": review_row.get("last_active_time"),
        "review_timestamp": review_row.get("review_timestamp"),
        "review_timestamp_source": review_row.get("review_timestamp_source"),
        "priority_score_source": (
            "ml_ordinal_scored_universe"
            if ml_label is not None
            else "review_time_state_only"
        ),
        "medication_standardized": review_row.get("medication_standardized"),
        "rxnorm_rxcui": review_row.get("rxnorm_rxcui"),
        "rxnorm_term_type": review_row.get("rxnorm_term_type"),
        "ingredient_standardized": review_row.get("ingredient_standardized"),
        "ml_priority_label": ml_label,
        "ml_priority_confidence": ml_confidence,
        "ml_probability_low": review_row.get("prediction__probability_low"),
        "ml_probability_medium": review_row.get("prediction__probability_medium"),
        "ml_probability_high": review_row.get("prediction__probability_high"),
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
        alignment_key, alignment_label = _pharmacist_review_alignment(review_row, clinician_review)
        queue_priority, queue_reasons = build_review_queue_hints(
            reviewable_row=reviewable_row,
            clinician_review=clinician_review,
            displayed_rule_level=str(base_record["deprescribing_priority_label"]),
        )
        base_record.update(
            {
                "medication_standardized": reviewable_row.get("medication_standardized"),
                "medication_classes": medication_classes if alignment_key is not None else [],
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
                "pharmacist_review_alignment": alignment_key,
                "pharmacist_review_alignment_label": alignment_label,
            }
        )
    else:
        base_record.update(
            {
                "medication_classes": [],
                "reviewable_flag": False,
                "review_queue_priority": "out_of_scope",
                "review_queue_reasons": ["not_in_phase5_review_scope"],
                "clinician_review": None,
                "pharmacist_review_alignment": None,
                "pharmacist_review_alignment_label": None,
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


def _build_clinician_review_queue_row(record: dict[str, Any]) -> dict[str, Any]:
    clinician_review = None
    review_submission_id = _normalize_value(record.get("review_submission_id"))
    if review_submission_id is not None:
        review_version = _normalize_value(record.get("review_version"))
        review_artifact_version = _normalize_value(record.get("review_artifact_version"))
        review_submission_timestamp = _normalize_value(record.get("review_submission_timestamp"))
        reviewer_id = _normalize_value(record.get("reviewer_id"))
        clinician_priority_level = _normalize_value(record.get("label__clinician_priority_level"))
        clinician_review_status = _normalize_value(record.get("label__clinician_review_status"))
        clinician_reason_tags = _normalize_string_list(record.get("label__clinician_reason_tags"))
        clinician_reviewed_flag = _normalize_value(record.get("label__clinician_reviewed_flag"))
        clinician_review = {
            "subject_id": int(record["subject_id"]),
            "encounter_id": str(record["encounter_id"]),
            "hadm_id": _optional_int(record.get("hadm_id")),
            "stay_id": _optional_int(record.get("stay_id")),
            "medication_standardized": str(record["medication_standardized"]),
            "medication_normalized": record.get("medication_normalized"),
            "review_timestamp": str(record["review_timestamp"]),
            "modeling__row_id": record.get("modeling__row_id"),
            "review_submission_id": str(review_submission_id),
            "review_version": int(review_version) if review_version is not None else 1,
            "review_artifact_version": (
                str(review_artifact_version) if review_artifact_version is not None else ""
            ),
            "review_submission_timestamp": (
                str(review_submission_timestamp)
                if review_submission_timestamp is not None
                else ""
            ),
            "reviewer_id": str(reviewer_id) if reviewer_id is not None else "",
            "label__clinician_priority_level": (
                str(clinician_priority_level) if clinician_priority_level is not None else "low"
            ),
            "label__clinician_priority_score": record.get("label__clinician_priority_score"),
            "label__clinician_priority_score_level": record.get(
                "label__clinician_priority_score_level"
            ),
            "label__clinician_review_status": (
                str(clinician_review_status)
                if clinician_review_status is not None
                else "reviewed"
            ),
            "label__clinician_reason_tags": clinician_reason_tags,
            "label__clinician_note": record.get("label__clinician_note"),
            "label__clinician_reviewed_flag": (
                int(clinician_reviewed_flag) if clinician_reviewed_flag is not None else 0
            ),
            "label__clinician_suggested_action": record.get(
                "label__clinician_suggested_action"
            ),
            "review_provenance_json": record.get("review_provenance_json"),
        }

    payload = {
        "subject_id": int(record["subject_id"]),
        "encounter_id": str(record["encounter_id"]),
        "hadm_id": _optional_int(record.get("hadm_id")),
        "stay_id": _optional_int(record.get("stay_id")),
        "review_timestamp": str(record["review_timestamp"]),
        "medication_standardized": str(record["medication_standardized"]),
        "medication_normalized": record.get("medication_normalized"),
        "medication_class_standardized": record.get("medication_class_standardized"),
        "first_scope_supported_class_flag": _optional_int(record.get("first_scope_supported_class_flag")),
        "medication_status_at_review": record.get("medication_status_at_review"),
        "active_at_review_flag": _optional_int(record.get("active_at_review_flag")),
        "dose_value": record.get("dose_value"),
        "dose_unit": record.get("dose_unit"),
        "route": record.get("route"),
        "frequency": record.get("frequency"),
        "modeling__row_id": record.get("modeling__row_id"),
        "benchmark__current_rule_score": record.get("benchmark__current_rule_score"),
        "benchmark__current_rule_score_level": record.get("benchmark__current_rule_score_level"),
        "benchmark__current_rule_available_flag": _optional_int(
            record.get("benchmark__current_rule_available_flag")
        ),
        "label__primary_action_label": record.get("label__primary_action_label"),
        "label__unknown_or_insufficient_evidence_flag": _optional_int(
            record.get("label__unknown_or_insufficient_evidence_flag")
        ),
        "meta__dataset_row_eligible_for_training_flag": _optional_int(
            record.get("meta__dataset_row_eligible_for_training_flag")
        ),
        "reviewable_flag": bool(record.get("reviewable_flag", True)),
        "current_clinician_reviewed_flag": int(
            record.get("current_clinician_reviewed_flag") or 0
        ),
        "queue_rank": int(record.get("queue_rank") or 0),
        "queue_priority_score": int(record.get("queue_priority_score") or 0),
        "queue_priority_band": str(record.get("queue_priority_band") or "review_backlog"),
        "queue_priority_reasons": _normalize_string_list(record.get("queue_priority_reasons")),
        "needs_review_justification": str(record.get("needs_review_justification") or ""),
        "class_reviewed_count": int(record.get("class_reviewed_count") or 0),
        "class_reviewable_count": int(record.get("class_reviewable_count") or 0),
        "subject_reviewed_count": int(record.get("subject_reviewed_count") or 0),
        "subject_reviewable_count": int(record.get("subject_reviewable_count") or 0),
        "encounter_reviewed_count": int(record.get("encounter_reviewed_count") or 0),
        "encounter_reviewable_count": int(record.get("encounter_reviewable_count") or 0),
        "clinician_review": clinician_review,
    }
    return _clean_record(payload)


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


def _optional_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
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
