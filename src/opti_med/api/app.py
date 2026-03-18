"""FastAPI app exposing the saved OPTI-MED scored output."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from opti_med.api.repository import ScoredDataRepository
from opti_med.api.schemas import (
    AdmissionDetailResponse,
    AdmissionSummariesResponse,
    AdmissionSummary,
    HealthResponse,
    MedicationRowSummary,
    RefreshScoresResponse,
    ScoredOutputSummary,
    ScoredRow,
    ScoredRowsResponse,
)
from opti_med.config import Settings
from opti_med.scoring.scorer import DeprescribingPriorityScorer


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
            total_medication_count=int(top_row["total_medication_count"]),
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


def _load_scored_dataframe() -> pd.DataFrame:
    repository = get_repository()
    try:
        return repository.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _clean_record(record: dict) -> dict:
    return {
        key: (_normalize_value(value))
        for key, value in record.items()
    }


def _normalize_value(value: object) -> object:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
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
                "total_medication_count": int(top_row["total_medication_count"]),
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


def _build_medication_summary(record: dict) -> MedicationRowSummary:
    return MedicationRowSummary(
        drug=str(record["drug"]),
        medication_classes=_medication_classes_from_record(record),
        starttime=str(record["starttime"]),
        stoptime=record.get("stoptime"),
        deprescribing_priority_score=int(record["deprescribing_priority_score"]),
        deprescribing_priority_label=str(record["deprescribing_priority_label"]),
        deprescribing_priority_explanation=str(record["deprescribing_priority_explanation"]),
        benzodiazepine_flag=int(record["benzodiazepine_flag"]),
        opioid_flag=int(record["opioid_flag"]),
        anticholinergic_flag=int(record["anticholinergic_flag"]),
        ppi_flag=int(record["ppi_flag"]),
        antipsychotic_flag=int(record["antipsychotic_flag"]),
    )


app = create_app()
