"""Phase 5 clinician review storage, lookup, and QC helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.provenance import TIMESTAMP_FORMAT, json_ready_value
from opti_med.scoring.priority_levels import derive_priority_level_series


CLINICIAN_REVIEW_EVENT_CONTRACT_VERSION = "clinician_review_events_phase5.v1"
CLINICIAN_REVIEW_SNAPSHOT_CONTRACT_VERSION = "clinician_review_labels_phase5.v1"
CLINICIAN_REVIEW_UI_SOURCE = "dossier_ui_phase5"
REVIEW_KEY_COLUMNS = [
    "subject_id",
    "encounter_id",
    "medication_standardized",
    "review_timestamp",
]
ALLOWED_PRIORITY_LEVELS = ("low", "medium", "high")
ALLOWED_REVIEW_STATUSES = ("reviewed", "uncertain", "insufficient_context", "skip")
ALLOWED_REASON_TAGS = (
    "polypharmacy",
    "duplication",
    "renal_risk",
    "fall_risk",
    "anticholinergic_burden",
    "interaction_risk",
    "questionable_indication",
    "monitoring_needed",
    "tapering_candidate",
    "insufficient_context",
    "other",
)
ALLOWED_SUGGESTED_ACTIONS = (
    "keep",
    "monitor",
    "reconsider",
    "deprescribe_candidate",
    "needs_more_info",
)
REVIEWABLE_COLUMNS = [
    *REVIEW_KEY_COLUMNS,
    "hadm_id",
    "stay_id",
    "selected_medication_event_id",
    "selected_medication_event_type",
    "medication_normalized",
    "medication_class_standardized",
    "first_scope_supported_class_flag",
    "medication_status_at_review",
    "active_at_review_flag",
    "dose_value",
    "dose_unit",
    "route",
    "frequency",
    "encounter_medication_first_scope_contract_version",
    "meta__dataset_contract_version",
    "benchmark__current_rule_score",
    "benchmark__current_rule_score_level",
    "benchmark__current_rule_available_flag",
    "benchmark__medication_class_only_medication_class_standardized",
    "benchmark__heuristic_any_supported_class_flag",
    "label__primary_action_label",
    "label__unknown_or_insufficient_evidence_flag",
    "meta__dataset_row_eligible_for_training_flag",
    "modeling__row_id",
]


@dataclass(frozen=True, slots=True)
class ClinicianReviewRepository:
    """File-backed append-only repository for Phase 5 pharmacist reviews."""

    settings: Settings

    def load_reviewable_universe(self) -> pd.DataFrame:
        """Return the analytical-grain reviewable rows available in this workspace."""
        first_scope_path = self.settings.encounter_medication_first_scope_output_path
        if not first_scope_path.exists():
            return pd.DataFrame(columns=REVIEWABLE_COLUMNS)

        first_scope = pd.read_parquet(first_scope_path).copy()
        if first_scope.empty:
            return pd.DataFrame(columns=REVIEWABLE_COLUMNS)

        base = first_scope.loc[
            :,
            [
                "subject_id",
                "encounter_id",
                "hadm_id",
                "stay_id",
                "review_timestamp",
                "medication_standardized",
                "selected_medication_event_id",
                "selected_medication_event_type",
                "medication_normalized",
                "medication_class_standardized",
                "first_scope_supported_class_flag",
                "medication_status_at_review",
                "active_at_review_flag",
                "dose_value",
                "dose_unit",
                "route",
                "frequency",
                "encounter_medication_first_scope_contract_version",
            ],
        ].copy()
        base["review_timestamp"] = base["review_timestamp"].map(_normalize_review_timestamp)
        base["hadm_id"] = pd.to_numeric(base["hadm_id"], errors="coerce")
        base["stay_id"] = pd.to_numeric(base["stay_id"], errors="coerce")
        base["meta__dataset_contract_version"] = pd.NA
        _assert_no_duplicate_review_keys(base, frame_name="encounter_medication_first_scope")

        dataset_path = self.settings.first_scope_dataset_output_path
        if dataset_path.exists():
            dataset = pd.read_parquet(dataset_path).copy()
            dataset["review_timestamp"] = dataset["review_timestamp"].map(_normalize_review_timestamp)
            dataset["modeling__row_id"] = dataset.apply(_build_modeling_row_id, axis=1)
            dataset["benchmark__current_rule_score_level"] = derive_priority_level_series(
                dataset.get("benchmark__current_rule_score")
            )
            dataset_frame = dataset.loc[
                :,
                [
                    *REVIEW_KEY_COLUMNS,
                    "meta__dataset_contract_version",
                    "benchmark__current_rule_score",
                    "benchmark__current_rule_score_level",
                    "benchmark__current_rule_available_flag",
                    "benchmark__medication_class_only_medication_class_standardized",
                    "benchmark__heuristic_any_supported_class_flag",
                    "label__primary_action_label",
                    "label__unknown_or_insufficient_evidence_flag",
                    "meta__dataset_row_eligible_for_training_flag",
                    "modeling__row_id",
                ],
            ].copy()
            _assert_no_duplicate_review_keys(dataset_frame, frame_name="first_scope_dataset")
            base = base.merge(
                dataset_frame,
                how="left",
                on=REVIEW_KEY_COLUMNS,
                validate="one_to_one",
                suffixes=("", "_dataset"),
            )
        else:
            for column_name in [
                "benchmark__current_rule_score",
                "benchmark__current_rule_score_level",
                "benchmark__current_rule_available_flag",
                "benchmark__medication_class_only_medication_class_standardized",
                "benchmark__heuristic_any_supported_class_flag",
                "label__primary_action_label",
                "label__unknown_or_insufficient_evidence_flag",
                "meta__dataset_row_eligible_for_training_flag",
                "modeling__row_id",
            ]:
                base[column_name] = pd.NA

        for column_name in REVIEWABLE_COLUMNS:
            if column_name not in base.columns:
                base[column_name] = pd.NA
        base = base.loc[:, REVIEWABLE_COLUMNS].copy()
        return base.sort_values(REVIEW_KEY_COLUMNS).reset_index(drop=True)

    def resolve_reviewable_row(
        self,
        *,
        subject_id: int,
        encounter_id: str,
        review_timestamp: str | None,
        medication_candidates: list[str],
        selected_event_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a dossier medication row back to the analytical grain when possible."""
        reviewable = self.load_reviewable_universe()
        if reviewable.empty:
            return None

        matched = reviewable.loc[
            (pd.to_numeric(reviewable["subject_id"], errors="coerce") == int(subject_id))
            & (reviewable["encounter_id"].astype(str) == str(encounter_id))
        ].copy()
        if review_timestamp:
            normalized_review_timestamp = _normalize_review_timestamp(review_timestamp)
            matched = matched.loc[
                matched["review_timestamp"].astype(str) == normalized_review_timestamp
            ].copy()
        if matched.empty:
            return None

        if selected_event_id:
            selected_event_matches = matched.loc[
                matched["selected_medication_event_id"].astype(str) == str(selected_event_id)
            ].copy()
            if len(selected_event_matches) == 1:
                resolved = _json_ready_dict(selected_event_matches.iloc[0].to_dict())
                resolved["traceability_resolution_method"] = "selected_medication_event_id"
                return resolved
            if len(selected_event_matches) > 1:
                return None

        normalized_candidates = {
            _normalized_medication_text(candidate)
            for candidate in medication_candidates
            if _normalized_medication_text(candidate)
        }
        if not normalized_candidates:
            return None

        matched["_candidate_match"] = matched.apply(
            lambda row: _row_matches_any_candidate(row, normalized_candidates),
            axis=1,
        )
        matched = matched.loc[matched["_candidate_match"]].copy()
        if matched.empty:
            return None
        unique_keys = matched.loc[:, REVIEW_KEY_COLUMNS].drop_duplicates()
        if len(unique_keys) != 1:
            return None

        matched = matched.sort_values(
            ["active_at_review_flag", "first_scope_supported_class_flag", "medication_standardized"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        resolved = _json_ready_dict(matched.iloc[0].drop(labels="_candidate_match").to_dict())
        resolved["traceability_resolution_method"] = "medication_text_fallback"
        return resolved

    def load_latest_reviews(self) -> pd.DataFrame:
        """Return the latest active clinician review rows keyed to the analytical grain."""
        snapshot_path = self.settings.clinician_review_snapshot_output_path
        if snapshot_path.exists():
            dataframe = pd.read_parquet(snapshot_path).copy()
            return _normalize_review_snapshot(dataframe)

        events = self.load_review_events()
        if not events.empty:
            return self._materialize_latest_snapshot(events)
        return pd.DataFrame()

    def load_review_events(self) -> pd.DataFrame:
        """Return the append-only clinician review event log."""
        path = self.settings.clinician_review_event_log_path
        if not path.exists():
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
        if not rows:
            return pd.DataFrame()
        return _normalize_review_snapshot(pd.DataFrame(rows))

    def save_review(
        self,
        *,
        submission: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Append one clinician review event and refresh the latest-active artifacts."""
        reviewable = self.load_reviewable_universe()
        if reviewable.empty:
            raise ValueError("No reviewable analytical rows are available in this workspace.")

        normalized_review_timestamp = _normalize_review_timestamp(submission.get("review_timestamp"))
        matched = reviewable.loc[
            (pd.to_numeric(reviewable["subject_id"], errors="coerce") == int(submission["subject_id"]))
            & (reviewable["encounter_id"].astype(str) == str(submission["encounter_id"]))
            & (
                reviewable["medication_standardized"].astype(str)
                == str(submission["medication_standardized"])
            )
            & (reviewable["review_timestamp"].astype(str) == normalized_review_timestamp)
        ].copy()
        if matched.empty:
            raise KeyError(
                "No reviewable analytical-grain row matched the submitted subject, encounter, medication, and review timestamp."
            )

        current_reviews = self.load_latest_reviews()
        review_version = 1
        if not current_reviews.empty:
            existing_versions = current_reviews.loc[
                (pd.to_numeric(current_reviews["subject_id"], errors="coerce") == int(submission["subject_id"]))
                & (current_reviews["encounter_id"].astype(str) == str(submission["encounter_id"]))
                & (
                    current_reviews["medication_standardized"].astype(str)
                    == str(submission["medication_standardized"])
                )
                & (current_reviews["review_timestamp"].astype(str) == normalized_review_timestamp),
                "review_version",
            ]
            if not existing_versions.empty and existing_versions.notna().any():
                review_version = int(pd.to_numeric(existing_versions, errors="coerce").max()) + 1

        reviewer_id = str(
            submission.get("reviewer_id") or self.settings.default_clinician_reviewer_id
        ).strip()
        submitted_modeling_row_id = _optional_text(submission.get("modeling__row_id"))
        matched_modeling_row_id = _optional_text(matched.iloc[0].get("modeling__row_id"))
        if submitted_modeling_row_id and matched_modeling_row_id and submitted_modeling_row_id != matched_modeling_row_id:
            raise ValueError(
                "Submitted modeling__row_id does not match the resolved analytical-grain row."
            )
        priority_level = str(submission["label__clinician_priority_level"]).strip().lower()
        if priority_level not in ALLOWED_PRIORITY_LEVELS:
            raise ValueError("Unsupported clinician priority level.")
        review_status = str(submission["label__clinician_review_status"]).strip().lower()
        if review_status not in ALLOWED_REVIEW_STATUSES:
            raise ValueError("Unsupported clinician review status.")

        raw_reason_tags = submission.get("label__clinician_reason_tags") or []
        reason_tags = sorted(
            {
                str(item).strip()
                for item in raw_reason_tags
                if str(item).strip() in ALLOWED_REASON_TAGS
            }
        )
        invalid_reason_tags = sorted(
            {
                str(item).strip()
                for item in raw_reason_tags
                if str(item).strip() and str(item).strip() not in ALLOWED_REASON_TAGS
            }
        )
        if invalid_reason_tags:
            raise ValueError(
                "Unsupported clinician reason tags: " + ", ".join(invalid_reason_tags)
            )

        suggested_action = submission.get("label__clinician_suggested_action")
        if suggested_action is not None:
            suggested_action = str(suggested_action).strip()
            if suggested_action and suggested_action not in ALLOWED_SUGGESTED_ACTIONS:
                raise ValueError("Unsupported clinician suggested action.")
            if not suggested_action:
                suggested_action = None

        priority_score = _coerce_optional_numeric_score(
            submission.get("label__clinician_priority_score")
        )
        priority_score_level = (
            str(derive_priority_level_series(pd.Series([priority_score])).iloc[0])
            if priority_score is not None
            else None
        )
        note = _optional_text(submission.get("label__clinician_note"))
        matched_row = _json_ready_dict(matched.iloc[0].to_dict())

        submission_timestamp = (
            datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        )
        submission_id = _build_submission_id(
            subject_id=int(submission["subject_id"]),
            encounter_id=str(submission["encounter_id"]),
            medication_standardized=str(submission["medication_standardized"]),
            review_timestamp=normalized_review_timestamp,
            review_version=review_version,
            reviewer_id=reviewer_id,
        )
        event = {
            "subject_id": int(submission["subject_id"]),
            "encounter_id": str(submission["encounter_id"]),
            "hadm_id": matched_row.get("hadm_id"),
            "stay_id": matched_row.get("stay_id"),
            "medication_standardized": str(submission["medication_standardized"]),
            "medication_normalized": matched_row.get("medication_normalized"),
            "review_timestamp": normalized_review_timestamp,
            "modeling__row_id": matched_row.get("modeling__row_id"),
            "benchmark__current_rule_score": matched_row.get("benchmark__current_rule_score"),
            "benchmark__current_rule_score_level": matched_row.get("benchmark__current_rule_score_level"),
            "benchmark__current_rule_available_flag": matched_row.get("benchmark__current_rule_available_flag"),
            "benchmark__medication_class_only_medication_class_standardized": matched_row.get(
                "benchmark__medication_class_only_medication_class_standardized"
            ),
            "benchmark__heuristic_any_supported_class_flag": matched_row.get(
                "benchmark__heuristic_any_supported_class_flag"
            ),
            "review_submission_id": submission_id,
            "review_version": review_version,
            "review_artifact_version": CLINICIAN_REVIEW_EVENT_CONTRACT_VERSION,
            "review_submission_timestamp": submission_timestamp,
            "reviewer_id": reviewer_id,
            "label__clinician_priority_level": priority_level,
            "label__clinician_priority_score": priority_score,
            "label__clinician_priority_score_level": priority_score_level,
            "label__clinician_review_status": review_status,
            "label__clinician_reason_tags": reason_tags,
            "label__clinician_note": note,
            "label__clinician_reviewed_flag": 1,
            "label__clinician_suggested_action": suggested_action,
            "review_provenance_json": {
                "submission_source": CLINICIAN_REVIEW_UI_SOURCE,
                "latest_as_active_semantics": "append_only_latest_review_version_per_analytical_key",
                "reviewable_row_match_status": "matched",
                "reviewable_row_contract_version": matched_row.get(
                    "encounter_medication_first_scope_contract_version"
                ),
                "dataset_contract_version": matched_row.get("meta__dataset_contract_version"),
                "dataset_row_available_flag": int(matched_row.get("modeling__row_id") is not None),
            },
        }

        _append_jsonl_line(self.settings.clinician_review_event_log_path, event)
        latest_snapshot = self._materialize_latest_snapshot()
        report = self.write_qc_artifacts(
            latest_reviews=latest_snapshot,
            reviewable_universe=reviewable,
        )
        latest_matched = latest_snapshot.loc[
            latest_snapshot["review_submission_id"].astype(str) == submission_id
        ].copy()
        latest_event = event if latest_matched.empty else _json_ready_dict(latest_matched.iloc[0].to_dict())
        return _json_ready_dict(latest_event), report

    def write_qc_artifacts(
        self,
        *,
        latest_reviews: pd.DataFrame | None = None,
        reviewable_universe: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Write the Phase 5 QC summary artifacts and return the structured payload."""
        reviewable = (
            reviewable_universe.copy()
            if reviewable_universe is not None
            else self.load_reviewable_universe()
        )
        latest = (
            latest_reviews.copy()
            if latest_reviews is not None
            else self.load_latest_reviews()
        )
        summary = build_clinician_review_workflow_report(
            latest_reviews=latest,
            reviewable_universe=reviewable,
            event_log_path=self.settings.clinician_review_event_log_path,
            snapshot_path=self.settings.clinician_review_snapshot_output_path,
        )

        qc_json_path = self.settings.clinician_review_qc_summary_path
        qc_json_path.parent.mkdir(parents=True, exist_ok=True)
        qc_json_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        qc_report_path = self.settings.clinician_review_qc_report_path
        qc_report_path.parent.mkdir(parents=True, exist_ok=True)
        qc_report_path.write_text(
            render_clinician_review_workflow_report(summary),
            encoding="utf-8",
        )
        return summary

    def _materialize_latest_snapshot(
        self,
        events: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        events_frame = events.copy() if events is not None else self.load_review_events()
        if events_frame.empty:
            snapshot_path = self.settings.clinician_review_snapshot_output_path
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_parquet(snapshot_path, index=False)
            return pd.DataFrame()

        normalized = _normalize_review_snapshot(events_frame)
        normalized["_submission_sort_time"] = pd.to_datetime(
            normalized["review_submission_timestamp"],
            utc=True,
            errors="coerce",
        )
        latest = normalized.sort_values(
            [*REVIEW_KEY_COLUMNS, "review_version", "_submission_sort_time", "review_submission_id"],
            ascending=[True, True, True, True, True, True, True],
            na_position="last",
        ).drop_duplicates(subset=REVIEW_KEY_COLUMNS, keep="last")
        latest = latest.drop(columns="_submission_sort_time").copy()
        latest["clinician_review_snapshot_contract_version"] = (
            CLINICIAN_REVIEW_SNAPSHOT_CONTRACT_VERSION
        )
        latest["active_review_flag"] = 1
        latest = latest.sort_values(REVIEW_KEY_COLUMNS).reset_index(drop=True)

        snapshot_path = self.settings.clinician_review_snapshot_output_path
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        latest.to_parquet(snapshot_path, index=False)
        return latest


def build_review_queue_hints(
    *,
    reviewable_row: dict[str, Any] | None,
    clinician_review: dict[str, Any] | None,
    displayed_rule_level: str | None,
) -> tuple[str, list[str]]:
    """Return a simple queue priority label and explanatory reasons for one row."""
    if reviewable_row is None:
        return "out_of_scope", ["not_in_phase5_review_scope"]

    reasons: list[str] = []
    clinician_level = None
    if clinician_review is None:
        reasons.append("lacks_clinician_review")
    else:
        clinician_level = _optional_text(
            clinician_review.get("label__clinician_priority_level")
        )
        reasons.append("already_reviewed")

    if clinician_level and displayed_rule_level and clinician_level != displayed_rule_level:
        reasons.append("clinician_rule_disagreement")
    supported_class_flag = pd.to_numeric(
        pd.Series([reviewable_row.get("first_scope_supported_class_flag")]),
        errors="coerce",
    ).fillna(0)
    if int(supported_class_flag.iloc[0]) == 1:
        reasons.append("supported_medication_class")
    if str(reviewable_row.get("label__primary_action_label") or "") == "action_undetermined":
        reasons.append("constructed_label_ambiguous")
    if displayed_rule_level in {"medium", "high"}:
        reasons.append("rule_signal_present")

    if "clinician_rule_disagreement" in reasons:
        priority = "disagreement_candidate"
    elif clinician_review is not None:
        priority = "reviewed"
    elif "rule_signal_present" in reasons or "constructed_label_ambiguous" in reasons:
        priority = "priority"
    else:
        priority = "reviewable"
    return priority, reasons[:4]


def summarize_review_queue(cards: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize simple review queue ergonomics for one patient dossier."""
    reviewable_cards = [card for card in cards if bool(card.get("reviewable_flag"))]
    reviewed_cards = [card for card in reviewable_cards if card.get("clinician_review")]
    disagreement_cards = [
        card
        for card in reviewable_cards
        if card.get("review_queue_priority") == "disagreement_candidate"
    ]
    priority_cards = [
        card for card in reviewable_cards if card.get("review_queue_priority") == "priority"
    ]
    return {
        "reviewable_rows": len(reviewable_cards),
        "reviewed_rows": len(reviewed_cards),
        "unreviewed_rows": len(reviewable_cards) - len(reviewed_cards),
        "priority_rows": len(priority_cards),
        "disagreement_candidate_rows": len(disagreement_cards),
    }


def build_clinician_review_workflow_report(
    *,
    latest_reviews: pd.DataFrame,
    reviewable_universe: pd.DataFrame,
    event_log_path: Path,
    snapshot_path: Path,
) -> dict[str, Any]:
    """Build the structured Phase 5 review workflow report payload."""
    latest = _normalize_review_snapshot(latest_reviews.copy())
    reviewable = reviewable_universe.copy()
    if not reviewable.empty:
        reviewable["review_timestamp"] = reviewable["review_timestamp"].map(
            _normalize_review_timestamp
        )

    reason_tag_counts: dict[str, int] = {}
    if not latest.empty and "label__clinician_reason_tags" in latest.columns:
        for items in latest["label__clinician_reason_tags"].tolist():
            for tag in items or []:
                reason_tag_counts[str(tag)] = reason_tag_counts.get(str(tag), 0) + 1

    review_status_counts: dict[str, int] = {}
    if not latest.empty and "label__clinician_review_status" in latest.columns:
        review_status_counts = (
            latest["label__clinician_review_status"]
            .fillna("null")
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
    priority_level_counts: dict[str, int] = {}
    if not latest.empty and "label__clinician_priority_level" in latest.columns:
        priority_level_counts = (
            latest["label__clinician_priority_level"]
            .fillna("null")
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )

    if latest.empty:
        matched_review_count = 0
        unmatched_review_count = 0
    elif reviewable.empty:
        matched_review_count = 0
        unmatched_review_count = len(latest)
    else:
        validation = latest.merge(
            reviewable.loc[:, REVIEW_KEY_COLUMNS].copy(),
            how="left",
            on=REVIEW_KEY_COLUMNS,
            indicator=True,
            validate="one_to_one",
        )
        matched_review_count = int((validation["_merge"] == "both").sum())
        unmatched_review_count = int((validation["_merge"] != "both").sum())

    required_level_populated = (
        int(latest["label__clinician_priority_level"].notna().sum())
        if not latest.empty and "label__clinician_priority_level" in latest.columns
        else 0
    )
    numeric_score_coverage = (
        int(latest["label__clinician_priority_score"].notna().sum())
        if not latest.empty and "label__clinician_priority_score" in latest.columns
        else 0
    )
    reason_tag_row_coverage_count = (
        int(
            latest["label__clinician_reason_tags"].map(lambda value: len(value or []) > 0).sum()
        )
        if not latest.empty and "label__clinician_reason_tags" in latest.columns
        else 0
    )
    reviewable_duplicate_key_count = (
        int(reviewable.duplicated(subset=REVIEW_KEY_COLUMNS, keep=False).sum())
        if not reviewable.empty
        else 0
    )
    latest_duplicate_key_count = (
        int(latest.duplicated(subset=REVIEW_KEY_COLUMNS, keep=False).sum())
        if not latest.empty
        else 0
    )

    return {
        "contract_version": CLINICIAN_REVIEW_SNAPSHOT_CONTRACT_VERSION,
        "generated_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "phase_scope_statement": (
            "Phase 5 is a clinician-label collection milestone. It captures pharmacist-reviewed "
            "low/medium/high decisions and audit trails at the analytical grain. It is not a "
            "model-performance validation milestone, because the current workspace still lacks a "
            "populated clinician ordinal target and a canonical prediction__priority_score output."
        ),
        "artifact_paths": {
            "event_log_path": str(event_log_path),
            "latest_snapshot_path": str(snapshot_path),
        },
        "reviewable_row_count": int(len(reviewable)),
        "clinician_reviewed_row_count": int(len(latest)),
        "required_level_populated_count": required_level_populated,
        "numeric_score_populated_count": numeric_score_coverage,
        "reason_tag_row_coverage_count": reason_tag_row_coverage_count,
        "priority_level_frequencies": dict(sorted(priority_level_counts.items())),
        "reason_tag_frequencies": dict(sorted(reason_tag_counts.items())),
        "review_status_frequencies": dict(sorted(review_status_counts.items())),
        "traceability_validation": {
            "matched_to_reviewable_rows_count": matched_review_count,
            "unmatched_review_rows_count": unmatched_review_count,
            "matched_to_modeling_row_id_count": (
                int(latest["modeling__row_id"].notna().sum())
                if not latest.empty and "modeling__row_id" in latest.columns
                else 0
            ),
            "reviewable_duplicate_key_count": reviewable_duplicate_key_count,
            "latest_review_duplicate_key_count": latest_duplicate_key_count,
        },
    }


def render_clinician_review_workflow_report(summary: dict[str, Any]) -> str:
    """Render the compact Phase 5 clinician review markdown report."""
    status_lines = [
        f"- `{status}`: {count:,}"
        for status, count in summary.get("review_status_frequencies", {}).items()
    ] or ["- none"]
    level_lines = [
        f"- `{level}`: {count:,}"
        for level, count in summary.get("priority_level_frequencies", {}).items()
    ] or ["- none"]
    reason_lines = [
        f"- `{reason}`: {count:,}"
        for reason, count in summary.get("reason_tag_frequencies", {}).items()
    ] or ["- none"]
    traceability = summary.get("traceability_validation", {})

    return "\n".join(
        [
            "# Phase 5 Clinician Review Workflow QC",
            "",
            f"- Generated at: `{summary.get('generated_at')}`",
            f"- Contract version: `{summary.get('contract_version')}`",
            f"- Scope note: {summary.get('phase_scope_statement')}",
            f"- Reviewable rows surfaced: {int(summary.get('reviewable_row_count', 0)):,}",
            f"- Clinician-reviewed rows saved: {int(summary.get('clinician_reviewed_row_count', 0)):,}",
            f"- Rows with required clinician level populated: {int(summary.get('required_level_populated_count', 0)):,}",
            f"- Rows with optional numeric score populated: {int(summary.get('numeric_score_populated_count', 0)):,}",
            f"- Rows with at least one reason tag: {int(summary.get('reason_tag_row_coverage_count', 0)):,}",
            "",
            "## Clinician Priority Level Frequencies",
            *level_lines,
            "",
            "## Review Status Frequencies",
            *status_lines,
            "",
            "## Reason Tag Frequencies",
            *reason_lines,
            "",
            "## Traceability Validation",
            f"- Matched to reviewable analytical rows: {int(traceability.get('matched_to_reviewable_rows_count', 0)):,}",
            f"- Matched to modeling row ids: {int(traceability.get('matched_to_modeling_row_id_count', 0)):,}",
            f"- Unmatched review rows: {int(traceability.get('unmatched_review_rows_count', 0)):,}",
            f"- Reviewable-universe duplicate analytical keys: {int(traceability.get('reviewable_duplicate_key_count', 0)):,}",
            f"- Latest-review duplicate analytical keys: {int(traceability.get('latest_review_duplicate_key_count', 0)):,}",
            "",
            "## Artifact Paths",
            f"- Event log: `{summary.get('artifact_paths', {}).get('event_log_path')}`",
            f"- Latest snapshot: `{summary.get('artifact_paths', {}).get('latest_snapshot_path')}`",
        ]
    ).strip() + "\n"


def _append_jsonl_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready_dict(payload), sort_keys=True) + "\n")


def _normalize_review_snapshot(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()
    normalized = dataframe.copy()
    if "review_timestamp" in normalized.columns:
        normalized["review_timestamp"] = normalized["review_timestamp"].map(
            _normalize_review_timestamp
        )
    if "label__clinician_reason_tags" in normalized.columns:
        normalized["label__clinician_reason_tags"] = normalized[
            "label__clinician_reason_tags"
        ].map(_normalize_reason_tag_list)
    if "review_provenance_json" in normalized.columns:
        normalized["review_provenance_json"] = normalized["review_provenance_json"].map(
            _normalize_dict_cell
        )
    return normalized.where(pd.notna(normalized), None)


def _normalize_reason_tag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(item) for item in converted]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            return [str(item) for item in json.loads(stripped)]
        return [stripped]
    return [str(value)]


def _normalize_dict_cell(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): json_ready_value(item) for key, item in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("{"):
            return json.loads(stripped)
    return None


def _normalize_review_timestamp(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError("Review timestamp is required and must be parseable.")
    return timestamp.strftime(TIMESTAMP_FORMAT)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_optional_numeric_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    numeric = float(value)
    if numeric < 0 or numeric > 10:
        raise ValueError("Clinician priority score must be between 0 and 10.")
    return numeric


def _build_modeling_row_id(row: pd.Series) -> str:
    parts: list[str] = []
    for column_name in REVIEW_KEY_COLUMNS:
        value = row[column_name]
        if isinstance(value, pd.Timestamp):
            value_text = value.isoformat()
        else:
            value_text = str(value)
        parts.append(f"{column_name}={value_text}")
    return "|".join(parts)


def _build_submission_id(
    *,
    subject_id: int,
    encounter_id: str,
    medication_standardized: str,
    review_timestamp: str,
    review_version: int,
    reviewer_id: str,
) -> str:
    payload = (
        f"{subject_id}|{encounter_id}|{medication_standardized}|{review_timestamp}|"
        f"{review_version}|{reviewer_id}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"review-{digest}"


def _assert_no_duplicate_review_keys(dataframe: pd.DataFrame, *, frame_name: str) -> None:
    duplicate_mask = dataframe.duplicated(subset=REVIEW_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        raise ValueError(
            f"{frame_name} contains duplicate analytical-grain rows for the clinician review workflow."
        )


def _normalized_medication_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _row_matches_any_candidate(row: pd.Series, candidates: set[str]) -> bool:
    for field_name in ("medication_standardized", "medication_normalized"):
        normalized = _normalized_medication_text(row.get(field_name))
        if normalized and normalized in candidates:
            return True
    return False


def _json_ready_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_ready_value(item) for key, item in payload.items()}


def _json_ready_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready_value(item) for item in value]
    return json_ready_value(value)
