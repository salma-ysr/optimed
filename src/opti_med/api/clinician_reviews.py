"""Clinician review storage, queueing, and densification QC helpers."""

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
from opti_med.medication_semantics import FIRST_SCOPE_SUPPORTED_CLASSES
from opti_med.scoring.priority_levels import derive_priority_level_series


CLINICIAN_REVIEW_EVENT_CONTRACT_VERSION = "clinician_review_events_phase5.v1"
CLINICIAN_REVIEW_SNAPSHOT_CONTRACT_VERSION = "clinician_review_labels_phase5.v1"
CLINICIAN_REVIEW_DENSIFICATION_QUEUE_CONTRACT_VERSION = "clinician_review_queue_phase6.v1"
CLINICIAN_REVIEW_DENSIFICATION_REPORT_CONTRACT_VERSION = (
    "clinician_review_densification_phase6.v1"
)
CLINICIAN_REVIEW_UI_SOURCE = "dossier_ui_phase6"
ALLOWED_REVIEW_SUBMISSION_SOURCES = (
    "dossier_ui_phase5",
    "dossier_ui_phase6",
    "review_queue_ui_phase6",
    "blind_eval_slice_ui_phase8",
)
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
REVIEW_QUEUE_ARTIFACT_COLUMNS = [
    *REVIEWABLE_COLUMNS,
    "reviewable_flag",
    "current_clinician_reviewed_flag",
    "review_submission_id",
    "review_version",
    "review_artifact_version",
    "review_submission_timestamp",
    "reviewer_id",
    "label__clinician_priority_level",
    "label__clinician_priority_score",
    "label__clinician_priority_score_level",
    "label__clinician_review_status",
    "label__clinician_reason_tags",
    "label__clinician_note",
    "label__clinician_suggested_action",
    "review_provenance_json",
    "class_reviewable_count",
    "class_reviewed_count",
    "subject_reviewable_count",
    "subject_reviewed_count",
    "encounter_reviewable_count",
    "encounter_reviewed_count",
    "class_zero_review_flag",
    "subject_zero_review_flag",
    "encounter_zero_review_flag",
    "rule_signal_strength_rank",
    "rule_disagreement_flag",
    "informative_unbenchmarked_flag",
    "ambiguous_primary_action_flag",
    "excluded_from_training_flag",
    "queue_priority_score",
    "queue_priority_band",
    "queue_priority_reasons",
    "needs_review_justification",
    "queue_rank_within_subject",
    "queue_rank_within_medication_class",
    "queue_rank",
    "review_queue_contract_version",
]
PHASE6_QUEUE_GENERATION_LOGIC = (
    "Start from the existing analytical-grain first-scope reviewable universe only.",
    "Do not enqueue non-reviewable dossier rows; they stay visible in the dossier with explicit non-reviewable messaging because they do not reconcile cleanly to the analytical grain.",
    "Rank unreviewed rows ahead of already reviewed rows.",
    "Boost stronger current-rule signals when benchmark__current_rule_score_level is present.",
    "Boost disagreement candidates when clinician and current-rule ordinal levels differ.",
    "Keep reviewable but benchmark-sparse or training-excluded rows visible when they remain clinically informative.",
    "Prefer medication classes, subjects, and encounters with less reviewed coverage, then interleave by subject and class to avoid one patient dominating the queue.",
)
BENCHMARK_LEVEL_RANKS = {"low": 1, "medium": 2, "high": 3}
CLASS_ORDER_LOOKUP = {
    class_label: index for index, class_label in enumerate(FIRST_SCOPE_SUPPORTED_CLASSES, start=1)
}


@dataclass(frozen=True, slots=True)
class ClinicianReviewRepository:
    """File-backed append-only repository for clinician review state and Phase 6 queueing."""

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

    def load_review_queue(
        self,
        *,
        latest_reviews: pd.DataFrame | None = None,
        reviewable_universe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Return the deterministic Phase 6 review queue dataframe."""
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
        return build_clinician_review_queue(
            latest_reviews=latest,
            reviewable_universe=reviewable,
        )

    def filter_review_queue(
        self,
        queue: pd.DataFrame,
        *,
        unreviewed_only: bool = False,
        medication_class: str | None = None,
        review_status: str | None = None,
        subject_id: int | None = None,
        reason_tag_presence: str | None = None,
    ) -> pd.DataFrame:
        """Apply deterministic server-side queue filters."""
        filtered = queue.copy()
        if filtered.empty:
            return filtered

        if unreviewed_only:
            filtered = filtered.loc[filtered["current_clinician_reviewed_flag"] == 0].copy()
        if medication_class:
            filtered = filtered.loc[
                filtered["medication_class_standardized"].astype(str).str.lower()
                == str(medication_class).strip().lower()
            ].copy()
        normalized_review_status = str(review_status or "all").strip().lower()
        if normalized_review_status == "unreviewed":
            filtered = filtered.loc[filtered["current_clinician_reviewed_flag"] == 0].copy()
        elif normalized_review_status in ALLOWED_REVIEW_STATUSES:
            filtered = filtered.loc[
                filtered["label__clinician_review_status"].astype(str) == normalized_review_status
            ].copy()
        if subject_id is not None:
            filtered = filtered.loc[
                pd.to_numeric(filtered["subject_id"], errors="coerce") == int(subject_id)
            ].copy()

        normalized_reason_tag_presence = str(reason_tag_presence or "all").strip().lower()
        if normalized_reason_tag_presence == "has_reason_tags":
            filtered = filtered.loc[
                filtered["label__clinician_reason_tags"].map(lambda value: len(value or []) > 0)
            ].copy()
        elif normalized_reason_tag_presence == "no_reason_tags":
            filtered = filtered.loc[
                filtered["label__clinician_reason_tags"].map(lambda value: len(value or []) == 0)
            ].copy()

        return filtered.reset_index(drop=True)

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
        review_submission_source = str(
            submission.get("review_submission_source") or CLINICIAN_REVIEW_UI_SOURCE
        ).strip()
        if review_submission_source not in ALLOWED_REVIEW_SUBMISSION_SOURCES:
            raise ValueError("Unsupported clinician review submission source.")
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
                "submission_source": review_submission_source,
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
        review_queue: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Write the legacy Phase 5 QC plus the Phase 6 densification artifacts."""
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
        phase5_summary = build_clinician_review_workflow_report(
            latest_reviews=latest,
            reviewable_universe=reviewable,
            event_log_path=self.settings.clinician_review_event_log_path,
            snapshot_path=self.settings.clinician_review_snapshot_output_path,
        )
        phase5_qc_json_path = self.settings.clinician_review_qc_summary_path
        phase5_qc_json_path.parent.mkdir(parents=True, exist_ok=True)
        phase5_qc_json_path.write_text(
            json.dumps(phase5_summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        phase5_qc_report_path = self.settings.clinician_review_qc_report_path
        phase5_qc_report_path.parent.mkdir(parents=True, exist_ok=True)
        phase5_qc_report_path.write_text(
            render_clinician_review_workflow_report(phase5_summary),
            encoding="utf-8",
        )

        queue = review_queue.copy() if review_queue is not None else self.load_review_queue(
            latest_reviews=latest,
            reviewable_universe=reviewable,
        )
        reviewable_universe_artifact = build_clinician_reviewable_universe_artifact(
            review_queue=queue
        )
        universe_path = self.settings.clinician_review_phase6_universe_output_path
        universe_path.parent.mkdir(parents=True, exist_ok=True)
        reviewable_universe_artifact.to_parquet(universe_path, index=False)

        queue_path = self.settings.clinician_review_phase6_queue_output_path
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue.to_parquet(queue_path, index=False)

        phase6_summary = build_clinician_review_densification_report(
            latest_reviews=latest,
            reviewable_universe=reviewable,
            review_queue=queue,
            event_log_path=self.settings.clinician_review_event_log_path,
            snapshot_path=self.settings.clinician_review_snapshot_output_path,
            universe_path=universe_path,
            queue_path=queue_path,
            phase5_summary_path=phase5_qc_json_path,
            phase5_report_path=phase5_qc_report_path,
            phase6_summary_path=self.settings.clinician_review_phase6_qc_summary_path,
            phase6_report_path=self.settings.clinician_review_phase6_qc_report_path,
        )
        phase6_qc_json_path = self.settings.clinician_review_phase6_qc_summary_path
        phase6_qc_json_path.parent.mkdir(parents=True, exist_ok=True)
        phase6_qc_json_path.write_text(
            json.dumps(phase6_summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        phase6_qc_report_path = self.settings.clinician_review_phase6_qc_report_path
        phase6_qc_report_path.parent.mkdir(parents=True, exist_ok=True)
        phase6_qc_report_path.write_text(
            render_clinician_review_densification_report(phase6_summary),
            encoding="utf-8",
        )
        return phase6_summary

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


def overlay_latest_reviews_on_queue_rows(
    *,
    queue_rows: pd.DataFrame,
    latest_reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Refresh latest-review fields on a fixed queue without changing membership or order."""
    queue = queue_rows.copy()
    if queue.empty:
        return queue

    latest = _normalize_review_snapshot(latest_reviews.copy())
    latest_columns = [
        *REVIEW_KEY_COLUMNS,
        "review_submission_id",
        "review_version",
        "review_artifact_version",
        "review_submission_timestamp",
        "reviewer_id",
        "label__clinician_priority_level",
        "label__clinician_priority_score",
        "label__clinician_priority_score_level",
        "label__clinician_review_status",
        "label__clinician_reason_tags",
        "label__clinician_note",
        "label__clinician_reviewed_flag",
        "label__clinician_suggested_action",
        "review_provenance_json",
    ]
    refresh_columns = [column_name for column_name in latest_columns if column_name not in REVIEW_KEY_COLUMNS]
    if latest.empty:
        latest_frame = pd.DataFrame(columns=latest_columns)
    else:
        latest_frame = latest.loc[:, latest_columns].copy()

    queue["_fixed_session_order"] = range(len(queue))
    queue = queue.drop(columns=refresh_columns, errors="ignore")
    queue = queue.merge(
        latest_frame,
        how="left",
        on=REVIEW_KEY_COLUMNS,
        validate="one_to_one",
    )
    queue["label__clinician_reason_tags"] = queue["label__clinician_reason_tags"].map(
        _normalize_reason_tag_list
    )
    queue["review_provenance_json"] = queue["review_provenance_json"].map(_normalize_dict_cell)
    queue["current_clinician_reviewed_flag"] = queue["review_submission_id"].notna().astype(int)
    queue = queue.sort_values("_fixed_session_order", kind="mergesort").drop(
        columns="_fixed_session_order"
    )
    return queue.reset_index(drop=True)


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


def build_clinician_review_queue(
    *,
    latest_reviews: pd.DataFrame,
    reviewable_universe: pd.DataFrame,
) -> pd.DataFrame:
    """Build the deterministic Phase 6 clinician review queue artifact."""
    reviewable = reviewable_universe.copy()
    if reviewable.empty:
        return pd.DataFrame(columns=REVIEW_QUEUE_ARTIFACT_COLUMNS)

    reviewable["review_timestamp"] = reviewable["review_timestamp"].map(_normalize_review_timestamp)
    latest = _normalize_review_snapshot(latest_reviews.copy())

    latest_columns = [
        *REVIEW_KEY_COLUMNS,
        "review_submission_id",
        "review_version",
        "review_artifact_version",
        "review_submission_timestamp",
        "reviewer_id",
        "label__clinician_priority_level",
        "label__clinician_priority_score",
        "label__clinician_priority_score_level",
        "label__clinician_review_status",
        "label__clinician_reason_tags",
        "label__clinician_note",
        "label__clinician_reviewed_flag",
        "label__clinician_suggested_action",
        "review_provenance_json",
    ]
    if latest.empty:
        latest_frame = pd.DataFrame(columns=latest_columns)
    else:
        latest_frame = latest.loc[:, latest_columns].copy()

    queue = reviewable.merge(
        latest_frame,
        how="left",
        on=REVIEW_KEY_COLUMNS,
        validate="one_to_one",
    )
    queue["reviewable_flag"] = True
    queue["label__clinician_reason_tags"] = queue["label__clinician_reason_tags"].map(
        _normalize_reason_tag_list
    )
    queue["review_provenance_json"] = queue["review_provenance_json"].map(_normalize_dict_cell)
    queue["current_clinician_reviewed_flag"] = queue["review_submission_id"].notna().astype(int)

    class_reviewable_counts = (
        queue["medication_class_standardized"]
        .fillna("unresolved")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    class_reviewed_counts = (
        queue.loc[queue["current_clinician_reviewed_flag"] == 1, "medication_class_standardized"]
        .fillna("unresolved")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    subject_reviewable_counts = (
        queue["subject_id"].astype(str).value_counts(dropna=False).to_dict()
    )
    subject_reviewed_counts = (
        queue.loc[queue["current_clinician_reviewed_flag"] == 1, "subject_id"]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    encounter_reviewable_counts = (
        queue["encounter_id"].astype(str).value_counts(dropna=False).to_dict()
    )
    encounter_reviewed_counts = (
        queue.loc[queue["current_clinician_reviewed_flag"] == 1, "encounter_id"]
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )

    queue["class_reviewable_count"] = queue["medication_class_standardized"].map(
        lambda value: int(
            class_reviewable_counts.get("unresolved" if pd.isna(value) else str(value), 0)
        )
    )
    queue["class_reviewed_count"] = queue["medication_class_standardized"].map(
        lambda value: int(
            class_reviewed_counts.get("unresolved" if pd.isna(value) else str(value), 0)
        )
    )
    queue["subject_reviewable_count"] = queue["subject_id"].map(
        lambda value: int(subject_reviewable_counts.get(str(value), 0))
    )
    queue["subject_reviewed_count"] = queue["subject_id"].map(
        lambda value: int(subject_reviewed_counts.get(str(value), 0))
    )
    queue["encounter_reviewable_count"] = queue["encounter_id"].map(
        lambda value: int(encounter_reviewable_counts.get(str(value), 0))
    )
    queue["encounter_reviewed_count"] = queue["encounter_id"].map(
        lambda value: int(encounter_reviewed_counts.get(str(value), 0))
    )

    queue["class_zero_review_flag"] = (queue["class_reviewed_count"] == 0).astype(int)
    queue["subject_zero_review_flag"] = (queue["subject_reviewed_count"] == 0).astype(int)
    queue["encounter_zero_review_flag"] = (queue["encounter_reviewed_count"] == 0).astype(int)
    queue["rule_signal_strength_rank"] = queue["benchmark__current_rule_score_level"].map(
        lambda value: BENCHMARK_LEVEL_RANKS.get(
            "" if pd.isna(value) else str(value).strip().lower(),
            0,
        )
    )
    queue["rule_disagreement_flag"] = queue.apply(_rule_disagreement_flag, axis=1)
    queue["informative_unbenchmarked_flag"] = (
        pd.to_numeric(queue["benchmark__current_rule_available_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(0)
    ).astype(int)
    queue["ambiguous_primary_action_flag"] = queue["label__primary_action_label"].map(
        lambda value: int(
            ("" if pd.isna(value) else str(value))
            in {"action_undetermined", "window_censored"}
        )
    )
    queue["excluded_from_training_flag"] = (
        pd.to_numeric(queue["meta__dataset_row_eligible_for_training_flag"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(0)
    ).astype(int)
    queue["queue_priority_score"] = queue.apply(_queue_priority_score, axis=1)
    queue["queue_priority_band"] = queue.apply(_queue_priority_band, axis=1)
    queue["queue_priority_reasons"] = queue.apply(_queue_priority_reasons, axis=1)
    queue["needs_review_justification"] = queue["queue_priority_reasons"].map(
        _queue_reason_summary
    )

    queue = queue.sort_values(
        [
            "queue_priority_score",
            "rule_signal_strength_rank",
            "class_zero_review_flag",
            "subject_zero_review_flag",
            "encounter_zero_review_flag",
            "class_reviewable_count",
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
        ],
        ascending=[False, False, False, False, False, True, True, True, True, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    queue["queue_rank_within_subject"] = queue.groupby("subject_id").cumcount() + 1
    queue["queue_rank_within_medication_class"] = queue.groupby(
        "medication_class_standardized",
        dropna=False,
    ).cumcount() + 1

    queue = queue.sort_values(
        [
            "current_clinician_reviewed_flag",
            "queue_priority_score",
            "queue_rank_within_subject",
            "queue_rank_within_medication_class",
            "class_reviewed_count",
            "subject_reviewed_count",
            "class_reviewable_count",
            "subject_id",
            "encounter_id",
            "medication_standardized",
            "review_timestamp",
        ],
        ascending=[True, False, True, True, True, True, True, True, True, True, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    queue["queue_rank"] = range(1, len(queue) + 1)
    queue["review_queue_contract_version"] = CLINICIAN_REVIEW_DENSIFICATION_QUEUE_CONTRACT_VERSION

    for column_name in REVIEW_QUEUE_ARTIFACT_COLUMNS:
        if column_name not in queue.columns:
            queue[column_name] = pd.NA
    return queue.loc[:, REVIEW_QUEUE_ARTIFACT_COLUMNS].copy()


def build_clinician_reviewable_universe_artifact(
    *,
    review_queue: pd.DataFrame,
) -> pd.DataFrame:
    """Return the stable reviewable-universe artifact derived from the Phase 6 queue."""
    if review_queue.empty:
        return pd.DataFrame(columns=REVIEW_QUEUE_ARTIFACT_COLUMNS)
    return review_queue.sort_values(
        REVIEW_KEY_COLUMNS,
        ascending=[True, True, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def build_clinician_review_densification_report(
    *,
    latest_reviews: pd.DataFrame,
    reviewable_universe: pd.DataFrame,
    review_queue: pd.DataFrame,
    event_log_path: Path,
    snapshot_path: Path,
    universe_path: Path,
    queue_path: Path,
    phase5_summary_path: Path,
    phase5_report_path: Path,
    phase6_summary_path: Path,
    phase6_report_path: Path,
) -> dict[str, Any]:
    """Build the structured Phase 6 densification QC payload."""
    latest = _normalize_review_snapshot(latest_reviews.copy())
    reviewable = reviewable_universe.copy()
    if not reviewable.empty:
        reviewable["review_timestamp"] = reviewable["review_timestamp"].map(_normalize_review_timestamp)
    queue = review_queue.copy()

    priority_level_counts = _priority_level_frequency_dict(latest)
    review_status_counts = _review_status_frequency_dict(latest)
    reason_tag_counts = _reason_tag_frequency_dict(latest)
    reviewable_class_counts = _medication_class_count_dict(reviewable)
    reviewed_class_counts = _medication_class_count_dict(
        queue.loc[queue["current_clinician_reviewed_flag"] == 1].copy()
    )
    reviewed_patient_counts = (
        queue.loc[queue["current_clinician_reviewed_flag"] == 1, "subject_id"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    queue_priority_reason_counts = _queue_priority_reason_frequency_dict(
        queue.loc[queue["current_clinician_reviewed_flag"] == 0].copy()
    )
    traceability = _build_traceability_validation(
        latest=latest,
        reviewable=reviewable,
    )

    reviewable_row_count = int(len(reviewable))
    clinician_reviewed_row_count = int(len(latest))
    remaining_unreviewed_row_count = max(0, reviewable_row_count - clinician_reviewed_row_count)
    reviewed_subject_count = (
        int(queue.loc[queue["current_clinician_reviewed_flag"] == 1, "subject_id"].nunique())
        if not queue.empty
        else 0
    )
    reviewed_encounter_count = (
        int(queue.loc[queue["current_clinician_reviewed_flag"] == 1, "encounter_id"].nunique())
        if not queue.empty
        else 0
    )
    reviewable_with_modeling_row_id_count = (
        int(reviewable["modeling__row_id"].notna().sum())
        if not reviewable.empty and "modeling__row_id" in reviewable.columns
        else 0
    )
    benchmark_available_reviewable_count = (
        int(
            pd.to_numeric(reviewable["benchmark__current_rule_available_flag"], errors="coerce")
            .fillna(0)
            .sum()
        )
        if not reviewable.empty and "benchmark__current_rule_available_flag" in reviewable.columns
        else 0
    )
    reviewable_subject_count = int(reviewable["subject_id"].nunique()) if not reviewable.empty else 0
    reviewable_encounter_count = (
        int(reviewable["encounter_id"].nunique()) if not reviewable.empty else 0
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
    required_level_populated = (
        int(latest["label__clinician_priority_level"].notna().sum())
        if not latest.empty and "label__clinician_priority_level" in latest.columns
        else 0
    )
    clinician_reviewed_medication_class_count = int(
        sum(count > 0 for count in reviewed_class_counts.values())
    )
    top_queue_preview = [
        {
            "queue_rank": int(record["queue_rank"]),
            "subject_id": int(record["subject_id"]),
            "hadm_id": _json_ready_value(record.get("hadm_id")),
            "medication_standardized": str(record.get("medication_standardized")),
            "medication_class_standardized": record.get("medication_class_standardized"),
            "current_clinician_reviewed_flag": int(record.get("current_clinician_reviewed_flag") or 0),
            "queue_priority_band": str(record.get("queue_priority_band") or "review_backlog"),
            "queue_priority_reasons": list(record.get("queue_priority_reasons") or [])[:5],
        }
        for record in queue.head(10).to_dict(orient="records")
    ]

    label_balance_assessment = _build_label_balance_assessment(
        priority_level_counts=priority_level_counts,
        clinician_reviewed_row_count=clinician_reviewed_row_count,
        numeric_score_coverage=numeric_score_coverage,
    )
    label_balance_assessment["minimally_credible_reviewed_label_base_flag"] = bool(
        clinician_reviewed_row_count >= 24
        and reviewed_subject_count >= 8
        and reviewed_encounter_count >= 8
        and clinician_reviewed_medication_class_count >= 3
        and len(label_balance_assessment.get("missing_priority_levels", [])) == 0
    )
    label_balance_assessment["assessment_statement"] = (
        "The current reviewed label base is still too sparse and too imbalanced for credible supervised training."
        if not label_balance_assessment["minimally_credible_reviewed_label_base_flag"]
        else "The reviewed label base is approaching a minimally credible starting point for modest supervised experiments, but still requires careful caveats."
    )
    milestone_status = _build_milestone_status(
        clinician_reviewed_row_count=clinician_reviewed_row_count,
        reviewed_subject_count=reviewed_subject_count,
        reviewed_encounter_count=reviewed_encounter_count,
        reviewed_class_counts=reviewed_class_counts,
        priority_level_counts=priority_level_counts,
    )

    return {
        "contract_version": CLINICIAN_REVIEW_DENSIFICATION_REPORT_CONTRACT_VERSION,
        "generated_at": datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        ),
        "phase_scope_statement": (
            "Phase 6 is a clinician-label densification milestone. It expands the amount, diversity, "
            "and auditability of clinician-approved low/medium/high labels on the existing analytical "
            "grain. It is not a model-performance milestone: benchmark__current_rule_score remains "
            "sparse in the current workspace and there is still no populated canonical "
            "prediction__priority_score output."
        ),
        "artifact_paths": {
            "event_log_path": str(event_log_path),
            "latest_snapshot_path": str(snapshot_path),
            "phase5_qc_summary_path": str(phase5_summary_path),
            "phase5_qc_report_path": str(phase5_report_path),
            "phase6_reviewable_universe_path": str(universe_path),
            "phase6_queue_path": str(queue_path),
            "phase6_qc_summary_path": str(phase6_summary_path),
            "phase6_qc_report_path": str(phase6_report_path),
        },
        "queue_generation_logic": list(PHASE6_QUEUE_GENERATION_LOGIC),
        "reviewable_row_count": reviewable_row_count,
        "clinician_reviewed_row_count": clinician_reviewed_row_count,
        "remaining_unreviewed_row_count": remaining_unreviewed_row_count,
        "clinician_review_coverage_rate": _safe_rate(
            clinician_reviewed_row_count,
            reviewable_row_count,
        ),
        "unlabeled_reviewable_rate": _safe_rate(
            remaining_unreviewed_row_count,
            reviewable_row_count,
        ),
        "required_level_populated_count": required_level_populated,
        "numeric_score_populated_count": numeric_score_coverage,
        "reason_tag_row_coverage_count": reason_tag_row_coverage_count,
        "reviewable_distinct_subject_count": reviewable_subject_count,
        "reviewable_distinct_encounter_count": reviewable_encounter_count,
        "clinician_reviewed_distinct_subject_count": reviewed_subject_count,
        "clinician_reviewed_distinct_encounter_count": reviewed_encounter_count,
        "clinician_reviewed_medication_class_count": clinician_reviewed_medication_class_count,
        "reviewable_rows_with_modeling_row_id_count": reviewable_with_modeling_row_id_count,
        "reviewable_rows_with_modeling_row_id_rate": _safe_rate(
            reviewable_with_modeling_row_id_count,
            reviewable_row_count,
        ),
        "benchmark_available_reviewable_row_count": benchmark_available_reviewable_count,
        "benchmark_available_reviewable_row_rate": _safe_rate(
            benchmark_available_reviewable_count,
            reviewable_row_count,
        ),
        "priority_level_frequencies": priority_level_counts,
        "reason_tag_frequencies": dict(sorted(reason_tag_counts.items())),
        "review_status_frequencies": review_status_counts,
        "reviewable_rows_by_medication_class": reviewable_class_counts,
        "clinician_reviewed_rows_by_medication_class": reviewed_class_counts,
        "clinician_reviewed_rows_by_patient": reviewed_patient_counts,
        "queue_priority_reason_frequencies": dict(sorted(queue_priority_reason_counts.items())),
        "label_balance_assessment": label_balance_assessment,
        "milestone_status": milestone_status,
        "traceability_validation": traceability,
        "top_queue_preview": top_queue_preview,
    }


def render_clinician_review_densification_report(summary: dict[str, Any]) -> str:
    """Render the compact Phase 6 densification markdown report."""
    queue_logic_lines = [
        f"- {item}" for item in summary.get("queue_generation_logic", [])
    ] or ["- none"]
    level_lines = [
        f"- `{level}`: {count:,}"
        for level, count in summary.get("priority_level_frequencies", {}).items()
    ] or ["- none"]
    status_lines = [
        f"- `{status}`: {count:,}"
        for status, count in summary.get("review_status_frequencies", {}).items()
    ] or ["- none"]
    reason_lines = [
        f"- `{reason}`: {count:,}"
        for reason, count in summary.get("reason_tag_frequencies", {}).items()
    ] or ["- none"]
    class_lines = [
        f"- `{class_label}`: {count:,} reviewed / {int(summary.get('reviewable_rows_by_medication_class', {}).get(class_label, 0)):,} reviewable"
        for class_label, count in summary.get("clinician_reviewed_rows_by_medication_class", {}).items()
    ] or ["- none"]
    patient_lines = [
        f"- patient `{patient_id}`: {count:,}"
        for patient_id, count in list(summary.get("clinician_reviewed_rows_by_patient", {}).items())[:10]
    ] or ["- none"]
    milestone_lines = [
        f"- {item.get('label')}: {'met' if item.get('met') else 'pending'}"
        f" ({item.get('target')}; current {item.get('current_value')})"
        for item in summary.get("milestone_status", {}).values()
    ] or ["- none"]
    top_queue_lines = [
        f"- rank {int(item.get('queue_rank', 0))}: patient `{item.get('subject_id')}` / "
        f"`{item.get('medication_standardized')}` / `{item.get('medication_class_standardized')}` / "
        + ", ".join(str(reason) for reason in item.get("queue_priority_reasons", []))
        for item in summary.get("top_queue_preview", [])
    ] or ["- none"]
    label_balance = summary.get("label_balance_assessment", {})
    traceability = summary.get("traceability_validation", {})

    return "\n".join(
        [
            "# Phase 6 Clinician Review Densification QC",
            "",
            f"- Generated at: `{summary.get('generated_at')}`",
            f"- Contract version: `{summary.get('contract_version')}`",
            f"- Scope note: {summary.get('phase_scope_statement')}",
            "",
            "## Queue Generation Logic",
            *queue_logic_lines,
            "",
            "## Densification Coverage",
            f"- Reviewable rows: {int(summary.get('reviewable_row_count', 0)):,}",
            f"- Clinician-reviewed rows: {int(summary.get('clinician_reviewed_row_count', 0)):,}",
            f"- Coverage rate: {float(summary.get('clinician_review_coverage_rate', 0.0)):.2%}",
            f"- Remaining unreviewed rows: {int(summary.get('remaining_unreviewed_row_count', 0)):,}",
            f"- Unlabeled share of reviewable universe: {float(summary.get('unlabeled_reviewable_rate', 0.0)):.2%}",
            f"- Reviewable distinct subjects: {int(summary.get('reviewable_distinct_subject_count', 0)):,}",
            f"- Reviewable distinct encounters: {int(summary.get('reviewable_distinct_encounter_count', 0)):,}",
            f"- Reviewed distinct subjects: {int(summary.get('clinician_reviewed_distinct_subject_count', 0)):,}",
            f"- Reviewed distinct encounters: {int(summary.get('clinician_reviewed_distinct_encounter_count', 0)):,}",
            f"- Reviewed medication classes with coverage: {int(summary.get('clinician_reviewed_medication_class_count', 0)):,}",
            f"- Reviewable rows with modeling__row_id: {int(summary.get('reviewable_rows_with_modeling_row_id_count', 0)):,} ({float(summary.get('reviewable_rows_with_modeling_row_id_rate', 0.0)):.2%})",
            f"- Reviewable rows with benchmark__current_rule_score available: {int(summary.get('benchmark_available_reviewable_row_count', 0)):,} ({float(summary.get('benchmark_available_reviewable_row_rate', 0.0)):.2%})",
            f"- Optional numeric score coverage: {int(summary.get('numeric_score_populated_count', 0)):,}",
            "",
            "## Reviewed Low / Medium / High Distribution",
            *level_lines,
            "",
            "## Review Status Frequencies",
            *status_lines,
            "",
            "## Clinician Reason Tag Frequencies",
            *reason_lines,
            "",
            "## Reviewed Coverage by Medication Class",
            *class_lines,
            "",
            "## Reviewed Coverage by Patient",
            *patient_lines,
            "",
            "## Queue Front Preview",
            *top_queue_lines,
            "",
            "## Milestones",
            *milestone_lines,
            "",
            "## Remaining Densification Gaps",
            f"- Medium labels rare: {bool(label_balance.get('medium_labels_rare_flag', False))}",
            f"- High labels rare: {bool(label_balance.get('high_labels_rare_flag', False))}",
            f"- Missing reviewed priority levels: {', '.join(label_balance.get('missing_priority_levels', [])) or 'none'}",
            f"- Approaching minimally credible reviewed label base: {bool(label_balance.get('minimally_credible_reviewed_label_base_flag', False))}",
            f"- Assessment: {label_balance.get('assessment_statement')}",
            "- The deterministic queue only includes analytically reviewable rows. Non-reviewable dossier rows remain dossier-visible and explicitly marked as non-reviewable instead of entering the queue.",
            "- This phase expands the label base and coverage audit trail. It does not claim model readiness or model performance.",
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
            f"- Legacy Phase 5 QC summary: `{summary.get('artifact_paths', {}).get('phase5_qc_summary_path')}`",
            f"- Legacy Phase 5 QC report: `{summary.get('artifact_paths', {}).get('phase5_qc_report_path')}`",
            f"- Phase 6 reviewable universe: `{summary.get('artifact_paths', {}).get('phase6_reviewable_universe_path')}`",
            f"- Phase 6 queue: `{summary.get('artifact_paths', {}).get('phase6_queue_path')}`",
            f"- Phase 6 QC summary: `{summary.get('artifact_paths', {}).get('phase6_qc_summary_path')}`",
            f"- Phase 6 QC report: `{summary.get('artifact_paths', {}).get('phase6_qc_report_path')}`",
        ]
    ).strip() + "\n"


def _rule_disagreement_flag(row: pd.Series) -> int:
    clinician_level = _optional_text(row.get("label__clinician_priority_level"))
    benchmark_level = _optional_text(row.get("benchmark__current_rule_score_level"))
    if clinician_level and benchmark_level and clinician_level != benchmark_level:
        return 1
    return 0


def _queue_priority_score(row: pd.Series) -> int:
    return (
        (1000 if int(row.get("current_clinician_reviewed_flag") or 0) == 0 else 0)
        + (200 if int(row.get("rule_disagreement_flag") or 0) == 1 else 0)
        + (40 * int(row.get("rule_signal_strength_rank") or 0))
        + (60 if int(row.get("informative_unbenchmarked_flag") or 0) == 1 else 0)
        + (30 if int(row.get("ambiguous_primary_action_flag") or 0) == 1 else 0)
        + (20 if int(row.get("excluded_from_training_flag") or 0) == 1 else 0)
        + (45 if int(row.get("class_zero_review_flag") or 0) == 1 else 0)
        + (25 if int(row.get("subject_zero_review_flag") or 0) == 1 else 0)
        + (15 if int(row.get("encounter_zero_review_flag") or 0) == 1 else 0)
        + (10 if int(pd.to_numeric(pd.Series([row.get("first_scope_supported_class_flag")]), errors="coerce").fillna(0).iloc[0]) == 1 else 0)
    )


def _queue_priority_band(row: pd.Series) -> str:
    if int(row.get("current_clinician_reviewed_flag") or 0) == 1:
        if int(row.get("rule_disagreement_flag") or 0) == 1:
            return "reviewed_disagreement_followup"
        return "reviewed_backlog"
    if int(row.get("rule_disagreement_flag") or 0) == 1:
        return "disagreement_candidate"
    if int(row.get("rule_signal_strength_rank") or 0) >= 2:
        return "high_signal_unreviewed"
    if int(row.get("class_zero_review_flag") or 0) == 1 or int(row.get("subject_zero_review_flag") or 0) == 1:
        return "coverage_expansion_unreviewed"
    return "review_backlog"


def _queue_priority_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    if int(row.get("current_clinician_reviewed_flag") or 0) == 0:
        reasons.append("unreviewed_row")
    else:
        reasons.append("already_reviewed")
    if int(row.get("rule_disagreement_flag") or 0) == 1:
        reasons.append("clinician_rule_disagreement")
    benchmark_level = _optional_text(row.get("benchmark__current_rule_score_level"))
    if benchmark_level in BENCHMARK_LEVEL_RANKS:
        reasons.append(f"benchmark_signal_{benchmark_level}")
    if int(row.get("informative_unbenchmarked_flag") or 0) == 1:
        reasons.append("benchmark_sparse_but_reviewable")
    if int(row.get("ambiguous_primary_action_flag") or 0) == 1:
        reasons.append("primary_action_ambiguous")
    if int(row.get("excluded_from_training_flag") or 0) == 1:
        reasons.append("training_excluded_but_reviewable")
    if int(row.get("class_zero_review_flag") or 0) == 1:
        reasons.append("expand_medication_class_coverage")
    if int(row.get("subject_zero_review_flag") or 0) == 1:
        reasons.append("expand_patient_coverage")
    if int(row.get("encounter_zero_review_flag") or 0) == 1:
        reasons.append("expand_encounter_coverage")
    if int(pd.to_numeric(pd.Series([row.get("first_scope_supported_class_flag")]), errors="coerce").fillna(0).iloc[0]) == 1:
        reasons.append("supported_first_scope_class")
    deduped: list[str] = []
    for item in reasons:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _queue_reason_summary(reasons: list[str]) -> str:
    if not reasons:
        return "Deterministic review backlog row on the analytical grain."
    labels = [_queue_reason_label(reason) for reason in reasons[:3]]
    return "Prioritized because " + "; ".join(labels) + "."


def _queue_reason_label(reason: str) -> str:
    labels = {
        "unreviewed_row": "it has no clinician review yet",
        "already_reviewed": "it already has a saved clinician review",
        "clinician_rule_disagreement": "clinician and current-rule ordinal levels disagree",
        "benchmark_signal_low": "a low current-rule signal is available",
        "benchmark_signal_medium": "a medium current-rule signal is available",
        "benchmark_signal_high": "a high current-rule signal is available",
        "benchmark_sparse_but_reviewable": "benchmark comparison is sparse but the row remains reviewable",
        "primary_action_ambiguous": "the current constructed action label is ambiguous",
        "training_excluded_but_reviewable": "the row is excluded from confident training comparison but still clinically informative",
        "expand_medication_class_coverage": "this medication class still lacks reviewed coverage",
        "expand_patient_coverage": "this patient still lacks reviewed coverage",
        "expand_encounter_coverage": "this encounter still lacks reviewed coverage",
        "supported_first_scope_class": "it maps cleanly to a supported first-scope medication class",
    }
    return labels.get(reason, reason.replace("_", " "))


def _priority_level_frequency_dict(latest: pd.DataFrame) -> dict[str, int]:
    counts = {level: 0 for level in ALLOWED_PRIORITY_LEVELS}
    if latest.empty or "label__clinician_priority_level" not in latest.columns:
        return counts
    observed = (
        latest["label__clinician_priority_level"]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    for level in ALLOWED_PRIORITY_LEVELS:
        counts[level] = int(observed.get(level, 0))
    return counts


def _review_status_frequency_dict(latest: pd.DataFrame) -> dict[str, int]:
    counts = {status: 0 for status in ALLOWED_REVIEW_STATUSES}
    if latest.empty or "label__clinician_review_status" not in latest.columns:
        return counts
    observed = (
        latest["label__clinician_review_status"]
        .fillna("null")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    for status in ALLOWED_REVIEW_STATUSES:
        counts[status] = int(observed.get(status, 0))
    return counts


def _reason_tag_frequency_dict(latest: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if latest.empty or "label__clinician_reason_tags" not in latest.columns:
        return counts
    for items in latest["label__clinician_reason_tags"].tolist():
        for tag in items or []:
            counts[str(tag)] = counts.get(str(tag), 0) + 1
    return counts


def _medication_class_count_dict(frame: pd.DataFrame) -> dict[str, int]:
    counts = {class_label: 0 for class_label in FIRST_SCOPE_SUPPORTED_CLASSES}
    if frame.empty or "medication_class_standardized" not in frame.columns:
        return counts
    observed = (
        frame["medication_class_standardized"]
        .fillna("unresolved")
        .astype(str)
        .value_counts(dropna=False)
        .to_dict()
    )
    for class_label in FIRST_SCOPE_SUPPORTED_CLASSES:
        counts[class_label] = int(observed.get(class_label, 0))
    extras = sorted(
        key for key in observed.keys() if key not in counts and str(key) != "unresolved"
    )
    for class_label in extras:
        counts[str(class_label)] = int(observed.get(class_label, 0))
    return counts


def _queue_priority_reason_frequency_dict(queue: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if queue.empty or "queue_priority_reasons" not in queue.columns:
        return counts
    for items in queue["queue_priority_reasons"].tolist():
        for reason in items or []:
            counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def _build_traceability_validation(
    *,
    latest: pd.DataFrame,
    reviewable: pd.DataFrame,
) -> dict[str, int]:
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
    return {
        "matched_to_reviewable_rows_count": matched_review_count,
        "unmatched_review_rows_count": unmatched_review_count,
        "matched_to_modeling_row_id_count": (
            int(latest["modeling__row_id"].notna().sum())
            if not latest.empty and "modeling__row_id" in latest.columns
            else 0
        ),
        "reviewable_duplicate_key_count": (
            int(reviewable.duplicated(subset=REVIEW_KEY_COLUMNS, keep=False).sum())
            if not reviewable.empty
            else 0
        ),
        "latest_review_duplicate_key_count": (
            int(latest.duplicated(subset=REVIEW_KEY_COLUMNS, keep=False).sum())
            if not latest.empty
            else 0
        ),
    }


def _build_label_balance_assessment(
    *,
    priority_level_counts: dict[str, int],
    clinician_reviewed_row_count: int,
    numeric_score_coverage: int,
) -> dict[str, object]:
    missing_levels = [
        level for level in ALLOWED_PRIORITY_LEVELS if int(priority_level_counts.get(level, 0)) == 0
    ]
    medium_count = int(priority_level_counts.get("medium", 0))
    high_count = int(priority_level_counts.get("high", 0))
    return {
        "missing_priority_levels": missing_levels,
        "medium_labels_rare_flag": medium_count < 5,
        "high_labels_rare_flag": high_count < 5,
        "numeric_score_coverage_rate": _safe_rate(
            numeric_score_coverage,
            clinician_reviewed_row_count,
        ),
    }


def _build_milestone_status(
    *,
    clinician_reviewed_row_count: int,
    reviewed_subject_count: int,
    reviewed_encounter_count: int,
    reviewed_class_counts: dict[str, int],
    priority_level_counts: dict[str, int],
) -> dict[str, dict[str, object]]:
    reviewed_class_count = int(sum(count > 0 for count in reviewed_class_counts.values()))
    milestone_1_met = clinician_reviewed_row_count >= 24
    milestone_2_met = (
        clinician_reviewed_row_count >= 24
        and reviewed_subject_count >= 8
        and reviewed_encounter_count >= 8
        and reviewed_class_count >= 3
    )
    milestone_3_met = (
        clinician_reviewed_row_count >= 60
        and reviewed_subject_count >= 15
        and reviewed_encounter_count >= 15
        and reviewed_class_count >= 4
        and int(priority_level_counts.get("low", 0)) >= 5
        and int(priority_level_counts.get("medium", 0)) >= 5
        and int(priority_level_counts.get("high", 0)) >= 5
    )
    return {
        "milestone_1_label_volume": {
            "label": "Milestone 1",
            "target": "24 reviewed rows",
            "current_value": clinician_reviewed_row_count,
            "met": milestone_1_met,
        },
        "milestone_2_diversity": {
            "label": "Milestone 2",
            "target": "24 reviewed rows, 8 subjects, 8 encounters, 3 medication classes",
            "current_value": (
                f"{clinician_reviewed_row_count} rows / {reviewed_subject_count} subjects / "
                f"{reviewed_encounter_count} encounters / {reviewed_class_count} classes"
            ),
            "met": milestone_2_met,
        },
        "milestone_3_modest_supervised_start": {
            "label": "Milestone 3",
            "target": "60 reviewed rows plus at least 5 low, 5 medium, and 5 high labels across diverse subjects, encounters, and classes",
            "current_value": (
                f"{clinician_reviewed_row_count} rows / low {int(priority_level_counts.get('low', 0))} / "
                f"medium {int(priority_level_counts.get('medium', 0))} / high {int(priority_level_counts.get('high', 0))}"
            ),
            "met": milestone_3_met,
        },
    }


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


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
    if value is None or _is_missing_scalar(value):
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
    if value is None or _is_missing_scalar(value):
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
    if value is None or _is_missing_scalar(value):
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


def _is_missing_scalar(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
