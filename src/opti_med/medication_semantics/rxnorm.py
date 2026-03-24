"""RxNorm-backed medication semantics using the RxNav REST API."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from opti_med.medication_semantics.first_scope import FirstScopeRxNormClassAssigner
from opti_med.medication_semantics.contracts import (
    MappingConfidence,
    MedicationClassAssignment,
    MedicationClassIdentity,
    MedicationIdentityResolution,
    MedicationScheduleSemantics,
    MedicationSemanticMapper,
    NormalizedDose,
    NormalizedMedicationName,
    StandardizedMedicationIdentity,
)
from opti_med.medication_semantics.normalization import (
    RxNormQueryNormalizer,
    build_rxnorm_query_text,
)


RXNORM_API_BASE_URL = "https://rxnav.nlm.nih.gov/REST"
RXNORM_API_VERSION = "rxnav_rest_v1"
RXNORM_BOOTSTRAP_MAPPER_NAME = "rxnorm_api_bootstrap_mapper"
RXNORM_BOOTSTRAP_MAPPER_VERSION = "0.1"
INGREDIENT_TERM_TYPES = ("IN", "PIN", "MIN")
SCHEDULED_TOKEN_PATTERN = re.compile(
    r"(^|[\s/;,])"
    r"(bid|tid|qid|qam|qpm|qhs|hs|daily|weekly|monthly|nightly|scheduled|once|stat|"
    r"q\d+h|q\d+hr|q\d+hrs|every\s+\d+\s*h|every\s+\d+\s*hour|every\s+day|"
    r"\d+\s*x\s*(daily|day))"
    r"($|[\s/;,])"
)
PRN_TOKEN_PATTERN = re.compile(r"(^|[\s/;,])(prn|as needed|as-needed)($|[\s/;,])")


class RxNormApiError(RuntimeError):
    """Raised when the RxNorm API cannot be reached or parsed."""


JsonTransport = Callable[[str, float], dict[str, object]]


class MedicationClassAssignerProtocol(Protocol):
    """Structural interface for medication class assignment helpers."""

    def assign(
        self,
        resolution: MedicationIdentityResolution,
    ) -> MedicationClassAssignment:
        """Assign supported classes for one resolved medication identity."""
        ...


@dataclass(frozen=True, slots=True)
class ResolvedRxNormConcept:
    """Resolved RxNorm concept details for one candidate RxCUI."""

    rxcui: str | None
    matched_term: str | None
    matched_term_type: str | None
    medication_standardized: str | None
    medication_standardized_source: str | None
    ingredient_rxcui: str | None
    ingredient_standardized: str | None
    ingredient_resolution_status: str


class RxNormApiClient:
    """Thin JSON client for the subset of RxNav endpoints used today."""

    def __init__(
        self,
        *,
        base_url: str = RXNORM_API_BASE_URL,
        timeout_seconds: float = 10.0,
        transport: JsonTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport or _default_json_transport
        self._cached_rxnorm_version: str | None = None

    def find_rxcui_by_string(self, term: str) -> list[str]:
        payload = self._request_json("rxcui.json", {"name": term, "search": 2})
        group = payload.get("idGroup", {})
        if not isinstance(group, dict):
            return []
        values = group.get("rxnormId", [])
        if not isinstance(values, list):
            return []
        return [str(value) for value in values if value]

    def get_approximate_match(self, term: str, *, max_entries: int = 5) -> list[dict[str, object]]:
        payload = self._request_json(
            "approximateTerm.json",
            {"term": term, "maxEntries": max_entries, "option": 1},
        )
        group = payload.get("approximateGroup", {})
        if not isinstance(group, dict):
            return []
        candidates = group.get("candidate", [])
        if not isinstance(candidates, list):
            return []
        return [candidate for candidate in candidates if isinstance(candidate, dict)]

    def get_rxconcept_properties(self, rxcui: str) -> dict[str, object]:
        payload = self._request_json(f"rxcui/{rxcui}/properties.json")
        properties = payload.get("properties", {})
        return properties if isinstance(properties, dict) else {}

    def get_related_by_type(
        self,
        rxcui: str,
        *,
        tty: tuple[str, ...] = INGREDIENT_TERM_TYPES,
    ) -> list[dict[str, object]]:
        payload = self._request_json(
            f"rxcui/{rxcui}/related.json",
            {"tty": "+".join(tty)},
        )
        related_group = payload.get("relatedGroup", {})
        if not isinstance(related_group, dict):
            return []
        concept_groups = related_group.get("conceptGroup", [])
        if not isinstance(concept_groups, list):
            return []

        concepts: list[dict[str, object]] = []
        for group in concept_groups:
            if not isinstance(group, dict):
                continue
            concept_properties = group.get("conceptProperties", [])
            if isinstance(concept_properties, list):
                concepts.extend(
                    concept for concept in concept_properties if isinstance(concept, dict)
                )
        return concepts

    def get_rxnorm_version(self) -> str | None:
        if self._cached_rxnorm_version is not None:
            return self._cached_rxnorm_version
        payload = self._request_json("version.json")
        version = payload.get("version")
        self._cached_rxnorm_version = str(version) if version else None
        return self._cached_rxnorm_version

    def _request_json(
        self,
        endpoint: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        query = urlencode(
            {key: value for key, value in (params or {}).items() if value is not None}
        )
        url = f"{self.base_url}/{endpoint}"
        if query:
            url = f"{url}?{query}"
        try:
            payload = self.transport(url, self.timeout_seconds)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise RxNormApiError(f"RxNorm API request failed for '{url}': {exc}") from exc
        if not isinstance(payload, dict):
            raise RxNormApiError(f"RxNorm API returned a non-object payload for '{url}'.")
        return payload


class PlaceholderMedicationClassAssigner:
    """Class-enrichment hook kept explicit while RxNorm identity work boots up."""

    def assign(
        self,
        resolution: MedicationIdentityResolution,
    ) -> MedicationClassAssignment:
        return MedicationClassAssignment(
            standardized_ingredient_id=resolution.ingredient_rxcui,
            standardized_ingredient_label=resolution.ingredient_standardized,
            assignment_status="class_assignment_pending",
            provenance={
                "class_hook_implemented": False,
                "class_enrichment_scope": "partial_placeholder_only",
            },
        )


class RxNormIdentityResolver:
    """Resolve normalized medication text to RxNorm-backed identities."""

    def __init__(
        self,
        *,
        client: RxNormApiClient | None = None,
        class_assigner: MedicationClassAssignerProtocol | None = None,
        normalizer: RxNormQueryNormalizer | None = None,
    ) -> None:
        self.client = client or RxNormApiClient()
        self.class_assigner = class_assigner or FirstScopeRxNormClassAssigner()
        self.normalizer = normalizer or RxNormQueryNormalizer()

    def resolve(
        self,
        *,
        raw_name: object,
        normalized_hint: object = None,
    ) -> MedicationIdentityResolution:
        normalized_name = self.normalizer.normalize_name(
            raw_name,
            normalized_hint=normalized_hint,
        )
        query_candidates = _query_candidates(normalized_name, normalized_hint=normalized_hint)
        if not query_candidates:
            return _unresolved_resolution(
                raw_name=normalized_name.raw_name,
                normalized_query=normalized_name.normalized_text,
                lookup_strategy_used="no_query_available",
                ingredient_resolution_status="unresolved_no_match",
                ambiguity_note="No usable RxNorm query text could be constructed.",
            )

        last_error: RxNormApiError | None = None
        for query_text, strategy in query_candidates:
            try:
                rxcuis = self.client.find_rxcui_by_string(query_text)
            except RxNormApiError as exc:
                last_error = exc
                break
            if rxcuis:
                return self._resolve_candidate_rxcuis(
                    raw_name=normalized_name.raw_name,
                    normalized_query=normalized_name.normalized_text,
                    candidate_rxcuis=rxcuis,
                    lookup_strategy_used=strategy,
                    approximate_rank=None,
                )

        if last_error is not None:
            return _unresolved_resolution(
                raw_name=normalized_name.raw_name,
                normalized_query=normalized_name.normalized_text,
                lookup_strategy_used="exact_lookup_api_error",
                ingredient_resolution_status="unresolved_api_error",
                ambiguity_note=str(last_error),
            )

        approximate_query = (
            normalized_name.normalized_text or build_rxnorm_query_text(raw_name)
        )
        if approximate_query is None:
            return _unresolved_resolution(
                raw_name=normalized_name.raw_name,
                normalized_query=normalized_name.normalized_text,
                lookup_strategy_used="approximate_lookup_skipped",
                ingredient_resolution_status="unresolved_no_match",
                ambiguity_note="No normalized query text was available for approximate lookup.",
            )
        try:
            approximate_candidates = self.client.get_approximate_match(approximate_query)
        except RxNormApiError as exc:
            return _unresolved_resolution(
                raw_name=normalized_name.raw_name,
                normalized_query=normalized_name.normalized_text,
                lookup_strategy_used="approximate_lookup_api_error",
                ingredient_resolution_status="unresolved_api_error",
                ambiguity_note=str(exc),
            )
        if not approximate_candidates:
            return _unresolved_resolution(
                raw_name=normalized_name.raw_name,
                normalized_query=normalized_name.normalized_text,
                lookup_strategy_used="approximate_lookup_no_match",
                ingredient_resolution_status="unresolved_no_match",
                ambiguity_note="RxNorm returned no approximate candidates.",
            )

        candidate_ranks = [
            int(candidate.get("rank", 999999))
            for candidate in approximate_candidates
            if candidate.get("rxcui")
        ]
        if not candidate_ranks:
            return _unresolved_resolution(
                raw_name=normalized_name.raw_name,
                normalized_query=normalized_name.normalized_text,
                lookup_strategy_used="approximate_lookup_no_match",
                ingredient_resolution_status="unresolved_no_match",
                ambiguity_note="Approximate candidates did not include usable RxCUIs.",
            )
        top_rank = min(candidate_ranks)
        candidate_rxcuis = [
            str(candidate.get("rxcui"))
            for candidate in approximate_candidates
            if candidate.get("rxcui") and int(candidate.get("rank", 999999)) == top_rank
        ]
        return self._resolve_candidate_rxcuis(
            raw_name=normalized_name.raw_name,
            normalized_query=normalized_name.normalized_text,
            candidate_rxcuis=candidate_rxcuis,
            lookup_strategy_used="approximate_fallback",
            approximate_rank=top_rank,
        )

    def _resolve_candidate_rxcuis(
        self,
        *,
        raw_name: str | None,
        normalized_query: str | None,
        candidate_rxcuis: list[str],
        lookup_strategy_used: str,
        approximate_rank: int | None,
    ) -> MedicationIdentityResolution:
        resolved_candidates = [
            self._resolve_rxcui(rxcui)
            for rxcui in candidate_rxcuis
        ]
        resolved_candidates = [
            candidate for candidate in resolved_candidates if candidate is not None
        ]
        if not resolved_candidates:
            return _unresolved_resolution(
                raw_name=raw_name,
                normalized_query=normalized_query,
                lookup_strategy_used=lookup_strategy_used,
                ingredient_resolution_status="unresolved_no_match",
                candidate_rxcuis=candidate_rxcuis,
                ambiguity_note="RxNorm returned RxCUIs, but no usable concept properties were resolved.",
                approximate_rank=approximate_rank,
            )

        distinct_standardizations = {
            (
                candidate.medication_standardized,
                candidate.ingredient_standardized,
                candidate.ingredient_resolution_status,
            )
            for candidate in resolved_candidates
        }
        chosen = resolved_candidates[0]
        confidence = _mapping_confidence_for_resolution(
            strategy=lookup_strategy_used,
            candidate_count=len(candidate_rxcuis),
            ingredient_resolution_status=chosen.ingredient_resolution_status,
        )
        if len(distinct_standardizations) > 1:
            return _unresolved_resolution(
                raw_name=raw_name,
                normalized_query=normalized_query,
                lookup_strategy_used=lookup_strategy_used,
                ingredient_resolution_status="unresolved_ambiguous_multi_hit",
                candidate_rxcuis=candidate_rxcuis,
                ambiguity_note="Multiple RxNorm candidates resolved to different standardized identities.",
                approximate_rank=approximate_rank,
            )

        class_assignment = self.class_assigner.assign(
            MedicationIdentityResolution(
                raw_name=raw_name,
                normalized_query=normalized_query,
                lookup_strategy_used=lookup_strategy_used,
                returned_rxcui=chosen.rxcui,
                matched_term=chosen.matched_term,
                matched_term_type=chosen.matched_term_type,
                medication_standardized=chosen.medication_standardized,
                medication_standardized_source=chosen.medication_standardized_source,
                ingredient_rxcui=chosen.ingredient_rxcui,
                ingredient_standardized=chosen.ingredient_standardized,
                ingredient_resolution_status=chosen.ingredient_resolution_status,  # type: ignore[arg-type]
                mapping_confidence=confidence,  # type: ignore[arg-type]
                ambiguous_match_flag=len(candidate_rxcuis) > 1,
                multi_hit_candidate_count=len(candidate_rxcuis),
                candidate_rxcuis=tuple(candidate_rxcuis),
                provenance={},
            )
        )
        return MedicationIdentityResolution(
            raw_name=raw_name,
            normalized_query=normalized_query,
            lookup_strategy_used=lookup_strategy_used,
            returned_rxcui=chosen.rxcui,
            matched_term=chosen.matched_term,
            matched_term_type=chosen.matched_term_type,
            medication_standardized=chosen.medication_standardized,
            medication_standardized_source=chosen.medication_standardized_source,
            ingredient_rxcui=chosen.ingredient_rxcui,
            ingredient_standardized=chosen.ingredient_standardized,
            ingredient_resolution_status=chosen.ingredient_resolution_status,  # type: ignore[arg-type]
            mapping_confidence=confidence,  # type: ignore[arg-type]
            ambiguous_match_flag=len(candidate_rxcuis) > 1,
            multi_hit_candidate_count=len(candidate_rxcuis),
            candidate_rxcuis=tuple(candidate_rxcuis),
            provenance={
                "approximate_rank": approximate_rank,
                "class_assignment_status": class_assignment.assignment_status,
                "class_ids_json": json.dumps(list(class_assignment.class_ids), sort_keys=True),
                "class_labels_json": json.dumps(list(class_assignment.class_labels), sort_keys=True),
                "rxnorm_version": _safe_rxnorm_version(self.client),
            },
        )

    def _resolve_rxcui(self, rxcui: str) -> ResolvedRxNormConcept | None:
        properties = self.client.get_rxconcept_properties(rxcui)
        if not properties:
            return None
        matched_term = _string_or_none(properties.get("name"))
        matched_term_type = _string_or_none(properties.get("tty"))
        if matched_term_type in INGREDIENT_TERM_TYPES:
            return ResolvedRxNormConcept(
                rxcui=str(rxcui),
                matched_term=matched_term,
                matched_term_type=matched_term_type,
                medication_standardized=matched_term,
                medication_standardized_source="rxnorm_ingredient",
                ingredient_rxcui=str(rxcui),
                ingredient_standardized=matched_term,
                ingredient_resolution_status="resolved_to_ingredient",
            )

        ingredient_candidates = self.client.get_related_by_type(rxcui)
        preferred_ingredient = _select_preferred_ingredient(ingredient_candidates)
        if preferred_ingredient is not None:
            return ResolvedRxNormConcept(
                rxcui=str(rxcui),
                matched_term=matched_term,
                matched_term_type=matched_term_type,
                medication_standardized=preferred_ingredient["name"],
                medication_standardized_source="rxnorm_ingredient",
                ingredient_rxcui=preferred_ingredient["rxcui"],
                ingredient_standardized=preferred_ingredient["name"],
                ingredient_resolution_status="resolved_via_related_concept",
            )

        if matched_term is None:
            return None
        return ResolvedRxNormConcept(
            rxcui=str(rxcui),
            matched_term=matched_term,
            matched_term_type=matched_term_type,
            medication_standardized=matched_term,
            medication_standardized_source="rxnorm_term",
            ingredient_rxcui=None,
            ingredient_standardized=None,
            ingredient_resolution_status="resolved_term_only_no_ingredient",
        )


class RxNormBackedMedicationSemanticMapper(MedicationSemanticMapper):
    """Medication semantics adapter backed by the RxNorm API."""

    mapper_name = RXNORM_BOOTSTRAP_MAPPER_NAME
    mapper_version = RXNORM_BOOTSTRAP_MAPPER_VERSION

    def __init__(
        self,
        *,
        normalizer: RxNormQueryNormalizer | None = None,
        resolver: RxNormIdentityResolver | None = None,
        class_assigner: MedicationClassAssignerProtocol | None = None,
    ) -> None:
        self.normalizer = normalizer or RxNormQueryNormalizer()
        self.class_assigner = class_assigner or FirstScopeRxNormClassAssigner()
        self.resolver = resolver or RxNormIdentityResolver(
            class_assigner=self.class_assigner,
            normalizer=self.normalizer,
        )

    def normalize_name(self, raw_name: object) -> NormalizedMedicationName:
        return self.normalizer.normalize_name(raw_name)

    def standardize_ingredient(
        self,
        normalized_name: NormalizedMedicationName | str | None,
    ) -> StandardizedMedicationIdentity:
        resolution = self.resolve_identity(normalized_name)
        return StandardizedMedicationIdentity(
            normalized_text=resolution.normalized_query,
            standardized_ingredient_id=resolution.ingredient_rxcui or resolution.returned_rxcui,
            standardized_ingredient_label=resolution.ingredient_standardized
            or resolution.medication_standardized,
            identity_system="RxNorm" if resolution.returned_rxcui else None,
            identity_system_version=_safe_rxnorm_version(self.resolver.client),
            standardization_status=(
                "ontology_backed"
                if resolution.returned_rxcui is not None
                else "not_standardized"
            ),
            provenance={
                "lookup_strategy_used": resolution.lookup_strategy_used,
                "medication_standardized_source": resolution.medication_standardized_source,
                "mapping_confidence": resolution.mapping_confidence,
                "ambiguous_match_flag": resolution.ambiguous_match_flag,
            },
        )

    def infer_class(
        self,
        standardized_ingredient: StandardizedMedicationIdentity | str | None,
    ) -> tuple[MedicationClassIdentity, ...]:
        del standardized_ingredient
        return tuple()

    def detect_prn_vs_scheduled(
        self,
        *,
        frequency: str | None = None,
        status: str | None = None,
        route: str | None = None,
    ) -> MedicationScheduleSemantics:
        del route
        return infer_prn_vs_scheduled(
            frequency=frequency,
            status=status,
        )

    def normalize_dose(
        self,
        *,
        dose_value: object = None,
        dose_unit: str | None = None,
        normalized_name: NormalizedMedicationName | None = None,
        standardized_ingredient: StandardizedMedicationIdentity | None = None,
    ) -> NormalizedDose:
        del normalized_name, standardized_ingredient
        return NormalizedDose(
            input_dose_value=None if dose_value is None else str(dose_value),
            input_dose_unit=dose_unit,
            normalized_dose_value=None,
            normalized_dose_unit=None,
            normalization_status="not_standardized",
            provenance={
                "rxnorm_bootstrap_scope": "identity_first_no_dose_normalization",
            },
        )

    def resolve_identity(
        self,
        normalized_name: NormalizedMedicationName | str | None,
    ) -> MedicationIdentityResolution:
        if isinstance(normalized_name, NormalizedMedicationName):
            raw_name = normalized_name.raw_name
            normalized_hint = normalized_name.normalized_text
        else:
            raw_name = normalized_name
            normalized_hint = normalized_name
        return self.resolver.resolve(
            raw_name=raw_name,
            normalized_hint=normalized_hint,
        )


def _default_json_transport(url: str, timeout_seconds: float) -> dict[str, object]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _query_candidates(
    normalized_name: NormalizedMedicationName,
    *,
    normalized_hint: object = None,
) -> list[tuple[str, str]]:
    query_candidates: list[tuple[str, str]] = []
    raw_query = build_rxnorm_query_text(
        normalized_name.raw_name,
        normalized_hint=normalized_hint,
    )
    if raw_query:
        query_candidates.append((raw_query, "exact_or_normalized_raw_query"))
    normalized_query = normalized_name.normalized_text
    if normalized_query and normalized_query != raw_query:
        query_candidates.append(
            (normalized_query, "exact_or_normalized_normalized_query")
        )
    return query_candidates


def _select_preferred_ingredient(
    concepts: list[dict[str, object]],
) -> dict[str, str] | None:
    typed_candidates: dict[str, list[dict[str, str]]] = {tty: [] for tty in INGREDIENT_TERM_TYPES}
    for concept in concepts:
        tty = _string_or_none(concept.get("tty"))
        name = _string_or_none(concept.get("name"))
        rxcui = _string_or_none(concept.get("rxcui"))
        if tty not in typed_candidates or name is None or rxcui is None:
            continue
        typed_candidates[tty].append({"rxcui": rxcui, "name": name, "tty": tty})

    for tty in ("MIN", "IN", "PIN"):
        unique_candidates = {
            (candidate["rxcui"], candidate["name"]): candidate
            for candidate in typed_candidates[tty]
        }
        if len(unique_candidates) == 1:
            return next(iter(unique_candidates.values()))
        if len(unique_candidates) > 1:
            return None
    return None


def _mapping_confidence_for_resolution(
    *,
    strategy: str,
    candidate_count: int,
    ingredient_resolution_status: str,
) -> MappingConfidence:
    if ingredient_resolution_status.startswith("unresolved"):
        return "none"
    if strategy.startswith("exact_or_normalized") and candidate_count == 1:
        return "high"
    if strategy.startswith("exact_or_normalized"):
        return "medium"
    if ingredient_resolution_status == "resolved_term_only_no_ingredient":
        return "low"
    return "medium"


def _safe_rxnorm_version(client: RxNormApiClient) -> str | None:
    try:
        return client.get_rxnorm_version()
    except RxNormApiError:
        return None


def _unresolved_resolution(
    *,
    raw_name: str | None,
    normalized_query: str | None,
    lookup_strategy_used: str,
    ingredient_resolution_status: str,
    candidate_rxcuis: list[str] | None = None,
    ambiguity_note: str | None = None,
    approximate_rank: int | None = None,
) -> MedicationIdentityResolution:
    return MedicationIdentityResolution(
        raw_name=raw_name,
        normalized_query=normalized_query,
        lookup_strategy_used=lookup_strategy_used,
        returned_rxcui=None,
        matched_term=None,
        matched_term_type=None,
        medication_standardized=normalized_query,
        medication_standardized_source="normalized_text_fallback",
        ingredient_rxcui=None,
        ingredient_standardized=None,
        ingredient_resolution_status=ingredient_resolution_status,  # type: ignore[arg-type]
        mapping_confidence="none",
        ambiguous_match_flag=ingredient_resolution_status == "unresolved_ambiguous_multi_hit",
        multi_hit_candidate_count=len(candidate_rxcuis or []),
        candidate_rxcuis=tuple(candidate_rxcuis or ()),
        provenance={
            "ambiguity_note": ambiguity_note,
            "approximate_rank": approximate_rank,
            "lookup_timestamp_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
        },
    )


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def infer_prn_vs_scheduled(
    *,
    frequency: str | None = None,
    status: str | None = None,
) -> MedicationScheduleSemantics:
    """Infer PRN-vs-scheduled semantics from structured order text when defensible."""
    combined_parts = [
        _normalize_schedule_text(frequency),
        _normalize_schedule_text(status),
    ]
    combined = " ".join(part for part in combined_parts if part).strip()
    if not combined:
        return "unknown"
    if PRN_TOKEN_PATTERN.search(combined):
        return "prn"
    if combined.isdigit():
        return "scheduled"
    if SCHEDULED_TOKEN_PATTERN.search(combined):
        return "scheduled"
    return "unknown"


def _normalize_schedule_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())
