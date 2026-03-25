"""Tests for the RxNorm mapper, cache builder, and state-table enrichment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    ENCOUNTER_MEDICATION_STATE_COLUMNS,
    ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION,
    MEDICATION_EVENT_COLUMNS,
    MEDICATION_EVENTS_CONTRACT_VERSION,
    MEDICATION_RXNORM_MAPPING_COLUMNS,
    MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION,
)
from opti_med.data_access.encounter_medication_state import build_encounter_medication_state
from opti_med.data_access.encounter_medication_state_rxnorm import (
    build_encounter_medication_state_rxnorm,
)
from opti_med.data_access.medication_rxnorm_mapping import (
    LOOKUP_MODE_CACHE_FIRST,
    LOOKUP_MODE_CACHE_ONLY,
    MedicationRxNormMappingBuilder,
    extract_medication_rxnorm_queries_from_encounter_medication_state,
)
from opti_med.medication_semantics.contracts import MedicationIdentityResolution
from opti_med.medication_semantics.rxnorm import (
    PlaceholderMedicationClassAssigner,
    RxNormApiClient,
    RxNormBackedMedicationSemanticMapper,
    RxNormIdentityResolver,
)
from opti_med.pipeline.qc import build_medication_rxnorm_mapping_qc_report


class RxNormResolverTests(unittest.TestCase):
    def test_related_lookup_encodes_tty_as_space_delimited_query_parameter(self) -> None:
        seen_urls: list[str] = []

        def transport(url: str, timeout_seconds: float) -> dict[str, object]:
            del timeout_seconds
            seen_urls.append(url)
            return {"relatedGroup": {"conceptGroup": []}}

        client = RxNormApiClient(transport=transport)

        related = client.get_related_by_type("123")

        self.assertEqual(related, [])
        self.assertEqual(len(seen_urls), 1)
        self.assertIn("tty=IN+PIN+MIN", seen_urls[0])
        self.assertNotIn("%2B", seen_urls[0])

    def test_exact_lookup_resolves_to_related_ingredient(self) -> None:
        client = RxNormApiClient(transport=_fake_exact_transport)
        resolver = RxNormIdentityResolver(
            client=client,
            class_assigner=PlaceholderMedicationClassAssigner(),
        )

        resolution = resolver.resolve(raw_name="Lasix 20 MG Oral Tablet", normalized_hint="lasix")

        self.assertEqual(resolution.returned_rxcui, "123")
        self.assertEqual(resolution.matched_term, "Lasix 20 MG Oral Tablet")
        self.assertEqual(resolution.ingredient_standardized, "Furosemide")
        self.assertEqual(resolution.ingredient_resolution_status, "resolved_via_related_concept")
        self.assertEqual(resolution.medication_standardized_source, "rxnorm_ingredient")
        self.assertEqual(resolution.mapping_confidence, "high")

    def test_approximate_lookup_only_runs_after_exact_miss(self) -> None:
        client = RxNormApiClient(transport=_fake_approximate_transport)
        resolver = RxNormIdentityResolver(client=client)

        resolution = resolver.resolve(raw_name="Tylenol 325 mg", normalized_hint="tylenol")

        self.assertEqual(resolution.lookup_strategy_used, "approximate_fallback")
        self.assertEqual(resolution.returned_rxcui, "555")
        self.assertEqual(resolution.ingredient_standardized, "Acetaminophen")
        self.assertEqual(resolution.mapping_confidence, "medium")

    def test_related_lookup_api_error_yields_unresolved_api_error_instead_of_crashing(self) -> None:
        client = RxNormApiClient(transport=_fake_related_lookup_error_transport)
        resolver = RxNormIdentityResolver(client=client)

        resolution = resolver.resolve(
            raw_name="Lasix 20 MG Oral Tablet",
            normalized_hint="lasix",
        )

        self.assertEqual(resolution.lookup_strategy_used, "exact_or_normalized_raw_query")
        self.assertEqual(resolution.ingredient_resolution_status, "unresolved_api_error")
        self.assertEqual(resolution.mapping_confidence, "none")
        self.assertIn("related.json", resolution.provenance.get("ambiguity_note", ""))


class MedicationRxNormMappingBuilderTests(unittest.TestCase):
    def test_cache_hit_reuses_existing_mapping_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(analytical_root=Path(temp_dir) / "analytical")
            mapper = _CountingMapper()
            builder = MedicationRxNormMappingBuilder(
                settings,
                semantic_mapper=mapper,
                lookup_mode=LOOKUP_MODE_CACHE_FIRST,
            )
            medication_events = _minimal_medication_events(
                medication_normalized="furosemide",
                raw_medication_name="Furosemide 20 mg",
            )
            existing_cache = pd.DataFrame([_mapping_row("furosemide")])

            mapping = builder.build(
                medication_events=medication_events,
                existing_cache=existing_cache,
            )

            self.assertEqual(mapper.call_count, 0)
            self.assertEqual(mapping.iloc[0]["lookup_status"], "resolved")
            self.assertEqual(mapping.iloc[0]["medication_standardized"], "furosemide")

    def test_cache_first_retries_cached_unresolved_api_error_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(analytical_root=Path(temp_dir) / "analytical")
            mapper = _CountingMapper()
            builder = MedicationRxNormMappingBuilder(
                settings,
                semantic_mapper=mapper,
                lookup_mode=LOOKUP_MODE_CACHE_FIRST,
            )
            medication_events = _minimal_medication_events(
                medication_normalized="furosemide",
                raw_medication_name="Furosemide 20 mg",
            )
            existing_cache = pd.DataFrame(
                [
                    _mapping_row(
                        "furosemide",
                        medication_standardized="furosemide",
                        medication_standardized_source="normalized_text_fallback",
                        ingredient_standardized=None,
                        ingredient_resolution_status="unresolved_api_error",
                        rxnorm_rxcui=None,
                        matched_term=None,
                        matched_term_type=None,
                    )
                ]
            )

            mapping = builder.build(
                medication_events=medication_events,
                existing_cache=existing_cache,
            )

            self.assertEqual(mapper.call_count, 1)
            self.assertEqual(mapping.iloc[0]["lookup_status"], "resolved")
            self.assertEqual(mapping.iloc[0]["api_called_flag"], 1)

    def test_cache_first_retries_cached_cache_only_miss_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(analytical_root=Path(temp_dir) / "analytical")
            mapper = _CountingMapper()
            builder = MedicationRxNormMappingBuilder(
                settings,
                semantic_mapper=mapper,
                lookup_mode=LOOKUP_MODE_CACHE_FIRST,
            )
            medication_events = _minimal_medication_events(
                medication_normalized="unknownmed",
                raw_medication_name="UnknownMed 10 mg",
            )
            existing_cache = pd.DataFrame(
                [
                    _mapping_row(
                        "unknownmed",
                        medication_standardized="unknownmed",
                        medication_standardized_source="normalized_text_fallback",
                        ingredient_standardized=None,
                        ingredient_resolution_status="unresolved_cache_only_miss",
                        rxnorm_rxcui=None,
                        matched_term=None,
                        matched_term_type=None,
                    )
                ]
            )

            mapping = builder.build(
                medication_events=medication_events,
                existing_cache=existing_cache,
            )

            self.assertEqual(mapper.call_count, 1)
            self.assertEqual(mapping.iloc[0]["lookup_status"], "resolved")
            self.assertEqual(mapping.iloc[0]["api_called_flag"], 1)

    def test_cache_only_reuses_cached_retryable_rows_without_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(analytical_root=Path(temp_dir) / "analytical")
            mapper = _CountingMapper()
            builder = MedicationRxNormMappingBuilder(
                settings,
                semantic_mapper=mapper,
                lookup_mode=LOOKUP_MODE_CACHE_ONLY,
            )
            medication_events = _minimal_medication_events(
                medication_normalized="furosemide",
                raw_medication_name="Furosemide 20 mg",
            )
            existing_cache = pd.DataFrame(
                [
                    _mapping_row(
                        "furosemide",
                        medication_standardized="furosemide",
                        medication_standardized_source="normalized_text_fallback",
                        ingredient_standardized=None,
                        ingredient_resolution_status="unresolved_api_error",
                        rxnorm_rxcui=None,
                        matched_term=None,
                        matched_term_type=None,
                    )
                ]
            )

            mapping = builder.build(
                medication_events=medication_events,
                existing_cache=existing_cache,
            )

            self.assertEqual(mapper.call_count, 0)
            self.assertEqual(mapping.iloc[0]["lookup_status"], "unresolved_api_error")
            self.assertEqual(mapping.iloc[0]["api_called_flag"], 1)

    def test_cache_only_mode_emits_explicit_unresolved_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(analytical_root=Path(temp_dir) / "analytical")
            mapper = _CountingMapper()
            builder = MedicationRxNormMappingBuilder(
                settings,
                semantic_mapper=mapper,
                lookup_mode=LOOKUP_MODE_CACHE_ONLY,
            )
            medication_events = _minimal_medication_events(
                medication_normalized="unknownmed",
                raw_medication_name="UnknownMed 10 mg",
            )

            mapping = builder.build(medication_events=medication_events)

            self.assertEqual(mapper.call_count, 0)
            self.assertEqual(mapping.iloc[0]["lookup_status"], "unresolved_cache_only_miss")
            self.assertEqual(
                mapping.iloc[0]["ingredient_resolution_status"],
                "unresolved_cache_only_miss",
            )
            self.assertEqual(mapping.iloc[0]["api_called_flag"], 0)

    def test_state_builder_groups_by_rxnorm_standardized_key_when_mapping_is_available(self) -> None:
        encounter_index = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "sex": "F",
                    "age_proxy": 77,
                    "age_group": "75-84",
                    "encounter_id": "hadm:10",
                    "hadm_id": 10,
                    "stay_id": None,
                    "encounter_source": "hospital_only",
                    "encounter_start": "2125-03-19 18:00:00",
                    "encounter_end": "2125-03-20 10:00:00",
                    "admittime": "2125-03-19 18:00:00",
                    "dischtime": "2125-03-20 10:00:00",
                    "intime": None,
                    "outtime": None,
                }
            ]
        )
        medication_events = pd.concat(
            [
                _minimal_medication_events(
                    medication_normalized="lasix",
                    raw_medication_name="Lasix 20 mg",
                    medication_event_id="event-1",
                    starttime="2125-03-19 19:00:00",
                    stoptime="2125-03-20 12:00:00",
                ),
                _minimal_medication_events(
                    medication_normalized="furosemide",
                    raw_medication_name="Furosemide 20 mg",
                    medication_event_id="event-2",
                    starttime="2125-03-19 20:00:00",
                    stoptime="2125-03-20 12:00:00",
                ),
            ],
            ignore_index=True,
        )
        mapping = pd.DataFrame(
            [
                _mapping_row(
                    "lasix",
                    raw_medication_string="Lasix 20 mg",
                    medication_standardized="Furosemide",
                    ingredient_standardized="Furosemide",
                    ingredient_resolution_status="resolved_via_related_concept",
                    medication_standardized_source="rxnorm_ingredient",
                    lookup_strategy_used="exact_or_normalized_raw_query",
                    rxnorm_rxcui="123",
                    matched_term="Lasix 20 MG Oral Tablet",
                ),
                _mapping_row(
                    "furosemide",
                    raw_medication_string="Furosemide 20 mg",
                    medication_standardized="Furosemide",
                    ingredient_standardized="Furosemide",
                    ingredient_resolution_status="resolved_to_ingredient",
                    medication_standardized_source="rxnorm_ingredient",
                    lookup_strategy_used="exact_or_normalized_normalized_query",
                    rxnorm_rxcui="321",
                    matched_term="Furosemide",
                    matched_term_type="IN",
                ),
            ]
        )

        state = build_encounter_medication_state(
            encounter_index=encounter_index,
            medication_events=medication_events,
            medication_rxnorm_mapping=mapping,
            review_time_policy="discharge_capped_latest_available",
        )

        self.assertEqual(len(state), 1)
        self.assertEqual(state.iloc[0]["medication_standardized"], "Furosemide")
        self.assertEqual(state.iloc[0]["medication_standardized_source"], "rxnorm_ingredient")
        self.assertEqual(state.iloc[0]["ingredient_standardized"], "Furosemide")
        self.assertEqual(state.iloc[0]["mapping_candidate_count"], 2)
        self.assertEqual(state.iloc[0]["active_at_review_flag"], 1)

    def test_extract_queries_from_encounter_state_deduplicates_on_normalized_candidates(self) -> None:
        state = pd.DataFrame(
            [
                _minimal_encounter_medication_state_row(
                    medication_standardized="acetaminophen",
                    medication_raw="Tylenol 325 mg",
                    medication_normalized="acetaminophen",
                    encounter_id="hadm:10",
                    hadm_id=10,
                    selected_medication_event_id="event-1",
                ),
                _minimal_encounter_medication_state_row(
                    medication_standardized="acetaminophen tablet",
                    medication_raw="Acetaminophen 325 mg Tablet",
                    medication_normalized="acetaminophen",
                    encounter_id="hadm:11",
                    hadm_id=11,
                    selected_medication_event_id="event-2",
                ),
                _minimal_encounter_medication_state_row(
                    medication_standardized="furosemide",
                    medication_raw="Lasix 20 mg",
                    medication_normalized="lasix",
                    encounter_id="hadm:12",
                    hadm_id=12,
                    selected_medication_event_id="event-3",
                ),
            ]
        ).loc[:, ENCOUNTER_MEDICATION_STATE_COLUMNS]

        queries = extract_medication_rxnorm_queries_from_encounter_medication_state(state)

        self.assertEqual(len(queries), 2)
        self.assertEqual(set(queries["medication_query_key"].tolist()), {"acetaminophen", "lasix"})
        acetaminophen = queries.loc[queries["medication_query_key"] == "acetaminophen"].iloc[0]
        self.assertEqual(acetaminophen["medication_normalized"], "acetaminophen")
        self.assertIn(
            acetaminophen["medication_raw"],
            {"Tylenol 325 mg", "Acetaminophen 325 mg Tablet"},
        )

    def test_state_rxnorm_join_enriches_persisted_state_without_rebuild(self) -> None:
        state = pd.DataFrame(
            [
                _minimal_encounter_medication_state_row(
                    medication_standardized="furosemide",
                    medication_raw="Lasix 20 mg",
                    medication_normalized="lasix",
                    medication_standardized_source="normalized_text_fallback",
                    rxnorm_rxcui=None,
                    ingredient_standardized=None,
                    ingredient_resolution_status="unresolved_no_match",
                    mapping_confidence="none",
                    medication_class_standardized="unresolved",
                    medication_mapping_lookup_strategy="normalized_text_fallback",
                    selected_medication_event_id="event-1",
                )
            ]
        ).loc[:, ENCOUNTER_MEDICATION_STATE_COLUMNS]
        mapping = pd.DataFrame(
            [
                _mapping_row(
                    "lasix",
                    medication_raw="Lasix 20 mg",
                    medication_normalized="lasix",
                    medication_standardized="Furosemide",
                    ingredient_standardized="Furosemide",
                    ingredient_resolution_status="resolved_via_related_concept",
                    medication_standardized_source="rxnorm_ingredient",
                    lookup_strategy_used="exact_or_normalized_raw_query",
                    rxnorm_rxcui="123",
                    matched_term="Lasix 20 MG Oral Tablet",
                    class_labels_json='["opioid"]',
                )
            ]
        )

        enriched = build_encounter_medication_state_rxnorm(
            encounter_medication_state=state,
            medication_rxnorm_mapping=mapping,
        )

        self.assertEqual(enriched.iloc[0]["medication_standardized"], "Furosemide")
        self.assertEqual(enriched.iloc[0]["medication_standardized_source"], "rxnorm_ingredient")
        self.assertEqual(enriched.iloc[0]["rxnorm_rxcui"], "123")
        self.assertEqual(enriched.iloc[0]["ingredient_standardized"], "Furosemide")
        self.assertEqual(enriched.iloc[0]["ingredient_resolution_status"], "resolved_via_related_concept")
        self.assertEqual(enriched.iloc[0]["medication_class_standardized"], "opioid")
        self.assertEqual(
            enriched.iloc[0]["medication_mapping_lookup_strategy"],
            "exact_or_normalized_raw_query",
        )

    def test_mapping_qc_report_uses_state_frequency_for_top_unresolved_strings(self) -> None:
        state = pd.DataFrame(
            [
                _minimal_encounter_medication_state_row(
                    medication_standardized="unknownmed",
                    medication_raw="UnknownMed 10 mg",
                    medication_normalized="unknownmed",
                    encounter_id="hadm:10",
                    hadm_id=10,
                    selected_medication_event_id="event-1",
                ),
                _minimal_encounter_medication_state_row(
                    medication_standardized="unknownmed repeat",
                    medication_raw="UnknownMed 10 mg",
                    medication_normalized="unknownmed",
                    encounter_id="hadm:11",
                    hadm_id=11,
                    selected_medication_event_id="event-2",
                ),
                _minimal_encounter_medication_state_row(
                    medication_standardized="othermed",
                    medication_raw="OtherMed 5 mg",
                    medication_normalized="othermed",
                    encounter_id="hadm:12",
                    hadm_id=12,
                    selected_medication_event_id="event-3",
                ),
            ]
        ).loc[:, ENCOUNTER_MEDICATION_STATE_COLUMNS]
        mapping = pd.DataFrame(
            [
                _mapping_row(
                    "unknownmed",
                    medication_raw="UnknownMed 10 mg",
                    medication_normalized="unknownmed",
                    medication_standardized="unknownmed",
                    medication_standardized_source="normalized_text_fallback",
                    ingredient_standardized=None,
                    ingredient_resolution_status="unresolved_no_match",
                    rxnorm_rxcui=None,
                    matched_term=None,
                    matched_term_type=None,
                ),
                _mapping_row(
                    "othermed",
                    medication_raw="OtherMed 5 mg",
                    medication_normalized="othermed",
                    medication_standardized="othermed",
                    medication_standardized_source="normalized_text_fallback",
                    ingredient_standardized=None,
                    ingredient_resolution_status="unresolved_no_match",
                    rxnorm_rxcui=None,
                    matched_term=None,
                    matched_term_type=None,
                ),
            ]
        )

        report = build_medication_rxnorm_mapping_qc_report(
            medication_rxnorm_mapping=mapping,
            encounter_medication_state=state,
        )

        self.assertIn("- UnknownMed 10 mg: 2", report)
        self.assertIn("- OtherMed 5 mg: 1", report)

    def test_state_rxnorm_join_does_not_downgrade_existing_enriched_state(self) -> None:
        state = pd.DataFrame(
            [
                _minimal_encounter_medication_state_row(
                    medication_standardized="Furosemide",
                    medication_raw="Lasix 20 mg",
                    medication_normalized="lasix",
                    medication_standardized_source="rxnorm_ingredient",
                    rxnorm_rxcui="123",
                    rxnorm_matched_term="Lasix 20 MG Oral Tablet",
                    rxnorm_term_type="SCD",
                    ingredient_standardized="Furosemide",
                    ingredient_resolution_status="resolved_via_related_concept",
                    mapping_confidence="high",
                    medication_class_standardized="unresolved",
                    medication_mapping_lookup_strategy="exact_or_normalized_raw_query",
                    selected_medication_event_id="event-1",
                )
            ]
        ).loc[:, ENCOUNTER_MEDICATION_STATE_COLUMNS]
        mapping = pd.DataFrame(
            [
                _mapping_row(
                    "lasix",
                    medication_raw="Lasix 20 mg",
                    medication_normalized="lasix",
                    medication_standardized="lasix",
                    medication_standardized_source="normalized_text_fallback",
                    ingredient_standardized=None,
                    ingredient_resolution_status="unresolved_cache_only_miss",
                    rxnorm_rxcui=None,
                    matched_term=None,
                    matched_term_type=None,
                )
            ]
        )

        enriched = build_encounter_medication_state_rxnorm(
            encounter_medication_state=state,
            medication_rxnorm_mapping=mapping,
        )

        self.assertEqual(enriched.iloc[0]["medication_standardized"], "Furosemide")
        self.assertEqual(enriched.iloc[0]["medication_standardized_source"], "rxnorm_ingredient")
        self.assertEqual(enriched.iloc[0]["rxnorm_rxcui"], "123")
        self.assertEqual(enriched.iloc[0]["ingredient_standardized"], "Furosemide")
        self.assertEqual(enriched.iloc[0]["ingredient_resolution_status"], "resolved_via_related_concept")


class _CountingMapper:
    mapper_name = "fake-counting-mapper"
    mapper_version = "0.1"

    def __init__(self) -> None:
        self.call_count = 0
        self.class_assigner = PlaceholderMedicationClassAssigner()
        self.resolver = self

    def resolve(self, *, raw_name: object, normalized_hint: object = None) -> MedicationIdentityResolution:
        self.call_count += 1
        query = str(normalized_hint or raw_name)
        return MedicationIdentityResolution(
            raw_name=str(raw_name),
            normalized_query=query,
            lookup_strategy_used="fake_lookup",
            returned_rxcui="111",
            matched_term=query,
            matched_term_type="IN",
            medication_standardized=query,
            medication_standardized_source="rxnorm_ingredient",
            ingredient_rxcui="111",
            ingredient_standardized=query,
            ingredient_resolution_status="resolved_to_ingredient",
            mapping_confidence="high",
            provenance={},
        )


def _minimal_medication_events(
    *,
    medication_normalized: str,
    raw_medication_name: str,
    medication_event_id: str = "event-1",
    starttime: str = "2125-03-19 19:00:00",
    stoptime: str = "2125-03-20 12:00:00",
) -> pd.DataFrame:
    row = {
        "subject_id": 1,
        "hadm_id": 10,
        "stay_id": None,
        "encounter_id": "hadm:10",
        "encounter_source": "hospital_only",
        "encounter_start": "2125-03-19 18:00:00",
        "encounter_end": "2125-03-20 10:00:00",
        "medication_event_id": medication_event_id,
        "medication_event_type": "hospital_order",
        "event_source_category": "hospital_medication_order",
        "event_source_table": "prescriptions",
        "raw_medication_name": raw_medication_name,
        "medication_name": raw_medication_name,
        "medication_normalized": medication_normalized,
        "medication_prestandardized_text": medication_normalized,
        "event_time": starttime,
        "starttime": starttime,
        "stoptime": stoptime,
        "route": "PO",
        "frequency": "BID",
        "status": "active",
        "dose_value": "20",
        "dose_unit": "mg",
        "pharmacy_id": 500,
        "poe_id": 700,
        "emar_id": None,
        "emar_seq": None,
        "source_home_medrecon": 0,
        "source_ed_pyxis": 0,
        "source_hospital_order": 1,
        "source_hospital_admin": 0,
        "pharmacy_enriched_flag": 1,
        "order_enrichment_applied_flag": 1,
        "order_enrichment_source_table": "pharmacy",
        "continued_from_home_inferred": 0,
        "continued_from_home_inferred_flag": 0,
        "newly_started_during_encounter_inferred": 1,
        "newly_started_during_encounter_inferred_flag": 1,
        "continuity_inference_rule": None,
        "medication_episode_id": "episode-1",
        "prescription_segment_count": 1,
        "prescription_segments_json": "[]",
        "source_tables_json": "[\"prescriptions\"]",
        "source_record_provenance_json": "{\"role\": \"prescriptions\"}",
        "medication_event_build_run_id": "medication-events-test",
        "medication_event_contract_version": MEDICATION_EVENTS_CONTRACT_VERSION,
    }
    dataframe = pd.DataFrame([row])
    return dataframe.loc[:, MEDICATION_EVENT_COLUMNS]


def _mapping_row(
    medication_query_key: str,
    *,
    medication_raw: str | None = None,
    medication_normalized: str | None = None,
    raw_medication_string: str | None = None,
    medication_standardized: str | None = None,
    ingredient_standardized: str | None = None,
    ingredient_resolution_status: str = "resolved_to_ingredient",
    medication_standardized_source: str = "rxnorm_ingredient",
    lookup_strategy_used: str = "cached_prior_result",
    rxnorm_rxcui: str | None = "111",
    matched_term: str | None = None,
    matched_term_type: str | None = "IN",
    class_labels_json: str = "[]",
) -> dict[str, object]:
    lookup_status = (
        "resolved"
        if ingredient_resolution_status in {"resolved_to_ingredient", "resolved_via_related_concept"}
        else ingredient_resolution_status
    )
    row = {
        "medication_query_key": medication_query_key,
        "medication_raw": medication_raw or raw_medication_string or medication_query_key,
        "medication_normalized": medication_normalized or medication_query_key,
        "raw_medication_string": medication_raw or raw_medication_string or medication_query_key,
        "normalized_query_string": medication_normalized or medication_query_key,
        "lookup_mode": LOOKUP_MODE_CACHE_FIRST,
        "lookup_strategy_used": lookup_strategy_used,
        "lookup_status": lookup_status,
        "rxnorm_rxcui": rxnorm_rxcui,
        "matched_term": matched_term or medication_standardized or medication_query_key,
        "matched_term_type": matched_term_type,
        "ingredient_rxcui": rxnorm_rxcui,
        "ingredient_standardized": ingredient_standardized,
        "ingredient_resolution_status": ingredient_resolution_status,
        "medication_standardized": medication_standardized or medication_query_key,
        "medication_standardized_source": medication_standardized_source,
        "mapping_confidence": "high"
        if ingredient_resolution_status in {"resolved_to_ingredient", "resolved_via_related_concept"}
        else "none",
        "ambiguous_match_flag": 0,
        "candidate_match_count": 1,
        "candidate_rxcuis_json": "[\"111\"]" if rxnorm_rxcui else "[]",
        "ambiguity_note": None,
        "unresolved_reason": None if lookup_status == "resolved" else lookup_status,
        "class_assignment_status": "class_assignment_pending",
        "class_ids_json": "[]",
        "class_labels_json": class_labels_json,
        "lookup_timestamp": "2125-03-20 10:00:00",
        "api_called_flag": 1,
        "mapper_name": "rxnorm_api_bootstrap_mapper",
        "mapper_version": "0.1",
        "rxnorm_version": "2026-03-02",
        "rxnorm_api_version": "rxnav_rest_v1",
        "mapping_provenance_json": "{\"lookup_mode\": \"cache_first\"}",
        "medication_rxnorm_mapping_build_run_id": "medication-rxnorm-mapping-test",
        "medication_rxnorm_mapping_contract_version": MEDICATION_RXNORM_MAPPING_CONTRACT_VERSION,
    }
    return {column_name: row.get(column_name) for column_name in MEDICATION_RXNORM_MAPPING_COLUMNS}


def _minimal_encounter_medication_state_row(**overrides: object) -> dict[str, object]:
    row = {
        "subject_id": 1,
        "encounter_id": "hadm:10",
        "hadm_id": 10,
        "stay_id": None,
        "review_timestamp": "2125-03-20 10:00:00",
        "review_timestamp_source": "encounter_end",
        "age_proxy": 77,
        "age_group": "75-84",
        "review_time_policy_name": "discharge_capped_latest_available",
        "review_timestamp_candidate": "2125-03-20 10:00:00",
        "review_timestamp_candidate_source": "encounter_end",
        "review_time_capped_to_discharge_flag": 0,
        "review_time_validated_flag": 1,
        "discharge_boundary": "2125-03-20 10:00:00",
        "medication_raw": "Furosemide 20 mg",
        "medication_normalized": "furosemide",
        "medication_standardized": "furosemide",
        "medication_standardized_source": "normalized_text_fallback",
        "rxnorm_rxcui": None,
        "rxnorm_matched_term": None,
        "rxnorm_term_type": None,
        "ingredient_standardized": None,
        "ingredient_resolution_status": "unresolved_no_match",
        "mapping_confidence": "none",
        "ambiguous_mapping_flag": 0,
        "mapping_candidate_count": 1,
        "medication_mapping_lookup_strategy": "normalized_text_fallback",
        "medication_class_standardized": "unresolved",
        "medication_status_at_review": "active_at_review_time",
        "active_at_review_flag": 1,
        "continued_from_home_inferred": 0,
        "newly_started_during_encounter_inferred": 1,
        "route": "PO",
        "frequency": "BID",
        "status": "active",
        "selected_medication_event_id": "event-1",
        "selected_medication_event_type": "hospital_order",
        "active_event_count_at_review": 1,
        "candidate_event_count": 1,
        "source_home_medrecon_flag": 0,
        "source_ed_pyxis_flag": 0,
        "source_hospital_order_flag": 1,
        "source_hospital_admin_flag": 0,
        "source_tables_json": "[\"prescriptions\"]",
        "source_record_provenance_json": "{\"role\": \"prescriptions\"}",
        "encounter_medication_state_build_run_id": "encounter-medication-state-test",
        "encounter_medication_state_contract_version": ENCOUNTER_MEDICATION_STATE_CONTRACT_VERSION,
    }
    row.update(overrides)
    return row


def _fake_exact_transport(url: str, timeout_seconds: float) -> dict[str, object]:
    del timeout_seconds
    if "rxcui.json" in url and "Lasix" in url:
        return {"idGroup": {"rxnormId": ["123"]}}
    if "rxcui/123/properties.json" in url:
        return {"properties": {"name": "Lasix 20 MG Oral Tablet", "tty": "SCD"}}
    if "rxcui/123/related.json" in url:
        return {
            "relatedGroup": {
                "conceptGroup": [
                    {
                        "tty": "IN",
                        "conceptProperties": [
                            {"rxcui": "321", "name": "Furosemide", "tty": "IN"}
                        ],
                    }
                ]
            }
        }
    if "version.json" in url:
        return {"version": "2026-03-02"}
    raise AssertionError(f"Unexpected URL: {url}")


def _fake_approximate_transport(url: str, timeout_seconds: float) -> dict[str, object]:
    del timeout_seconds
    if "rxcui.json" in url:
        return {"idGroup": {}}
    if "approximateTerm.json" in url:
        return {"approximateGroup": {"candidate": [{"rxcui": "555", "rank": "1"}]}}
    if "rxcui/555/properties.json" in url:
        return {"properties": {"name": "Acetaminophen 325 MG Oral Tablet", "tty": "SCD"}}
    if "rxcui/555/related.json" in url:
        return {
            "relatedGroup": {
                "conceptGroup": [
                    {
                        "tty": "IN",
                        "conceptProperties": [
                            {"rxcui": "777", "name": "Acetaminophen", "tty": "IN"}
                        ],
                    }
                ]
            }
        }
    if "version.json" in url:
        return {"version": "2026-03-02"}
    raise AssertionError(f"Unexpected URL: {url}")


def _fake_related_lookup_error_transport(url: str, timeout_seconds: float) -> dict[str, object]:
    del timeout_seconds
    if "rxcui.json" in url and "Lasix" in url:
        return {"idGroup": {"rxnormId": ["123"]}}
    if "rxcui/123/properties.json" in url:
        return {"properties": {"name": "Lasix 20 MG Oral Tablet", "tty": "SCD"}}
    if "rxcui/123/related.json" in url:
        raise HTTPError(url, 400, "Bad Request", hdrs=None, fp=None)
    raise AssertionError(f"Unexpected URL: {url}")


if __name__ == "__main__":
    unittest.main()
