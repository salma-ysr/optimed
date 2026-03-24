"""Canonical medication event extraction from persisted standardized tables."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.artifact_schemas import (
    MEDICATION_EVENT_COLUMNS,
    MEDICATION_EVENTS_CONTRACT_VERSION,
    validate_medication_events_artifact,
)
from opti_med.data_access.encounters import EncounterIndexBuilder
from opti_med.data_access.medication_consolidation import (
    collapse_continuation_intervals,
    normalize_medication_name,
)
from opti_med.data_access.provenance import (
    dumps_json,
    loads_json_or_none,
    source_provenance_payload,
)
from opti_med.standardized import SOURCE_MANIFEST_METADATA_COLUMNS, StandardizedParquetRepository


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
CONTINUITY_INFERENCE_RULE = (
    "home_medrecon_match_within_subject_hadm_or_stay_after_simple_text_canonicalization"
)


@dataclass(frozen=True)
class MedicationEventsBuildResult:
    """Built medication events and its output path."""

    dataframe: pd.DataFrame
    output_path: Path


class CanonicalMedicationEventBuilder:
    """Build a unified medication history layer from persisted standardized inputs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = StandardizedParquetRepository(settings)
        self.encounter_builder = EncounterIndexBuilder(settings)

    def build(self, encounter_index: pd.DataFrame | None = None) -> pd.DataFrame:
        """Build the canonical medication-events table with stable columns."""
        build_run_id = f"medication-events-{uuid4().hex[:12]}"
        encounter_index = encounter_index if encounter_index is not None else self.encounter_builder.build()

        prescriptions = self.repository.load_source_table(
            "clinical",
            "prescriptions",
            columns=[
                "subject_id",
                "hadm_id",
                "pharmacy_id",
                "poe_id",
                "starttime",
                "stoptime",
                "drug",
                "route",
                "dose_val_rx",
                "dose_unit_rx",
                "doses_per_24_hrs",
                *SOURCE_MANIFEST_METADATA_COLUMNS,
            ],
        )
        medrecon = self.repository.load_optional_source_table(
            "ed",
            "medrecon",
            columns=["subject_id", "stay_id", "charttime", "name", *SOURCE_MANIFEST_METADATA_COLUMNS],
        )
        pyxis = self.repository.load_optional_source_table(
            "ed",
            "pyxis",
            columns=[
                "subject_id",
                "stay_id",
                "charttime",
                "name",
                "med_rn",
                *SOURCE_MANIFEST_METADATA_COLUMNS,
            ],
        )
        pharmacy = self.repository.load_optional_source_table(
            "clinical",
            "pharmacy",
            columns=[
                "subject_id",
                "hadm_id",
                "pharmacy_id",
                "poe_id",
                "medication",
                "status",
                "route",
                "frequency",
                "starttime",
                "stoptime",
                *SOURCE_MANIFEST_METADATA_COLUMNS,
            ],
        )
        emar = self.repository.load_optional_source_table(
            "clinical",
            "emar",
            columns=[
                "subject_id",
                "hadm_id",
                "emar_id",
                "emar_seq",
                "poe_id",
                "pharmacy_id",
                "charttime",
                "medication",
                "event_txt",
                "scheduletime",
                *SOURCE_MANIFEST_METADATA_COLUMNS,
            ],
        )
        emar_detail = self.repository.load_optional_source_table(
            "clinical",
            "emar_detail",
            columns=[
                "subject_id",
                "emar_id",
                "emar_seq",
                "pharmacy_id",
                "dose_given",
                "dose_given_unit",
                "route",
                *SOURCE_MANIFEST_METADATA_COLUMNS,
            ],
        )

        home_events = extract_home_medications(
            medrecon.dataframe if medrecon else None,
            encounter_index=encounter_index,
            standardized_reference=medrecon.standardized_reference if medrecon else None,
        )
        ed_events = extract_ed_medications(
            pyxis.dataframe if pyxis else None,
            encounter_index=encounter_index,
            standardized_reference=pyxis.standardized_reference if pyxis else None,
        )
        hospital_order_events = extract_hospital_med_orders(
            prescriptions=prescriptions.dataframe,
            encounter_index=encounter_index,
            standardized_reference=prescriptions.standardized_reference,
        )
        hospital_order_events = enrich_hospital_meds_with_pharmacy(
            hospital_order_events,
            pharmacy.dataframe if pharmacy else None,
            standardized_reference=pharmacy.standardized_reference if pharmacy else None,
        )
        hospital_order_events = collapse_continuation_intervals(
            hospital_order_events,
            group_columns=["subject_id", "encounter_id", "medication_normalized"],
            start_column="starttime",
            stop_column="stoptime",
            segment_fields=[
                "medication_event_id",
                "event_source_category",
                "event_source_table",
                "raw_medication_name",
                "medication_name",
                "medication_prestandardized_text",
                "starttime",
                "stoptime",
                "route",
                "frequency",
                "dose_value",
                "dose_unit",
                "status",
                "pharmacy_id",
                "poe_id",
                "pharmacy_enriched_flag",
                "order_enrichment_applied_flag",
                "order_enrichment_source_table",
                "source_tables_json",
                "source_record_provenance_json",
            ],
            episode_id_prefix="event-episode",
        ).reindex(columns=MEDICATION_EVENT_COLUMNS)
        hospital_admin_events = extract_hospital_admin_events(
            emar=emar.dataframe if emar else None,
            emar_detail=emar_detail.dataframe if emar_detail else None,
            encounter_index=encounter_index,
            emar_reference=emar.standardized_reference if emar else None,
            emar_detail_reference=emar_detail.standardized_reference if emar_detail else None,
        )

        event_frames = [
            frame
            for frame in [
                home_events,
                ed_events,
                hospital_order_events,
                hospital_admin_events,
            ]
            if not frame.empty
        ]
        if event_frames:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
                    category=FutureWarning,
                )
                medication_events = pd.concat(event_frames, ignore_index=True, sort=False)
        else:
            medication_events = _empty_medication_events()

        medication_events = medication_events.loc[:, MEDICATION_EVENT_COLUMNS].copy()
        medication_events = _fill_medication_event_defaults(medication_events)
        medication_events = infer_home_medication_continuity(medication_events)
        medication_events["medication_event_build_run_id"] = build_run_id
        medication_events["medication_event_contract_version"] = MEDICATION_EVENTS_CONTRACT_VERSION
        medication_events = medication_events.drop_duplicates(
            subset=["medication_event_id"],
        ).reset_index(drop=True)
        medication_events = medication_events.sort_values(
            ["subject_id", "encounter_start", "event_time", "medication_normalized"],
            na_position="last",
        ).reset_index(drop=True)
        validate_medication_events_artifact(medication_events)
        return medication_events

    def save(
        self,
        dataframe: pd.DataFrame,
        output_path: Path | None = None,
    ) -> MedicationEventsBuildResult:
        """Persist the medication-events artifact to Parquet."""
        validate_medication_events_artifact(dataframe)
        target_path = output_path or self.settings.medication_events_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(target_path, index=False)
        return MedicationEventsBuildResult(dataframe=dataframe, output_path=target_path)


def extract_home_medications(
    medrecon: pd.DataFrame | None,
    *,
    encounter_index: pd.DataFrame,
    standardized_reference: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Extract pre-admission medication history from ED medication reconciliation."""
    if medrecon is None or medrecon.empty:
        return _empty_medication_events()

    home = medrecon.loc[:, ["subject_id", "stay_id", "charttime", "name", *SOURCE_MANIFEST_METADATA_COLUMNS]].copy()
    home["charttime"] = pd.to_datetime(home["charttime"], errors="coerce")
    home["medication_normalized"] = home["name"].map(normalize_medication_name)
    home = home.loc[home["medication_normalized"].notna()].copy().reset_index(drop=True)
    if home.empty:
        return _empty_medication_events()
    home = _attach_ed_encounter_context(home, encounter_index=encounter_index)
    home["medication_event_id"] = (
        "home-medrecon-"
        + home["subject_id"].astype(str)
        + "-"
        + home["stay_id"].astype("Int64").astype(str)
        + "-"
        + home.index.astype(str)
    )
    source_tables_json = dumps_json(["medrecon"])
    source_record_provenance_json = _single_source_provenance_json(
        home,
        role_name="medrecon",
        standardized_reference=standardized_reference,
    )
    return _finalize_medication_events(
        home.rename(
            columns={
                "name": "raw_medication_name",
                "charttime": "event_time",
            }
        ).assign(
            medication_event_type="home_medrecon",
            event_source_category="home_medication_reconciliation",
            event_source_table="medrecon",
            medication_name=lambda frame: frame["raw_medication_name"].astype(str).str.strip(),
            medication_prestandardized_text=lambda frame: frame["medication_normalized"],
            starttime=pd.NaT,
            stoptime=pd.NaT,
            route=pd.NA,
            frequency=pd.NA,
            status=pd.NA,
            dose_value=pd.NA,
            dose_unit=pd.NA,
            pharmacy_id=pd.NA,
            poe_id=pd.NA,
            emar_id=pd.NA,
            emar_seq=pd.NA,
            source_home_medrecon=1,
            source_ed_pyxis=0,
            source_hospital_order=0,
            source_hospital_admin=0,
            pharmacy_enriched_flag=0,
            order_enrichment_applied_flag=0,
            order_enrichment_source_table=pd.NA,
            continued_from_home_inferred=0,
            continued_from_home_inferred_flag=0,
            newly_started_during_encounter_inferred=0,
            newly_started_during_encounter_inferred_flag=0,
            continuity_inference_rule=pd.NA,
            medication_episode_id=pd.NA,
            prescription_segment_count=1,
            prescription_segments_json=pd.NA,
            source_tables_json=source_tables_json,
            source_record_provenance_json=source_record_provenance_json,
            medication_event_build_run_id=pd.NA,
            medication_event_contract_version=pd.NA,
        )
    )


def extract_ed_medications(
    pyxis: pd.DataFrame | None,
    *,
    encounter_index: pd.DataFrame,
    standardized_reference: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Extract ED medication dispense events from standardized Pyxis data."""
    if pyxis is None or pyxis.empty:
        return _empty_medication_events()

    ed = pyxis.loc[
        :,
        ["subject_id", "stay_id", "charttime", "name", "med_rn", *SOURCE_MANIFEST_METADATA_COLUMNS],
    ].copy()
    ed["raw_name"] = ed["name"].where(ed["name"].notna() & (ed["name"] != ""), ed["med_rn"])
    ed["charttime"] = pd.to_datetime(ed["charttime"], errors="coerce")
    ed["medication_normalized"] = ed["raw_name"].map(normalize_medication_name)
    ed = ed.loc[ed["medication_normalized"].notna()].copy().reset_index(drop=True)
    if ed.empty:
        return _empty_medication_events()
    ed = _attach_ed_encounter_context(ed, encounter_index=encounter_index)
    ed["medication_event_id"] = (
        "ed-pyxis-"
        + ed["subject_id"].astype(str)
        + "-"
        + ed["stay_id"].astype("Int64").astype(str)
        + "-"
        + ed.index.astype(str)
    )
    source_tables_json = dumps_json(["pyxis"])
    source_record_provenance_json = _single_source_provenance_json(
        ed,
        role_name="pyxis",
        standardized_reference=standardized_reference,
    )
    return _finalize_medication_events(
        ed.rename(
            columns={
                "raw_name": "raw_medication_name",
                "charttime": "event_time",
            }
        ).assign(
            medication_event_type="ed_pyxis",
            event_source_category="ed_medication_event",
            event_source_table="pyxis",
            medication_name=lambda frame: frame["raw_medication_name"].astype(str).str.strip(),
            medication_prestandardized_text=lambda frame: frame["medication_normalized"],
            starttime=lambda frame: frame["event_time"],
            stoptime=pd.NaT,
            route=pd.NA,
            frequency=pd.NA,
            status=pd.NA,
            dose_value=pd.NA,
            dose_unit=pd.NA,
            pharmacy_id=pd.NA,
            poe_id=pd.NA,
            emar_id=pd.NA,
            emar_seq=pd.NA,
            source_home_medrecon=0,
            source_ed_pyxis=1,
            source_hospital_order=0,
            source_hospital_admin=0,
            pharmacy_enriched_flag=0,
            order_enrichment_applied_flag=0,
            order_enrichment_source_table=pd.NA,
            continued_from_home_inferred=0,
            continued_from_home_inferred_flag=0,
            newly_started_during_encounter_inferred=0,
            newly_started_during_encounter_inferred_flag=0,
            continuity_inference_rule=pd.NA,
            medication_episode_id=pd.NA,
            prescription_segment_count=1,
            prescription_segments_json=pd.NA,
            source_tables_json=source_tables_json,
            source_record_provenance_json=source_record_provenance_json,
            medication_event_build_run_id=pd.NA,
            medication_event_contract_version=pd.NA,
        )
    )


def extract_hospital_med_orders(
    *,
    prescriptions: pd.DataFrame,
    encounter_index: pd.DataFrame,
    standardized_reference: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Extract hospital medication orders from standardized prescriptions."""
    orders = prescriptions.loc[
        :,
        [
            "subject_id",
            "hadm_id",
            "pharmacy_id",
            "poe_id",
            "starttime",
            "stoptime",
            "drug",
            "route",
            "dose_val_rx",
            "dose_unit_rx",
            "doses_per_24_hrs",
            *SOURCE_MANIFEST_METADATA_COLUMNS,
        ],
    ].copy()
    orders["starttime"] = pd.to_datetime(orders["starttime"], errors="coerce")
    orders["stoptime"] = pd.to_datetime(orders["stoptime"], errors="coerce")
    orders["medication_normalized"] = orders["drug"].map(normalize_medication_name)
    orders = orders.loc[orders["medication_normalized"].notna()].copy().reset_index(drop=True)
    if orders.empty:
        return _empty_medication_events()
    orders = _attach_hospital_encounter_context(orders, encounter_index=encounter_index)
    orders["medication_event_id"] = (
        "hospital-order-"
        + orders["subject_id"].astype(str)
        + "-"
        + orders["hadm_id"].astype("Int64").astype(str)
        + "-"
        + orders.index.astype(str)
    )
    source_tables_json = dumps_json(["prescriptions"])
    source_record_provenance_json = _single_source_provenance_json(
        orders,
        role_name="prescriptions",
        standardized_reference=standardized_reference,
    )
    return _finalize_medication_events(
        orders.rename(
            columns={
                "drug": "raw_medication_name",
                "doses_per_24_hrs": "frequency",
                "dose_val_rx": "dose_value",
                "dose_unit_rx": "dose_unit",
            }
        ).assign(
            medication_event_type="hospital_order",
            event_source_category="hospital_medication_order",
            event_source_table="prescriptions",
            medication_name=lambda frame: frame["raw_medication_name"].astype(str).str.strip(),
            medication_prestandardized_text=lambda frame: frame["medication_normalized"],
            event_time=lambda frame: frame["starttime"],
            status=pd.NA,
            emar_id=pd.NA,
            emar_seq=pd.NA,
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=1,
            source_hospital_admin=0,
            pharmacy_enriched_flag=0,
            order_enrichment_applied_flag=0,
            order_enrichment_source_table=pd.NA,
            continued_from_home_inferred=0,
            continued_from_home_inferred_flag=0,
            newly_started_during_encounter_inferred=0,
            newly_started_during_encounter_inferred_flag=0,
            continuity_inference_rule=pd.NA,
            medication_episode_id=pd.NA,
            prescription_segment_count=1,
            prescription_segments_json=pd.NA,
            source_tables_json=source_tables_json,
            source_record_provenance_json=source_record_provenance_json,
            medication_event_build_run_id=pd.NA,
            medication_event_contract_version=pd.NA,
        )
    )


def enrich_hospital_meds_with_pharmacy(
    hospital_orders: pd.DataFrame,
    pharmacy: pd.DataFrame | None,
    *,
    standardized_reference: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Enrich hospital orders with pharmacy metadata when available."""
    if pharmacy is None or pharmacy.empty or hospital_orders.empty:
        return hospital_orders

    pharmacy_frame = pharmacy.loc[
        :,
        [
            "subject_id",
            "hadm_id",
            "pharmacy_id",
            "poe_id",
            "medication",
            "status",
            "route",
            "frequency",
            "starttime",
            "stoptime",
            *SOURCE_MANIFEST_METADATA_COLUMNS,
        ],
    ].copy()
    pharmacy_frame["starttime"] = pd.to_datetime(pharmacy_frame["starttime"], errors="coerce")
    pharmacy_frame["stoptime"] = pd.to_datetime(pharmacy_frame["stoptime"], errors="coerce")
    pharmacy_frame["medication_normalized_pharmacy"] = pharmacy_frame["medication"].map(
        normalize_medication_name
    )
    pharmacy_frame["medication_match_key_pharmacy"] = pharmacy_frame["medication_normalized_pharmacy"].map(
        _simple_medication_match_key
    )

    enriched = hospital_orders.merge(
        pharmacy_frame,
        how="left",
        on=["subject_id", "hadm_id", "pharmacy_id"],
        suffixes=("", "_pharmacy"),
    )
    enriched["medication_match_key"] = enriched["medication_normalized"].map(_simple_medication_match_key)
    medication_match = (
        enriched["medication_match_key_pharmacy"].isna()
        | (enriched["medication_match_key_pharmacy"] == enriched["medication_match_key"])
    )
    enriched = enriched.loc[medication_match].copy()
    if enriched.empty:
        return hospital_orders

    if "status_pharmacy" in enriched:
        enrichment_rank = enriched[
            ["medication", "route_pharmacy", "frequency_pharmacy", "status_pharmacy"]
        ].notna().sum(axis=1)
        enriched = (
            enriched.assign(_enrichment_rank=enrichment_rank)
            .sort_values("_enrichment_rank", ascending=False)
            .drop_duplicates(subset=["medication_event_id"], keep="first")
            .drop(columns="_enrichment_rank")
        )

    enriched["medication_name"] = enriched["medication_name"].where(
        enriched["medication_name"].notna() & (enriched["medication_name"] != ""),
        enriched["medication"],
    )
    enriched["route"] = enriched["route"].where(
        enriched["route"].notna() & (enriched["route"] != ""),
        enriched["route_pharmacy"],
    )
    enriched["frequency"] = enriched["frequency"].where(
        enriched["frequency"].notna() & (enriched["frequency"] != ""),
        enriched["frequency_pharmacy"],
    )
    enriched["status"] = enriched["status"].where(
        enriched["status"].notna() & (enriched["status"] != ""),
        enriched["status_pharmacy"],
    )
    enrichment_applied = (
        enriched[["medication", "route_pharmacy", "frequency_pharmacy", "status_pharmacy"]]
        .notna()
        .any(axis=1)
        .astype(int)
    )
    enriched["pharmacy_enriched_flag"] = enrichment_applied
    enriched["order_enrichment_applied_flag"] = enrichment_applied
    enriched["order_enrichment_source_table"] = pd.Series(
        [
            "pharmacy" if value == 1 else pd.NA
            for value in enrichment_applied.tolist()
        ],
        dtype="string",
    )
    if int(enrichment_applied.max()) == 1:
        pharmacy_provenance = _single_source_provenance_json(
            pharmacy_frame,
            role_name="pharmacy",
            standardized_reference=standardized_reference,
        )
        enriched["source_tables_json"] = [
            dumps_json(["prescriptions", "pharmacy"]) if flag == 1 else tables_json
            for flag, tables_json in zip(
                enrichment_applied.tolist(),
                enriched["source_tables_json"].tolist(),
            )
        ]
        enriched["source_record_provenance_json"] = [
            _merge_provenance_json(
                primary_json=source_json,
                secondary_role_name="pharmacy",
                secondary_json=pharmacy_provenance,
            )
            if flag == 1
            else source_json
            for flag, source_json in zip(
                enrichment_applied.tolist(),
                enriched["source_record_provenance_json"].tolist(),
            )
        ]
    return _finalize_medication_events(enriched)


def extract_hospital_admin_events(
    *,
    emar: pd.DataFrame | None,
    emar_detail: pd.DataFrame | None,
    encounter_index: pd.DataFrame,
    emar_reference: dict[str, object] | None = None,
    emar_detail_reference: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Extract hospital administration events from standardized eMAR tables."""
    if emar is None or emar.empty:
        return _empty_medication_events()

    admin = emar.loc[
        :,
        [
            "subject_id",
            "hadm_id",
            "emar_id",
            "emar_seq",
            "poe_id",
            "pharmacy_id",
            "charttime",
            "medication",
            "event_txt",
            "scheduletime",
            *SOURCE_MANIFEST_METADATA_COLUMNS,
        ],
    ].copy()
    admin["charttime"] = pd.to_datetime(admin["charttime"], errors="coerce")
    admin["scheduletime"] = pd.to_datetime(admin["scheduletime"], errors="coerce")
    admin["medication_normalized"] = admin["medication"].map(normalize_medication_name)
    admin = admin.loc[admin["medication_normalized"].notna()].copy().reset_index(drop=True)
    if admin.empty:
        return _empty_medication_events()
    admin["source_tables_json"] = dumps_json(["emar"])
    admin["source_record_provenance_json"] = _single_source_provenance_json(
        admin,
        role_name="emar",
        standardized_reference=emar_reference,
    )

    if emar_detail is not None and not emar_detail.empty:
        detail = emar_detail.loc[
            :,
            [
                "subject_id",
                "emar_id",
                "emar_seq",
                "pharmacy_id",
                "dose_given",
                "dose_given_unit",
                "route",
                *SOURCE_MANIFEST_METADATA_COLUMNS,
            ],
        ].copy()
        detail = detail.rename(columns={"route": "route_detail"})
        admin = admin.merge(
            detail,
            how="left",
            on=["subject_id", "emar_id", "emar_seq"],
            suffixes=("", "_detail"),
        )
        detail_match = admin[["dose_given", "dose_given_unit", "route_detail"]].notna().any(axis=1).astype(int)
        if int(detail_match.max()) == 1:
            detail_provenance = _single_source_provenance_json(
                detail,
                role_name="emar_detail",
                standardized_reference=emar_detail_reference,
            )
            admin["source_tables_json"] = [
                dumps_json(["emar", "emar_detail"]) if flag == 1 else tables_json
                for flag, tables_json in zip(
                    detail_match.tolist(),
                    admin["source_tables_json"].tolist(),
                )
            ]
            admin["source_record_provenance_json"] = [
                _merge_provenance_json(
                    primary_json=source_json,
                    secondary_role_name="emar_detail",
                    secondary_json=detail_provenance,
                )
                if flag == 1
                else source_json
                for flag, source_json in zip(
                    detail_match.tolist(),
                    admin["source_record_provenance_json"].tolist(),
                )
            ]

    admin = _attach_hospital_encounter_context(admin, encounter_index=encounter_index)
    admin["medication_event_id"] = (
        "hospital-admin-"
        + admin["subject_id"].astype(str)
        + "-"
        + admin["hadm_id"].astype("Int64").astype(str)
        + "-"
        + admin["emar_id"].astype("Int64").astype(str)
        + "-"
        + admin["emar_seq"].astype("Int64").astype(str)
    )
    return _finalize_medication_events(
        admin.rename(
            columns={
                "medication": "raw_medication_name",
                "event_txt": "status",
                "dose_given": "dose_value",
                "dose_given_unit": "dose_unit",
                "route_detail": "route",
            }
        ).assign(
            medication_event_type="hospital_admin",
            event_source_category="hospital_administration_event",
            event_source_table="emar",
            medication_name=lambda frame: frame["raw_medication_name"].astype(str).str.strip(),
            medication_prestandardized_text=lambda frame: frame["medication_normalized"],
            event_time=lambda frame: frame["charttime"],
            starttime=lambda frame: frame["scheduletime"].where(
                frame["scheduletime"].notna(),
                frame["charttime"],
            ),
            stoptime=pd.NaT,
            frequency=pd.NA,
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=0,
            source_hospital_admin=1,
            pharmacy_enriched_flag=0,
            order_enrichment_applied_flag=0,
            order_enrichment_source_table=pd.NA,
            continued_from_home_inferred=0,
            continued_from_home_inferred_flag=0,
            newly_started_during_encounter_inferred=0,
            newly_started_during_encounter_inferred_flag=0,
            continuity_inference_rule=pd.NA,
            stay_id=pd.NA,
            medication_episode_id=pd.NA,
            prescription_segment_count=1,
            prescription_segments_json=pd.NA,
            medication_event_build_run_id=pd.NA,
            medication_event_contract_version=pd.NA,
        )
    )


def infer_home_medication_continuity(medication_events: pd.DataFrame) -> pd.DataFrame:
    """Infer whether encounter medications look continued from home or newly started."""
    if medication_events.empty:
        return medication_events

    inferred = medication_events.copy()
    inferred["_continuity_match_key"] = inferred["medication_normalized"].map(_simple_medication_match_key)
    home_by_hadm = (
        inferred.loc[
            (inferred["source_home_medrecon"] == 1)
            & inferred["hadm_id"].notna()
            & inferred["_continuity_match_key"].notna(),
            ["subject_id", "hadm_id", "_continuity_match_key"],
        ]
        .drop_duplicates()
        .assign(_continued_home_hadm=1)
    )
    home_by_stay = (
        inferred.loc[
            (inferred["source_home_medrecon"] == 1)
            & inferred["stay_id"].notna()
            & inferred["_continuity_match_key"].notna(),
            ["subject_id", "stay_id", "_continuity_match_key"],
        ]
        .drop_duplicates()
        .assign(_continued_home_stay=1)
    )

    inferred = inferred.merge(
        home_by_hadm,
        how="left",
        on=["subject_id", "hadm_id", "_continuity_match_key"],
    )
    inferred = inferred.merge(
        home_by_stay,
        how="left",
        on=["subject_id", "stay_id", "_continuity_match_key"],
    )

    inferred["continued_from_home_inferred"] = (
        (
            inferred["_continued_home_hadm"].fillna(0).astype(int)
            | inferred["_continued_home_stay"].fillna(0).astype(int)
        )
        & (inferred["source_home_medrecon"] == 0)
    ).astype(int)
    inferred["continued_from_home_inferred_flag"] = inferred["continued_from_home_inferred"]
    inferred["newly_started_during_encounter_inferred"] = (
        (inferred["source_home_medrecon"] == 0)
        & (inferred["medication_normalized"].notna())
        & (inferred["continued_from_home_inferred"] == 0)
    ).astype(int)
    inferred["newly_started_during_encounter_inferred_flag"] = inferred[
        "newly_started_during_encounter_inferred"
    ]
    inferred["continuity_inference_rule"] = pd.Series(
        [
            CONTINUITY_INFERENCE_RULE if flag == 0 and pd.notna(name) else pd.NA
            for flag, name in zip(
                inferred["source_home_medrecon"].tolist(),
                inferred["medication_normalized"].tolist(),
            )
        ],
        dtype="string",
    )
    return inferred.drop(
        columns=["_continuity_match_key", "_continued_home_hadm", "_continued_home_stay"]
    )


def summarize_medication_events(dataframe: pd.DataFrame) -> list[str]:
    """Return compact summaries for the medication-events artifact."""
    return [
        f"rows={len(dataframe):,}, columns={dataframe.shape[1]}",
        f"unique_subjects={dataframe['subject_id'].nunique():,}",
        f"home_medrecon_rows={int(dataframe['source_home_medrecon'].sum()):,}",
        f"ed_pyxis_rows={int(dataframe['source_ed_pyxis'].sum()):,}",
        f"hospital_order_rows={int(dataframe['source_hospital_order'].sum()):,}",
        f"hospital_admin_rows={int(dataframe['source_hospital_admin'].sum()):,}",
        f"order_enriched_rows={int(dataframe['order_enrichment_applied_flag'].sum()):,}",
    ]


def _attach_ed_encounter_context(
    dataframe: pd.DataFrame,
    *,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame:
    encounter_lookup = encounter_index.loc[
        :,
        ["subject_id", "hadm_id", "stay_id", "encounter_id", "encounter_source", "encounter_start", "encounter_end"],
    ].drop_duplicates(subset=["subject_id", "stay_id"])
    return dataframe.merge(
        encounter_lookup,
        how="left",
        on=["subject_id", "stay_id"],
        validate="many_to_one",
    )


def _attach_hospital_encounter_context(
    dataframe: pd.DataFrame,
    *,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame:
    encounter_lookup = encounter_index.loc[
        :,
        ["subject_id", "hadm_id", "encounter_id", "encounter_source", "encounter_start", "encounter_end"],
    ].drop_duplicates(subset=["subject_id", "hadm_id"])
    return dataframe.merge(
        encounter_lookup,
        how="left",
        on=["subject_id", "hadm_id"],
        validate="many_to_one",
    )


def _finalize_medication_events(dataframe: pd.DataFrame) -> pd.DataFrame:
    finalized = dataframe.copy()
    for column in ["event_time", "starttime", "stoptime", "encounter_start", "encounter_end"]:
        finalized[column] = _format_timestamp_series(finalized[column])
    if "medication_name" in finalized:
        finalized["medication_name"] = finalized["medication_name"].replace("", pd.NA)
    if "raw_medication_name" in finalized:
        finalized["raw_medication_name"] = finalized["raw_medication_name"].replace("", pd.NA)
    return finalized.reindex(columns=MEDICATION_EVENT_COLUMNS)


def _fill_medication_event_defaults(dataframe: pd.DataFrame) -> pd.DataFrame:
    filled = dataframe.copy()
    for column in [
        "source_home_medrecon",
        "source_ed_pyxis",
        "source_hospital_order",
        "source_hospital_admin",
        "pharmacy_enriched_flag",
        "order_enrichment_applied_flag",
        "continued_from_home_inferred",
        "continued_from_home_inferred_flag",
        "newly_started_during_encounter_inferred",
        "newly_started_during_encounter_inferred_flag",
        "prescription_segment_count",
    ]:
        filled[column] = pd.to_numeric(filled[column], errors="coerce").fillna(0).astype(int)
    filled["medication_prestandardized_text"] = filled["medication_normalized"]
    filled["order_enrichment_source_table"] = filled["order_enrichment_source_table"].where(
        filled["order_enrichment_source_table"].notna(),
        pd.NA,
    )
    filled["prescription_segments_json"] = filled["prescription_segments_json"].where(
        filled["prescription_segments_json"].notna(),
        pd.NA,
    )
    filled["medication_episode_id"] = filled["medication_episode_id"].where(
        filled["medication_episode_id"].notna(),
        pd.NA,
    )
    return filled


def _empty_medication_events() -> pd.DataFrame:
    return pd.DataFrame(columns=MEDICATION_EVENT_COLUMNS)


def _single_source_provenance_json(
    dataframe: pd.DataFrame,
    *,
    role_name: str,
    standardized_reference: dict[str, object] | None,
) -> str:
    if dataframe.empty:
        return dumps_json({})
    payload = {
        role_name: source_provenance_payload(
            dataframe.iloc[0],
            role_name=role_name,
            standardized_reference=standardized_reference,
            source_metadata_columns=SOURCE_MANIFEST_METADATA_COLUMNS,
        )
    }
    return dumps_json(payload)


def _merge_provenance_json(
    *,
    primary_json: object,
    secondary_role_name: str,
    secondary_json: object,
) -> str:
    payload = loads_json_or_none(primary_json) or {}
    secondary_payload = loads_json_or_none(secondary_json) or {}
    if secondary_role_name in secondary_payload:
        payload[secondary_role_name] = secondary_payload[secondary_role_name]
    return dumps_json(payload)


def _format_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime(TIMESTAMP_FORMAT)


def _simple_medication_match_key(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    tokens = [
        token
        for token in str(value).strip().split()
        if token and not re.fullmatch(r"\d+(\.\d+)?", token)
    ]
    if not tokens:
        return None
    return " ".join(tokens)
