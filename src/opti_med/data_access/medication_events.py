"""Canonical medication event extraction across home, ED, and hospital sources."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from opti_med.config import Settings
from opti_med.data_access.medication_consolidation import (
    collapse_continuation_intervals,
    normalize_medication_name,
)
from opti_med.data_access.encounters import EncounterIndexBuilder
from opti_med.data_access.loaders import MimicDualDemoLoader


MEDICATION_EVENT_COLUMNS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "encounter_id",
    "encounter_source",
    "encounter_start",
    "encounter_end",
    "medication_event_id",
    "medication_event_type",
    "raw_medication_name",
    "medication_name",
    "medication_normalized",
    "event_time",
    "starttime",
    "stoptime",
    "route",
    "frequency",
    "status",
    "dose_value",
    "dose_unit",
    "pharmacy_id",
    "poe_id",
    "emar_id",
    "emar_seq",
    "source_home_medrecon",
    "source_ed_pyxis",
    "source_hospital_order",
    "source_hospital_admin",
    "pharmacy_enriched_flag",
    "continued_from_home_inferred",
    "newly_started_during_encounter_inferred",
    "medication_episode_id",
    "prescription_segment_count",
    "prescription_segments_json",
]


@dataclass(frozen=True)
class MedicationEventsBuildResult:
    """Built medication events and its output path."""

    dataframe: pd.DataFrame
    output_path: Path


class CanonicalMedicationEventBuilder:
    """Build a unified medication history layer from the available raw sources."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loader = MimicDualDemoLoader(settings)
        self.encounter_builder = EncounterIndexBuilder(settings)

    def build(self) -> pd.DataFrame:
        """Build the canonical medication events table with stable columns."""
        encounter_index = self.encounter_builder.build()
        clinical_tables = self.loader.clinical_loader.load_all()
        edstays = self.loader.load_ed_table("edstays").dataframe

        medrecon_loaded = self.loader.load_optional_ed_table("medrecon")
        pyxis_loaded = self.loader.load_optional_ed_table("pyxis")
        pharmacy_loaded = self.loader.clinical_loader.load_optional_table("pharmacy")
        emar_loaded = self.loader.clinical_loader.load_optional_table("emar")
        emar_detail_loaded = self.loader.clinical_loader.load_optional_table("emar_detail")

        home_events = extract_home_medications(
            medrecon_loaded.dataframe if medrecon_loaded else None,
            edstays=edstays,
            encounter_index=encounter_index,
        )
        ed_events = extract_ed_medications(
            pyxis_loaded.dataframe if pyxis_loaded else None,
            edstays=edstays,
            encounter_index=encounter_index,
        )
        hospital_order_events = extract_hospital_med_orders(
            prescriptions=clinical_tables["prescriptions"],
            encounter_index=encounter_index,
        )
        hospital_order_events = enrich_hospital_meds_with_pharmacy(
            hospital_order_events,
            pharmacy_loaded.dataframe if pharmacy_loaded else None,
        )
        hospital_order_events = collapse_continuation_intervals(
            hospital_order_events,
            group_columns=["subject_id", "encounter_id", "medication_normalized"],
            start_column="starttime",
            stop_column="stoptime",
            segment_fields=[
                "medication_event_id",
                "raw_medication_name",
                "medication_name",
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
            ],
            episode_id_prefix="event-episode",
        ).reindex(columns=MEDICATION_EVENT_COLUMNS)
        hospital_admin_events = extract_hospital_admin_events(
            emar=emar_loaded.dataframe if emar_loaded else None,
            emar_detail=emar_detail_loaded.dataframe if emar_detail_loaded else None,
            encounter_index=encounter_index,
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
        medication_events = medication_events.drop_duplicates(
            subset=[
                "subject_id",
                "hadm_id",
                "stay_id",
                "medication_event_type",
                "medication_normalized",
                "event_time",
                "pharmacy_id",
                "emar_id",
                "emar_seq",
            ]
        ).reset_index(drop=True)
        return medication_events.sort_values(
            ["subject_id", "encounter_start", "event_time", "medication_normalized"],
            na_position="last",
        ).reset_index(drop=True)

    def save(
        self, dataframe: pd.DataFrame, output_path: Path | None = None
    ) -> MedicationEventsBuildResult:
        """Persist the canonical medication event layer to a CSV file."""
        target_path = output_path or self.settings.medication_events_output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(target_path, index=False)
        return MedicationEventsBuildResult(dataframe=dataframe, output_path=target_path)


def extract_home_medications(
    medrecon: pd.DataFrame | None,
    *,
    edstays: pd.DataFrame,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame:
    """Extract pre-admission medication history from ED medication reconciliation."""
    if medrecon is None or medrecon.empty:
        return _empty_medication_events()

    home = medrecon.loc[:, ["subject_id", "stay_id", "charttime", "name"]].copy()
    home["charttime"] = pd.to_datetime(home["charttime"], errors="coerce")
    home["medication_normalized"] = home["name"].map(normalize_medication_name)
    home = home.loc[home["medication_normalized"].notna()].copy()
    home = _attach_ed_encounter_context(home, edstays=edstays, encounter_index=encounter_index)
    home["medication_event_id"] = (
        "home-medrecon-"
        + home["subject_id"].astype(str)
        + "-"
        + home["stay_id"].astype("Int64").astype(str)
        + "-"
        + home.index.astype(str)
    )
    return _finalize_medication_events(
        home.rename(
            columns={
                "name": "raw_medication_name",
                "charttime": "event_time",
            }
        ).assign(
            medication_event_type="home_medrecon",
            medication_name=lambda frame: frame["raw_medication_name"].astype(str).str.strip(),
            starttime=pd.NA,
            stoptime=pd.NA,
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
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=0,
            medication_episode_id=pd.NA,
            prescription_segment_count=1,
            prescription_segments_json=pd.NA,
        )
    )


def extract_ed_medications(
    pyxis: pd.DataFrame | None,
    *,
    edstays: pd.DataFrame,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame:
    """Extract ED medication dispense events from Pyxis."""
    if pyxis is None or pyxis.empty:
        return _empty_medication_events()

    ed = pyxis.loc[:, ["subject_id", "stay_id", "charttime", "name"]].copy()
    ed["charttime"] = pd.to_datetime(ed["charttime"], errors="coerce")
    ed["medication_normalized"] = ed["name"].map(normalize_medication_name)
    ed = ed.loc[ed["medication_normalized"].notna()].copy()
    ed = _attach_ed_encounter_context(ed, edstays=edstays, encounter_index=encounter_index)
    ed["medication_event_id"] = (
        "ed-pyxis-"
        + ed["subject_id"].astype(str)
        + "-"
        + ed["stay_id"].astype("Int64").astype(str)
        + "-"
        + ed.index.astype(str)
    )
    return _finalize_medication_events(
        ed.rename(
            columns={
                "name": "raw_medication_name",
                "charttime": "event_time",
            }
        ).assign(
            medication_event_type="ed_pyxis",
            medication_name=lambda frame: frame["raw_medication_name"].astype(str).str.strip(),
            starttime=lambda frame: frame["event_time"],
            stoptime=pd.NA,
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
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=0,
            medication_episode_id=pd.NA,
            prescription_segment_count=1,
            prescription_segments_json=pd.NA,
        )
    )


def extract_hospital_med_orders(
    *,
    prescriptions: pd.DataFrame,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame:
    """Extract hospital medication orders from prescriptions."""
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
        ],
    ].copy()
    orders["starttime"] = pd.to_datetime(orders["starttime"], errors="coerce")
    orders["stoptime"] = pd.to_datetime(orders["stoptime"], errors="coerce")
    orders["medication_normalized"] = orders["drug"].map(normalize_medication_name)
    orders = orders.loc[orders["medication_normalized"].notna()].copy()
    orders = _attach_hospital_encounter_context(orders, encounter_index=encounter_index)
    orders["medication_event_id"] = (
        "hospital-order-"
        + orders["subject_id"].astype(str)
        + "-"
        + orders["hadm_id"].astype("Int64").astype(str)
        + "-"
        + orders.index.astype(str)
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
            medication_name=lambda frame: frame["raw_medication_name"].astype(str).str.strip(),
            event_time=lambda frame: frame["starttime"],
            status=pd.NA,
            emar_id=pd.NA,
            emar_seq=pd.NA,
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=1,
            source_hospital_admin=0,
            pharmacy_enriched_flag=0,
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=0,
            medication_episode_id=pd.NA,
            prescription_segment_count=1,
            prescription_segments_json=pd.NA,
        )
    )


def enrich_hospital_meds_with_pharmacy(
    hospital_orders: pd.DataFrame,
    pharmacy: pd.DataFrame | None,
) -> pd.DataFrame:
    """Enrich hospital orders with pharmacy metadata when the optional table is available."""
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
        ],
    ].copy()
    pharmacy_frame["starttime"] = pd.to_datetime(pharmacy_frame["starttime"], errors="coerce")
    pharmacy_frame["stoptime"] = pd.to_datetime(pharmacy_frame["stoptime"], errors="coerce")
    pharmacy_frame["medication_normalized_pharmacy"] = pharmacy_frame["medication"].map(
        normalize_medication_name
    )

    enriched = hospital_orders.merge(
        pharmacy_frame,
        how="left",
        on=["subject_id", "hadm_id", "pharmacy_id"],
        suffixes=("", "_pharmacy"),
    )
    medication_match = (
        enriched["medication_normalized_pharmacy"].isna()
        | (enriched["medication_normalized_pharmacy"] == enriched["medication_normalized"])
    )
    enriched = enriched.loc[medication_match].copy()
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
    enriched["status"] = enriched["status_pharmacy"]
    enriched["pharmacy_enriched_flag"] = enriched["medication"].notna().astype(int)
    return _finalize_medication_events(enriched)


def extract_hospital_admin_events(
    *,
    emar: pd.DataFrame | None,
    emar_detail: pd.DataFrame | None,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame:
    """Extract hospital administration events from eMAR, optionally enriched by eMAR detail."""
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
        ],
    ].copy()
    admin["charttime"] = pd.to_datetime(admin["charttime"], errors="coerce")
    admin["scheduletime"] = pd.to_datetime(admin["scheduletime"], errors="coerce")
    admin["medication_normalized"] = admin["medication"].map(normalize_medication_name)
    admin = admin.loc[admin["medication_normalized"].notna()].copy()

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
            ],
        ].copy()
        admin = admin.merge(
            detail,
            how="left",
            on=["subject_id", "emar_id", "emar_seq"],
            suffixes=("", "_detail"),
        )

    admin = _attach_hospital_encounter_context(admin, encounter_index=encounter_index)
    admin["medication_event_id"] = (
        "hospital-admin-"
        + admin["subject_id"].astype(str)
        + "-"
        + admin["hadm_id"].astype("Int64").astype(str)
        + "-"
        + admin["emar_id"].astype(str)
        + "-"
        + admin["emar_seq"].astype(str)
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
            medication_name=lambda frame: frame["raw_medication_name"].astype(str).str.strip(),
            event_time=lambda frame: frame["charttime"],
            starttime=lambda frame: frame["scheduletime"].where(
                frame["scheduletime"].notna(),
                frame["charttime"],
            ),
            stoptime=pd.NA,
            frequency=pd.NA,
            source_home_medrecon=0,
            source_ed_pyxis=0,
            source_hospital_order=0,
            source_hospital_admin=1,
            pharmacy_enriched_flag=0,
            continued_from_home_inferred=0,
            newly_started_during_encounter_inferred=0,
            stay_id=pd.NA,
            medication_episode_id=pd.NA,
            prescription_segment_count=1,
            prescription_segments_json=pd.NA,
        )
    )


def infer_home_medication_continuity(medication_events: pd.DataFrame) -> pd.DataFrame:
    """Infer whether encounter medications look continued from home or newly started."""
    if medication_events.empty:
        return medication_events

    inferred = medication_events.copy()
    home_by_hadm = set(
        zip(
            inferred.loc[inferred["source_home_medrecon"] == 1, "subject_id"],
            inferred.loc[inferred["source_home_medrecon"] == 1, "hadm_id"],
            inferred.loc[inferred["source_home_medrecon"] == 1, "medication_normalized"],
        )
    )
    home_by_stay = set(
        zip(
            inferred.loc[inferred["source_home_medrecon"] == 1, "subject_id"],
            inferred.loc[inferred["source_home_medrecon"] == 1, "stay_id"],
            inferred.loc[inferred["source_home_medrecon"] == 1, "medication_normalized"],
        )
    )

    def _continued_from_home(row: pd.Series) -> int:
        if int(row.get("source_home_medrecon", 0)) == 1:
            return 0
        normalized_name = row.get("medication_normalized")
        if normalized_name is None or pd.isna(normalized_name):
            return 0
        hadm_key = (row.get("subject_id"), row.get("hadm_id"), normalized_name)
        stay_key = (row.get("subject_id"), row.get("stay_id"), normalized_name)
        return int(hadm_key in home_by_hadm or stay_key in home_by_stay)

    inferred["continued_from_home_inferred"] = inferred.apply(_continued_from_home, axis=1)
    inferred["newly_started_during_encounter_inferred"] = (
        (
            inferred["source_home_medrecon"] == 0
        )
        & (inferred["medication_normalized"].notna())
        & (inferred["continued_from_home_inferred"] == 0)
    ).astype(int)
    return inferred


def summarize_medication_events(dataframe: pd.DataFrame) -> list[str]:
    """Return compact summaries for the canonical medication event layer."""
    return [
        f"rows={len(dataframe):,}, columns={dataframe.shape[1]}",
        f"unique_subjects={dataframe['subject_id'].nunique():,}",
        f"home_medrecon_rows={int(dataframe['source_home_medrecon'].sum()):,}",
        f"ed_pyxis_rows={int(dataframe['source_ed_pyxis'].sum()):,}",
        f"hospital_order_rows={int(dataframe['source_hospital_order'].sum()):,}",
        f"hospital_admin_rows={int(dataframe['source_hospital_admin'].sum()):,}",
    ]


def _attach_ed_encounter_context(
    dataframe: pd.DataFrame,
    *,
    edstays: pd.DataFrame,
    encounter_index: pd.DataFrame,
) -> pd.DataFrame:
    ed_context = edstays.loc[:, ["subject_id", "stay_id", "hadm_id"]].drop_duplicates().copy()
    enriched = dataframe.merge(
        ed_context,
        how="left",
        on=["subject_id", "stay_id"],
        validate="many_to_one",
    )
    encounter_lookup = encounter_index.loc[
        :, ["subject_id", "hadm_id", "stay_id", "encounter_id", "encounter_source", "encounter_start", "encounter_end"]
    ].drop_duplicates()
    return enriched.merge(
        encounter_lookup,
        how="left",
        on=["subject_id", "hadm_id", "stay_id"],
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
        finalized[column] = pd.to_datetime(finalized[column], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    if "medication_name" in finalized:
        finalized["medication_name"] = finalized["medication_name"].replace("", pd.NA)
    return finalized.reindex(columns=MEDICATION_EVENT_COLUMNS)


def _fill_medication_event_defaults(dataframe: pd.DataFrame) -> pd.DataFrame:
    filled = dataframe.copy()
    for column in [
        "source_home_medrecon",
        "source_ed_pyxis",
        "source_hospital_order",
        "source_hospital_admin",
        "pharmacy_enriched_flag",
        "continued_from_home_inferred",
        "newly_started_during_encounter_inferred",
        "prescription_segment_count",
    ]:
        filled[column] = pd.to_numeric(filled[column], errors="coerce").fillna(0).astype(int)
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
