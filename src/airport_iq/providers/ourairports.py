"""OurAirports reference data: names, cities, regions, coordinates.

Supplies the human-facing identity of an airport (used by `resolve_airport`)
and the lat/lon we need for great-circle distance in the long-haul metric.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from .base import HttpCache, fetch_text

log = logging.getLogger(__name__)

AIRPORTS_CSV = "https://davidmegginson.github.io/ourairports-data/airports.csv"
RUNWAYS_CSV = "https://davidmegginson.github.io/ourairports-data/runways.csv"

# Airport types that can plausibly host scheduled commercial service.
COMMERCIAL_TYPES = {"large_airport", "medium_airport", "small_airport"}


def fetch_airports(
    *, cache: HttpCache | None = None, refresh: bool = False
) -> tuple[pd.DataFrame, str]:
    """US airports with an IATA code, as a (frame, source) pair."""
    result = fetch_text(AIRPORTS_CSV, cache=cache, refresh=refresh)
    df = pd.read_csv(io.StringIO(result.payload), low_memory=False)

    df = df[df["iso_country"] == "US"]
    df = df[df["type"].isin(COMMERCIAL_TYPES)]
    df = df[df["iata_code"].notna() & (df["iata_code"].astype(str).str.len() == 3)]

    df = df.rename(
        columns={
            "iata_code": "iata",
            "name": "name",
            "municipality": "city",
            "latitude_deg": "lat",
            "longitude_deg": "lon",
            "iso_region": "region_code",
            "type": "airport_type",
            "icao_code": "icao",
        }
    )
    df["iata"] = df["iata"].astype(str).str.strip().str.upper()
    # "US-CA" -> "CA"
    df["state_ourairports"] = df["region_code"].astype(str).str.split("-").str[-1]
    df["has_scheduled_service"] = df["scheduled_service"].astype(str).str.lower().eq("yes")

    # The FAA location identifier. It usually equals the IATA code but not
    # always (AZA/IWA, FCA/GPI, SCE/UNV, ...), and the FAA runway and facility
    # layers are keyed on this, not on IATA. Without it those joins silently
    # drop a few hundred airports - including some with millions of passengers.
    df["faa_id"] = (
        df["local_code"].astype(str).str.strip().str.upper().replace({"NAN": ""})
    )
    df.loc[df["faa_id"] == "", "faa_id"] = df["iata"]

    keep = [
        "iata",
        "icao",
        "faa_id",
        "name",
        "city",
        "state_ourairports",
        "lat",
        "lon",
        "airport_type",
        "has_scheduled_service",
    ]
    df = df[keep].drop_duplicates(subset="iata", keep="first").reset_index(drop=True)

    log.info("ourairports -> %d US airports with IATA codes (%s)", len(df), result.source)
    return df, result.source


def fetch_runways_fallback(
    *, cache: HttpCache | None = None, refresh: bool = False
) -> tuple[pd.DataFrame, str]:
    """Runway geometry from OurAirports, keyed on ICAO.

    Secondary source for airports the FAA NTAD layer omits - Palm Beach (PBI,
    4.1M enplanements) is absent from it entirely. This file carries no
    pavement-condition field, so airports filled from here get a null renewal
    component rather than an invented one.
    """
    result = fetch_text(RUNWAYS_CSV, cache=cache, refresh=refresh)
    df = pd.read_csv(io.StringIO(result.payload), low_memory=False)

    df = df[df["closed"].fillna(0).astype(int) == 0]
    df["length_ft"] = pd.to_numeric(df["length_ft"], errors="coerce")
    df["width_ft"] = pd.to_numeric(df["width_ft"], errors="coerce")
    df = df[df["length_ft"].notna() & (df["length_ft"] > 0)]

    surface = df["surface"].astype(str).str.upper()
    df = df.assign(_paved=surface.str.contains("ASP|CON|PEM|BIT", regex=True, na=False))

    agg = df.groupby("airport_ident").agg(
        runway_count=("le_ident", "nunique"),
        runway_len_total=("length_ft", "sum"),
        runway_len_max=("length_ft", "max"),
        runway_width_max=("width_ft", "max"),
        paved_runways=("_paved", "sum"),
    )
    agg["pct_runways_paved"] = (agg["paved_runways"] / agg["runway_count"]).fillna(0.0)
    agg = agg.drop(columns=["paved_runways"]).reset_index()
    agg = agg.rename(columns={"airport_ident": "icao"})

    log.info("ourairports runways -> %d airports (%s)", len(agg), result.source)
    return agg, result.source
