"""BTS / FAA data published as ArcGIS FeatureServer layers.

Three layers, all key-free and all verified live:

  T-100        annual passengers / departures / arrivals per US airport (2024)
  Runways      per-runway length, width, surface and pavement condition
  Facilities   FAA airport master record: acreage, ownership, location

FeatureServer caps a response at `maxRecordCount` (2000 here), so every query
pages with `resultOffset` until the server stops setting `exceededTransferLimit`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .base import HttpCache, fetch_json

log = logging.getLogger(__name__)

ARCGIS_ROOT = "https://services.arcgis.com/xOi1kZaI0eWDREZv/ArcGIS/rest/services"

T100_URL = f"{ARCGIS_ROOT}/T100_Domestic_Market_and_Segment_Data/FeatureServer/1/query"
RUNWAYS_URL = f"{ARCGIS_ROOT}/Runways_View/FeatureServer/0/query"
FACILITIES_URL = f"{ARCGIS_ROOT}/NTAD_Aviation_Facilities/FeatureServer/0/query"

PAGE_SIZE = 2000

# FAA pavement condition codes, mapped to how much renewal work each implies.
# Anything not listed here (including a blank code, which appears on ~1,700 of
# the ~8,800 published runways) is treated as unknown rather than as bad.
CONDITION_SEVERITY = {
    "EXCELLENT": 0.0,
    "GOOD": 0.0,
    "FAIR": 0.5,
    "POOR": 0.9,
    "FAILED": 1.0,
}


@dataclass
class LayerResult:
    frame: pd.DataFrame
    source: str  # "live" | "cache" | mixed pages report the weakest source


def query_layer(
    url: str,
    out_fields: str,
    *,
    where: str = "1=1",
    cache: HttpCache | None = None,
    refresh: bool = False,
    order_by: str | None = None,
) -> LayerResult:
    """Page through a FeatureServer layer and return every matching row."""
    cache = cache or HttpCache()
    rows: list[dict[str, Any]] = []
    sources: set[str] = set()
    offset = 0

    # ArcGIS pagination is only stable under an explicit sort. Without one the
    # server may repeat or drop rows between pages.
    order = order_by or "OBJECTID"

    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "returnGeometry": "false",
            "orderByFields": order,
            "resultOffset": offset,
            "resultRecordCount": PAGE_SIZE,
            "f": "json",
        }
        result = fetch_json(url, params, cache=cache, refresh=refresh)
        sources.add(result.source)
        payload = result.payload

        if "error" in payload:
            raise RuntimeError(f"ArcGIS error from {url}: {payload['error']}")

        features = payload.get("features", [])
        rows.extend(f["attributes"] for f in features)

        if not payload.get("exceededTransferLimit") or not features:
            break
        offset += len(features)

    source = "live" if sources == {"live"} else ("cache" if "cache" in sources else "live")
    log.info("%s -> %d rows (%s)", url.rsplit("/", 3)[1], len(rows), source)
    return LayerResult(frame=pd.DataFrame(rows), source=source)


def fetch_t100(*, cache: HttpCache | None = None, refresh: bool = False) -> LayerResult:
    """Annual traffic per airport. One row per (origin, year)."""
    result = query_layer(
        T100_URL,
        "origin,year,enplanements,passengers,departures,arrivals,freight,mail",
        cache=cache,
        refresh=refresh,
    )
    df = result.frame
    if df.empty:
        return result

    df = df.rename(columns={"origin": "iata"})
    df["iata"] = df["iata"].astype(str).str.strip().str.upper()

    numeric = ["enplanements", "passengers", "departures", "arrivals", "freight", "mail"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep the most recent year per airport in case the layer ever gains history.
    df = df.sort_values("year").groupby("iata", as_index=False).last()
    return LayerResult(frame=df, source=result.source)


def fetch_runways(*, cache: HttpCache | None = None, refresh: bool = False) -> LayerResult:
    """Per-runway physical characteristics, aggregated to one row per airport."""
    result = query_layer(
        RUNWAYS_URL,
        "ARPT_ID,RWY_ID,RWY_LEN,RWY_WIDTH,SURFACE_TYPE_CODE,COND",
        cache=cache,
        refresh=refresh,
    )
    df = result.frame
    if df.empty:
        return result

    df["ARPT_ID"] = df["ARPT_ID"].astype(str).str.strip().str.upper()
    df["RWY_LEN"] = pd.to_numeric(df["RWY_LEN"], errors="coerce")
    df["RWY_WIDTH"] = pd.to_numeric(df["RWY_WIDTH"], errors="coerce")
    df["COND"] = df["COND"].astype(str).str.strip().str.upper()
    df["SURFACE_TYPE_CODE"] = df["SURFACE_TYPE_CODE"].astype(str).str.strip().str.upper()

    # Drop runways with no usable length - they cannot inform capacity.
    df = df[df["RWY_LEN"].notna() & (df["RWY_LEN"] > 0)]

    # A paved surface is a prerequisite for commercial service; treat anything
    # containing ASPH or CONC as paved.
    paved = df["SURFACE_TYPE_CODE"].str.contains("ASPH|CONC", regex=True, na=False)

    # Severity of pavement work implied by each published condition code.
    # EXCELLENT and GOOD both mean no near-term work; treating GOOD as the only
    # acceptable value would rank the best-maintained airports as the most in
    # need of renewal. A blank code means unknown, NOT bad - it maps to NaN so
    # the airport drops out of the renewal component instead of being charged
    # for a gap in FAA reporting.
    df = df.assign(
        _paved=paved,
        _severity=df["COND"].map(CONDITION_SEVERITY),
    )
    known = df[df["_severity"].notna()].copy()
    known["_weighted_len"] = known["RWY_LEN"] * known["_severity"]

    agg = df.groupby("ARPT_ID").agg(
        runway_count=("RWY_ID", "nunique"),
        runway_len_total=("RWY_LEN", "sum"),
        runway_len_max=("RWY_LEN", "max"),
        runway_width_max=("RWY_WIDTH", "max"),
        paved_runways=("_paved", "sum"),
    )

    # Renewal need is the length-weighted mean severity across runways whose
    # condition is actually published.
    cond = known.groupby("ARPT_ID").agg(
        _weighted=("_weighted_len", "sum"),
        _len_known=("RWY_LEN", "sum"),
    )
    agg = agg.join(cond)
    agg["pct_runway_len_not_good"] = agg["_weighted"] / agg["_len_known"]
    agg["pct_runway_len_cond_known"] = (agg["_len_known"] / agg["runway_len_total"]).fillna(0.0)

    agg["pct_runways_paved"] = (agg["paved_runways"] / agg["runway_count"]).fillna(0.0)
    agg = agg.drop(columns=["paved_runways", "_weighted", "_len_known"]).reset_index()
    agg = agg.rename(columns={"ARPT_ID": "iata"})

    return LayerResult(frame=agg, source=result.source)


def fetch_facilities(*, cache: HttpCache | None = None, refresh: bool = False) -> LayerResult:
    """FAA airport master record. Filtered to public-use airports."""
    result = query_layer(
        FACILITIES_URL,
        (
            "ARPT_ID,ARPT_NAME,CITY,STATE_CODE,STATE_NAME,COUNTY_NAME,"
            "OWNERSHIP_TYPE_CODE,FACILITY_USE_CODE,SITE_TYPE_CODE,ACREAGE,"
            "LAT_DECIMAL,LONG_DECIMAL,TWR_TYPE_CODE,FAR_139_TYPE_CODE,"
            "DIST_CITY_TO_AIRPORT"
        ),
        where="SITE_TYPE_CODE='A' AND FACILITY_USE_CODE='PU'",
        cache=cache,
        refresh=refresh,
    )
    df = result.frame
    if df.empty:
        return result

    df["ARPT_ID"] = df["ARPT_ID"].astype(str).str.strip().str.upper()
    df["ACREAGE"] = pd.to_numeric(df["ACREAGE"], errors="coerce")
    for col in ("LAT_DECIMAL", "LONG_DECIMAL", "DIST_CITY_TO_AIRPORT"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.rename(
        columns={
            "ARPT_ID": "iata",
            "ARPT_NAME": "faa_name",
            "CITY": "faa_city",
            "STATE_CODE": "state",
            "STATE_NAME": "state_name",
            "COUNTY_NAME": "county",
            "OWNERSHIP_TYPE_CODE": "ownership",
            "ACREAGE": "acreage",
            "LAT_DECIMAL": "faa_lat",
            "LONG_DECIMAL": "faa_lon",
            "TWR_TYPE_CODE": "tower_type",
            "FAR_139_TYPE_CODE": "far139",
        }
    )
    df = df.drop(columns=["SITE_TYPE_CODE", "FACILITY_USE_CODE"], errors="ignore")

    # One FAA record per identifier; keep the largest by acreage if duplicated.
    df = df.sort_values("acreage").groupby("iata", as_index=False).last()
    return LayerResult(frame=df, source=result.source)
