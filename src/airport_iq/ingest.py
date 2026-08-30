"""Build the unified airport table the scoring engine reads.

Joins four public sources on the airport identifier and records, per airport,
which of them actually contributed. That `data_completeness` figure is what
lets the agent say how much to trust a given score.

Run:  uv run python -m airport_iq.ingest [--refresh]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .providers.arcgis import fetch_facilities, fetch_runways, fetch_t100
from .providers.base import HttpCache
from .providers.ourairports import fetch_airports, fetch_runways_fallback

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AIRPORTS_PARQUET = DATA_DIR / "airports.parquet"

# Identifier drift between data vintages.
#
# T-100 reports 2024 traffic under the code in use that year; the FAA and
# OurAirports reference files track present-day codes. Where an airport has
# been recoded in between, the join silently loses it - so the handful of
# known cases are reconciled explicitly rather than left to fall through.
#
#   PBI -> DJT  Palm Beach Intl renamed to President Donald J Trump Intl.
#               4.1M enplanements; absent from every FAA layer under "PBI".
IATA_ALIASES = {"PBI": "DJT"}

# Fields that must be present for an airport to be scorable at all.
REQUIRED_FIELDS = ["passengers", "departures", "arrivals", "runway_count"]

# Fields that contribute to a component but can be absent without disqualifying
# the airport - each missing one lowers `data_completeness`.
OPTIONAL_FIELDS = ["acreage", "runway_len_max", "pct_runway_len_not_good", "ownership"]


@dataclass
class IngestReport:
    rows: int
    sources: dict[str, str]
    dropped_no_traffic: int
    dropped_no_runway: int


def build(*, refresh: bool = False) -> tuple[pd.DataFrame, IngestReport]:
    cache = HttpCache()

    t100 = fetch_t100(cache=cache, refresh=refresh)
    runways = fetch_runways(cache=cache, refresh=refresh)
    facilities = fetch_facilities(cache=cache, refresh=refresh)
    reference, ref_source = fetch_airports(cache=cache, refresh=refresh)

    # T-100 is the spine: an airport with no reported traffic cannot be scored
    # on demand pressure, which is the heart of the model.
    df = t100.frame.copy()
    before = len(df)
    df = df[(df["departures"].fillna(0) > 0) & (df["passengers"].fillna(0) > 0)]
    dropped_no_traffic = before - len(df)

    df["reported_iata"] = df["iata"]
    df["iata"] = df["iata"].replace(IATA_ALIASES)

    # T-100 is keyed on IATA; the two FAA layers are keyed on the FAA location
    # identifier. Bring in the reference table first so we have `faa_id` to
    # join them on, defaulting to the IATA code where OurAirports has no entry.
    df = df.merge(reference, on="iata", how="left")
    df["faa_id"] = df["faa_id"].fillna(df["iata"])

    faa_runways = runways.frame.rename(columns={"iata": "faa_id"})
    faa_facilities = facilities.frame.rename(columns={"iata": "faa_id"})
    df = df.merge(faa_runways, on="faa_id", how="left")
    df = df.merge(faa_facilities, on="faa_id", how="left")

    df["runway_source"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["runway_count"].notna(), "runway_source"] = "faa_ntad"
    df = _backfill_runways(df, cache=cache, refresh=refresh)

    before = len(df)
    df = df[df["runway_count"].notna() & (df["runway_count"] > 0)]
    dropped_no_runway = before - len(df)

    # Prefer OurAirports for the public-facing name/city (better formatted),
    # fall back to the FAA record where OurAirports has no entry.
    df["name"] = df["name"].fillna(df["faa_name"])
    df["city"] = df["city"].fillna(df["faa_city"])
    df["state"] = df["state"].fillna(df["state_ourairports"])
    df["lat"] = df["lat"].fillna(df["faa_lat"])
    df["lon"] = df["lon"].fillna(df["faa_lon"])
    df = df.drop(
        columns=["faa_name", "faa_city", "state_ourairports", "faa_lat", "faa_lon"],
        errors="ignore",
    )

    df["data_completeness"] = _completeness(df)

    df = df.sort_values("enplanements", ascending=False).reset_index(drop=True)

    report = IngestReport(
        rows=len(df),
        sources={
            "t100": t100.source,
            "runways": runways.source,
            "facilities": facilities.source,
            "ourairports": ref_source,
        },
        dropped_no_traffic=dropped_no_traffic,
        dropped_no_runway=dropped_no_runway,
    )
    return df, report


def _backfill_runways(
    df: pd.DataFrame, *, cache: HttpCache, refresh: bool
) -> pd.DataFrame:
    """Fill runway geometry from OurAirports where the FAA layer has no record.

    Only geometry is backfilled. `pct_runway_len_not_good` stays null for these
    airports because OurAirports publishes no pavement condition - the renewal
    component is then dropped and its weight redistributed, rather than being
    silently treated as "pavement is fine".
    """
    gaps = df["runway_count"].isna()
    if not gaps.any():
        return df

    fallback, _ = fetch_runways_fallback(cache=cache, refresh=refresh)
    fallback = fallback.set_index("icao")

    geometry = ["runway_count", "runway_len_total", "runway_len_max",
                "runway_width_max", "pct_runways_paved"]
    icao = df.loc[gaps, "icao"]
    matched = icao[icao.isin(fallback.index)]

    for col in geometry:
        df.loc[matched.index, col] = fallback.loc[matched.values, col].to_numpy()
    df.loc[matched.index, "runway_source"] = "ourairports"

    log.info("backfilled runway geometry for %d airports from OurAirports", len(matched))
    return df


def _completeness(df: pd.DataFrame) -> pd.Series:
    """Fraction of scoring inputs present, weighting required fields double."""
    required = sum(df[c].notna().astype(int) * 2 for c in REQUIRED_FIELDS)
    optional = sum(df[c].notna().astype(int) for c in OPTIONAL_FIELDS if c in df)
    total = len(REQUIRED_FIELDS) * 2 + len(OPTIONAL_FIELDS)
    return ((required + optional) / total).round(3)


def save(df: pd.DataFrame, path: Path = AIRPORTS_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load(path: Path = AIRPORTS_PARQUET) -> pd.DataFrame:
    """Read the built table, building it on first use if absent."""
    if not path.exists():
        log.info("no airports.parquet yet - building from cache/network")
        df, _ = build()
        save(df, path)
        return df
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unified airport table.")
    parser.add_argument(
        "--refresh", action="store_true", help="bypass the HTTP cache and refetch"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df, report = build(refresh=args.refresh)
    path = save(df)

    print(f"\nwrote {report.rows} airports -> {path}")
    print(f"  sources:            {report.sources}")
    print(f"  dropped (no traffic): {report.dropped_no_traffic}")
    print(f"  dropped (no runway):  {report.dropped_no_runway}")
    print(f"  mean completeness:  {df['data_completeness'].mean():.3f}")


if __name__ == "__main__":
    main()
