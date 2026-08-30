"""Turning a score into a defensible explanation.

The agent is not allowed to invent reasoning about why an airport ranks where
it does. It calls this, which reports each component's score, the weight
applied to it, the raw published figure underneath, and what that figure means.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .model import COMPONENTS, load_config

# What each component measures, in one line, for an analyst who has not read
# the methodology doc.
DEFINITIONS = {
    "demand_pressure": "Runway-system saturation: annual operations per runway.",
    "terminal_strain": (
        "Gate and terminal saturation, measured through average aircraft gauge "
        "(passengers per departure). Rises when an airport cannot add flights "
        "and airlines respond with larger aircraft."
    ),
    "unmet_demand": (
        "Demand pressure amplified by how binding the physical constraint is. "
        "An airport with little spare land carries its pressure at full weight; "
        "one with room to expand is damped."
    ),
    "renewal_need": (
        "Share of runway length not in sound condition, length-weighted by "
        "severity of the FAA's published pavement condition code."
    ),
    "expansion_feasibility": (
        "Whether a project can physically and financially happen: land per unit "
        "of traffic, public ownership (AIP grant eligibility), and whether the "
        "longest runway supports widebody operations."
    ),
}

RAW_FIELDS = {
    "demand_pressure": ("ops_per_runway", "annual operations per runway"),
    "terminal_strain": ("pax_per_departure", "passengers per departure"),
    "unmet_demand": (None, None),
    "renewal_need": ("pct_runway_len_not_good", "severity-weighted share of runway length"),
    "expansion_feasibility": ("acres_per_100k_ops", "acres per 100k annual operations"),
}


def _clean(value: Any) -> Any:
    """Make a value JSON-safe (NaN -> None, numpy scalar -> Python scalar)."""
    if value is None:
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 4)
    return value


def explain(scored: pd.DataFrame, iata: str) -> dict[str, Any]:
    """Full breakdown for one airport, ready to hand to the model verbatim."""
    match = scored[scored["iata"].str.upper() == iata.upper()]
    if match.empty:
        raise KeyError(f"{iata!r} is not in the scored dataset")
    row = match.iloc[0]

    profile = row["profile_used"]
    weights = load_config().weights(profile)

    total_airports = len(scored)
    components: list[dict[str, Any]] = []

    for name in COMPONENTS:
        value = row[name]
        available = pd.notna(value)
        raw_field, raw_label = RAW_FIELDS[name]

        entry: dict[str, Any] = {
            "component": name,
            "definition": DEFINITIONS[name],
            "score_0_100": _clean(value) if available else None,
            "weight_in_profile": weights[name],
            "points_contributed": _clean(row.get(f"contrib_{name}")),
            "available": bool(available),
        }
        if raw_field and raw_field in row.index:
            entry["raw_value"] = _clean(row[raw_field])
            entry["raw_units"] = raw_label
        if not available:
            entry["why_unavailable"] = (
                "The FAA does not publish the underlying figure for this airport. "
                "Its weight was redistributed across the remaining components "
                "rather than being scored as zero."
            )
        components.append(entry)

    components.sort(key=lambda c: c["points_contributed"] or 0.0, reverse=True)

    notes: list[str] = []
    if row.get("runway_source") == "ourairports":
        notes.append(
            "Runway geometry came from OurAirports because the FAA NTAD layer "
            "has no record for this airport; pavement condition is unavailable, "
            "so the renewal component was dropped."
        )
    if pd.notna(row.get("pct_runway_len_cond_known")) and row["pct_runway_len_cond_known"] < 1.0:
        notes.append(
            f"Pavement condition is published for only "
            f"{row['pct_runway_len_cond_known']:.0%} of runway length; the renewal "
            "score reflects that portion only."
        )
    if row.get("reported_iata") and row["reported_iata"] != row["iata"]:
        notes.append(
            f"BTS reports 2024 traffic for this airport under the older code "
            f"{row['reported_iata']}; it has since been recoded to {row['iata']}."
        )

    return {
        "iata": row["iata"],
        "name": row["name"],
        "city": _clean(row.get("city")),
        "state": _clean(row.get("state")),
        "hub_class": row["hub_class"],
        "rank_overall": int(row["rank"]),
        "of_airports": total_airports,
        "investment_score": _clean(row["investment_score"]),
        "weight_profile": profile,
        "normalized_within": row["normalized_within"],
        "components": components,
        "raw_kpis": {
            "enplanements_2024": _clean(row.get("enplanements")),
            "passengers_2024": _clean(row.get("passengers")),
            "departures_2024": _clean(row.get("departures")),
            "annual_operations": _clean(row.get("annual_operations")),
            "runway_count": _clean(row.get("runway_count")),
            "longest_runway_ft": _clean(row.get("runway_len_max")),
            "acreage": _clean(row.get("acreage")),
            "ops_per_runway": _clean(row.get("ops_per_runway")),
            "pax_per_departure": _clean(row.get("pax_per_departure")),
            "long_haul_capable_runway": _clean(row.get("long_haul_capable")),
            "publicly_owned": _clean(row.get("publicly_owned")),
        },
        "confidence": {
            "band": row.get("confidence_band"),
            "score": _clean(row.get("confidence")),
            "data_completeness": _clean(row.get("data_completeness")),
            "profile_weight_applied": _clean(row.get("weight_coverage")),
            "notes": notes,
        },
        "data_vintage": {
            "traffic": "BTS T-100, calendar year 2024",
            "runways_and_land": "FAA NTAD (current publication)",
            "caveat": (
                "Traffic is a single year, so this measures current pressure, "
                "not a growth trend or a forecast."
            ),
        },
    }
