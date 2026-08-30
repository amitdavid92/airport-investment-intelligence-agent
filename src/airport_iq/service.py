"""Query façade over the scoring engine.

Both the HTTP API and the agent's tools call this and nothing lower. Keeping
one entry point is what guarantees `/rank` and a chat answer produce the same
numbers - the LLM path has no privileged access and no separate code path.
"""

from __future__ import annotations

import functools
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

from .ingest import load
from .scoring.explain import _clean, explain
from .scoring.model import COMPONENTS, load_config, load_regions, score

# Airports below this see so little traffic that including them in rankings
# adds noise rather than candidates. Overridable per query.
DEFAULT_MIN_ENPLANEMENTS = 10_000

# Colloquial names analysts actually use. Resolution tries these before fuzzy
# matching so "LA" cannot land on "Lake Charles".
COLLOQUIAL = {
    "la": "LAX",
    "los angeles": "LAX",
    "santa ana": "SNA",
    "orange county": "SNA",
    "john wayne": "SNA",
    "nyc": "JFK",
    "new york": "JFK",
    "sf": "SFO",
    "san francisco": "SFO",
    "bay area": "SFO",
    "chicago": "ORD",
    "ohare": "ORD",
    "o'hare": "ORD",
    "midway": "MDW",
    "dc": "DCA",
    "washington": "DCA",
    "reagan": "DCA",
    "dulles": "IAD",
    "boston": "BOS",
    "logan": "BOS",
    "anchorage": "ANC",
    "atlanta": "ATL",
    "denver": "DEN",
    "seattle": "SEA",
    "miami": "MIA",
    "newark": "EWR",
    "palm beach": "DJT",
    "west palm beach": "DJT",
    "sea-tac": "SEA",
    "seatac": "SEA",
    "philly": "PHL",
    "ft lauderdale": "FLL",
    "ft. lauderdale": "FLL",
    "fort lauderdale": "FLL",
    # An exact city match sends "Dallas" to Love Field, but an analyst asking
    # about Dallas almost always means the primary hub.
    "dallas": "DFW",
    "dallas fort worth": "DFW",
}


@functools.lru_cache(maxsize=8)
def scored_table(profile: str | None = None, within_hub_class: bool = False) -> pd.DataFrame:
    """Scored dataset, cached per (profile, normalization) pair."""
    return score(load(), profile=profile, within_hub_class=within_hub_class)


def clear_cache() -> None:
    scored_table.cache_clear()


def resolve(query: str) -> dict[str, Any]:
    """Map free text to an IATA code, reporting how confident the match is."""
    table = scored_table()
    text = (query or "").strip()
    if not text:
        return {"matched": False, "reason": "empty query"}

    lowered = text.lower()

    # 1. Exact IATA code.
    if len(text) == 3:
        hit = table[table["iata"].str.upper() == text.upper()]
        if not hit.empty:
            return _match(hit.iloc[0], "exact IATA code", 1.0)

    # 2. Known colloquial name.
    if lowered in COLLOQUIAL:
        hit = table[table["iata"] == COLLOQUIAL[lowered]]
        if not hit.empty:
            return _match(hit.iloc[0], f"known colloquial name for {COLLOQUIAL[lowered]}", 0.95)

    # 3. Exact city match.
    city = table[table["city"].fillna("").str.lower() == lowered]
    if len(city) == 1:
        return _match(city.iloc[0], "exact city match", 0.9)
    if len(city) > 1:
        biggest = city.sort_values("enplanements", ascending=False)
        return _match(
            biggest.iloc[0],
            f"city has {len(city)} airports; chose the busiest",
            0.7,
            alternatives=_brief(biggest.iloc[1:6]),
        )

    # 4. Fuzzy match against name and city.
    #
    # Generic words are stripped from both sides first. Almost every record
    # contains "airport", and leaving those in lets an arbitrary phrase score a
    # respectable similarity against any airport in the country.
    needle = _strip_generic(lowered)
    if not needle:
        return {"matched": False, "query": text, "reason": "no distinguishing words in query"}

    haystack = (table["name"].fillna("") + " " + table["city"].fillna("")).str.lower()
    stripped = haystack.map(_strip_generic)

    ratios = stripped.map(lambda s: SequenceMatcher(None, needle, s).ratio())
    # A whole-word containment is much stronger evidence than character overlap.
    contains = stripped.map(lambda s: _contains_word(s, needle))
    scores = ratios + contains.astype(float) * 0.4

    best_idx = scores.idxmax()
    best = float(scores[best_idx])
    if best < 0.62:
        return {"matched": False, "query": text, "reason": "no airport matched with confidence"}

    ordered = table.loc[scores.sort_values(ascending=False).index[:6]]
    return _match(
        table.loc[best_idx],
        "fuzzy name match",
        round(min(best, 0.85), 2),
        alternatives=_brief(ordered.iloc[1:5]),
    )


# Words that appear in so many airport names they carry no identifying signal.
GENERIC_TOKENS = {
    "airport", "airports", "international", "intl", "regional", "municipal",
    "field", "airfield", "national", "county", "metropolitan", "metro",
    "the", "of", "and", "at", "in", "a", "an",
}


def _strip_generic(text: str) -> str:
    tokens = [t.strip(".,'-/()") for t in text.lower().split()]
    kept = [t for t in tokens if t and t not in GENERIC_TOKENS]
    return " ".join(kept)


def _contains_word(haystack: str, needle: str) -> bool:
    """True when every word of `needle` appears in `haystack`."""
    words = set(haystack.split())
    return all(word in words for word in needle.split())


def _match(row: pd.Series, how: str, confidence: float, alternatives: list | None = None) -> dict:
    result = {
        "matched": True,
        "iata": row["iata"],
        "name": row["name"],
        "city": _clean(row.get("city")),
        "state": _clean(row.get("state")),
        "hub_class": row["hub_class"],
        "match_method": how,
        "match_confidence": confidence,
    }
    if alternatives:
        result["other_candidates"] = alternatives
    return result


def _brief(frame: pd.DataFrame) -> list[dict]:
    return [
        {
            "iata": r["iata"],
            "name": r["name"],
            "city": _clean(r.get("city")),
            "enplanements": _clean(r.get("enplanements")),
        }
        for _, r in frame.iterrows()
    ]


def rank(
    *,
    region: str | None = None,
    state: str | None = None,
    hub_class: str | None = None,
    profile: str | None = None,
    top_n: int = 5,
    min_enplanements: int = DEFAULT_MIN_ENPLANEMENTS,
    within_hub_class: bool = False,
) -> dict[str, Any]:
    """Rank airports under a filter, returning the applied scope explicitly."""
    table = scored_table(profile, within_hub_class)
    filtered = table
    applied: dict[str, Any] = {}
    warnings: list[str] = []

    if region:
        regions = load_regions()
        key = region.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in regions:
            return {
                "error": f"unknown region {region!r}",
                "available_regions": sorted(regions),
            }
        states = regions[key]["states"]
        filtered = filtered[filtered["state"].isin(states)]
        applied["region"] = regions[key]["label"]
        applied["region_states"] = states

    if state:
        filtered = filtered[filtered["state"].str.upper() == state.strip().upper()]
        applied["state"] = state.strip().upper()

    if hub_class:
        filtered = filtered[filtered["hub_class"] == hub_class.strip().lower()]
        applied["hub_class"] = hub_class.strip().lower()

    before = len(filtered)
    filtered = filtered[filtered["enplanements"].fillna(0) >= min_enplanements]
    applied["min_enplanements"] = min_enplanements
    excluded = before - len(filtered)
    if excluded:
        warnings.append(
            f"{excluded} airports in scope were excluded for having fewer than "
            f"{min_enplanements:,} enplanements in 2024."
        )

    if filtered.empty:
        return {
            "results": [],
            "scope_applied": applied,
            "warnings": warnings + ["No airports matched this scope."],
        }

    top = filtered.head(top_n)
    results = []
    for position, (_, row) in enumerate(top.iterrows(), start=1):
        entry = {
            "position_in_result": position,
            "iata": row["iata"],
            "name": row["name"],
            "city": _clean(row.get("city")),
            "state": _clean(row.get("state")),
            "hub_class": row["hub_class"],
            "investment_score": _clean(row["investment_score"]),
            "national_rank": int(row["rank"]),
            "confidence": row.get("confidence_band"),
            "components": {c: _clean(row[c]) for c in COMPONENTS},
            "key_figures": {
                "enplanements_2024": _clean(row.get("enplanements")),
                "ops_per_runway": _clean(row.get("ops_per_runway")),
                "pax_per_departure": _clean(row.get("pax_per_departure")),
                "runway_count": _clean(row.get("runway_count")),
                "acreage": _clean(row.get("acreage")),
            },
        }
        results.append(entry)

    classes = top["hub_class"].unique().tolist()
    if len(classes) > 1:
        warnings.append(
            "This ranking mixes FAA hub classes "
            f"({', '.join(sorted(classes))}). Scores are percentile-ranked "
            "nationally, so a small airport scoring highly is strained relative "
            "to all US airports, not only its size peers."
        )

    return {
        "results": results,
        "scope_applied": applied,
        "airports_in_scope": len(filtered),
        "weight_profile": top.iloc[0]["profile_used"],
        "normalized_within": top.iloc[0]["normalized_within"],
        "warnings": warnings,
    }


def profile_airport(iata: str, *, profile: str | None = None) -> dict[str, Any]:
    """Full scored breakdown for one airport."""
    table = scored_table(profile)
    try:
        return explain(table, iata)
    except KeyError:
        resolved = resolve(iata)
        if resolved.get("matched"):
            return explain(table, resolved["iata"])
        return {"error": f"no airport found for {iata!r}"}


def compare(
    codes: list[str], *, profile: str | None = None, metrics: list[str] | None = None
) -> dict[str, Any]:
    """Side-by-side comparison, flagging when it is not like-for-like."""
    table = scored_table(profile)
    resolved, unresolved = [], []
    for code in codes:
        hit = resolve(code)
        if hit.get("matched"):
            resolved.append(hit["iata"])
        else:
            unresolved.append(code)

    if len(resolved) < 2:
        return {
            "error": "need at least two resolvable airports to compare",
            "unresolved": unresolved,
        }

    rows = table[table["iata"].isin(resolved)]
    metrics = metrics or [
        "investment_score",
        "demand_pressure",
        "terminal_strain",
        "unmet_demand",
        "ops_per_runway",
        "pax_per_departure",
        "runway_count",
        "annual_operations",
        "enplanements",
        "acreage",
    ]

    comparison = []
    for _, row in rows.iterrows():
        comparison.append(
            {
                "iata": row["iata"],
                "name": row["name"],
                "hub_class": row["hub_class"],
                "national_rank": int(row["rank"]),
                "confidence": row.get("confidence_band"),
                **{m: _clean(row.get(m)) for m in metrics},
            }
        )
    comparison.sort(key=lambda r: r.get("investment_score") or 0, reverse=True)

    warnings = []
    classes = {r["hub_class"] for r in comparison}
    if len(classes) > 1:
        warnings.append(
            "These airports are in different FAA hub classes "
            f"({', '.join(sorted(classes))}). They serve different market sizes, "
            "so absolute traffic figures are not like-for-like; the percentile "
            "components are the fairer comparison."
        )
    if unresolved:
        warnings.append(f"Could not resolve: {', '.join(unresolved)}")

    return {
        "comparison": comparison,
        "metrics_returned": metrics,
        "weight_profile": rows.iloc[0]["profile_used"],
        "warnings": warnings,
    }


def available_profiles() -> list[str]:
    return sorted(load_config().profiles)


def available_regions() -> dict[str, str]:
    return {key: value["label"] for key, value in load_regions().items()}
