"""The agent's tool surface.

Every tool is a thin wrapper over `service`, which wraps the deterministic
scoring engine. The model chooses which tool to call and how to phrase the
result; it never computes a figure.

Gemini's `chats.create` / `send_message` supports automatic function calling
from plain typed Python functions with Google-style docstrings - the SDK
generates the schema from the signature and docstring itself, executes the
call, and feeds the result back internally within one `send_message` call.

Deliberately NOT using `from __future__ import annotations` here: the SDK's
automatic-function-calling argument conversion reads `inspect.signature(fn)`
directly rather than `typing.get_type_hints(fn)`, so a postponed-evaluation
annotation arrives as a bare string (e.g. `"str | None"`). Its converter then
does `isinstance(value, annotation)` on that string and raises `TypeError:
isinstance() arg 2 must be a type, a tuple of types, or a union` for any
explicitly-passed argument - the tool call fails, and the model is left to
paper over it in prose instead of reporting real data. Real type objects are
required here, not strings.
"""

import json
from typing import Any

from .. import service as svc
from ..ingest import load
from ..providers.routes import get_provider


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, default=str)


def resolve_airport(query: str) -> str:
    """Resolve a place name, city, colloquial name or code to a US airport.

    Call this first whenever the user names an airport in anything other than
    a three-letter IATA code. Reports how the match was made and how
    confident it is, and lists alternatives when a city has several airports.

    Args:
        query: What the user wrote, e.g. "LA", "Santa Ana", "Anchorage", "SFO".
    """
    return _json(svc.resolve(query))


def rank_airports(
    region: str | None = None,
    state: str | None = None,
    hub_class: str | None = None,
    weight_profile: str | None = None,
    top_n: int = 5,
    min_enplanements: int = 10_000,
) -> str:
    """Rank US airports by investment potential under an explicit scope.

    Returns each airport's composite score, its five component scores and the
    raw figures behind them, plus the exact scope that was applied and any
    warnings about the comparison.

    Choose weight_profile to match what the user is asking about:
      - "terminal"  terminal or gate expansion (weights passenger-per-departure strain)
      - "runway"    airfield and runway capacity
      - "renewal"   pavement rehabilitation and modernization
      - "balanced"  default when the user has not indicated a project type

    Args:
        region: Named region, e.g. "new_england", "california", "pacific_northwest".
        state: Two-letter state code, e.g. "MA".
        hub_class: One of "large_hub", "medium_hub", "small_hub", "nonhub".
        weight_profile: One of "balanced", "terminal", "runway", "renewal".
        top_n: How many airports to return.
        min_enplanements: Exclude airports below this 2024 enplanement count.
    """
    return _json(
        svc.rank(
            region=region,
            state=state,
            hub_class=hub_class,
            profile=weight_profile,
            top_n=top_n,
            min_enplanements=min_enplanements,
        )
    )


def get_airport_profile(iata: str, weight_profile: str | None = None) -> str:
    """Full scored breakdown for one airport, and the explanation of its score.

    Returns the composite score, every component ordered by how many points it
    contributed, with the weight applied and the raw published figure
    underneath it, plus the airport's national rank, the confidence band, and
    any caveats about data coverage. This is the tool for "why does X score
    what it scores" questions - the component ordering is the explanation, so
    the reasoning can be quoted rather than invented.

    Args:
        iata: Three-letter IATA code. Resolve a name to a code first if needed.
        weight_profile: One of "balanced", "terminal", "runway", "renewal".
    """
    return _json(svc.profile_airport(iata, profile=weight_profile))


def compare_airports(iata_codes: list[str], weight_profile: str | None = None) -> str:
    """Compare two or more airports side by side.

    Returns aligned metrics for each airport and warns when they sit in
    different FAA hub classes, which makes absolute traffic figures
    misleading. Accepts names as well as codes - they are resolved automatically.

    Args:
        iata_codes: Airports to compare, e.g. ["LAX", "SNA"].
        weight_profile: One of "balanced", "terminal", "runway", "renewal".
    """
    return _json(svc.compare(iata_codes, profile=weight_profile))


def get_long_haul_share(iata: str) -> str:
    """Share of departures from an airport that are long-haul (2,500+ miles).

    Important: with no OpenSky credentials configured this returns
    long_haul_share_available: false together with a runway-capability proxy.
    When that happens, say clearly that the actual percentage is not available
    from the public data in use, report the capability instead, and do not
    estimate a number.

    Args:
        iata: Three-letter IATA code.
    """
    provider = get_provider(load())
    return _json(provider.long_haul_share(iata).to_dict())


def list_scoring_options() -> str:
    """List the available weight profiles and named regions.

    Use this when the user asks what the agent can filter or weight by, or
    when a region name was not recognised.
    """
    return _json(
        {
            "weight_profiles": svc.available_profiles(),
            "regions": svc.available_regions(),
            "hub_classes": ["large_hub", "medium_hub", "small_hub", "nonhub"],
        }
    )


# Passed directly to GenerateContentConfig(tools=TOOLS) - the SDK derives each
# tool's schema from the function's type hints and Google-style docstring.
TOOLS = [
    resolve_airport,
    rank_airports,
    get_airport_profile,
    compare_airports,
    get_long_haul_share,
    list_scoring_options,
]
