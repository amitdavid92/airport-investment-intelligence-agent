"""Tests for the query façade and the HTTP API.

The service layer is what both the API and the agent's tools call, so these
tests cover the behaviour an analyst actually depends on: correct resolution of
colloquial names, honest scoping, and warnings that fire when a comparison is
not like-for-like.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from airport_iq import service as svc
from airport_iq.api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------
# Resolution - load-bearing for the brief's questions.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("SFO", "SFO"),
        ("sfo", "SFO"),
        ("LA", "LAX"),
        ("Santa Ana", "SNA"),
        ("Orange County", "SNA"),
        ("Anchorage", "ANC"),
        ("Boston", "BOS"),
        ("Newark", "EWR"),
    ],
)
def test_resolve_known_names(query, expected):
    result = svc.resolve(query)
    assert result["matched"], query
    assert result["iata"] == expected, query


def test_resolve_reports_method_and_confidence():
    result = svc.resolve("LA")
    assert result["match_confidence"] > 0
    assert "colloquial" in result["match_method"]


def test_resolve_rejects_nonsense():
    assert not svc.resolve("zzzzzzzz not an airport")["matched"]


def test_resolve_empty_query():
    assert not svc.resolve("")["matched"]


# --------------------------------------------------------------------------
# Ranking scope must be explicit and honest.
# --------------------------------------------------------------------------


def test_rank_region_filters_to_correct_states():
    result = svc.rank(region="new_england", top_n=10)
    new_england = {"CT", "ME", "MA", "NH", "RI", "VT"}
    assert result["scope_applied"]["region"] == "New England"
    assert all(r["state"] in new_england for r in result["results"])


def test_rank_reports_scope_it_applied():
    result = svc.rank(region="new_england", profile="terminal", top_n=3)
    assert result["scope_applied"]["region_states"]
    assert result["weight_profile"] == "terminal"
    assert "min_enplanements" in result["scope_applied"]


def test_rank_warns_when_mixing_hub_classes():
    result = svc.rank(region="new_england", profile="terminal", top_n=5)
    classes = {r["hub_class"] for r in result["results"]}
    if len(classes) > 1:
        assert any("hub class" in w for w in result["warnings"])


def test_rank_unknown_region_lists_valid_options():
    result = svc.rank(region="atlantis")
    assert "error" in result
    assert "new_england" in result["available_regions"]


def test_rank_respects_top_n():
    assert len(svc.rank(top_n=3)["results"]) == 3


def test_rank_state_filter():
    result = svc.rank(state="MA", top_n=5)
    assert all(r["state"] == "MA" for r in result["results"])


def test_boston_leads_new_england_terminal_ranking():
    """Logan is the obvious constrained New England candidate - a sanity anchor."""
    result = svc.rank(region="new_england", profile="terminal", top_n=3)
    assert result["results"][0]["iata"] == "BOS"


# --------------------------------------------------------------------------
# Comparison.
# --------------------------------------------------------------------------


def test_compare_resolves_colloquial_names():
    result = svc.compare(["LA", "Santa Ana"])
    codes = {r["iata"] for r in result["comparison"]}
    assert codes == {"LAX", "SNA"}


def test_compare_warns_across_hub_classes():
    result = svc.compare(["LAX", "SNA"])
    assert any("hub class" in w for w in result["warnings"])


def test_compare_needs_two_airports():
    assert "error" in svc.compare(["LAX"])


def test_lax_is_more_congested_than_sna():
    """The brief asks this directly; the answer must follow from the figures."""
    result = svc.compare(["LAX", "SNA"])
    by_code = {r["iata"]: r for r in result["comparison"]}
    assert by_code["LAX"]["ops_per_runway"] > by_code["SNA"]["ops_per_runway"]


# --------------------------------------------------------------------------
# Single-airport profile.
# --------------------------------------------------------------------------


def test_profile_includes_components_and_confidence():
    profile = svc.profile_airport("SFO")
    assert profile["iata"] == "SFO"
    assert len(profile["components"]) == 5
    assert profile["confidence"]["band"] in {"low", "medium", "high"}
    assert profile["raw_kpis"]["ops_per_runway"] > 0


def test_profile_components_sorted_by_contribution():
    components = svc.profile_airport("SFO")["components"]
    contributions = [c["points_contributed"] or 0 for c in components]
    assert contributions == sorted(contributions, reverse=True)


def test_profile_accepts_a_name_not_just_a_code():
    assert svc.profile_airport("Santa Ana")["iata"] == "SNA"


def test_profile_states_data_vintage():
    """Every answer must be able to say how old its inputs are."""
    vintage = svc.profile_airport("SFO")["data_vintage"]
    assert "2024" in vintage["traffic"]
    assert "not a growth trend" in vintage["caveat"]


# --------------------------------------------------------------------------
# Long-haul: must refuse to invent a percentage.
# --------------------------------------------------------------------------


def test_long_haul_reports_unavailable_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENSKY_CLIENT_ID", raising=False)
    monkeypatch.delenv("OPENSKY_CLIENT_SECRET", raising=False)

    from airport_iq.ingest import load
    from airport_iq.providers.routes import get_provider

    result = get_provider(load()).long_haul_share("ANC").to_dict()
    assert result["long_haul_share_available"] is False
    assert "long_haul_pct_of_departures" not in result
    assert "OPENSKY_CLIENT_ID" in result["caveat"]


def test_long_haul_capability_matches_runway_length():
    from airport_iq.ingest import load
    from airport_iq.providers.routes import RunwayCapabilityProvider

    provider = RunwayCapabilityProvider(load())
    anchorage = provider.long_haul_share("ANC").to_dict()["runway_capability"]
    santa_ana = provider.long_haul_share("SNA").to_dict()["runway_capability"]

    assert anchorage["widebody_long_haul_capable"] is True
    assert santa_ana["widebody_long_haul_capable"] is False


def test_haversine_known_distance():
    """LAX->JFK is about 2,470 statute miles."""
    from airport_iq.providers.routes import haversine_miles

    distance = haversine_miles(33.9425, -118.4081, 40.6413, -73.7781)
    assert 2400 < distance < 2550


# --------------------------------------------------------------------------
# HTTP surface.
# --------------------------------------------------------------------------


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["airports_scored"] > 900


def test_rank_endpoint(client):
    body = client.get("/rank", params={"region": "new_england", "top_n": 3}).json()
    assert len(body["results"]) == 3


def test_rank_endpoint_rejects_unknown_region(client):
    assert client.get("/rank", params={"region": "atlantis"}).status_code == 400


def test_airport_endpoint(client):
    assert client.get("/airport/SFO").json()["iata"] == "SFO"


def test_airport_endpoint_404(client):
    assert client.get("/airport/ZZZZZ").status_code == 404


def test_compare_endpoint(client):
    body = client.get("/compare", params={"codes": "LAX,SNA"}).json()
    assert len(body["comparison"]) == 2


def test_chat_requires_api_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/chat", json={"message": "hello"})
    assert response.status_code == 503


def test_api_and_service_agree(client):
    """The HTTP layer must not transform the numbers on their way out."""
    direct = svc.rank(region="new_england", top_n=3)
    over_http = client.get("/rank", params={"region": "new_england", "top_n": 3}).json()
    assert [r["investment_score"] for r in direct["results"]] == [
        r["investment_score"] for r in over_http["results"]
    ]
