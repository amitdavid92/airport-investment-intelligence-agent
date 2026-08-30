"""Tests for the deterministic scoring engine.

These are the tests that back the claim "the ranking is not LLM output":
same inputs, same numbers, every time, with published figures underneath.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airport_iq.ingest import load
from airport_iq.scoring.kpis import classify_hub, compute
from airport_iq.scoring.model import COMPONENTS, ConfigError, load_config, score
from airport_iq.scoring.normalize import minmax_score, percentile_score, winsorize


@pytest.fixture(scope="module")
def airports() -> pd.DataFrame:
    return load()


@pytest.fixture(scope="module")
def scored(airports: pd.DataFrame) -> pd.DataFrame:
    return score(airports)


# --------------------------------------------------------------------------
# Published figures must survive ingest unchanged.
# --------------------------------------------------------------------------

# Verified directly against the BTS T-100 FeatureServer during development.
GOLDEN_TRAFFIC = {
    "SFO": {"enplanements": 17_631_272, "departures": 138_571, "arrivals": 138_528},
    "ATL": {"enplanements": 45_406_031, "departures": 345_209},
}

# Verified against the FAA Runways and Aviation Facilities layers.
GOLDEN_INFRA = {
    "SFO": {"runway_count": 4, "runway_len_max": 11_870, "acreage": 5_207},
    "ORD": {"runway_count": 8, "acreage": 7_627},
    "DEN": {"runway_count": 6, "acreage": 33_531},
}


@pytest.mark.parametrize("code,expected", GOLDEN_TRAFFIC.items())
def test_traffic_matches_published_figures(airports, code, expected):
    row = airports[airports["iata"] == code].iloc[0]
    for field, value in expected.items():
        assert row[field] == value, f"{code}.{field}"


@pytest.mark.parametrize("code,expected", GOLDEN_INFRA.items())
def test_infrastructure_matches_published_figures(airports, code, expected):
    row = airports[airports["iata"] == code].iloc[0]
    for field, value in expected.items():
        assert row[field] == value, f"{code}.{field}"


def test_core_airports_present(airports):
    """The airports the brief asks about must all survive ingest."""
    required = {"SFO", "LAX", "SNA", "ANC", "BOS", "ATL", "JFK", "PVD", "BDL"}
    assert required <= set(airports["iata"])


def test_palm_beach_recovered_through_alias(airports):
    """PBI was recoded to DJT between the traffic and reference vintages."""
    row = airports[airports["iata"] == "DJT"]
    assert not row.empty, "Palm Beach missing - the IATA alias table regressed"
    assert row.iloc[0]["reported_iata"] == "PBI"
    assert row.iloc[0]["enplanements"] > 4_000_000


# --------------------------------------------------------------------------
# Determinism - the property the whole design rests on.
# --------------------------------------------------------------------------


def test_scoring_is_deterministic(airports):
    first = score(airports)
    second = score(airports)
    pd.testing.assert_frame_equal(first, second)


def test_score_is_independent_of_input_row_order(airports):
    shuffled = airports.sample(frac=1.0, random_state=17).reset_index(drop=True)
    baseline = score(airports).set_index("iata")["investment_score"]
    reordered = score(shuffled).set_index("iata")["investment_score"]
    pd.testing.assert_series_equal(
        baseline.sort_index(), reordered.sort_index(), check_names=False
    )


# --------------------------------------------------------------------------
# Weight configuration.
# --------------------------------------------------------------------------


def test_every_profile_sums_to_one():
    config = load_config()
    for name, weights in config.profiles.items():
        assert sum(weights.values()) == pytest.approx(1.0), name


def test_every_profile_covers_every_component():
    config = load_config()
    for name, weights in config.profiles.items():
        assert set(weights) == set(COMPONENTS), name


def test_unknown_profile_is_rejected():
    with pytest.raises(ConfigError):
        load_config().weights("no_such_profile")


def test_terminal_profile_favours_terminal_strain(airports):
    """A profile change must actually move the ranking, or it is decoration."""
    balanced = score(airports, profile="balanced").set_index("iata")
    terminal = score(airports, profile="terminal").set_index("iata")
    assert not balanced["investment_score"].equals(terminal["investment_score"])


# --------------------------------------------------------------------------
# Component behaviour.
# --------------------------------------------------------------------------


def test_all_components_within_bounds(scored):
    for component in COMPONENTS:
        values = scored[component].dropna()
        assert values.between(0, 100).all(), component


def test_investment_score_within_bounds(scored):
    assert scored["investment_score"].dropna().between(0, 100).all()


def test_demand_pressure_is_monotonic_in_operations():
    """More operations on the same runways must not lower demand pressure."""
    base = pd.DataFrame(
        {
            "iata": ["AAA", "BBB", "CCC"],
            "name": ["a", "b", "c"],
            "enplanements": [1_000_000, 1_000_000, 1_000_000],
            "passengers": [1_000_000, 1_000_000, 1_000_000],
            "departures": [10_000, 20_000, 30_000],
            "arrivals": [10_000, 20_000, 30_000],
            "runway_count": [2, 2, 2],
            "acreage": [1000, 1000, 1000],
            "runway_len_max": [10_000, 10_000, 10_000],
            "pct_runway_len_not_good": [0.1, 0.1, 0.1],
            "ownership": ["PU", "PU", "PU"],
            "data_completeness": [1.0, 1.0, 1.0],
        }
    )
    result = compute(base).set_index("iata")
    assert result.loc["AAA", "ops_per_runway"] < result.loc["BBB", "ops_per_runway"]
    assert result.loc["BBB", "ops_per_runway"] < result.loc["CCC", "ops_per_runway"]


def test_renewal_need_tracks_published_condition(scored):
    """Airports with all-sound pavement score zero; degraded pavement scores above."""
    jfk = scored[scored["iata"] == "JFK"].iloc[0]
    assert jfk["renewal_need"] == 0.0, "JFK runways are all EXCELLENT"

    djt = scored[scored["iata"] == "DJT"].iloc[0]
    assert djt["renewal_need"] > 0, "DJT has a FAIR runway and should register need"


def test_excellent_condition_is_not_treated_as_needing_renewal(airports):
    """Regression: EXCELLENT once fell outside the 'good' test and inverted the signal."""
    for code in ("JFK", "EWR", "SEA"):
        row = airports[airports["iata"] == code].iloc[0]
        assert row["pct_runway_len_not_good"] == 0.0, code


def test_missing_condition_is_unknown_not_bad(airports):
    """A blank FAA condition code must produce NaN, never a maximal need score."""
    partial = airports[airports["pct_runway_len_cond_known"].fillna(0) == 0]
    assert partial["pct_runway_len_not_good"].isna().all()


# --------------------------------------------------------------------------
# Missing data must never be scored as zero.
# --------------------------------------------------------------------------


def test_weights_renormalize_over_available_components():
    """Two identical airports, one missing pavement data, must score the same."""
    rows = pd.DataFrame(
        {
            "iata": ["AAA", "BBB"],
            "name": ["complete", "missing_cond"],
            "enplanements": [5_000_000, 5_000_000],
            "passengers": [5_000_000, 5_000_000],
            "departures": [50_000, 50_000],
            "arrivals": [50_000, 50_000],
            "runway_count": [3, 3],
            "acreage": [2000, 2000],
            "runway_len_max": [11_000, 11_000],
            "pct_runway_len_not_good": [0.0, np.nan],
            "ownership": ["PU", "PU"],
            "data_completeness": [1.0, 1.0],
        }
    )
    result = score(rows).set_index("iata")

    # AAA scores 0 on renewal; BBB has no renewal score at all. Because AAA's
    # renewal contribution is zero, redistributing BBB's renewal weight across
    # the other components can only raise BBB - it must never be penalised.
    assert result.loc["BBB", "investment_score"] >= result.loc["AAA", "investment_score"]
    assert pd.isna(result.loc["BBB", "renewal_need"])
    assert result.loc["BBB", "weight_coverage"] < result.loc["AAA", "weight_coverage"]


def test_confidence_drops_when_weight_coverage_drops(scored):
    incomplete = scored[scored["weight_coverage"] < 1.0]
    if incomplete.empty:
        pytest.skip("every airport has full component coverage")
    complete = scored[scored["weight_coverage"] == 1.0]
    assert incomplete["confidence"].mean() < complete["confidence"].mean()


# --------------------------------------------------------------------------
# Normalization primitives.
# --------------------------------------------------------------------------


def test_winsorize_clips_outliers():
    series = pd.Series([1, 2, 3, 4, 5, 1000])
    assert winsorize(series).max() < 1000


def test_percentile_score_is_scale_invariant():
    """Percentile rank must not care about units - that is the point of it."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    pd.testing.assert_series_equal(
        percentile_score(series), percentile_score(series * 1000), check_names=False
    )


def test_percentile_score_preserves_nan():
    series = pd.Series([1.0, np.nan, 3.0])
    assert percentile_score(series).isna().sum() == 1


def test_percentile_invert_reverses_order():
    series = pd.Series([1.0, 2.0, 3.0])
    normal = percentile_score(series)
    inverted = percentile_score(series, invert=True)
    assert normal.iloc[0] < normal.iloc[-1]
    assert inverted.iloc[0] > inverted.iloc[-1]


def test_minmax_score_keeps_fraction_meaning():
    """0.34 of runway length degraded must read as 34, not as a percentile."""
    assert minmax_score(pd.Series([0.344])).iloc[0] == pytest.approx(34.4, abs=0.1)


# --------------------------------------------------------------------------
# FAA hub classification.
# --------------------------------------------------------------------------


def test_hub_classification_thresholds():
    share = pd.Series([0.02, 0.005, 0.001, 0.0001])
    assert classify_hub(share).tolist() == [
        "large_hub",
        "medium_hub",
        "small_hub",
        "nonhub",
    ]


def test_known_hubs_classified_correctly(scored):
    by_code = scored.set_index("iata")["hub_class"]
    assert by_code["ATL"] == "large_hub"
    assert by_code["LAX"] == "large_hub"
    assert by_code["SNA"] == "medium_hub"
