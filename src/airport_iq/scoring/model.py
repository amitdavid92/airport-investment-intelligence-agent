"""The deterministic scoring engine.

Given the ingested table, produces an investment score per airport plus the
component breakdown behind it. No language model is involved at any point -
this module is the reason the system can claim its rankings are reproducible.

Thesis: investment value is highest where demand presses hardest against
physical supply AND something can actually be built.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from . import normalize as norm
from .kpis import compute as compute_kpis

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"
WEIGHTS_PATH = CONFIG_DIR / "weights.yaml"
REGIONS_PATH = CONFIG_DIR / "regions.yaml"

COMPONENTS = [
    "demand_pressure",
    "terminal_strain",
    "unmet_demand",
    "renewal_need",
    "expansion_feasibility",
]

# How much of the "how binding is the constraint" adjustment land headroom can
# account for. Capped at half so an airport with abundant land still registers
# unmet demand if its runways and gates are saturated.
HEADROOM_DAMPING = 0.5


class ConfigError(ValueError):
    """Weights or regions config is malformed."""


@dataclass(frozen=True)
class ScoringConfig:
    profiles: dict[str, dict[str, float]]
    default_profile: str

    def weights(self, profile: str | None = None) -> dict[str, float]:
        name = profile or self.default_profile
        if name not in self.profiles:
            raise ConfigError(
                f"unknown weight profile {name!r}; available: {sorted(self.profiles)}"
            )
        return dict(self.profiles[name])


@functools.lru_cache(maxsize=1)
def load_config(path: Path = WEIGHTS_PATH) -> ScoringConfig:
    raw = yaml.safe_load(path.read_text())
    profiles = raw["profiles"]

    for name, weights in profiles.items():
        missing = set(COMPONENTS) - set(weights)
        unknown = set(weights) - set(COMPONENTS)
        if missing:
            raise ConfigError(f"profile {name!r} is missing components: {sorted(missing)}")
        if unknown:
            raise ConfigError(f"profile {name!r} has unknown components: {sorted(unknown)}")
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(f"profile {name!r} weights sum to {total}, expected 1.0")

    return ScoringConfig(profiles=profiles, default_profile=raw["default_profile"])


@functools.lru_cache(maxsize=1)
def load_regions(path: Path = REGIONS_PATH) -> dict[str, dict]:
    return yaml.safe_load(path.read_text())["regions"]


def components(df: pd.DataFrame, *, within_hub_class: bool = False) -> pd.DataFrame:
    """Compute the five 0-100 component scores.

    By default components are ranked nationally so that scores are comparable
    across the whole dataset. `within_hub_class` ranks each airport against its
    FAA peer group instead, which answers "big for its size" questions.
    """
    out = compute_kpis(df)
    groups = out["hub_class"] if within_hub_class else None

    # 1. Runway-system saturation.
    out["demand_pressure"] = norm.percentile_score(out["ops_per_runway"], groups=groups)

    # 2. Gate/terminal saturation, via average aircraft gauge.
    out["terminal_strain"] = norm.percentile_score(out["pax_per_departure"], groups=groups)

    # 3. Unmet demand: pressure, amplified where the physical constraint binds.
    #    Abundant land means the constraint is relievable, so it damps the
    #    score; a boxed-in airport carries its pressure at full weight.
    headroom = norm.normalized_fraction(out["acres_per_100k_ops"], groups=groups)
    pressure = 0.5 * out["demand_pressure"] + 0.5 * out["terminal_strain"]
    constraint = 1.0 - HEADROOM_DAMPING * headroom.fillna(headroom.median())
    out["unmet_demand"] = (pressure * constraint).round(2)

    # 4. Renewal need, straight from published pavement condition.
    out["renewal_need"] = norm.minmax_score(out.get("pct_runway_len_not_good"))

    # 5. Expansion feasibility: can a project physically and financially happen?
    land = norm.percentile_score(out["acres_per_100k_ops"], groups=groups)
    public_bonus = out["publicly_owned"].astype(float) * 100.0
    longhaul_bonus = out["long_haul_capable"].astype(float) * 100.0
    out["expansion_feasibility"] = (
        0.60 * land.fillna(land.median())
        + 0.20 * public_bonus
        + 0.20 * longhaul_bonus
    ).round(2)

    return out


def score(
    df: pd.DataFrame,
    *,
    profile: str | None = None,
    within_hub_class: bool = False,
) -> pd.DataFrame:
    """Attach `investment_score`, `confidence` and per-component contributions."""
    config = load_config()
    weights = config.weights(profile)

    out = components(df, within_hub_class=within_hub_class)

    # Renormalize over the components that actually have a value for this
    # airport. A missing component must not be read as a zero - that would
    # quietly penalise an airport for a gap in the FAA's published data.
    present = pd.DataFrame(
        {c: out[c].notna().astype(float) * weights[c] for c in COMPONENTS},
        index=out.index,
    )
    weight_total = present.sum(axis=1)

    weighted = pd.DataFrame(
        {c: out[c].fillna(0.0) * present[c] for c in COMPONENTS}, index=out.index
    )
    out["investment_score"] = (weighted.sum(axis=1) / weight_total.where(weight_total > 0)).round(2)

    for component in COMPONENTS:
        out[f"contrib_{component}"] = (
            weighted[component] / weight_total.where(weight_total > 0)
        ).round(2)

    # Confidence blends how complete the inputs were with how much of the
    # chosen profile's weight could actually be applied.
    out["weight_coverage"] = weight_total.round(3)
    out["confidence"] = (
        0.5 * out["data_completeness"].fillna(0.0) + 0.5 * weight_total
    ).round(3)
    out["confidence_band"] = pd.cut(
        out["confidence"],
        bins=[-0.01, 0.60, 0.85, 1.01],
        labels=["low", "medium", "high"],
    ).astype("object")

    out["profile_used"] = profile or config.default_profile
    out["normalized_within"] = "hub_class" if within_hub_class else "national"

    out = out.sort_values("investment_score", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)
