"""Raw KPIs derived from the ingested table.

Everything here is arithmetic on published figures - no modelling, no
estimation, no LLM. Each KPI keeps its natural unit so it can be quoted
directly in an answer ("312 ops per runway", "88 passengers per departure")
rather than only as an index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# FAA OWNERSHIP_TYPE_CODE values that mean publicly owned. Public ownership is
# what makes an airport eligible for FAA Airport Improvement Program grants,
# which materially changes who funds a project.
PUBLIC_OWNERSHIP = {"PU", "MA", "MN", "MR", "CG"}

# A runway of this length can support widebody long-haul operations. Used as a
# capability flag, not as a claim about what actually flies there.
LONG_HAUL_RUNWAY_FT = 10_000


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Attach raw KPI columns. Pure function - returns a new frame."""
    out = df.copy()

    ops = out["departures"].fillna(0) + out["arrivals"].fillna(0)
    out["annual_operations"] = ops

    # Runway-system saturation. The denominator is runway count rather than a
    # modelled hourly capacity because runway count is published fact and an
    # hourly capacity figure would be an assumption dressed as data.
    out["ops_per_runway"] = _safe_divide(ops, out["runway_count"])

    # Average aircraft gauge. Rises when an airport cannot add frequency and
    # airlines respond with larger aircraft - the classic terminal-side
    # constraint signal.
    out["pax_per_departure"] = _safe_divide(out["passengers"], out["departures"])

    # Land available per unit of traffic. Low values mean a physically boxed-in
    # airport where expansion is expensive or impossible.
    out["acres_per_100k_ops"] = _safe_divide(out["acreage"], ops / 100_000)

    out["runway_len_max"] = pd.to_numeric(out.get("runway_len_max"), errors="coerce")
    out["long_haul_capable"] = out["runway_len_max"] >= LONG_HAUL_RUNWAY_FT

    ownership = out.get("ownership", pd.Series(index=out.index, dtype="object"))
    out["publicly_owned"] = ownership.isin(PUBLIC_OWNERSHIP)

    total_enplanements = out["enplanements"].sum()
    out["enplanement_share"] = out["enplanements"] / total_enplanements
    out["hub_class"] = classify_hub(out["enplanement_share"])

    return out


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, yielding NaN rather than inf where the denominator is 0/NaN."""
    denom = pd.to_numeric(denominator, errors="coerce")
    denom = denom.where(denom > 0)
    return pd.to_numeric(numerator, errors="coerce") / denom


def classify_hub(share: pd.Series) -> pd.Series:
    """FAA hub classification by share of total US enplanements.

    These thresholds are the FAA's own (14 CFR Part 158), not invented
    buckets - which matters when the agent has to justify why comparing a
    large hub to a medium hub is not like-for-like.
    """
    conditions = [
        share >= 0.01,
        share >= 0.0025,
        share >= 0.0005,
    ]
    labels = ["large_hub", "medium_hub", "small_hub"]
    return pd.Series(
        np.select(conditions, labels, default="nonhub"),
        index=share.index,
        dtype="object",
    )
