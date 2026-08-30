"""Turning raw KPIs into comparable 0-100 component scores.

Percentile rank is used rather than min-max because US airport traffic is
extremely long-tailed: Atlanta alone would compress every other airport into
the bottom few percent of a min-max scale, and the resulting ranking would
say nothing except "big airports are big".
"""

from __future__ import annotations

import pandas as pd

WINSOR_LOW = 0.01
WINSOR_HIGH = 0.99


def winsorize(series: pd.Series, low: float = WINSOR_LOW, high: float = WINSOR_HIGH) -> pd.Series:
    """Clip extreme tails so one outlier cannot dominate a scale."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() < 2:
        return values
    lower, upper = values.quantile([low, high])
    return values.clip(lower=lower, upper=upper)


def percentile_score(
    series: pd.Series,
    *,
    groups: pd.Series | None = None,
    invert: bool = False,
) -> pd.Series:
    """Percentile-rank a KPI onto 0-100, preserving NaN.

    `groups` ranks within a peer group instead of nationally. `invert` is for
    KPIs where a low raw value is the interesting one.
    """
    values = winsorize(series)

    if groups is None:
        ranked = values.rank(pct=True, na_option="keep")
    else:
        ranked = values.groupby(groups).rank(pct=True, na_option="keep")

    if invert:
        ranked = 1.0 - ranked
    return (ranked * 100).round(2)


def minmax_score(series: pd.Series) -> pd.Series:
    """Scale a value that is already a meaningful 0-1 fraction onto 0-100.

    Used for the renewal component. `pct_runway_len_not_good` is already
    directly interpretable - "34% of runway length is not in good condition" -
    and percentile-ranking it would destroy that meaning, since most airports
    sit at exactly zero and would be spread arbitrarily across the bottom half.
    """
    values = pd.to_numeric(series, errors="coerce")
    return (values.clip(lower=0.0, upper=1.0) * 100).round(2)


def normalized_fraction(series: pd.Series, *, groups: pd.Series | None = None) -> pd.Series:
    """Percentile rank expressed as a 0-1 fraction (not 0-100)."""
    return percentile_score(series, groups=groups) / 100.0
