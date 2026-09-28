"""Small statistics helpers (no scipy dependency)."""

from __future__ import annotations

import numpy as np


def rankdata(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged (as scipy.stats.rankdata)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        tied = x == value
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; nan if either input is constant."""
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def quantiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    qs = np.quantile(x, [0.0, 0.05, 0.5, 0.95, 1.0])
    return dict(zip(["min", "q05", "median", "q95", "max"], map(float, qs), strict=True))
