"""Small statistics helpers for CI-grade validation claims.

The repo's lesson (ARCHITECTURE 3.18: a 4-game pilot 'showed' quiescence hurts --
pure small-sample noise) is institutionalised here: comparative claims are asserted
on PAIRED per-seed differences with a t-confidence interval, not on point means.
All experiment harnesses pair naturally (the same world seed runs every config), so
the paired design removes the between-world variance for free.

Conventions:
  * ``paired_ci(a, b)`` -> stats of (a_i - b_i): a positive ``lo`` means "a beats b,
    significant at the level"; an interval straddling 0 is an honest null.
  * significance level defaults to 95% two-sided.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t as _t


def mean_ci(xs, level: float = 0.95) -> dict:
    """Mean with a two-sided t confidence interval."""
    xs = np.asarray(list(xs), float)
    n = xs.size
    m = float(xs.mean())
    if n < 2:
        return {"mean": m, "lo": m, "hi": m, "n": n}
    half = float(_t.ppf(0.5 + level / 2, n - 1) * xs.std(ddof=1) / np.sqrt(n))
    return {"mean": m, "lo": m - half, "hi": m + half, "n": n}


def paired_ci(a, b, level: float = 0.95) -> dict:
    """CI on the PAIRED differences a_i - b_i (same seed in both arms)."""
    a, b = np.asarray(list(a), float), np.asarray(list(b), float)
    assert a.size == b.size, "paired comparison needs equal, seed-aligned samples"
    d = mean_ci(a - b, level)
    d["wins"] = int(np.sum(a > b))
    return d


def verdict(ci: dict) -> str:
    """Human-readable significance verdict for a paired difference."""
    if ci["lo"] > 0:
        return "+ (sig)"
    if ci["hi"] < 0:
        return "- (sig)"
    return "0 (ns)"


def fmt_ci(ci: dict) -> str:
    return f"{ci['mean']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}]"
