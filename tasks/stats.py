# This code is part of the VQC/VQN architecture scorecard.
#
# (C) Copyright L. D. Gonzalez Alcocer, Tecnologico de Monterrey, 2026.
#
# History:
#   2026-10-01  Created. summary, holm and paired_p moved here unchanged from
#               the archived fair_report, plus the pre-registered decision
#               rules of pure-v1: paired contrasts against a smallest effect
#               size of interest (SESOI), the three verdicts (supported,
#               refuted, inconclusive), Friedman and Kendall's W.

"""Statistics of the task protocols, and the pre-registered decision rules.

Every hypothesis of pure-v1 is a statement about a **paired** contrast: the
same seed gives the same split, minibatch order and initial angles to both
arms, so the unit of analysis is the per-seed difference ``a_i - b_i``.

Decision rules (frozen in the pre-registration; do not edit after data):

* :func:`greater` - "``a`` beats ``b``". Supported when the Holm-adjusted
  one-sided Wilcoxon p-value is below ``alpha`` and the mean difference is
  positive. Refuted when the 95% interval of the mean difference lies entirely
  below ``+sesoi`` (no effect of interest is possible). Otherwise inconclusive.
* :func:`not_greater` - "``a`` does not beat ``b`` by more than the SESOI".
  Supported when the interval's upper end is at most ``+sesoi``; refuted when its
  lower end is above ``+sesoi``; otherwise inconclusive.
* :func:`equivalent` - "``a`` and ``b`` are within the SESOI" (two one-sided
  tests in interval form). Supported when the interval lies inside
  ``(-sesoi, +sesoi)``; refuted when it lies entirely outside; otherwise
  inconclusive.

The interval is the 95% t-interval of the mean paired difference, the same
interval :func:`summary` reports.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

ALPHA: float = 0.05
VERDICTS: tuple[str, ...] = ("supported", "refuted", "inconclusive")


def summary(values) -> dict:
    """Return mean, sd, count and the 95% t-interval half-width."""
    a = np.asarray(values, dtype=float)
    k = a.size
    sd = float(a.std(ddof=1)) if k > 1 else 0.0
    half = float(stats.t.ppf(0.975, k - 1) * sd / np.sqrt(k)) if k > 1 else 0.0
    return {"mean": float(a.mean()) if k else float("nan"), "sd": sd, "n": int(k), "ci95": half}


def holm(pvalues: dict) -> dict:
    """Return Holm-adjusted p-values, same keys."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[key] = running
    return out


def paired_p(a, b) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value; identical samples give 1."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(diff, 0.0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)


def paired_p_greater(a, b) -> float:
    """One-sided paired Wilcoxon p-value for ``a > b``; identical samples give 1."""
    diff = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(diff, 0.0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="zsplit", alternative="greater").pvalue)


def contrast(a, b) -> dict:
    """Return the paired difference ``a - b``: mean, sd, n, 95% interval, both p-values.

    Args:
        a: Per-seed values of the first arm.
        b: Per-seed values of the second arm, same seed order.

    Returns:
        ``{"mean", "sd", "n", "ci95", "lo", "hi", "p_two", "p_greater"}``.

    Raises:
        ValueError: If the two arms have different lengths.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired arms differ in length: {a.shape} vs {b.shape}")
    s = summary(a - b)
    return {**s, "lo": s["mean"] - s["ci95"], "hi": s["mean"] + s["ci95"],
            "p_two": paired_p(a, b), "p_greater": paired_p_greater(a, b)}


def greater(c: dict, sesoi: float, p_adj: float, alpha: float = ALPHA) -> str:
    """Verdict on "a beats b" from a :func:`contrast` and its adjusted p-value."""
    if p_adj < alpha and c["mean"] > 0:
        return "supported"
    if c["hi"] < sesoi:
        return "refuted"
    return "inconclusive"


def not_greater(c: dict, sesoi: float) -> str:
    """Verdict on "a does not beat b by more than the SESOI"."""
    if c["hi"] <= sesoi:
        return "supported"
    if c["lo"] > sesoi:
        return "refuted"
    return "inconclusive"


def equivalent(c: dict, sesoi: float) -> str:
    """Verdict on "a and b are within the SESOI" (interval TOST)."""
    if -sesoi < c["lo"] and c["hi"] < sesoi:
        return "supported"
    if c["lo"] >= sesoi or c["hi"] <= -sesoi:
        return "refuted"
    return "inconclusive"


def combine(verdicts: Sequence[str]) -> str:
    """Return the verdict of a hypothesis made of several pre-registered parts.

    All parts supported -> supported; any part refuted -> refuted; otherwise
    inconclusive.
    """
    if verdicts and all(v == "supported" for v in verdicts):
        return "supported"
    if any(v == "refuted" for v in verdicts):
        return "refuted"
    return "inconclusive"


def friedman(matrix: np.ndarray) -> dict:
    """Friedman test and Kendall's W over a ``(seeds, arms)`` matrix (higher is better).

    Kendall's coefficient of concordance ``W = chi2 / (N (k - 1))`` measures how
    consistently the seeds order the arms (0 = no agreement, 1 = identical
    order in every seed).

    Args:
        matrix: Rows are seeds (blocks), columns are arms.

    Returns:
        ``{"chi2", "p", "W", "N", "k"}``; with fewer than 3 arms, ``p`` and
        ``chi2`` are ``nan``.
    """
    m = np.asarray(matrix, float)
    n_blocks, k = m.shape
    if k < 3 or n_blocks < 2:
        return {"chi2": float("nan"), "p": float("nan"), "W": float("nan"), "N": n_blocks, "k": k}
    res = stats.friedmanchisquare(*[m[:, j] for j in range(k)])
    return {"chi2": float(res.statistic), "p": float(res.pvalue),
            "W": float(res.statistic / (n_blocks * (k - 1))), "N": n_blocks, "k": k}


def spearman(x, y) -> dict:
    """Spearman correlation ``rho`` and its two-sided p-value."""
    res = stats.spearmanr(np.asarray(x, float), np.asarray(y, float))
    return {"rho": float(res.statistic), "p": float(res.pvalue), "n": int(len(x))}
