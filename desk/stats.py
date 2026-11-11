"""stats.py: backtesting statistics and forecast scoring rules."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import chi2


def _xlogy(x: float, y: float) -> float:
    """x * log(y) with the convention 0 * log(0) = 0."""
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_pof(exceedances: int, n: int, p: float) -> Dict[str, Optional[float]]:
    """Kupiec (1995) proportion-of-failures test.

    H0: the exceedance probability equals p. LR ~ chi-square(1).
    """
    if n <= 0 or not 0 < p < 1 or not 0 <= exceedances <= n:
        raise ValueError("need n > 0, 0 < p < 1 and 0 <= exceedances <= n")
    x = exceedances
    rate = x / n
    log_null = _xlogy(n - x, 1 - p) + _xlogy(x, p)
    log_alt = _xlogy(n - x, 1 - rate) + _xlogy(x, rate)
    lr = max(0.0, -2.0 * (log_null - log_alt))
    return {"exceedances": x, "n": n, "expected": n * p, "rate": rate, "lr": lr,
            "p_value": float(chi2.sf(lr, 1))}


def christoffersen_independence(hits: Sequence[int]) -> Dict[str, Optional[float]]:
    """Christoffersen (1998) test that exceedances do not cluster.

    H0: an exceedance today is no more likely after an exceedance yesterday.
    LR ~ chi-square(1). p_value is None when the test is undefined (no
    exceedances, or no transitions of one kind to estimate from).
    """
    h = [int(bool(x)) for x in hits]
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(h[:-1], h[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    out = {"n00": n00, "n01": n01, "n10": n10, "n11": n11, "lr": None, "p_value": None}
    total = n00 + n01 + n10 + n11
    if total == 0 or (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return out
    pi = (n01 + n11) / total
    pi0 = n01 / (n00 + n01)
    pi1 = n11 / (n10 + n11)
    log_null = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    log_alt = _xlogy(n00, 1 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1 - pi1) + _xlogy(n11, pi1)
    lr = max(0.0, -2.0 * (log_null - log_alt))
    out.update({"lr": lr, "p_value": float(chi2.sf(lr, 1))})
    return out


def basel_zone(exceedances_99_in_250: int) -> str:
    """Basel traffic light for a 99% one-day VaR over 250 observations."""
    if exceedances_99_in_250 <= 4:
        return "green"
    if exceedances_99_in_250 <= 9:
        return "yellow"
    return "red"


def pinball_loss(returns: Sequence[float], var: Sequence[float], tail: float) -> float:
    """Average quantile (pinball) loss of VaR forecasts for the `tail` quantile.

    `var` is quoted as a positive loss, so the forecast quantile is -var. The
    quantile loss is a strictly consistent scoring rule: the true quantile
    minimises its expectation, which makes it a fair way to rank VaR models.
    """
    r = np.asarray(returns, float)
    q = -np.asarray(var, float)
    u = r - q
    return float(np.mean(np.where(u >= 0, tail * u, (tail - 1) * u)))


def qlike(realised_var: Sequence[float], forecast_var: Sequence[float]) -> float:
    """QLIKE loss for variance forecasts (robust to noise in the realised proxy)."""
    rv = np.asarray(realised_var, float)
    fv = np.asarray(forecast_var, float)
    if np.any(fv <= 0):
        raise ValueError("variance forecasts must be positive")
    return float(np.mean(np.log(fv) + rv / fv))


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks, ties shared."""
    a = np.asarray(values, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation; None when undefined."""
    if len(x) != len(y) or len(x) < 3:
        return None
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return None
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))
