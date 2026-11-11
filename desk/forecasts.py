"""forecasts.py: the forecasting models the ledger records and scores.

All models use only information available at the forecast date. One-day
value-at-risk is quoted as a positive loss.

  hist    historical simulation: the empirical tail quantile of the last 250 days
  normal  parametric: mean and standard deviation of the last 250 days
  ewma    RiskMetrics: zero mean, exponentially weighted variance (lambda 0.94)

Volatility forecasts (one-day variance): the same EWMA, a 60-day rolling
variance, and a GARCH(1,1) fitted by maximum likelihood with variance
targeting on the history available at the forecast date.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from . import stats

WINDOW = 250
EWMA_LAMBDA = 0.94
EWMA_SEED_DAYS = 30
ROLLING_VOL_DAYS = 60
CONFIDENCES = (0.95, 0.99)
VAR_MODELS = ("hist", "normal", "ewma")
VOL_MODELS = ("ewma", "rolling60", "garch")
MIN_HISTORY = WINDOW


def ewma_variance(returns: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    """EWMA variance forecast for each next day.

    out[t] is the forecast for day t + 1 using returns up to and including day
    t. The recursion is seeded with the variance of the first 30 observations.
    """
    r = np.asarray(returns, float)
    out = np.empty(len(r))
    var = float(np.var(r[:EWMA_SEED_DAYS]))
    for t in range(len(r)):
        var = lam * var + (1.0 - lam) * r[t] ** 2
        out[t] = var
    return out


def var_forecast(history: np.ndarray, model: str, conf: float) -> float:
    """One-day VaR for the day after the last observation in `history`."""
    h = np.asarray(history, float)
    if len(h) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} observations")
    tail = 1.0 - conf
    if model == "hist":
        return float(-np.quantile(h[-WINDOW:], tail))
    if model == "normal":
        w = h[-WINDOW:]
        return float(-(w.mean() + norm.ppf(tail) * w.std(ddof=1)))
    if model == "ewma":
        return float(-norm.ppf(tail) * np.sqrt(ewma_variance(h)[-1]))
    raise KeyError(model)


def garch_fit(returns: np.ndarray) -> Dict[str, float]:
    """GARCH(1,1) by Gaussian maximum likelihood with variance targeting.

    sigma2[t+1] = omega + alpha * r[t]^2 + beta * sigma2[t], with omega set so
    that the unconditional variance equals the sample variance. Returns alpha,
    beta, omega and the one-step-ahead variance forecast.
    """
    r = np.asarray(returns, float)
    r = r - r.mean()
    target = float(np.var(r))

    def path(alpha: float, beta: float) -> np.ndarray:
        omega = target * (1.0 - alpha - beta)
        s2 = np.empty(len(r) + 1)
        s2[0] = target
        for t in range(len(r)):
            s2[t + 1] = omega + alpha * r[t] ** 2 + beta * s2[t]
        return s2

    def neg_loglik(theta: np.ndarray) -> float:
        alpha, beta = theta
        if alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e12
        s2 = path(alpha, beta)[:-1]
        return float(0.5 * np.sum(np.log(s2) + r ** 2 / s2))

    best = None
    for start in ((0.05, 0.90), (0.10, 0.80), (0.03, 0.95)):
        res = minimize(neg_loglik, np.array(start), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    alpha, beta = (float(x) for x in best.x)
    return {"alpha": alpha, "beta": beta, "omega": target * (1.0 - alpha - beta),
            "forecast_var": float(path(alpha, beta)[-1])}


def vol_forecast(history: np.ndarray, model: str) -> float:
    """One-day variance forecast for the day after the last observation."""
    h = np.asarray(history, float)
    if model == "ewma":
        return float(ewma_variance(h)[-1])
    if model == "rolling60":
        return float(np.var(h[-ROLLING_VOL_DAYS:], ddof=1))
    if model == "garch":
        return garch_fit(h)["forecast_var"]
    raise KeyError(model)


# ---------------------------------------------------------------- walk-forward backtests

def var_backtest(returns: Sequence[float], dates: Sequence[str]) -> Dict:
    """Walk-forward VaR backtest of every model at every confidence level.

    For each day t >= 250 the forecast uses returns[:t] only and is compared
    with returns[t].
    """
    r = np.asarray(returns, float)
    n = len(r)
    ew = ewma_variance(r)
    out = {"start": dates[MIN_HISTORY], "end": dates[-1], "n": n - MIN_HISTORY, "models": {}}
    for model in VAR_MODELS:
        out["models"][model] = {}
        for conf in CONFIDENCES:
            tail = 1.0 - conf
            z = norm.ppf(tail)
            var = np.empty(n - MIN_HISTORY)
            for i, t in enumerate(range(MIN_HISTORY, n)):
                w = r[t - WINDOW:t]
                if model == "hist":
                    var[i] = -np.quantile(w, tail)
                elif model == "normal":
                    var[i] = -(w.mean() + z * w.std(ddof=1))
                else:
                    var[i] = -z * np.sqrt(ew[t - 1])
            realised = r[MIN_HISTORY:]
            hits = (realised < -var).astype(int)
            kup = stats.kupiec_pof(int(hits.sum()), len(hits), tail)
            chr_ = stats.christoffersen_independence(hits)
            entry = {
                "confidence": conf, "exceedances": int(hits.sum()), "expected": kup["expected"],
                "rate": kup["rate"], "kupiec_lr": kup["lr"], "kupiec_p": kup["p_value"],
                "christoffersen_lr": chr_["lr"], "christoffersen_p": chr_["p_value"],
                "pinball": stats.pinball_loss(realised, var, tail),
                "mean_var": float(var.mean()),
            }
            if conf == 0.99:
                last = int(hits[-250:].sum())
                entry["last_250_exceedances"] = last
                entry["basel_zone"] = stats.basel_zone(last)
            out["models"][model][f"{conf:.2f}"] = entry
    return out


def vol_backtest(returns: Sequence[float], dates: Sequence[str], garch_refit_every: int = 21) -> Dict:
    """Walk-forward one-day variance forecasts scored by QLIKE and MSE.

    GARCH parameters are re-estimated every `garch_refit_every` days on the
    history to date, and the variance recursion is run forward in between.
    """
    r = np.asarray(returns, float)
    n = len(r)
    ew = ewma_variance(r)
    realised = r[MIN_HISTORY:] ** 2
    forecasts = {"ewma": ew[MIN_HISTORY - 1:n - 1],
                 "rolling60": np.array([np.var(r[t - ROLLING_VOL_DAYS:t], ddof=1) for t in range(MIN_HISTORY, n)])}
    garch = np.empty(n - MIN_HISTORY)
    params, s2 = None, None
    for i, t in enumerate(range(MIN_HISTORY, n)):
        if i % garch_refit_every == 0:
            params = garch_fit(r[:t])
            s2 = params["forecast_var"]
            mean = float(r[:t].mean())
        else:
            s2 = params["omega"] + params["alpha"] * (r[t - 1] - mean) ** 2 + params["beta"] * s2
        garch[i] = s2
    forecasts["garch"] = garch
    out = {"start": dates[MIN_HISTORY], "end": dates[-1], "n": n - MIN_HISTORY, "models": {}}
    for model, fv in forecasts.items():
        out["models"][model] = {
            "qlike": stats.qlike(realised, fv),
            "mse": float(np.mean((realised - fv) ** 2)),
            "mean_vol_annual": float(np.sqrt(fv.mean() * 252)),
        }
    out["realised_vol_annual"] = float(np.sqrt(realised.mean() * 252))
    return out


def portfolio_returns(series: Dict[str, List[float]], keys: Sequence[str], weights: Sequence[float]) -> np.ndarray:
    """Daily returns of a constant-weight portfolio of the given sleeves."""
    mat = np.column_stack([np.asarray(series[k], float) for k in keys])
    return mat @ np.asarray(weights, float)
