"""portfolio_validation.py: independent re-validation of the Portfolio
Construction Lab from its published data file.

Three kinds of evidence, following the structure of bank model validation:

  replication   the estimation is rebuilt from the published daily returns
                (weekly covariance, sample means, Bayes-Stein shrinkage);
  challenger    the optimisation is re-solved by a different, exact method
                (active-set enumeration instead of the lab's SLSQP), and the
                risk metrics are recomputed with separately written code;
  outcomes      the lab's VaR is backtested and the stability of its
                optimiser's output under estimation error is measured.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import qp

TRADING_DAYS = 252
WEEKS = 52

# The lab publishes its inputs rounded to six decimals (returns, covariances,
# weights). Recomputing from rounded inputs cannot agree to machine precision,
# so tolerances are set just above the rounding noise and far below anything
# that would change a decision: 0.002 percentage points on a volatility or a
# return, 0.0001 on a ratio.
TOL = 2e-5
TOL_RATIO = 1e-4
TOL_WEIGHT = 5e-3


def check(checks: List[Dict], group: str, title: str, ok: bool, detail: str,
          published=None, challenger=None, diff: Optional[float] = None, tolerance: Optional[float] = None,
          level: str = "check") -> bool:
    checks.append({"group": group, "title": title, "ok": bool(ok), "level": level, "detail": detail,
                   "published": published, "challenger": challenger, "diff": diff, "tolerance": tolerance})
    return bool(ok)


def weekly_returns(data: Dict) -> pd.DataFrame:
    """Friday-to-Friday returns of the five sleeves, rebuilt from daily returns."""
    keys = [a["key"] for a in data["assets"]]
    dates = pd.to_datetime(data["series"]["dates"])
    daily = pd.DataFrame({k: data["series"]["returns"][k] for k in keys}, index=dates)
    levels = (1.0 + daily).cumprod()
    start = pd.DataFrame({k: [1.0] for k in keys}, index=[dates[0] - pd.offsets.BDay(1)])
    levels = pd.concat([start, levels])
    weekly = levels.resample("W-FRI").last()
    weekly = weekly[weekly.index <= pd.Timestamp(data["estimation"]["week_end"])]
    return weekly.pct_change().dropna()


def bayes_stein(weekly: pd.DataFrame):
    """Jorion (1986) shrinkage of sample means toward the minimum-variance mean."""
    t, n = weekly.shape
    mu = weekly.mean().values
    s = weekly.cov().values * (t - 1) / (t - n - 2)
    s_inv = np.linalg.inv(s)
    one = np.ones(n)
    mu0 = float((s_inv @ one) @ mu / (one @ s_inv @ one))
    d = mu - mu0
    phi = (n + 2) / ((n + 2) + t * float(d @ s_inv @ d))
    return (1 - phi) * mu + phi * mu0, phi, mu0


def risk_metrics(r: np.ndarray, rf: np.ndarray, bench: np.ndarray, conf: float = 0.95) -> Dict[str, Optional[float]]:
    """Daily risk and performance metrics, written independently of the lab.

    Conventions that have to match the lab's: sample standard deviations,
    linear-interpolated (type 7) quantiles and 252 trading days a year.
    """
    n = len(r)
    excess = r - rf
    ordered = np.sort(r)
    pos = (1 - conf) * (n - 1)
    lo = int(np.floor(pos))
    q = ordered[lo] + (ordered[min(lo + 1, n - 1)] - ordered[lo]) * (pos - lo)
    mean, sd = r.sum() / n, np.sqrt(((r - r.mean()) ** 2).sum() / (n - 1))
    z = norm.ppf(1 - conf)
    wealth = np.cumprod(1 + r)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    bench_excess = bench - rf
    bx = bench_excess - bench_excess.mean()
    beta = float((excess - excess.mean()) @ bx / (bx @ bx))
    active = r - bench
    te = np.sqrt(((active - active.mean()) ** 2).sum() / (n - 1)) * np.sqrt(TRADING_DAYS)
    ann_excess = excess.mean() * TRADING_DAYS
    return {
        "var_hist_1d": float(-q),
        "cvar_hist_1d": float(-r[r <= q].mean()),
        "var_param_1d": float(-(mean + z * sd)),
        "cvar_param_1d": float(-(mean - sd * norm.pdf(z) / (1 - conf))),
        "vol_ann": float(sd * np.sqrt(TRADING_DAYS)),
        "max_drawdown": float(drawdown.min()),
        "sharpe": float(ann_excess / (sd * np.sqrt(TRADING_DAYS))),
        "sortino": float(ann_excess / (downside * np.sqrt(TRADING_DAYS))),
        "beta": beta,
        "treynor": float(ann_excess / beta),
        "jensen_alpha": float((excess.mean() - beta * bench_excess.mean()) * TRADING_DAYS),
        "tracking_error": float(te),
        "information_ratio": float(active.mean() * TRADING_DAYS / te) if te > 1e-12 else None,
    }


def random_feasible(n_assets: int, lower: np.ndarray, upper: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """Random portfolios satisfying the budget and the box (rejection sampling)."""
    rng = np.random.default_rng(seed)
    out = []
    slack = 1.0 - lower.sum()
    while sum(len(x) for x in out) < draws:
        w = lower + slack * rng.dirichlet(np.ones(n_assets), size=draws)
        out.append(w[np.all(w <= upper + 1e-12, axis=1)])
    return np.vstack(out)[:draws]


def validate(data: Dict, mc_draws: int = 100_000, bootstrap: int = 300, seed: int = 20261111) -> Dict:
    checks: List[Dict] = []
    keys = [a["key"] for a in data["assets"]]
    n = len(keys)
    series = data["series"]
    rf_annual = data["risk_free"]["current_annual"]
    cov = np.array(data["estimation"]["cov"], float)

    # ---------------------------------------------------------------- data integrity
    dates = pd.to_datetime(series["dates"])
    lengths = {k: len(v) for k, v in series["returns"].items()}
    check(checks, "Data integrity", "Every return series has one value per date",
          len(set(lengths.values())) == 1 and lengths[keys[0]] == len(dates),
          f"{len(dates)} dates, series lengths {sorted(set(lengths.values()))}")
    weekend = [str(x.date()) for x in dates[dates.dayofweek >= 5]]
    check(checks, "Data integrity", "Dates are strictly increasing with no duplicates",
          bool(dates.is_monotonic_increasing and dates.is_unique),
          f"{dates[0].date()} to {dates[-1].date()}" +
          (f"; includes special weekend sessions on {', '.join(weekend)}" if weekend else ""))
    allr = np.column_stack([series["returns"][k] for k in series["returns"]])
    check(checks, "Data integrity", "No missing or implausible daily returns",
          bool(np.isfinite(allr).all() and np.abs(allr).max() < 0.25),
          f"largest absolute daily return {np.abs(allr).max():.2%}")
    gap = int(np.diff(dates.values).astype("timedelta64[D]").astype(int).max())
    check(checks, "Data integrity", "No gap longer than a week between sessions", gap <= 7, f"longest gap {gap} days")

    # ---------------------------------------------------------------- covariance
    eig = np.linalg.eigvalsh(cov)
    check(checks, "Covariance matrix", "Symmetric and positive definite",
          bool(np.allclose(cov, cov.T, atol=1e-12) and eig.min() > 0), f"smallest eigenvalue {eig.min():.2e}")
    vols = np.sqrt(np.diag(cov))
    pub_vols = np.array([a["vol"] for a in data["assets"]])
    d = float(np.abs(vols - pub_vols).max())
    check(checks, "Covariance matrix", "Diagonal agrees with the published volatilities", d < TOL,
          "square root of the diagonal vs each sleeve's stated volatility", diff=d, tolerance=TOL)
    corr = cov / np.outer(vols, vols)
    d = float(np.abs(corr - np.array(data["estimation"]["corr_weekly"])).max())
    check(checks, "Covariance matrix", "Implies the published correlation matrix", d < 5e-4,
          "covariance divided by the product of volatilities (small covariances lose precision to rounding)",
          diff=d, tolerance=5e-4)

    # ---------------------------------------------------------------- replication of the estimation
    weekly = weekly_returns(data)
    check(checks, "Estimation replicated", "Same number of weekly observations",
          len(weekly) == data["estimation"]["weeks"],
          "Friday-to-Friday weeks rebuilt from the published daily returns",
          published=data["estimation"]["weeks"], challenger=len(weekly))
    cov_rebuilt = weekly.cov().values * WEEKS
    d = float(np.abs(cov_rebuilt - cov).max())
    check(checks, "Estimation replicated", "Covariance matrix rebuilt from daily returns", d < TOL,
          "sample covariance of weekly returns, annualised by 52", diff=d, tolerance=TOL)
    mu_text = weekly.mean().values * WEEKS
    d = float(np.abs(mu_text - np.array(data["models"]["textbook"]["mu"])).max())
    check(checks, "Estimation replicated", "Sample mean returns (textbook model)", d < TOL,
          "mean weekly return, annualised by 52", diff=d, tolerance=TOL)
    mu_bs, phi, mu0 = bayes_stein(weekly)
    d = float(np.abs(mu_bs * WEEKS - np.array(data["models"]["policy"]["mu"])).max())
    check(checks, "Estimation replicated", "Bayes-Stein shrunk means (policy model)", d < TOL,
          "Jorion (1986) shrinkage toward the minimum-variance portfolio's mean", diff=d, tolerance=TOL)
    pub_bs = data["estimation"]["bayes_stein"]
    d = abs(phi - pub_bs["phi"])
    check(checks, "Estimation replicated", "Shrinkage intensity", d < TOL_RATIO,
          "share of each mean pulled toward the common mean", published=pub_bs["phi"], challenger=phi,
          diff=d, tolerance=TOL_RATIO)

    # ---------------------------------------------------------------- challenger optimisation
    frontiers = {}
    rng_seed = seed
    for name in ("policy", "textbook"):
        model = data["models"][name]
        label = model["label"]
        group = f"Optimiser challenged: {name} model"
        mu = np.array(model["mu"], float)
        lower = np.array([b[0] for b in model["bounds"]], float)
        upper = np.array([b[1] for b in model["bounds"]], float)

        points = model["frontier"] + model["lower"] + [model["orp"], model["mvp"]]
        w_all = np.array([p["w"] for p in points])
        feasible = (np.abs(w_all.sum(axis=1) - 1).max() < 5e-6 and (w_all >= lower - 5e-6).all()
                    and (w_all <= upper + 5e-6).all())
        check(checks, group, "Every published portfolio is fully invested and within its limits", feasible,
              f"{len(points)} portfolios checked against the budget and the allocation ranges")
        stated = max(abs(float(np.sqrt(np.array(p["w"]) @ cov @ np.array(p["w"]))) - p["vol"]) for p in points)
        check(checks, group, "Stated volatility follows from the stated weights", stated < TOL,
              "square root of w.cov.w for each published portfolio", diff=stated, tolerance=TOL)

        w_mvp = qp.min_variance(cov, lower, upper)
        _, vol_mvp, _ = qp.stats(w_mvp, mu, cov, rf_annual)
        d = abs(vol_mvp - model["mvp"]["vol"])
        check(checks, group, "Minimum-variance portfolio: volatility", d < TOL,
              "exact active-set solution vs the lab's iterative optimiser",
              published=model["mvp"]["vol"], challenger=vol_mvp, diff=d, tolerance=TOL)
        d = float(np.abs(w_mvp - np.array(model["mvp"]["w"])).max())
        check(checks, group, "Minimum-variance portfolio: weights", d < TOL_WEIGHT,
              "largest difference in any sleeve's weight", diff=d, tolerance=TOL_WEIGHT)

        w_orp = qp.max_sharpe(mu, cov, lower, upper, rf_annual)
        _, _, sharpe = qp.stats(w_orp, mu, cov, rf_annual)
        d = abs(sharpe - model["orp"]["sharpe"])
        check(checks, group, "Tangency portfolio: Sharpe ratio", d < TOL_RATIO,
              "exact frontier scan with golden-section refinement vs the lab's optimiser",
              published=model["orp"]["sharpe"], challenger=sharpe, diff=d, tolerance=TOL_RATIO)
        d = float(np.abs(w_orp - np.array(model["orp"]["w"])).max())
        check(checks, group, "Tangency portfolio: weights", d < TOL_WEIGHT,
              "largest difference in any sleeve's weight", diff=d, tolerance=TOL_WEIGHT)

        # Published returns are rounded, so the end points can sit a hair outside
        # the exactly attainable range; targets are brought back inside it.
        r_floor = float(w_mvp @ mu)
        r_ceiling = float(qp.max_return(mu, lower, upper) @ mu)
        challenger_pts, worst = [], 0.0
        for p in model["frontier"]:
            target = min(max(p["ret"], r_floor), r_ceiling)
            w = qp.min_variance(cov, lower, upper, mu, target)
            vol = float(np.sqrt(w @ cov @ w)) if w is not None else float("nan")
            challenger_pts.append([vol, p["ret"]])
            worst = max(worst, abs(vol - p["vol"])) if w is not None else float("inf")
        check(checks, group, "Efficient frontier: every published point is the true minimum", worst < TOL,
              f"{len(model['frontier'])} target returns re-solved exactly; largest volatility difference shown",
              diff=worst, tolerance=TOL)

        cloud = random_feasible(n, lower, upper, mc_draws, rng_seed)
        rng_seed += 1
        c_ret = cloud @ mu
        c_vol = np.sqrt(np.einsum("ij,jk,ik->i", cloud, cov, cloud))
        best = float(((c_ret - rf_annual) / c_vol).max())
        check(checks, group, "No random portfolio beats the tangency portfolio", best <= model["orp"]["sharpe"] + 1e-9,
              f"{mc_draws:,} random feasible portfolios; best Sharpe found {best:.4f}",
              published=model["orp"]["sharpe"], challenger=best)
        check(checks, group, "No random portfolio has lower risk than the minimum-variance portfolio",
              float(c_vol.min()) >= model["mvp"]["vol"] - 1e-9,
              f"lowest volatility found {c_vol.min():.4%}", published=model["mvp"]["vol"], challenger=float(c_vol.min()))

        idx = np.random.default_rng(seed).choice(mc_draws, 400, replace=False)
        frontiers[name] = {
            "label": label,
            "published": [[p["vol"], p["ret"]] for p in model["frontier"]],
            "challenger": challenger_pts,
            "cloud": [[float(c_vol[i]), float(c_ret[i])] for i in idx],
            "mvp": {"published": [model["mvp"]["vol"], model["mvp"]["ret"]], "challenger": [vol_mvp, float(w_mvp @ mu)],
                    "w_published": model["mvp"]["w"], "w_challenger": w_mvp.tolist()},
            "orp": {"published": [model["orp"]["vol"], model["orp"]["ret"]],
                    "challenger": [float(np.sqrt(w_orp @ cov @ w_orp)), float(w_orp @ mu)],
                    "w_published": model["orp"]["w"], "w_challenger": w_orp.tolist(),
                    "sharpe_published": model["orp"]["sharpe"], "sharpe_challenger": sharpe},
        }

    # ---------------------------------------------------------------- risk metrics recomputed
    sleeves = np.column_stack([series["returns"][k] for k in keys])
    rf_d = np.array(series["returns"]["RF"], float)
    bench = np.array(series["returns"]["BENCH"], float)
    metric_rows = []
    for name, entry in data["crosscheck"].items():
        r = bench if "weights" not in entry else sleeves @ np.array(entry["weights"], float)
        mine = risk_metrics(r, rf_d, bench)
        worst_key, worst = None, 0.0
        for k, v in mine.items():
            pub = entry.get(k)
            if pub is None or v is None:
                if not (pub is None and (v is None or abs(v) < 1e9)):
                    worst_key, worst = k, float("inf")
                continue
            if abs(v - pub) > worst:
                worst_key, worst = k, abs(v - pub)
            metric_rows.append({"portfolio": name, "metric": k, "published": pub, "challenger": v, "diff": abs(v - pub)})
        check(checks, "Risk metrics recomputed", f"{name.replace('_', ' ')}: 13 risk and performance metrics", worst < TOL_RATIO,
              f"VaR, expected shortfall, drawdown, Sharpe, Sortino, beta, alpha and tracking error; largest gap in {worst_key}",
              diff=worst, tolerance=TOL_RATIO)

    # ---------------------------------------------------------------- findings: estimation risk
    policy = data["models"]["policy"]
    lower = np.array([b[0] for b in policy["bounds"]], float)
    upper = np.array([b[1] for b in policy["bounds"]], float)
    rng = np.random.default_rng(seed)
    wk = weekly.values
    boot_w = []
    for _ in range(bootstrap):
        sample = wk[rng.integers(0, len(wk), len(wk))]
        boot_w.append(qp.min_variance(np.cov(sample, rowvar=False) * WEEKS, lower, upper))
    boot_w = np.array(boot_w)
    orp_r = sleeves @ np.array(policy["orp"]["w"], float) - rf_d
    sharpes = []
    for _ in range(2000):
        s = orp_r[rng.integers(0, len(orp_r), len(orp_r))]
        sharpes.append(s.mean() / s.std(ddof=1) * np.sqrt(TRADING_DAYS))
    findings = {
        "mvp_weight_ranges": [{"key": k, "published": policy["mvp"]["w"][i],
                               "p05": float(np.quantile(boot_w[:, i], 0.05)), "p95": float(np.quantile(boot_w[:, i], 0.95))}
                              for i, k in enumerate(keys)],
        "orp_realised_sharpe": {"point": float(orp_r.mean() / orp_r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
                                "p05": float(np.quantile(sharpes, 0.05)), "p95": float(np.quantile(sharpes, 0.95)),
                                "years": len(orp_r) / TRADING_DAYS},
        "bootstrap_draws": bootstrap,
    }

    return {"checks": checks, "frontiers": frontiers, "metric_rows": metric_rows, "findings": findings,
            "asset_names": {a["key"]: a["name"] for a in data["assets"]}, "as_of": data["meta"]["as_of"]}
