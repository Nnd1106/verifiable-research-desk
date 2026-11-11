"""Tests for the desk's own tools: a validator has to be validated too."""

import json
import math
import os
import sys

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from scipy import stats as sps
from scipy.optimize import linprog, minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from desk import forecasts, ledger, portfolio_validation as pv, qp, stats, valuation_validation as vv  # noqa: E402
import run  # noqa: E402


def random_problem(seed, n=5):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n + 3, n))
    cov = a.T @ a / (n + 3) * 0.02 + np.eye(n) * 1e-4
    mu = rng.uniform(0.03, 0.18, n)
    return mu, cov


# ---------------------------------------------------------------- statistics

def test_kupiec_matches_hand_calculation():
    out = stats.kupiec_pof(10, 250, 0.01)
    lr = -2 * ((240 * math.log(0.99) + 10 * math.log(0.01)) - (240 * math.log(0.96) + 10 * math.log(0.04)))
    assert out["lr"] == pytest.approx(lr)
    assert out["p_value"] == pytest.approx(float(sps.chi2.sf(lr, 1)))
    assert out["expected"] == pytest.approx(2.5)


def test_kupiec_is_zero_when_rate_equals_nominal_and_handles_edges():
    assert stats.kupiec_pof(5, 500, 0.01)["lr"] == pytest.approx(0.0, abs=1e-12)
    assert stats.kupiec_pof(5, 500, 0.01)["p_value"] == pytest.approx(1.0)
    assert stats.kupiec_pof(0, 250, 0.01)["lr"] == pytest.approx(-2 * 250 * math.log(0.99))
    assert stats.kupiec_pof(250, 250, 0.01)["p_value"] < 1e-10
    with pytest.raises(ValueError):
        stats.kupiec_pof(3, 0, 0.01)


@given(x=st.integers(0, 60), n=st.integers(100, 2000), p=st.sampled_from([0.01, 0.05]))
def test_kupiec_rejects_more_as_the_rate_moves_away(x, n, p):
    out = stats.kupiec_pof(x, n, p)
    assert out["lr"] >= 0 and 0 <= out["p_value"] <= 1
    further = stats.kupiec_pof(min(n, x + 5), n, p) if x / n >= p else stats.kupiec_pof(max(0, x - 5), n, p)
    if (x / n >= p and x + 5 <= n) or (x / n < p and x - 5 >= 0):
        assert further["lr"] >= out["lr"] - 1e-9


def test_christoffersen_detects_clustering():
    clustered = [0] * 200 + [1] * 10 + [0] * 200
    spread = ([0] * 40 + [1]) * 10
    c, s = stats.christoffersen_independence(clustered), stats.christoffersen_independence(spread)
    assert c["n11"] == 9 and s["n11"] == 0
    assert c["p_value"] < 0.001
    assert s["p_value"] > 0.3
    assert stats.christoffersen_independence([0] * 50)["p_value"] is None
    assert stats.christoffersen_independence([])["p_value"] is None


def test_basel_zones():
    assert [stats.basel_zone(x) for x in (0, 4, 5, 9, 10, 30)] == ["green", "green", "yellow", "yellow", "red", "red"]


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000), tail=st.sampled_from([0.01, 0.05, 0.25]))
def test_pinball_loss_is_minimised_at_the_true_quantile(seed, tail):
    r = np.random.default_rng(seed).normal(0, 0.01, 4000)
    true_var = -np.quantile(r, tail)
    best = stats.pinball_loss(r, np.full(len(r), true_var), tail)
    for shift in (-0.004, -0.001, 0.001, 0.004):
        assert stats.pinball_loss(r, np.full(len(r), true_var + shift), tail) >= best - 1e-12


def test_qlike_prefers_the_true_variance():
    rng = np.random.default_rng(1)
    realised = rng.normal(0, 0.02, 20_000) ** 2
    truth = stats.qlike(realised, np.full(20_000, 0.0004))
    assert truth < stats.qlike(realised, np.full(20_000, 0.0002))
    assert truth < stats.qlike(realised, np.full(20_000, 0.0008))
    with pytest.raises(ValueError):
        stats.qlike([0.1], [0.0])


@settings(max_examples=60)
@given(st.lists(st.tuples(st.integers(-20, 20), st.integers(-20, 20)), min_size=4, max_size=40))
def test_spearman_matches_scipy(pairs):
    x, y = [p[0] for p in pairs], [p[1] for p in pairs]
    ours = stats.spearman(x, y)
    if len(set(x)) == 1 or len(set(y)) == 1:
        assert ours is None
    else:
        assert ours == pytest.approx(sps.spearmanr(x, y)[0], abs=1e-12)
    assert list(stats.rankdata(x)) == pytest.approx(list(sps.rankdata(x)))


# ---------------------------------------------------------------- exact optimiser

@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 5000), lo=st.sampled_from([0.0, 0.05]), hi=st.sampled_from([0.5, 1.0]))
def test_exact_minimum_variance_matches_an_iterative_solver(seed, lo, hi):
    mu, cov = random_problem(seed)
    lower, upper = np.full(5, lo), np.full(5, hi)
    w = qp.min_variance(cov, lower, upper)
    res = minimize(lambda x: x @ cov @ x, np.full(5, 0.2), method="SLSQP", bounds=list(zip(lower, upper)),
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], options={"ftol": 1e-14, "maxiter": 1000})
    assert w.sum() == pytest.approx(1.0)
    assert np.all(w >= lower - 1e-12) and np.all(w <= upper + 1e-12)
    assert w @ cov @ w <= res.fun + 1e-10


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 5000))
def test_unconstrained_minimum_variance_has_the_textbook_closed_form(seed):
    _, cov = random_problem(seed)
    inv = np.linalg.inv(cov)
    closed = inv @ np.ones(5) / (np.ones(5) @ inv @ np.ones(5))
    w = qp.min_variance(cov, np.full(5, -50.0), np.full(5, 50.0))
    assert w == pytest.approx(closed, abs=1e-8)


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(0, 5000), frac=st.floats(0.1, 0.9))
def test_frontier_point_matches_an_iterative_solver(seed, frac):
    mu, cov = random_problem(seed)
    lower, upper = np.full(5, 0.05), np.full(5, 0.5)
    r_lo = float(qp.min_variance(cov, lower, upper) @ mu)
    r_hi = float(qp.max_return(mu, lower, upper) @ mu)
    target = r_lo + frac * (r_hi - r_lo)
    w = qp.min_variance(cov, lower, upper, mu, target)
    res = minimize(lambda x: x @ cov @ x, w * 0 + 0.2, method="SLSQP", bounds=list(zip(lower, upper)),
                   constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1},
                                {"type": "eq", "fun": lambda x: x @ mu - target}], options={"ftol": 1e-14, "maxiter": 1000})
    assert w @ mu == pytest.approx(target, abs=1e-9)
    assert w @ cov @ w <= res.fun + 1e-9


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 5000))
def test_max_return_matches_linear_programming(seed):
    mu, _ = random_problem(seed)
    lower, upper = np.full(5, 0.05), np.full(5, 0.5)
    res = linprog(-mu, A_eq=[np.ones(5)], b_eq=[1.0], bounds=list(zip(lower, upper)))
    assert qp.max_return(mu, lower, upper) @ mu == pytest.approx(-res.fun, abs=1e-9)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 5000))
def test_max_sharpe_is_not_beaten_by_random_or_iterative_search(seed):
    mu, cov = random_problem(seed)
    lower, upper = np.full(5, 0.0), np.full(5, 1.0)
    rf = 0.02
    w = qp.max_sharpe(mu, cov, lower, upper, rf)
    best = qp.stats(w, mu, cov, rf)[2]
    cloud = pv.random_feasible(5, lower, upper, 20_000, seed)
    sharpe = (cloud @ mu - rf) / np.sqrt(np.einsum("ij,jk,ik->i", cloud, cov, cloud))
    assert sharpe.max() <= best + 1e-7
    res = minimize(lambda x: -(x @ mu - rf) / np.sqrt(x @ cov @ x), np.full(5, 0.2), method="SLSQP",
                   bounds=list(zip(lower, upper)), constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 1000})
    assert best >= -res.fun - 1e-6


def test_infeasible_target_returns_none():
    mu, cov = random_problem(3)
    assert qp.min_variance(cov, np.full(5, 0.05), np.full(5, 0.5), mu, 5.0) is None


# ---------------------------------------------------------------- ledger

def records(n, start=0):
    return [{"id": f"r{i}", "kind": "var", "value": i * 0.5, "as_of": "2026-01-01"} for i in range(start, start + n)]


def test_ledger_appends_verifies_and_deduplicates(tmp_path):
    path = str(tmp_path / "ledger" / "f.jsonl")
    assert ledger.read(path) == [] and ledger.head([]) == ledger.GENESIS
    added = ledger.append(path, records(3))
    assert [e["record"]["seq"] for e in added] == [0, 1, 2]
    assert ledger.append(path, records(4)) and len(ledger.read(path)) == 4
    assert ledger.append(path, records(4)) == []
    entries = ledger.read(path)
    assert ledger.verify(entries) == (True, None)
    assert entries[0]["prev"] == ledger.GENESIS
    assert all(entries[i]["prev"] == entries[i - 1]["hash"] for i in range(1, 4))
    assert entries[2]["hash"] == ledger.link_hash(entries[1]["hash"], entries[2]["payload"])


@pytest.mark.parametrize("attack", ["edit", "delete", "swap", "rehash_one"])
def test_ledger_detects_tampering(tmp_path, attack):
    path = str(tmp_path / "f.jsonl")
    ledger.append(path, records(5))
    lines = open(path, encoding="utf-8").read().splitlines()
    if attack == "edit":
        entry = json.loads(lines[1])
        entry["payload"] = entry["payload"].replace('"value":0.5', '"value":9.5')
        lines[1] = json.dumps(entry)
    elif attack == "delete":
        del lines[2]
    elif attack == "swap":
        lines[1], lines[3] = lines[3], lines[1]
    else:
        entry = json.loads(lines[1])
        entry["payload"] = entry["payload"].replace('"value":0.5', '"value":9.5')
        entry["hash"] = ledger.link_hash(entry["prev"], entry["payload"])
        lines[1] = json.dumps(entry)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    ok, where = ledger.verify(ledger.read(path))
    assert not ok and where in (1, 2)
    with pytest.raises(RuntimeError):
        ledger.append(path, records(1, start=10))


def test_canonical_text_is_stable_and_ids_are_deterministic():
    assert ledger.canonical({"b": 1, "a": [1.5, None]}) == '{"a":[1.5,null],"b":1}'
    assert ledger.record_id("var|x|hist|0.95|2026-01-01") == ledger.record_id("var|x|hist|0.95|2026-01-01")
    assert len(ledger.record_id("k")) == 16
    with pytest.raises(ValueError):
        ledger.canonical({"x": float("nan")})


# ---------------------------------------------------------------- forecasting models

def test_ewma_recursion_by_hand():
    r = np.array([0.01, -0.02, 0.015] + [0.0] * 40)
    out = forecasts.ewma_variance(r)
    seed = np.var(r[:30])
    v0 = 0.94 * seed + 0.06 * 0.01 ** 2
    v1 = 0.94 * v0 + 0.06 * 0.02 ** 2
    assert out[0] == pytest.approx(v0) and out[1] == pytest.approx(v1)
    assert out[10] == pytest.approx(out[9] * 0.94)


def test_var_models_recover_the_normal_quantile():
    r = np.random.default_rng(7).normal(0.0, 0.01, 250)
    for conf in (0.95, 0.99):
        z = -sps.norm.ppf(1 - conf)
        assert forecasts.var_forecast(r, "normal", conf) == pytest.approx(-(r.mean() - z * r.std(ddof=1)))
        assert forecasts.var_forecast(r, "hist", conf) == pytest.approx(-np.quantile(r, 1 - conf))
        assert forecasts.var_forecast(r, "ewma", conf) == pytest.approx(z * math.sqrt(forecasts.ewma_variance(r)[-1]))
    assert forecasts.var_forecast(r, "hist", 0.99) > forecasts.var_forecast(r, "hist", 0.95)
    with pytest.raises(ValueError):
        forecasts.var_forecast(r[:100], "hist", 0.95)


def test_var_backtest_is_calibrated_on_well_behaved_data():
    r = np.random.default_rng(11).normal(0.0003, 0.01, 3000)
    dates = [f"d{i}" for i in range(3000)]
    bt = forecasts.var_backtest(r, dates)
    assert bt["n"] == 2750
    for model in forecasts.VAR_MODELS:
        e95, e99 = bt["models"][model]["0.95"], bt["models"][model]["0.99"]
        assert 0.035 < e95["rate"] < 0.065
        assert 0.004 < e99["rate"] < 0.02
        assert e99["basel_zone"] in ("green", "yellow")
        assert e95["mean_var"] < e99["mean_var"]


def test_var_backtest_uses_no_future_information():
    r = np.random.default_rng(5).normal(0, 0.01, 600)
    shocked = r.copy()
    shocked[-1] = -0.5
    a = forecasts.var_backtest(r, [str(i) for i in range(600)])
    b = forecasts.var_backtest(shocked, [str(i) for i in range(600)])
    for model in forecasts.VAR_MODELS:
        assert a["models"][model]["0.99"]["mean_var"] == pytest.approx(b["models"][model]["0.99"]["mean_var"])
        assert b["models"][model]["0.99"]["exceedances"] >= a["models"][model]["0.99"]["exceedances"]


def test_garch_recovers_persistence_on_simulated_data():
    rng = np.random.default_rng(3)
    omega, alpha, beta = 2e-6, 0.08, 0.90
    r, s2 = np.empty(6000), omega / (1 - alpha - beta)
    for t in range(6000):
        r[t] = rng.normal() * math.sqrt(s2)
        s2 = omega + alpha * r[t] ** 2 + beta * s2
    fit = forecasts.garch_fit(r)
    assert fit["alpha"] == pytest.approx(alpha, abs=0.03)
    assert fit["beta"] == pytest.approx(beta, abs=0.04)
    assert fit["forecast_var"] > 0
    bt = forecasts.vol_backtest(r[:900], [str(i) for i in range(900)], garch_refit_every=200)
    assert set(bt["models"]) == set(forecasts.VOL_MODELS)
    assert bt["models"]["garch"]["qlike"] <= bt["models"]["rolling60"]["qlike"] + 0.05


# ---------------------------------------------------------------- validators

def dcf_case(**over):
    a = {"n_years": 10, "base_revenue": 1000.0, "growth_start": 0.12, "growth_terminal": 0.05, "high_growth_years": 3,
         "margin_start": 0.18, "margin_target": 0.20, "margin_converge_years": 5, "tax_rate": 0.2517, "da_pct": 0.03,
         "capex_pct": 0.06, "capex_pct_terminal": 0.04, "nwc_pct": 0.10, "wacc": 0.11, "terminal_ronic": 0.15,
         "ronic_tied": False, "mid_year": True, "cash": 150.0, "lt_investments": 50.0, "debt": 200.0, "minority": 10.0,
         "shares": 25.0, "cost_equity": 0.125, "years_since_fye": 0.6, "dividends_since_fye": [[0.3, 2.0]]}
    a.update(over)
    return a


def plain_dcf(a):
    """A third, deliberately naive implementation: a year-by-year loop over free cash flow."""
    rev, total, w, n = a["base_revenue"], 0.0, a["wacc"], a["n_years"]
    for t in range(1, n + 1):
        h = a["high_growth_years"]
        f = 0.0 if t <= h else (t - h) / (n - h)
        g = a["growth_start"] + (a["growth_terminal"] - a["growth_start"]) * f
        cx = a["capex_pct"] + (a["capex_pct_terminal"] - a["capex_pct"]) * f
        m = a["margin_start"] + (a["margin_target"] - a["margin_start"]) * min(t, a["margin_converge_years"]) / a["margin_converge_years"]
        new = rev * (1 + g)
        nopat = new * m * (1 - a["tax_rate"])
        fcff = nopat + new * a["da_pct"] - new * cx - a["nwc_pct"] * (new - rev)
        total += fcff / (1 + w) ** (t - 0.5 if a["mid_year"] else t)
        rev = new
    ronic = w if a["ronic_tied"] else a["terminal_ronic"]
    tv = nopat * (1 + a["growth_terminal"]) * (1 - a["growth_terminal"] / ronic) / (w - a["growth_terminal"])
    return total + tv / (1 + w) ** n * ((1 + w) ** 0.5 if a["mid_year"] else 1.0)


@settings(max_examples=100, deadline=None)
@given(g=st.floats(0.0, 0.25), gt=st.floats(0.01, 0.06), spread=st.floats(0.02, 0.10), ronic=st.floats(0.06, 0.40),
       tied=st.booleans(), mid=st.booleans(), capital=st.floats(0.0, 5000.0))
def test_economic_profit_equals_discounted_cash_flow(g, gt, spread, ronic, tied, mid, capital):
    a = dcf_case(growth_start=g, growth_terminal=gt, wacc=gt + spread, terminal_ronic=ronic, ronic_tied=tied, mid_year=mid)
    ep = vv.economic_profit_value(a, capital)["enterprise_value"]
    assert ep == pytest.approx(plain_dcf(a), rel=1e-9)


def test_economic_profit_per_share_bridge():
    a = dcf_case(dividends_since_fye=[], years_since_fye=0.0)
    out = vv.economic_profit_value(a, 300.0)
    assert out["per_share"] == pytest.approx((out["enterprise_value"] + 150 + 50 - 200 - 10) / 25)


def test_risk_metrics_against_direct_numpy_formulas():
    rng = np.random.default_rng(2)
    bench = rng.normal(0.0004, 0.01, 1500)
    r = 0.0001 + 0.6 * bench + rng.normal(0, 0.004, 1500)
    rf = np.full(1500, 0.0002)
    m = pv.risk_metrics(r, rf, bench)
    assert m["var_hist_1d"] == pytest.approx(-np.quantile(r, 0.05))
    assert m["vol_ann"] == pytest.approx(r.std(ddof=1) * math.sqrt(252))
    assert m["beta"] == pytest.approx(np.cov(r - rf, bench - rf, ddof=1)[0, 1] / np.var(bench - rf, ddof=1))
    assert m["sharpe"] == pytest.approx((r - rf).mean() * 252 / (r.std(ddof=1) * math.sqrt(252)))
    assert m["tracking_error"] == pytest.approx(np.std(r - bench, ddof=1) * math.sqrt(252))
    assert m["cvar_hist_1d"] >= m["var_hist_1d"] and m["cvar_param_1d"] >= m["var_param_1d"]
    wealth = np.cumprod(1 + r)
    assert m["max_drawdown"] == pytest.approx((wealth / np.maximum.accumulate(wealth) - 1).min())
    assert pv.risk_metrics(bench, rf, bench)["beta"] == pytest.approx(1.0)
    assert pv.risk_metrics(bench, rf, bench)["information_ratio"] is None


def test_random_feasible_portfolios_respect_the_constraints():
    lower, upper = np.full(5, 0.05), np.full(5, 0.5)
    w = pv.random_feasible(5, lower, upper, 5000, 1)
    assert w.shape == (5000, 5)
    assert np.allclose(w.sum(axis=1), 1.0) and (w >= lower - 1e-12).all() and (w <= upper + 1e-12).all()


# ---------------------------------------------------------------- the run loop on synthetic data

def synthetic_sources(n_days):
    rng = np.random.default_rng(9)
    keys = ["A", "B"]
    dates = [f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(n_days)]
    series = {k: list(rng.normal(0.0004, 0.01, n_days)) for k in keys}
    series["BENCH"] = list(rng.normal(0.0004, 0.011, n_days))
    pdata = {"assets": [{"key": k} for k in keys], "series": {"dates": dates, "returns": series},
             "crosscheck": {"mix": {"weights": [0.6, 0.4]}, "benchmark": {}}}
    vdata = {"meta": {"valuation_date": "2026-11-11"},
             "companies": [{"ticker": t, "sector": "s", "price": 100.0 + i, "dcf": {"upside": 0.1 * i - 0.2, "result": {"per_share": 90.0 + 12 * i}},
                            "comps": {"valuation": {"ev_ebitda": {"implied_mid": 95.0}}},
                            "prices": {"dates": ["2026-11-13", "2026-12-11"], "close": [100.0 + i, 100.0 + i * 5]}}
                           for i, t in enumerate(["X", "Y", "Z", "W"])]}
    return pdata, {"sha256": "p" * 64}, vdata, {"sha256": "v" * 64}


def test_forecasts_are_recorded_once_and_scored_only_when_the_outcome_exists(tmp_path):
    fpath, opath = str(tmp_path / "f.jsonl"), str(tmp_path / "o.jsonl")
    pdata, pmeta, vdata, vmeta = synthetic_sources(300)
    recs = run.new_forecasts(pdata, pmeta, vdata, vmeta, [], "2026-11-11T13:00:00Z")
    assert len([r for r in recs if r["kind"] == "var"]) == 2 * 3 * 2
    assert len([r for r in recs if r["kind"] == "vol"]) == 3
    assert len([r for r in recs if r["kind"] == "valuation"]) == 4
    ledger.append(fpath, recs)
    fc = ledger.read(fpath)
    again = run.new_forecasts(pdata, pmeta, vdata, vmeta, fc, "2026-11-11T14:00:00Z")
    assert ledger.append(fpath, again) == []
    risk = lambda rows: [o for o in rows if o["kind"] in ("var", "vol")]
    assert risk(run.new_outcomes(fc, set(), pdata, pmeta, vdata, vmeta, "t")) == []

    # one more session arrives: the risk forecasts mature
    longer, _, _, _ = synthetic_sources(301)
    out = risk(run.new_outcomes(fc, set(), longer, pmeta, vdata, vmeta, "t"))
    assert len(out) == 15 and all(o["target_date"] == longer["series"]["dates"][300] for o in out)
    var = next(o for o in out if o["kind"] == "var")
    f = next(e["record"] for e in fc if e["record"]["id"] == var["id"])
    realised = 0.6 * longer["series"]["returns"]["A"][300] + 0.4 * longer["series"]["returns"]["B"][300] \
        if f["weights"] else longer["series"]["returns"]["BENCH"][300]
    assert var["realised"] == pytest.approx(realised, rel=1e-9)
    assert var["exceeded"] == (realised < -f["value"])
    ledger.append(opath, out)
    oc = ledger.read(opath)
    assert risk(run.new_outcomes(fc, {e["record"]["id"] for e in oc}, longer, pmeta, vdata, vmeta, "t")) == []
    cards = run.scorecards(fc, oc)
    assert len(cards["var"]) == 12 and all(r["n"] == 1 for r in cards["var"])
    assert len(cards["vol"]) == 3 and cards["valuation"] == []


def test_valuation_cohort_is_scored_at_its_horizon(tmp_path):
    pdata, pmeta, vdata, vmeta = synthetic_sources(300)
    fpath = str(tmp_path / "f.jsonl")
    ledger.append(fpath, run.new_forecasts(pdata, pmeta, vdata, vmeta, [], "t"))
    fc = ledger.read(fpath)
    # Calls made 11 Nov; four weeks later is 9 Dec and every company has a close on 11 Dec.
    out = [o for o in run.new_outcomes(fc, set(), pdata, pmeta, vdata, vmeta, "t") if o["kind"] == "valuation"]
    assert len(out) == 4 and {o["horizon_weeks"] for o in out} == {4}
    assert all(o["target_date"] == "2026-12-11" for o in out)
    rets = [(100.0 + 5 * i) / (100.0 + i) - 1.0 for i in range(4)]
    by_forecast = {o["forecast_id"]: o for o in out}
    for i, t in enumerate(["X", "Y", "Z", "W"]):
        f = next(e["record"] for e in fc if e["record"]["kind"] == "valuation" and e["record"]["subject"] == t)
        o = by_forecast[f["id"]]
        assert o["realised"] == pytest.approx(rets[i], rel=1e-9)
        assert o["universe"] == pytest.approx(np.mean(rets), rel=1e-9)
        assert o["relative"] == pytest.approx(rets[i] - np.mean(rets), rel=1e-8)
    opath = str(tmp_path / "o.jsonl")
    ledger.append(opath, out)
    oc = ledger.read(opath)
    card = run.scorecards(fc, oc)["valuation"]
    assert len(card) == 1 and card[0]["rank_ic"] == pytest.approx(1.0) and card[0]["hit_rate"] == 1.0
    assert [o for o in run.new_outcomes(fc, {e["record"]["id"] for e in oc}, pdata, pmeta, vdata, vmeta, "t")
            if o["kind"] == "valuation"] == []

    # A cohort is only scored when every member has a price at the horizon.
    vdata["companies"][0]["prices"]["dates"] = ["2026-11-13"]
    vdata["companies"][0]["prices"]["close"] = [100.0]
    assert [o for o in run.new_outcomes(fc, set(), pdata, pmeta, vdata, vmeta, "t") if o["kind"] == "valuation"] == []
