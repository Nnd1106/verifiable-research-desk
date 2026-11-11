"""valuation_validation.py: independent re-validation of the Valuation & Deal
Lab from its published data file.

The lab's engine is not imported. Each model is recomputed here with
separately written code and, where one exists, by a different method:

  DCF      valued again as invested capital plus the present value of
           economic profit (NOPAT less a capital charge). Discounted cash flow
           and economic profit are algebraically equivalent, so two
           independent implementations must give the same enterprise value;
  WACC     rebuilt from its published components;
  comps    multiples rebuilt from the published statements and prices, peer
           quartiles from numpy's percentile routine;
  LBO      the published schedule is audited line by line and the IRR is
           recomputed from the equity cash flows;
  M&A      earnings per share recomputed from a closed-form expression.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

REL_TOL = 1e-9


def _rel(a: Optional[float], b: Optional[float]) -> float:
    if a is None or b is None:
        return 0.0 if a is None and b is None else float("inf")
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


class Collector:
    """Aggregates one check across all companies, keeping the worst case."""

    def __init__(self) -> None:
        self.items: Dict[str, Dict] = {}
        self.order: List[str] = []

    def add(self, group: str, title: str, detail: str, tolerance: float, diff: float, subject: str) -> None:
        key = group + "|" + title
        if key not in self.items:
            self.items[key] = {"group": group, "title": title, "detail": detail, "tolerance": tolerance,
                               "diff": 0.0, "worst": None, "n": 0, "level": "check"}
            self.order.append(key)
        item = self.items[key]
        item["n"] += 1
        if diff > item["diff"] or item["worst"] is None:
            item["diff"], item["worst"] = float(diff), subject

    def results(self) -> List[Dict]:
        out = []
        for key in self.order:
            item = dict(self.items[key])
            item["ok"] = bool(item["diff"] <= item["tolerance"])
            item["detail"] = f"{item['detail']} ({item['n']} cases; largest difference at {item['worst']})"
            item["published"] = item["challenger"] = None
            out.append(item)
        return out


# ---------------------------------------------------------------- DCF by economic profit

def project(a: Dict) -> Dict[str, np.ndarray]:
    """Operating forecast from the published assumptions (vectorised)."""
    n = int(a["n_years"])
    t = np.arange(1, n + 1)
    hold = min(max(int(a["high_growth_years"]), 0), n - 1)
    fade = np.clip((t - hold) / (n - hold), 0.0, 1.0)
    growth = a["growth_start"] + (a["growth_terminal"] - a["growth_start"]) * fade
    capex_pct = a["capex_pct"] + (a["capex_pct_terminal"] - a["capex_pct"]) * fade
    k = max(1, int(a["margin_converge_years"]))
    margin = a["margin_start"] + (a["margin_target"] - a["margin_start"]) * np.minimum(t, k) / k
    revenue = a["base_revenue"] * np.cumprod(1.0 + growth)
    previous = np.concatenate([[a["base_revenue"]], revenue[:-1]])
    nopat = revenue * margin * (1.0 - a["tax_rate"])
    net_investment = revenue * capex_pct - revenue * a["da_pct"] + a["nwc_pct"] * (revenue - previous)
    return {"t": t, "revenue": revenue, "nopat": nopat, "net_investment": net_investment,
            "fcff": nopat - net_investment, "ebitda": revenue * (margin + a["da_pct"])}


def economic_profit_value(a: Dict, opening_capital: float) -> Dict[str, float]:
    """Enterprise value as opening invested capital plus discounted economic profit.

    Economic profit = NOPAT - WACC x opening capital of the year. The
    continuing value is the economic-profit form of the value-driver formula.
    The answer does not depend on the opening capital figure: a higher
    starting capital is offset exactly by higher capital charges.
    """
    p = project(a)
    w, g = a["wacc"], a["growth_terminal"]
    ronic = w if a["ronic_tied"] else a["terminal_ronic"]
    n = len(p["t"])
    capital = opening_capital + np.cumsum(p["net_investment"])
    opening = np.concatenate([[opening_capital], capital[:-1]])
    profit = p["nopat"] - w * opening
    pv_profit = float(np.sum(profit / (1.0 + w) ** p["t"]))
    nopat_next = p["nopat"][-1] * (1.0 + g)
    profit_next = nopat_next - w * capital[-1]
    continuing = profit_next / w + nopat_next * (g / ronic) * (ronic - w) / (w * (w - g))
    ev_end_year = opening_capital + pv_profit + continuing / (1.0 + w) ** n
    ev = ev_end_year * ((1.0 + w) ** 0.5 if a["mid_year"] else 1.0)
    equity = ev + a["cash"] + a["lt_investments"] - a["debt"] - a["minority"]
    per_share = equity / a["shares"] * (1.0 + a["cost_equity"]) ** a["years_since_fye"]
    for paid, amount in a["dividends_since_fye"]:
        if 0.0 <= paid <= a["years_since_fye"]:
            per_share -= amount * (1.0 + a["cost_equity"]) ** (a["years_since_fye"] - paid)
    return {"enterprise_value": float(ev), "per_share": float(per_share)}


# ---------------------------------------------------------------- the validation

def validate(data: Dict) -> Dict:
    col = Collector()
    mkt = data["market"]
    tax = mkt["tax_rate"]
    companies = data["companies"]
    by_ticker = {c["ticker"]: c for c in companies}
    rows = []

    # sector betas rebuilt from each company's regression beta
    sector_beta = {}
    for s in data["sectors"]:
        unlevered = [by_ticker[t]["beta"]["raw"] / (1 + (1 - tax) * by_ticker[t]["latest_bs"]["debt"] / by_ticker[t]["market_cap"])
                     for t in s["tickers"]]
        sector_beta[s["key"]] = float(np.median(unlevered))
        col.add("Cost of capital", "Sector beta is the median of its peers' unlevered betas",
                "each regression beta unlevered at the company's own debt-to-equity", REL_TOL,
                _rel(sector_beta[s["key"]], s["unlevered_beta"]), s["name"])

    for c in companies:
        t, a, k, bs = c["ticker"], c["dcf"]["assumptions"], c["cost"], c["latest_bs"]

        # ---- statements
        worst = max(abs(r["total_assets"] - r["total_liabilities"] - r["equity"] - r["minority"]) / r["total_assets"]
                    for r in c["annual"])
        col.add("Statements", "Balance sheets balance", "assets = liabilities + equity + minorities, every fiscal year",
                0.01, worst, t)
        worst = max(max(_rel(r["ebitda"], r["ebit"] + r["da"]),
                        _rel(r["invested_capital"], r["equity"] + r["minority"] + r["debt"] - r["cash"] - r["lt_investments"]),
                        _rel(r["nwc"], (r["current_assets"] - r["cash"]) - (r["current_liabilities"] - r["current_debt"])))
                    for r in c["annual"])
        col.add("Statements", "Derived lines follow their definitions",
                "EBITDA, invested capital and working capital rebuilt from the reported lines", REL_TOL, worst, t)
        col.add("Statements", "Market cap is price times shares", "share price x shares outstanding", REL_TOL,
                _rel(c["market_cap"], c["price"] * c["shares"]), t)

        # ---- cost of capital
        rf = mkt["gsec_10y"] - mkt["sovereign_spread"]
        de = bs["debt"] / c["market_cap"]
        beta = sector_beta[c["sector"]] * (1 + (1 - tax) * de)
        ke = rf + beta * mkt["erp"]
        cov = c["history"]["interest_coverage"]
        spread = mkt["rating_table"][0][2]
        if cov is not None:
            spread = mkt["rating_table"][-1][2]
            for lower, _rating, sp in mkt["rating_table"]:
                if lower is None or cov >= lower:
                    spread = sp
                    break
        kd = (mkt["gsec_10y"] + spread) * (1 - tax)
        e_w = c["market_cap"] / (c["market_cap"] + bs["debt"])
        wacc = e_w * ke + (1 - e_w) * kd
        col.add("Cost of capital", "Cost of equity from CAPM", "risk-free rate + relevered sector beta x equity risk premium",
                REL_TOL, _rel(ke, k["cost_equity"]), t)
        col.add("Cost of capital", "After-tax cost of debt from the synthetic rating",
                "G-sec yield + rating spread on interest coverage, after tax", REL_TOL, _rel(kd, k["after_tax_cost_debt"]), t)
        col.add("Cost of capital", "WACC from its components", "market-value weights on equity and debt", REL_TOL,
                _rel(wacc, k["wacc"]), t)

        # ---- DCF by a second method
        ep = economic_profit_value(a, c["annual"][-1]["invested_capital"])
        res = c["dcf"]["result"]
        col.add("DCF, by economic profit", "Enterprise value: economic profit agrees with discounted cash flow",
                "invested capital + PV of (NOPAT - capital charge), against the lab's PV of free cash flow",
                REL_TOL, _rel(ep["enterprise_value"], res["enterprise_value"]), t)
        col.add("DCF, by economic profit", "Value per share after the equity bridge and roll-forward",
                "net of debt and minorities, plus cash and investments, rolled to the valuation date",
                REL_TOL, _rel(ep["per_share"], res["per_share"]), t)
        other_capital = economic_profit_value(a, 0.0)
        col.add("DCF, by economic profit", "Result does not depend on the opening capital figure",
                "the same valuation started from zero invested capital", 1e-8,
                _rel(other_capital["enterprise_value"], ep["enterprise_value"]), t)
        ig = c["dcf"]["implied_growth"]
        if ig is not None:
            repriced = economic_profit_value(dict(a, growth_start=ig), 0.0)["per_share"]
            col.add("DCF, by economic profit", "Reverse DCF: the implied growth rate reprices the share to market",
                    "economic-profit valuation at the lab's implied growth vs the market price", 1e-6,
                    _rel(repriced, c["price"]), t)
        im = c["dcf"]["implied_margin"]
        if im is not None:
            repriced = economic_profit_value(dict(a, margin_target=im), 0.0)["per_share"]
            col.add("DCF, by economic profit", "Reverse DCF: the implied margin reprices the share to market",
                    "economic-profit valuation at the lab's implied margin vs the market price", 1e-6,
                    _rel(repriced, c["price"]), t)

        # ---- comps
        tr, m = c["trailing"], c["comps"]["multiples"]
        ev = c["price"] * c["shares"] + bs["debt"] + bs["minority"] - bs["cash"] - bs["lt_investments"]
        mine = {"ev_ebitda": ev / tr["ebitda"] if tr["ebitda"] > 0 and ev > 0 else None,
                "ev_ebit": ev / tr["ebit"] if tr["ebit"] > 0 and ev > 0 else None,
                "ev_sales": ev / tr["revenue"] if tr["revenue"] > 0 and ev > 0 else None,
                "pe": c["price"] / tr["eps"] if tr["eps"] and tr["eps"] > 0 else None,
                "pb": c["price"] / (bs["equity"] / c["shares"]) if bs["equity"] > 0 else None}
        col.add("Trading comps", "Enterprise value from the balance sheet",
                "market cap + debt + minorities - cash - financial investments", REL_TOL, _rel(ev, m["enterprise_value"]), t)
        col.add("Trading comps", "Five multiples from statements and price", "EV/EBITDA, EV/EBIT, EV/Sales, P/E, P/B",
                REL_TOL, max(_rel(mine[x], m[x]) for x in mine), t)
        worst = 0.0
        for name, block in c["comps"]["valuation"].items():
            if block is None:
                continue
            peer_vals = [by_ticker[p]["comps"]["multiples"][name] for p in c["comps"]["peers"]]
            peer_vals = [v for v in peer_vals if v is not None]
            q1, med, q3 = np.percentile(peer_vals, [25, 50, 75])
            worst = max(worst, _rel(q1, block["stats"]["q1"]), _rel(med, block["stats"]["median"]),
                        _rel(q3, block["stats"]["q3"]))
        col.add("Trading comps", "Peer quartiles and medians", "numpy percentiles of the peer group's multiples",
                REL_TOL, worst, t)

        # ---- LBO audit
        lbo = c["lbo"]
        if lbo["result"]:
            la, r = lbo["assumptions"], lbo["result"]
            su, sched = r["sources_uses"], r["rows"]
            col.add("LBO audit", "Sources equal uses", "debt + sponsor equity = enterprise value + fees", REL_TOL,
                    _rel(su["total_sources"], su["total_uses"]), t)
            worst, cash, senior = 0.0, 0.0, su["senior_debt"]
            for x in sched:
                revolver_open = max(0.0, -cash)
                interest = (la["senior_rate"] * 0.5 * (x["senior_beg"] + x["senior_end"]) if la["interest_on_average"]
                            else la["senior_rate"] * x["senior_beg"])
                interest += la["sub_rate"] * x["sub_debt"] + la["senior_rate"] * revolver_open
                tax_due = max(x["ebit"] - x["interest"], 0.0) * la["tax_rate"]
                cfads = x["ebit"] - x["interest"] - tax_due + x["da"] - x["capex"] - x["delta_nwc"]
                cash = cash + x["cfads"] - x["mandatory"] - x["sweep"]
                worst = max(worst, _rel(interest, x["interest"]), _rel(tax_due, x["tax"]), _rel(cfads, x["cfads"]),
                            _rel(senior, x["senior_beg"]), _rel(x["senior_beg"] - x["mandatory"] - x["sweep"], x["senior_end"]),
                            abs(cash - x["cash_end"]) / max(1.0, su["total_uses"]))
                senior = x["senior_end"]
            col.add("LBO audit", "Debt schedule audited line by line",
                    "interest on average balances, tax, cash available for debt service, repayments and cash",
                    1e-9, worst, t)
            last = sched[-1]
            exit_equity = la["exit_multiple"] * last["ebitda"] - (last["senior_end"] + last["sub_debt"] - last["cash_end"])
            col.add("LBO audit", "Exit equity", "exit multiple x final-year EBITDA less net debt", REL_TOL,
                    _rel(exit_equity, r["exit_equity"]), t)
            if r["irr"] is not None:
                irr = (r["exit_equity"] / su["sponsor_equity"]) ** (1.0 / la["hold_years"]) - 1.0
                col.add("LBO audit", "IRR from the equity cash flows", "(exit equity / equity invested)^(1/years) - 1",
                        1e-9, abs(irr - r["irr"]), t)
            col.add("LBO audit", "Value-creation bridge sums to the equity gain",
                    "EBITDA growth + multiple change + debt paydown - fees", 1e-9,
                    abs(sum(r["attribution"].values()) - r["value_creation"]) / max(1.0, abs(r["value_creation"])), t)

        ig_gap = None if ig is None else ig - a["growth_start"]
        rows.append({
            "ticker": t, "name": c["name"], "sector": c["sector"], "price": c["price"],
            "ev_published": res["enterprise_value"], "ev_challenger": ep["enterprise_value"],
            "ev_rel_diff": _rel(ep["enterprise_value"], res["enterprise_value"]),
            "per_share_published": res["per_share"], "per_share_challenger": ep["per_share"],
            "upside": c["dcf"]["upside"], "terminal_share": res["terminal_share"],
            "implied_growth": ig, "model_growth": a["growth_start"], "implied_growth_gap": ig_gap,
            "wacc": k["wacc"], "warnings": len(c["warnings"]),
        })

    # ---- M&A presets by closed form
    for p in data["merger_presets"]:
        a, r = p["assumptions"], p["result"]
        price = a["tgt_price"] * (1 + a["premium"]) * a["tgt_shares"]
        cash_needed = price * a["pct_cash"] + price * a["fee_pct"]
        from_cash = cash_needed * a["cash_from_balance_sheet_pct"]
        intangibles = max(0.0, price - a["tgt_book_equity"]) * a["intangibles_pct"]
        keep = 1 - a["tax_rate"]
        income = (a["acq_net_income"] + a["tgt_net_income"] + a["synergies_pretax"] * a["synergy_phase"] * keep
                  - (cash_needed - from_cash) * a["cost_of_debt"] * keep - from_cash * a["yield_on_cash"] * keep
                  - intangibles / a["amort_years"] * keep)
        shares = a["acq_shares"] + price * (1 - a["pct_cash"]) / a["acq_price"]
        accretion = (income / shares) / (a["acq_net_income"] / a["acq_shares"]) - 1
        col.add("M&A", "EPS accretion from a closed-form expression", "pro forma net income / pro forma shares vs standalone EPS",
                1e-9, abs(accretion - r["accretion"]), f"{p['acquirer']}+{p['target']}")
        col.add("M&A", "Purchase price is fully allocated", "book equity + intangibles + goodwill = price paid", REL_TOL,
                _rel(a["tgt_book_equity"] + r["intangibles"] + r["goodwill"], price), f"{p['acquirer']}+{p['target']}")

    # ---- model-risk observations (not pass/fail)
    below = [r for r in rows if r["upside"] < 0]
    heavy_tv = [r["ticker"] for r in rows if r["terminal_share"] > 0.75]
    no_value = [r["ticker"] for r in rows if r["per_share_published"] <= 0]
    stretched = [r["ticker"] for r in rows if r["implied_growth"] is None or r["implied_growth"] > 0.30]
    observations = [
        {"title": "Most companies screen below their market price",
         "text": f"{len(below)} of {len(rows)} DCF baselines are below market (median gap "
                 f"{float(np.median([r['upside'] for r in rows])):.0%}). The defaults are conservative by construction "
                 "(history-driven forecasts, no excess returns after year ten), so this measures expectations in prices "
                 "more than mispricing. It should not be read as a set of sell calls."},
        {"title": "Terminal value dominates some valuations",
         "text": (f"Terminal value is over 75% of enterprise value for {', '.join(heavy_tv)}. " if heavy_tv
                  else "No company has a terminal value above 75% of enterprise value. ") +
                 "Where it does, the explicit forecast produces little cash and the result leans on the continuing-value assumptions."},
        {"title": "The price implies growth far outside history for some names",
         "text": (f"Reconciling to market needs more than 30% a year, or no growth rate in the tested range, for {', '.join(stretched)}. "
                  if stretched else "No company needs more than 30% a year to reconcile to its price. ") +
                 "For these the single-driver reverse DCF is a blunt tool: margin expansion or a lower discount rate is the more plausible market view."},
        {"title": "Inputs that are reviewed by hand",
         "text": "The G-sec yield and the equity risk premium are fixed in configuration with a source and a date. " +
                 ("Current warnings: " + "; ".join(data["quality"]["assumption_warnings"]) + "."
                  if data["quality"]["assumption_warnings"] else "Neither is past its review date.")},
    ]
    if no_value:
        observations.append({"title": "Non-positive DCF values",
                             "text": f"The baseline gives no positive equity value for {', '.join(no_value)}; the model is not usable for them at default settings."})

    return {"checks": col.results(), "rows": rows, "observations": observations,
            "valuation_date": data["meta"]["valuation_date"],
            "lab_quality": data["quality"]}
