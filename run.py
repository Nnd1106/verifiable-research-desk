"""run.py: one desk run.

    python run.py                 validate both labs, record today's forecasts,
                                  score matured ones, publish docs/data
    python run.py --assert-clean  exit 1 if the last published run has a failed
                                  check or a broken ledger (used by CI after the
                                  results are committed, so a failure is published
                                  first and then turns the workflow red)

The desk publishes failures instead of hiding them: a failed validation is a
finding, and withholding the page would defeat the purpose.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from desk import __version__, forecasts, ledger, portfolio_validation, sources, stats, valuation_validation  # noqa: E402

DOCS = os.path.join(ROOT, "docs")
FORECASTS = os.path.join(DOCS, "ledger", "forecasts.jsonl")
OUTCOMES = os.path.join(DOCS, "ledger", "outcomes.jsonl")
OUT_JSON = os.path.join(DOCS, "data", "desk-data.json")
OUT_JS = os.path.join(DOCS, "data", "desk-data.js")
VALUATION_HORIZONS_WEEKS = (4, 13, 26, 52)


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def round_sig(x: float, digits: int = 10) -> float:
    """Ledger values are stored to 10 significant digits so the record is
    stable text, independent of the last bits of a floating-point result."""
    return float(f"{x:.{digits}g}")


# ---------------------------------------------------------------- forecasts to record

def subjects(pdata: Dict) -> Dict[str, Dict]:
    """The reference portfolios whose risk is forecast."""
    keys = [a["key"] for a in pdata["assets"]]
    series = pdata["series"]["returns"]
    out = {}
    for name, entry in pdata["crosscheck"].items():
        if "weights" in entry:
            out[name] = {"weights": entry["weights"],
                         "returns": forecasts.portfolio_returns(series, keys, entry["weights"])}
        else:
            out[name] = {"weights": None, "returns": np.asarray(series["BENCH"], float)}
    return out


def new_forecasts(pdata: Dict, pmeta: Dict, vdata: Dict, vmeta: Dict, existing: List[Dict], now: str) -> List[Dict]:
    records = []
    as_of = pdata["series"]["dates"][-1]
    subj = subjects(pdata)
    for name, s in subj.items():
        for model in forecasts.VAR_MODELS:
            for conf in forecasts.CONFIDENCES:
                key = f"var|{name}|{model}|{conf:.2f}|{as_of}"
                records.append({
                    "id": ledger.record_id(key), "key": key, "kind": "var", "subject": name, "model": model,
                    "confidence": conf, "as_of": as_of, "target": "next trading session",
                    "value": round_sig(forecasts.var_forecast(s["returns"], model, conf)),
                    "weights": s["weights"], "made_utc": now, "source_sha256": pmeta["sha256"],
                })
    bench = subj["benchmark"]["returns"]
    for model in forecasts.VOL_MODELS:
        key = f"vol|benchmark|{model}|{as_of}"
        records.append({
            "id": ledger.record_id(key), "key": key, "kind": "vol", "subject": "benchmark", "model": model,
            "as_of": as_of, "target": "next trading session",
            "value": round_sig(forecasts.vol_forecast(bench, model)),
            "made_utc": now, "source_sha256": pmeta["sha256"],
        })

    # One cohort of valuation calls per ISO week.
    v_date = vdata["meta"]["valuation_date"]
    iso = date.fromisoformat(v_date).isocalendar()
    cohort = f"{iso[0]}-W{iso[1]:02d}"
    have = {e["record"].get("cohort") for e in existing if e["record"]["kind"] == "valuation"}
    if cohort not in have:
        for c in vdata["companies"]:
            key = f"valuation|{c['ticker']}|{cohort}"
            peer = c["comps"]["valuation"].get("ev_ebitda")
            records.append({
                "id": ledger.record_id(key), "key": key, "kind": "valuation", "subject": c["ticker"],
                "model": "dcf_baseline", "cohort": cohort, "as_of": v_date, "sector": c["sector"],
                "target": "relative return over 4, 13, 26 and 52 weeks",
                "value": round_sig(c["dcf"]["upside"]),
                "price": round_sig(c["price"]), "dcf_value": round_sig(c["dcf"]["result"]["per_share"]),
                "peer_value": None if not peer or peer["implied_mid"] is None else round_sig(peer["implied_mid"]),
                "made_utc": now, "source_sha256": vmeta["sha256"],
            })
    return records


# ---------------------------------------------------------------- scoring matured forecasts

def new_outcomes(fc: List[Dict], scored: set, pdata: Dict, pmeta: Dict, vdata: Dict, vmeta: Dict, now: str) -> List[Dict]:
    out = []
    dates = pdata["series"]["dates"]
    keys = [a["key"] for a in pdata["assets"]]
    series = pdata["series"]["returns"]
    index_of = {d: i for i, d in enumerate(dates)}

    for e in fc:
        r = e["record"]
        if r["kind"] not in ("var", "vol") or r["id"] in scored:
            continue
        i = index_of.get(r["as_of"])
        if i is None or i + 1 >= len(dates):
            continue
        if r.get("weights"):
            realised = float(sum(w * series[k][i + 1] for k, w in zip(keys, r["weights"])))
        else:
            realised = float(series["BENCH"][i + 1])
        rec = {"id": r["id"], "kind": r["kind"], "target_date": dates[i + 1], "realised": round_sig(realised),
               "scored_utc": now, "source_sha256": pmeta["sha256"]}
        if r["kind"] == "var":
            rec["exceeded"] = bool(realised < -r["value"])
        out.append(rec)

    # valuation cohorts: scored together once every member has a price at the horizon
    closes = {c["ticker"]: dict(zip(c["prices"]["dates"], c["prices"]["close"])) for c in vdata["companies"]}
    cohorts: Dict[str, List[Dict]] = {}
    for e in fc:
        if e["record"]["kind"] == "valuation":
            cohorts.setdefault(e["record"]["cohort"], []).append(e["record"])
    for cohort, members in cohorts.items():
        start = date.fromisoformat(members[0]["as_of"])
        for weeks in VALUATION_HORIZONS_WEEKS:
            due = (start + timedelta(weeks=weeks)).isoformat()
            if all(f"{m['id']}|{weeks}w" in scored for m in members):
                continue
            exits = {}
            for m in members:
                later = sorted(d for d in closes.get(m["subject"], {}) if d >= due)
                if later:
                    exits[m["subject"]] = (later[0], closes[m["subject"]][later[0]])
            if len(exits) < len(members):
                continue
            rets = {m["subject"]: exits[m["subject"]][1] / m["price"] - 1.0 for m in members}
            universe = float(np.mean(list(rets.values())))
            for m in members:
                rec_id = f"{m['id']}|{weeks}w"
                if rec_id in scored:
                    continue
                out.append({"id": rec_id, "forecast_id": m["id"], "kind": "valuation", "cohort": cohort,
                            "horizon_weeks": weeks, "target_date": exits[m["subject"]][0],
                            "realised": round_sig(rets[m["subject"]]), "universe": round_sig(universe),
                            "relative": round_sig(rets[m["subject"]] - universe),
                            "scored_utc": now, "source_sha256": vmeta["sha256"]})
    return out


def scorecards(fc: List[Dict], oc: List[Dict]) -> Dict:
    """Live track record: only forecasts whose outcomes are in the ledger."""
    by_id = {e["record"]["id"]: e["record"] for e in fc}
    var_groups: Dict[str, Dict] = {}
    vol_groups: Dict[str, Dict] = {}
    val_groups: Dict[str, List] = {}
    for e in oc:
        o = e["record"]
        if o["kind"] == "var":
            f = by_id[o["id"]]
            g = var_groups.setdefault(f"{f['subject']}|{f['model']}|{f['confidence']:.2f}",
                                      {"subject": f["subject"], "model": f["model"], "confidence": f["confidence"],
                                       "realised": [], "var": [], "hits": []})
            g["realised"].append(o["realised"]); g["var"].append(f["value"]); g["hits"].append(int(o["exceeded"]))
        elif o["kind"] == "vol":
            f = by_id[o["id"]]
            g = vol_groups.setdefault(f["model"], {"model": f["model"], "realised": [], "forecast": []})
            g["realised"].append(o["realised"] ** 2); g["forecast"].append(f["value"])
        else:
            f = by_id[o["forecast_id"]]
            val_groups.setdefault(f"{o['cohort']}|{o['horizon_weeks']}", []).append((f["value"], o["relative"]))
    var_rows = []
    for g in var_groups.values():
        n, x = len(g["hits"]), int(sum(g["hits"]))
        tail = 1 - g["confidence"]
        k = stats.kupiec_pof(x, n, tail)
        var_rows.append({"subject": g["subject"], "model": g["model"], "confidence": g["confidence"], "n": n,
                         "exceedances": x, "expected": k["expected"], "kupiec_p": k["p_value"],
                         "pinball": stats.pinball_loss(g["realised"], g["var"], tail)})
    vol_rows = [{"model": g["model"], "n": len(g["realised"]), "qlike": stats.qlike(g["realised"], g["forecast"])}
                for g in vol_groups.values()]
    val_rows = []
    for key, pairs in val_groups.items():
        cohort, weeks = key.split("|")
        ups, rels = [p[0] for p in pairs], [p[1] for p in pairs]
        med = float(np.median(ups))
        hits = [int((u > med) == (r > 0)) for u, r in pairs]
        val_rows.append({"cohort": cohort, "horizon_weeks": int(weeks), "n": len(pairs),
                         "rank_ic": stats.spearman(ups, rels), "hit_rate": float(np.mean(hits))})
    return {"var": sorted(var_rows, key=lambda r: (r["subject"], r["confidence"], r["model"])),
            "vol": sorted(vol_rows, key=lambda r: r["model"]),
            "valuation": sorted(val_rows, key=lambda r: (r["cohort"], r["horizon_weeks"]))}


def ledger_summary(path: str, entries: List[Dict]) -> Dict:
    ok, bad = ledger.verify(entries)
    made = [e["record"].get("made_utc") or e["record"].get("scored_utc") for e in entries]
    return {"file": "ledger/" + os.path.basename(path), "entries": len(entries), "verified": ok,
            "first_broken": bad, "head": ledger.head(entries),
            "first_utc": min(made) if made else None, "last_utc": max(made) if made else None}


# ---------------------------------------------------------------- main

def assert_clean() -> int:
    with open(OUT_JSON, encoding="utf-8") as fh:
        d = json.load(fh)
    s = d["summary"]
    problems = []
    if s["checks_failed"]:
        problems.append(f"{s['checks_failed']} validation checks failed")
    if not (d["ledger"]["forecasts"]["verified"] and d["ledger"]["outcomes"]["verified"]):
        problems.append("ledger hash chain does not verify")
    for p in problems:
        print("FAILED:", p)
    if not problems:
        print(f"clean: {s['checks_passed']}/{s['checks_total']} checks pass; ledger chains verify")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert-clean", action="store_true")
    parser.add_argument("--local", action="store_true", help="read sibling checkouts instead of the published files")
    args = parser.parse_args()
    if args.assert_clean:
        return assert_clean()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("Loading published data...")
    psrc = sources.load("portfolio", prefer_local=args.local)
    vsrc = sources.load("valuation", prefer_local=args.local)
    pdata, vdata = psrc["data"], vsrc["data"]
    print(f"  portfolio lab: {psrc['meta']['origin']}, as of {pdata['meta']['as_of']}")
    print(f"  valuation lab: {vsrc['meta']['origin']}, as of {vdata['meta']['valuation_date']}")

    print("Validating the portfolio lab...")
    pv = portfolio_validation.validate(pdata)
    print("Validating the valuation lab...")
    vv = valuation_validation.validate(vdata)

    print("Backtesting the forecast models...")
    dates = pdata["series"]["dates"]
    subj = subjects(pdata)
    pv["var_backtest"] = {name: forecasts.var_backtest(s["returns"], dates) for name, s in subj.items()}
    pv["vol_backtest"] = forecasts.vol_backtest(subj["benchmark"]["returns"], dates, garch_refit_every=63)

    print("Updating the ledger...")
    fc = ledger.read(FORECASTS)
    added_fc = ledger.append(FORECASTS, new_forecasts(pdata, psrc["meta"], vdata, vsrc["meta"], fc, now))
    fc = ledger.read(FORECASTS)
    oc = ledger.read(OUTCOMES)
    scored = {e["record"]["id"] for e in oc}
    added_oc = ledger.append(OUTCOMES, new_outcomes(fc, scored, pdata, psrc["meta"], vdata, vsrc["meta"], now))
    oc = ledger.read(OUTCOMES)
    print(f"  {len(added_fc)} forecasts recorded, {len(added_oc)} outcomes scored")

    # Risk forecasts and valuation calls are dated off different sources, so the
    # "latest" of each is found separately.
    last_risk = max(e["record"]["as_of"] for e in fc if e["record"]["kind"] in ("var", "vol"))
    cohorts = sorted({e["record"]["cohort"] for e in fc if e["record"]["kind"] == "valuation"})
    latest = [e["record"] for e in fc
              if (e["record"]["kind"] in ("var", "vol") and e["record"]["as_of"] == last_risk)
              or (e["record"]["kind"] == "valuation" and e["record"]["cohort"] == cohorts[-1])]
    scored = {e["record"]["id"] for e in oc}
    pending = sum(1 for e in fc if e["record"]["kind"] in ("var", "vol") and e["record"]["id"] not in scored)
    first_cohort_date = min((e["record"]["as_of"] for e in fc if e["record"]["kind"] == "valuation"), default=None)

    checks = pv["checks"] + vv["checks"]
    failed = [c for c in checks if not c["ok"]]
    data = {
        "meta": {"generated_utc": now, "desk_version": __version__},
        "sources": {"portfolio": psrc["meta"], "valuation": vsrc["meta"]},
        "portfolio": pv,
        "valuation": vv,
        "ledger": {
            "forecasts": ledger_summary(FORECASTS, fc),
            "outcomes": ledger_summary(OUTCOMES, oc),
            "latest": latest,
            "scorecards": scorecards(fc, oc),
            "pending_risk_forecasts": pending,
            "valuation_cohorts": cohorts,
            "valuation_horizons_weeks": list(VALUATION_HORIZONS_WEEKS),
            "first_valuation_score_due": None if first_cohort_date is None else
                (date.fromisoformat(first_cohort_date) + timedelta(weeks=VALUATION_HORIZONS_WEEKS[0])).isoformat(),
            "started": min(e["record"]["made_utc"] for e in fc)[:10] if fc else None,
        },
        "summary": {
            "checks_total": len(checks), "checks_passed": len(checks) - len(failed), "checks_failed": len(failed),
            "portfolio_checks": len(pv["checks"]), "valuation_checks": len(vv["checks"]),
            "failed": [f"{c['group']}: {c['title']}" for c in failed],
        },
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = json.dumps(clean(data), separators=(",", ":"), allow_nan=False)
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(payload + "\n")
    with open(OUT_JS, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("window.DESK_DATA = " + payload + ";\n")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks pass; "
          f"ledger: {len(fc)} forecasts, {len(oc)} outcomes; wrote docs/data/desk-data.json ({len(payload) / 1024:.0f} KB)")
    for c in failed:
        print(f"  FAILED: {c['group']}: {c['title']} ({c['detail']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
