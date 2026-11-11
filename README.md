# Verifiable Research Desk

Anyone can build a model that prints numbers. This project asks the question a bank's model-validation team asks next: **are the numbers right, and do the forecasts hold up?**

It does two things for my two analytics platforms, the [Portfolio Construction Lab](https://github.com/Nnd1106/portfolio-construction-lab) and the [Valuation & Deal Lab](https://github.com/Nnd1106/valuation-deal-lab):

1. **Independent validation.** It re-derives their published results from their published data alone, with separately written code and, wherever one exists, a different method. It imports nothing from the projects it checks.
2. **A forecast ledger.** Every weekday it seals forecasts in a hash-chained file *before* their outcomes exist, commits them through GitHub Actions, and scores them when the outcomes arrive.

**Live:** https://nnd1106.github.io/verifiable-research-desk/

No runtime AI, no API keys, no backend. Python for the checks, vanilla HTML/CSS/JS + Chart.js for the site.

---

## 1. Independent validation

The structure follows the three parts of model validation in the US Federal Reserve's SR 11-7 guidance: replicate the model, challenge it with an alternative, and analyse outcomes.

### Portfolio Construction Lab: 33 checks

| What | How the desk checks it |
|---|---|
| **Estimation** | Rebuilds Friday-to-Friday weekly returns from the published daily returns, then the covariance matrix, the sample means and the Bayes-Stein shrinkage (Jorion 1986). |
| **Optimiser** | The lab solves its frontier with an iterative optimiser (SLSQP). The desk solves the same problems **exactly, with no iteration**: for a convex quadratic programme the optimum lies on a face of the feasible box, so it tries every assignment of the five assets to {lower bound, upper bound, free} and solves each case in closed form (243 linear systems per portfolio). Minimum-variance portfolio, tangency portfolio and all 120 frontier points are compared for both of the lab's models. |
| **Dominance** | 100,000 random feasible portfolios per model: none may beat the published tangency Sharpe ratio or undercut the minimum variance. |
| **Risk metrics** | VaR, expected shortfall, drawdown, Sharpe, Sortino, beta, Treynor, Jensen's alpha, tracking error and information ratio recomputed with separately written code for three portfolios. |
| **Data** | One value per date, strictly increasing dates, no implausible returns, a positive-definite covariance matrix consistent with the published correlations and volatilities. |

The lab publishes its inputs rounded to six decimals, so recomputation cannot agree to machine precision. Tolerances sit just above that rounding noise: 0.002 percentage points on a return or volatility, 0.0001 on a ratio. The exact solver and the lab's optimiser currently agree to about 0.0005 percentage points.

### Valuation & Deal Lab: 22 checks

| What | How the desk checks it |
|---|---|
| **DCF** | Every company is valued a second way: **invested capital plus the present value of economic profit** (NOPAT less a charge for capital). Discounted cash flow and economic profit are algebraically equivalent, so two independently written implementations must produce the same enterprise value. They agree to about 2e-15. The result must also be invariant to the opening capital figure, and the reverse DCF's implied growth and margin must reprice each share to market. |
| **Cost of capital** | Sector betas, CAPM cost of equity, synthetic-rating cost of debt and WACC rebuilt from their published components. |
| **Comps** | Enterprise value and five multiples rebuilt from the published statements and prices; peer quartiles from NumPy's percentile routine. |
| **LBO** | The published debt schedule is audited line by line (interest on average balances, the revolver, tax, cash available for debt service, repayments), and the IRR is recomputed from the equity cash flows. |
| **M&A** | EPS accretion recomputed from a closed-form expression; the purchase price must be fully allocated. |

### Findings, not just ticks

Agreement proves the arithmetic. It does not prove the model is useful, so the desk also reports what it learns:

- In a walk-forward backtest the portfolio lab's 99% one-day VaR was breached 22 to 24 times, depending on the model, against 14.5 expected; the parametric model is rejected by the Kupiec test at the 5% level, and the 95% breaches cluster in time (Christoffersen test). That is the usual fat-tail, volatility-clustering failure of unconditional VaR, measured.
- The realised Sharpe ratio of the lab's policy portfolio has a 90% bootstrap range wide enough that under seven years of data cannot distinguish skill from luck.
- The lab's minimum-variance weights sit on their allocation limits in almost every resampled history: the constraints, not the optimiser, determine the answer.
- Most of the valuation lab's DCF baselines sit below market price, some far below. The desk frames this as a statement about expectations in prices and puts it to the test in the ledger instead of asserting it.

## 2. The forecast ledger

A model can agree with itself perfectly and still be wrong about the world. The only remedy is to write the forecast down before the outcome exists, in a form that cannot be quietly edited, and keep score.

**What is sealed**

| Forecast | Cadence | Scored by |
|---|---|---|
| One-day VaR at 95% and 99% for three reference portfolios, by three models (historical simulation, parametric normal, EWMA) | every weekday | exceedance counts, Kupiec coverage test, quantile (pinball) loss |
| One-day variance of the Nifty 50, by three models (EWMA, 60-day rolling, GARCH(1,1) by maximum likelihood) | every weekday | QLIKE |
| A valuation call on each of 24 companies: the gap between DCF baseline and price | once a week | rank correlation and hit rate against returns relative to the universe at 4, 13, 26 and 52 weeks |

**How it is sealed**

Each line of `docs/ledger/forecasts.jsonl` is

```json
{"hash": "...", "prev": "...", "payload": "<the record as a JSON string>"}
```

with `hash = SHA-256(prev + "\n" + payload)`. Changing, deleting or reordering any earlier record changes every later hash. The payload is stored as text so the chain can be verified byte for byte in any language; the site does it in your browser with the Web Crypto API.

A hash chain on its own proves only that the record is internally consistent. The evidence of *timing* is the public commit history: every run is committed by GitHub Actions with a server-side timestamp, and each commit's chain head is the next commit's `prev`.

Verify it yourself with nothing but Python:

```python
import hashlib, json
prev = "0" * 64
for line in open("docs/ledger/forecasts.jsonl"):
    e = json.loads(line)
    assert e["prev"] == prev
    assert e["hash"] == hashlib.sha256((prev + "\n" + e["payload"]).encode()).hexdigest()
    prev = e["hash"]
print("chain intact, head", prev)
```

**Live record versus backtest.** The ledger started on 11 November 2026, so the live record is short and the site says so. A track record cannot be backdated, which is the point. Until it accumulates, the site also shows a *reconstructed* walk-forward backtest of the same models on the portfolio lab's history. It uses no future information, but it was computed after the fact, so it is labelled separately and carries less weight.

## 3. The validator is validated

`tests/` checks the desk's own tools before every run:

- the exact optimiser against an iterative solver, the closed-form unconstrained solution and linear programming, on randomly generated problems;
- Kupiec and Christoffersen statistics against hand calculations and SciPy; the quantile loss is minimised at the true quantile; QLIKE prefers the true variance; Spearman matches SciPy;
- the economic-profit valuation against a third, deliberately naive DCF loop over hundreds of random assumption sets;
- the ledger against four kinds of tampering (edit a record, delete one, swap two, edit one and recompute its own hash), plus refusal to append to a broken chain;
- the VaR backtest uses no future information: shocking the final day cannot change any earlier forecast;
- GARCH estimation recovers the parameters of simulated data.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests -q        # the desk's own tools
python run.py                    # validate both projects, update the ledger, publish docs/data
python run.py --assert-clean     # exit 1 if any check failed or a chain is broken
```

`.github/workflows/desk.yml` runs this every weekday at 7:00 PM IST, after both projects have refreshed. **Results are committed before the run is judged**, so a failed check is published on the site as a finding and then turns the workflow red; it is never hidden by a failed build.

## What this does not prove

- That the inputs are right. Results are validated against published inputs, not the inputs against audited filings.
- That the models are the right models. Economic profit and DCF agree because they are one model in two forms; both inherit the same assumptions.
- That the forecasts will hold. That is what the ledger is for, and it needs time: with a 1% tail a model must be watched for months before a coverage test means anything.

## References

- Board of Governors of the Federal Reserve System and OCC (2011), *Supervisory Guidance on Model Risk Management* (SR 11-7)
- Kupiec (1995); Christoffersen (1998); Basel Committee on Banking Supervision (1996), backtesting framework
- Gneiting and Raftery (2007), strictly proper scoring rules; Patton (2011), volatility forecast comparison
- Jorion (1986), Bayes-Stein estimation; Koller, Goedhart and Wessels, *Valuation*

---

*Educational project. Nothing here is investment advice.*
