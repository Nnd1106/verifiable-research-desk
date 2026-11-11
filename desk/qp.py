"""qp.py: exact mean-variance optimisation by active-set enumeration.

The portfolio lab solves its frontier with an iterative optimiser (SLSQP).
This module reaches the same optimum by a different route, with no iteration
and no starting point. For a convex quadratic programme the optimum lies on
some face of the feasible box, so every assignment of each asset to
{lower bound, upper bound, free} is tried, the equality-constrained problem
on the free assets is solved in closed form (one linear system), and the best
feasible candidate is the global optimum. With five assets that is 243 small
linear solves per portfolio.
"""

from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

import numpy as np

TOL = 1e-10


def min_variance(cov: np.ndarray, lower: Sequence[float], upper: Sequence[float],
                 mu: Optional[np.ndarray] = None, target: Optional[float] = None) -> Optional[np.ndarray]:
    """Minimise w.cov.w subject to sum(w) = 1, lower <= w <= upper and, when
    `target` is given, w.mu = target. Returns None if infeasible."""
    cov = np.asarray(cov, float)
    n = cov.shape[0]
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    rows = [np.ones(n)]
    rhs = [1.0]
    if target is not None:
        rows.append(np.asarray(mu, float))
        rhs.append(float(target))
    a_mat, b_vec = np.vstack(rows), np.asarray(rhs)
    m = len(rhs)

    best_w, best_val = None, np.inf
    for state in itertools.product((0, 1, 2), repeat=n):   # 0 lower, 1 upper, 2 free
        free = [i for i, s in enumerate(state) if s == 2]
        w = np.where(np.asarray(state) == 0, lo, hi).astype(float)
        if free:
            w[free] = 0.0
            resid = b_vec - a_mat @ w
            k = len(free)
            kkt = np.zeros((k + m, k + m))
            kkt[:k, :k] = 2.0 * cov[np.ix_(free, free)]
            kkt[:k, k:] = a_mat[:, free].T
            kkt[k:, :k] = a_mat[:, free]
            fixed = [i for i in range(n) if i not in free]
            rhs_vec = np.concatenate([-2.0 * cov[np.ix_(free, fixed)] @ w[fixed], resid])
            sol, *_ = np.linalg.lstsq(kkt, rhs_vec, rcond=None)
            w[free] = sol[:k]
        if np.any(np.abs(a_mat @ w - b_vec) > 1e-9):
            continue
        if np.any(w < lo - TOL) or np.any(w > hi + TOL):
            continue
        val = float(w @ cov @ w)
        if val < best_val - 1e-16:
            best_w, best_val = w, val
    return None if best_w is None else np.clip(best_w, lo, hi)


def max_return(mu: np.ndarray, lower: Sequence[float], upper: Sequence[float]) -> np.ndarray:
    """Maximise w.mu subject to the budget and the box: fill from the highest mean down."""
    mu = np.asarray(mu, float)
    w = np.asarray(lower, float).copy()
    budget = 1.0 - w.sum()
    for i in np.argsort(-mu):
        add = min(budget, upper[i] - w[i])
        w[i] += add
        budget -= add
    return w


def stats(w: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> Tuple[float, float, float]:
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol, (ret - rf) / vol


def max_sharpe(mu: np.ndarray, cov: np.ndarray, lower: Sequence[float], upper: Sequence[float],
               rf: float, grid: int = 200) -> np.ndarray:
    """Tangency portfolio: scan the exact efficient frontier, then refine the
    best bracket by golden-section search on the target return."""
    mu, cov = np.asarray(mu, float), np.asarray(cov, float)
    w_mvp = min_variance(cov, lower, upper)
    r_lo = float(w_mvp @ mu)
    r_hi = float(max_return(mu, lower, upper) @ mu)

    def sharpe_at(target: float) -> float:
        w = min_variance(cov, lower, upper, mu, target)
        return -np.inf if w is None else stats(w, mu, cov, rf)[2]

    targets = np.linspace(r_lo, r_hi, grid)
    values = [sharpe_at(t) for t in targets]
    i = int(np.argmax(values))
    a, b = targets[max(0, i - 1)], targets[min(grid - 1, i + 1)]
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = sharpe_at(c), sharpe_at(d)
    for _ in range(60):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = sharpe_at(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = sharpe_at(d)
    return min_variance(cov, lower, upper, mu, 0.5 * (a + b))
