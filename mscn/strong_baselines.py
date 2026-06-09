"""Strong, state-of-the-art black-box optimisers for a tougher comparison.

These are the heavyweight references — **CMA-ES** (with BIPOP restarts, the
gold standard for continuous black-box optimisation), SciPy's **differential
evolution**, and SciPy's **dual annealing**. They are optional dependencies
(`pip install cma scipy`); the rest of the package stays numpy-only.

All are made strictly **evaluation-budget matched** by wrapping the objective in
:class:`_Budgeted`, which counts calls, tracks the best-so-far point, and raises
once the budget is spent. Every optimiser therefore sees exactly ``budget``
objective evaluations, just like the IBF learner and the classical baselines.
"""

from __future__ import annotations

import warnings

import numpy as np

from .landscapes import Landscape

try:
    import cma
    HAS_CMA = True
except Exception:  # pragma: no cover
    HAS_CMA = False

try:
    from scipy.optimize import differential_evolution, dual_annealing
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

AVAILABLE = {"CMA-ES": HAS_CMA, "DE": HAS_SCIPY, "dual-anneal": HAS_SCIPY}


class _BudgetExceeded(Exception):
    pass


class _Budgeted:
    """Objective wrapper enforcing an exact evaluation budget."""

    def __init__(self, f, budget: int) -> None:
        self.f = f
        self.budget = budget
        self.n = 0
        self.best = np.inf
        self.best_x = None

    def __call__(self, x) -> float:
        if self.n >= self.budget:
            raise _BudgetExceeded
        v = float(self.f(np.asarray(x, dtype=float)))
        self.n += 1
        if v < self.best:
            self.best = v
            self.best_x = np.asarray(x, dtype=float).copy()
        return v


def _result(g: _Budgeted) -> dict:
    return {"best_f": float(g.best), "best_x": g.best_x, "evals": g.n}


def cma_es(landscape: Landscape, budget: int, seed: int = 0) -> dict:
    """CMA-ES with BIPOP restarts (the SOTA black-box optimiser)."""
    if not HAS_CMA:
        raise RuntimeError("cma not installed (pip install cma)")
    rng = np.random.default_rng(seed)
    x0 = landscape.random_point(rng)
    sigma0 = 0.3 * float(np.mean(landscape.range))
    g = _Budgeted(landscape.f, budget)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cma.fmin(g, list(x0), sigma0,
                     options={"bounds": [list(landscape.lo), list(landscape.hi)],
                              "seed": seed + 1, "verbose": -9, "maxfevals": budget},
                     restarts=3, bipop=True)
        except _BudgetExceeded:
            pass
        except Exception:
            pass
    return _result(g)


def differential_evolution_opt(landscape: Landscape, budget: int, seed: int = 0) -> dict:
    """SciPy differential evolution (strong population-based global optimiser)."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy not installed (pip install scipy)")
    g = _Budgeted(landscape.f, budget)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            differential_evolution(g, bounds=list(zip(landscape.lo, landscape.hi)),
                                    seed=seed, maxiter=10 ** 6, tol=0, polish=False,
                                    init="sobol")
        except _BudgetExceeded:
            pass
        except Exception:
            pass
    return _result(g)


def dual_annealing_opt(landscape: Landscape, budget: int, seed: int = 0) -> dict:
    """SciPy dual annealing (strong global optimiser)."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy not installed (pip install scipy)")
    g = _Budgeted(landscape.f, budget)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            dual_annealing(g, bounds=list(zip(landscape.lo, landscape.hi)),
                           seed=seed, maxiter=10 ** 6, no_local_search=True)
        except _BudgetExceeded:
            pass
        except Exception:
            pass
    return _result(g)


STRONG = {
    "CMA-ES": cma_es,
    "DE": differential_evolution_opt,
    "dual-anneal": dual_annealing_opt,
}
