"""Classical optimisation baselines for the Pilot 1 comparison.

These are deliberately plain, equal-budget reference algorithms (random search,
hill climbing with restarts, simulated annealing). All operate on the raw
objective ``landscape.f`` (minimisation) and return the best objective value
found within ``n_evals`` function evaluations, so the comparison against the IBF
learner is budget-matched.
"""

from __future__ import annotations

import numpy as np

from .landscapes import Landscape

ArrayF = np.ndarray


def random_search(landscape: Landscape, n_evals: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = landscape.random_points(n_evals, rng)
    vals = landscape.f_batch(X)
    i = int(np.argmin(vals))
    return {"best_f": float(vals[i]), "best_x": X[i]}


def hill_climbing(
    landscape: Landscape,
    n_evals: int,
    seed: int = 0,
    n_restarts: int = 10,
    step_frac: float = 0.1,
) -> dict:
    rng = np.random.default_rng(seed)
    step = step_frac * landscape.range
    budget = max(n_evals // n_restarts, 1)
    best_f, best_x = np.inf, None
    for _ in range(n_restarts):
        x = landscape.random_point(rng)
        fx = landscape.f(x)
        for _ in range(budget):
            y = landscape.clip(x + rng.normal(0.0, 1.0, landscape.dim) * step)
            fy = landscape.f(y)
            if fy < fx:
                x, fx = y, fy
        if fx < best_f:
            best_f, best_x = fx, x
    return {"best_f": float(best_f), "best_x": best_x}


def simulated_annealing(
    landscape: Landscape,
    n_evals: int,
    seed: int = 0,
    t0: float = 5.0,
    t_min: float = 1e-3,
    step_frac: float = 0.1,
) -> dict:
    rng = np.random.default_rng(seed)
    step = step_frac * landscape.range
    x = landscape.random_point(rng)
    fx = landscape.f(x)
    best_f, best_x = fx, x.copy()
    cool = (t_min / t0) ** (1.0 / max(n_evals, 1))
    t = t0
    for _ in range(n_evals):
        y = landscape.clip(x + rng.normal(0.0, 1.0, landscape.dim) * step)
        fy = landscape.f(y)
        if fy < fx or rng.random() < np.exp(-(fy - fx) / max(t, 1e-12)):
            x, fx = y, fy
            if fx < best_f:
                best_f, best_x = fx, x.copy()
        t *= cool
    return {"best_f": float(best_f), "best_x": best_x}
