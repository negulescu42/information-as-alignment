"""Coherence landscapes for the MSCN toy model.

In the IBF picture an agent does not minimise a loss; it *climbs a coherence
landscape*. To reuse the standard global-optimisation benchmarks we simply
define coherence as the negated objective::

    R_hat(x) = - f(x)

so that maximising coherence is the same as minimising ``f``. Every landscape
exposes the raw objective ``f`` (to compare against classical optimisers) and
the coherence ``coherence`` (what the IBF learner actually senses).

The four benchmarks (sphere, Rastrigin, Ackley, Schwefel) are the ones named in
the pilot design (Part III). Sphere is the trivial unimodal control; the other
three are multimodal traps used to stress basin expansion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayF = np.ndarray


@dataclass
class Landscape:
    """A coherence landscape ``R_hat = -f`` over a box ``[lo, hi]^d``."""

    name: str
    dim: int
    lo: ArrayF
    hi: ArrayF
    f: Callable[[ArrayF], float]          # objective to MINIMISE
    f_batch: Callable[[ArrayF], ArrayF]   # vectorised objective, X: (n, d) -> (n,)
    x_opt: ArrayF                         # location of the global minimum
    f_opt: float                          # value of the global minimum

    # ----- coherence view (what the IBF agent senses) -----
    def coherence(self, x: ArrayF) -> float:
        return float(-self.f(np.asarray(x, dtype=float)))

    def coherence_batch(self, X: ArrayF) -> ArrayF:
        return -self.f_batch(np.asarray(X, dtype=float))

    # ----- helpers -----
    def clip(self, x: ArrayF) -> ArrayF:
        return np.clip(x, self.lo, self.hi)

    def random_point(self, rng: np.random.Generator) -> ArrayF:
        return rng.uniform(self.lo, self.hi)

    def random_points(self, n: int, rng: np.random.Generator) -> ArrayF:
        return rng.uniform(self.lo, self.hi, size=(n, self.dim))

    @property
    def range(self) -> ArrayF:
        return self.hi - self.lo


def _box(dim: int, half_width: float) -> tuple[ArrayF, ArrayF]:
    lo = np.full(dim, -half_width, dtype=float)
    hi = np.full(dim, half_width, dtype=float)
    return lo, hi


def sphere(dim: int = 2) -> Landscape:
    """f(x) = sum x_i^2. Unimodal control; optimum 0 at the origin."""
    lo, hi = _box(dim, 5.12)

    def f(x: ArrayF) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * x))

    def f_batch(X: ArrayF) -> ArrayF:
        X = np.asarray(X, dtype=float)
        return np.sum(X * X, axis=1)

    return Landscape("sphere", dim, lo, hi, f, f_batch, np.zeros(dim), 0.0)


def rastrigin(dim: int = 2) -> Landscape:
    """Highly multimodal; a regular lattice of local minima. Optimum 0 at 0."""
    lo, hi = _box(dim, 5.12)
    A = 10.0

    def f(x: ArrayF) -> float:
        x = np.asarray(x, dtype=float)
        return float(A * dim + np.sum(x * x - A * np.cos(2 * np.pi * x)))

    def f_batch(X: ArrayF) -> ArrayF:
        X = np.asarray(X, dtype=float)
        return A * dim + np.sum(X * X - A * np.cos(2 * np.pi * X), axis=1)

    return Landscape("rastrigin", dim, lo, hi, f, f_batch, np.zeros(dim), 0.0)


def ackley(dim: int = 2) -> Landscape:
    """Nearly flat outer region with a deep central well. Optimum 0 at 0."""
    lo, hi = _box(dim, 32.768)
    a, b, c = 20.0, 0.2, 2 * np.pi

    def f(x: ArrayF) -> float:
        x = np.asarray(x, dtype=float)
        d = x.size
        t1 = -a * np.exp(-b * np.sqrt(np.sum(x * x) / d))
        t2 = -np.exp(np.sum(np.cos(c * x)) / d)
        return float(t1 + t2 + a + np.e)

    def f_batch(X: ArrayF) -> ArrayF:
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        t1 = -a * np.exp(-b * np.sqrt(np.sum(X * X, axis=1) / d))
        t2 = -np.exp(np.sum(np.cos(c * X), axis=1) / d)
        return t1 + t2 + a + np.e

    return Landscape("ackley", dim, lo, hi, f, f_batch, np.zeros(dim), 0.0)


def schwefel(dim: int = 2) -> Landscape:
    """Deceptive: the global optimum is far from the next-best minima.

    Optimum 0 at x_i = 420.9687.
    """
    lo, hi = _box(dim, 500.0)
    const = 418.9829

    def f(x: ArrayF) -> float:
        x = np.asarray(x, dtype=float)
        return float(const * x.size - np.sum(x * np.sin(np.sqrt(np.abs(x)))))

    def f_batch(X: ArrayF) -> ArrayF:
        X = np.asarray(X, dtype=float)
        return const * X.shape[1] - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)

    x_opt = np.full(dim, 420.9687, dtype=float)
    return Landscape("schwefel", dim, lo, hi, f, f_batch, x_opt, 0.0)


def rosenbrock(dim: int = 2) -> Landscape:
    """Non-separable banana valley. Optimum 0 at the all-ones point.

    Used inside the hierarchy demo because its variables are *coupled*: the right
    level of coarse-graining (block scale) matters, unlike a fully separable
    function.
    """
    lo, hi = _box(dim, 2.048)

    def f(x: ArrayF) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

    def f_batch(X: ArrayF) -> ArrayF:
        X = np.asarray(X, dtype=float)
        return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)

    return Landscape("rosenbrock", dim, lo, hi, f, f_batch, np.ones(dim), 0.0)


BENCHMARKS: dict[str, Callable[[int], Landscape]] = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "schwefel": schwefel,
    "rosenbrock": rosenbrock,
}


def make(name: str, dim: int = 2) -> Landscape:
    return BENCHMARKS[name](dim)
