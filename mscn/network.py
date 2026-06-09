"""Layer 2 (part A) -- network coherence and coupling graphs.

N agents live on a coupling graph with symmetric, non-negative weights
``J_ij = J_ji >= 0``. The network coherence adds pairwise interaction terms to
the sum of individual effective coherences::

    C_net = sum_i R_eff(x_i)  +  sum_{i<j} J_ij * R_pair(x_i, x_j)

The proven structural facts (NetworkCoherence.lean) demonstrated here:

* **Coupling only helps** -- with ``J >= 0`` and ``R_pair >= 0`` the network
  coherence is at least the sum of the individual coherences.
* **Isolated nodes evolve independently** -- a zero row/column of ``J``
  contributes nothing.
* **Coherence propagates through connected components**.
* **Mean-field fluctuation suppression** -- the spread of the mean field shrinks
  like ``sigma / sqrt(N)`` (MeanFieldTheory.lean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ArrayF = np.ndarray


# ---------------------------------------------------------------------------
#  Coupling graphs
# ---------------------------------------------------------------------------

def ring_graph(n: int, weight: float = 1.0) -> ArrayF:
    J = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        J[i, j] = J[j, i] = weight
    return J


def complete_graph(n: int, weight: float = 1.0) -> ArrayF:
    J = np.full((n, n), weight)
    np.fill_diagonal(J, 0.0)
    return J


def scale_free_graph(n: int, m: int = 2, weight: float = 1.0, seed: int = 0) -> ArrayF:
    """Barabasi-Albert preferential-attachment graph (power-law degrees)."""
    rng = np.random.default_rng(seed)
    J = np.zeros((n, n))
    m = max(1, min(m, n - 1))
    # seed clique
    for i in range(m):
        for j in range(i + 1, m):
            J[i, j] = J[j, i] = weight
    degrees = J.sum(axis=1).copy()
    for new in range(m, n):
        deg = degrees[:new].copy()
        if deg.sum() <= 0:
            targets = rng.choice(new, size=min(m, new), replace=False)
        else:
            probs = deg / deg.sum()
            targets = rng.choice(new, size=min(m, new), replace=False, p=probs)
        for t in targets:
            J[new, t] = J[t, new] = weight
        degrees[:new] = J[:new, :new].sum(axis=1)
        degrees[new] = J[new].sum()
    return J


def connected_components(J: ArrayF) -> list[list[int]]:
    n = J.shape[0]
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if J[u, v] != 0 and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


# ---------------------------------------------------------------------------
#  Network coherence
# ---------------------------------------------------------------------------

@dataclass
class CoherenceNetwork:
    """A coupling graph plus per-node effective coherence and pairwise term."""

    J: ArrayF
    R_eff: Callable[[int, ArrayF], float]            # node i, its state -> coherence
    R_pair: Callable[[ArrayF, ArrayF], float]        # pairwise coherence, >= 0

    def __post_init__(self) -> None:
        self.J = np.asarray(self.J, dtype=float)
        assert np.allclose(self.J, self.J.T), "coupling must be symmetric"
        assert np.all(self.J >= 0), "coupling must be non-negative"
        self.n = self.J.shape[0]

    def individual_total(self, states: list[ArrayF]) -> float:
        return float(sum(self.R_eff(i, states[i]) for i in range(self.n)))

    def coupling_total(self, states: list[ArrayF]) -> float:
        total = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.J[i, j] != 0.0:
                    total += self.J[i, j] * self.R_pair(states[i], states[j])
        return float(total)

    def network_coherence(self, states: list[ArrayF]) -> float:
        return self.individual_total(states) + self.coupling_total(states)


# ---------------------------------------------------------------------------
#  Mean-field fluctuation suppression
# ---------------------------------------------------------------------------

def mean_field_spread(n: int, n_trials: int = 2000, sigma: float = 1.0, seed: int = 0) -> float:
    """Empirical std of the mean of ``n`` iid unit-variance fields ~ sigma/sqrt(N)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, sigma, size=(n_trials, n)).mean(axis=1)
    return float(np.std(means))
