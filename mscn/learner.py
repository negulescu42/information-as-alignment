"""Layer 1 — the IBF Coherence-Gradient Learner.

This is the non-neural learner of the pilot design (Part III, Pilot 1) and the
``IBFLearner.lean`` formalisation. Knowledge is stored as *coherence
modifications* ``delta_R`` on configuration space, never as weights. The agent
loop is::

    1. SENSE   R_eff(i) = R_hat(i) + delta_R(i)
    2. SELECT  Boltzmann policy  P(a) ~ exp(k * delta R_eff(a))
    3. ACT     move to the selected configuration
    4. MODIFY  delta_R(next) += alpha * max(discrepancy, 0) - mu * delta_R(next)
    5. ADAPT   if coherence improved, k += k_adapt

Two concrete realisations live here:

* :class:`DiscreteIBFLearner` -- a 1-D ``Fin n`` learner that mirrors the Lean
  ``floatLearnerStepV2`` exactly. It is the cleanest demonstration of the three
  Intelligence-Theorem (Thm 8) components.
* :class:`IBFLearner` -- a continuous ``R^d`` learner whose modification map is a
  sum of Gaussian kernels (the "non-neural associative memory"). This is the
  agent used for the optimisation benchmarks and as the Layer-1 unit inside the
  network, hierarchy and full MSCN.

The module also exposes the small analytic helpers that the formal guarantees
are stated about (Boltzmann probability, Euler step, decay/equilibrium of the
modification ODE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

ArrayF = np.ndarray


# ===========================================================================
#  Theorem-tied analytic helpers (Thm 7, Thm 10, Thm 11)
# ===========================================================================

def softmax(logits: ArrayF) -> ArrayF:
    z = np.asarray(logits, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def boltzmann_prob(k: float, increments: ArrayF) -> ArrayF:
    """Boltzmann selection probabilities P(a) ~ exp(k * increment(a))."""
    return softmax(k * np.asarray(increments, dtype=float))


def boltzmann_prob_best(k: float, delta: float) -> float:
    """P(best action) for a two-action gap ``delta`` > 0 (Thm 7).

    Equals ``exp(k*delta) / (exp(k*delta) + 1)``; strictly increasing in ``k``
    (Boltzmann monotonicity) and -> 1 as k -> infinity (greedy limit).
    """
    return float(1.0 / (1.0 + np.exp(-k * delta)))


def euler_step(f_t: float, alpha: float, mu: float, dt: float = 1.0) -> float:
    """One Euler step of the modification ODE  f' = alpha - mu*f  (Thm 11)."""
    return f_t + dt * (alpha - mu * f_t)


def modification_equilibrium(alpha: float, mu: float) -> float:
    """Steady state of  f' = alpha - mu*f  is alpha/mu."""
    return alpha / mu


def modification_decay(f0: float, mu: float, t: float) -> float:
    """Unreinforced modification decays as  f0 * exp(-mu t)  (Thm 3a / 10a)."""
    return f0 * np.exp(-mu * t)


# ===========================================================================
#  Discrete 1-D learner  (mirror of IBFLearner.lean floatLearnerStepV2)
# ===========================================================================

@dataclass
class DiscreteIBFLearner:
    """1-D coherence-gradient learner over ``Fin n`` with left/stay/right moves.

    Faithful Python port of the Lean ``floatLearnerStepV2`` demo: the
    reinforcement signal is the *improvement in the raw landscape* (so
    modifications never manufacture artificial traps), unvisited neighbours get
    a small exploration bonus, and unreinforced modifications decay.
    """

    landscape: ArrayF                 # baseline coherence over Fin n
    alpha: float = 0.2                # driving / learning rate
    mu: float = 0.02                  # decay rate
    k: float = 1.0                    # responsiveness
    k_adapt: float = 0.1
    explore_bonus: float = 0.05
    policy: str = "boltzmann"         # "boltzmann" | "greedy"
    position: int = 0
    seed: int = 0

    modification: ArrayF = field(init=False)
    visits: ArrayF = field(init=False)
    step_count: int = field(default=0, init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.landscape = np.asarray(self.landscape, dtype=float)
        n = self.landscape.size
        self.modification = np.zeros(n)
        self.visits = np.zeros(n, dtype=int)
        self.position = self.position % n
        self.visits[self.position] = 1
        self.rng = np.random.default_rng(self.seed)

    @property
    def n(self) -> int:
        return self.landscape.size

    def effective(self, i: int) -> float:
        return float(self.landscape[i] + self.modification[i])

    def viable_count(self, theta: float) -> int:
        return int(np.sum(self.landscape + self.modification > theta))

    def _neighbours(self) -> tuple[int, int, int]:
        cur = self.position
        left = cur - 1 if cur > 0 else cur
        right = cur + 1 if cur + 1 < self.n else cur
        return left, cur, right

    def step(self) -> None:
        if self.n <= 1:
            return
        left, cur, right = self._neighbours()
        cur_base = self.landscape[cur]

        def score(i: int) -> float:
            bonus = self.explore_bonus if self.visits[i] == 0 else 0.0
            return float(self.landscape[i] + bonus)

        actions = [left, cur, right]
        scores = np.array([score(left), score(cur), score(right)])

        if self.policy == "greedy":
            choice = int(np.argmax(scores))
        else:
            increments = scores - score(cur)
            probs = boltzmann_prob(self.k, increments)
            choice = int(self.rng.choice(3, p=probs))
        new = actions[choice]

        # MODIFY: reinforcement = positive improvement in the raw landscape.
        improvement = max(self.landscape[new] - cur_base, 0.0)
        old_mod = self.modification[new]
        self.modification[new] = max(old_mod + self.alpha * improvement - self.mu * old_mod, 0.0)

        # Selective retention: unreinforced modifications decay (forgetting).
        for j in range(self.n):
            if j != new:
                self.modification[j] = max((1.0 - self.mu) * self.modification[j], 0.0)

        # ADAPT responsiveness on genuine improvement.
        if self.landscape[new] > cur_base:
            self.k += self.k_adapt

        self.visits[new] += 1
        self.position = new
        self.step_count += 1

    def run(self, n_steps: int, theta: float | None = None) -> dict:
        viable = []
        coherence = []
        for _ in range(n_steps):
            self.step()
            coherence.append(self.effective(self.position))
            if theta is not None:
                viable.append(self.viable_count(theta))
        return {
            "position": self.position,
            "modification": self.modification.copy(),
            "k": self.k,
            "coherence_trace": np.array(coherence),
            "viable_trace": np.array(viable) if theta is not None else None,
        }


# ===========================================================================
#  Continuous kernel learner  (R^d, Gaussian-kernel associative memory)
# ===========================================================================

@dataclass
class _Center:
    z: ArrayF          # position in R^d
    v: float           # modification value (>= 0)


class IBFLearner:
    """Continuous coherence-gradient learner on ``R^d``.

    The modification map is a sum of Gaussian kernels centred on visited,
    reinforced configurations::

        delta_R(x) = sum_c  v_c * exp(-||x - z_c||^2 / (2 sigma^2))

    With ``v_c >= 0`` the modification is non-negative, so the viable set only
    grows (Basin Expansion, Thm 8a). Reinforced centres approach the
    equilibrium ``alpha*posD/mu`` while unreinforced ones decay (Selective
    Retention, Thm 8b). Responsiveness ``k`` grows on improvement, concentrating
    the Boltzmann policy on the best move (Agency Advantage, Thm 8c).
    """

    def __init__(
        self,
        coherence: Callable[[ArrayF], float],
        lo: ArrayF,
        hi: ArrayF,
        *,
        sigma: float | None = None,
        alpha: float = 0.3,
        mu: float = 0.02,
        k: float = 1.0,
        k_adapt: float = 0.05,
        step0: float | None = None,
        step_min: float | None = None,
        n_local_proposals: int = 4,
        n_global_proposals: int = 2,
        merge_frac: float = 0.5,
        nonneg: bool = True,
        seed: int = 0,
        x0: ArrayF | None = None,
    ) -> None:
        self.coherence = coherence
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.dim = self.lo.size
        rng_range = self.hi - self.lo
        self.sigma = float(sigma if sigma is not None else 0.12 * float(np.mean(rng_range)))
        self.alpha = alpha
        self.mu = mu
        self.k = k
        self.k_adapt = k_adapt
        self.step0 = float(step0 if step0 is not None else 0.20 * float(np.mean(rng_range)))
        self.step_min = float(step_min if step_min is not None else 0.01 * float(np.mean(rng_range)))
        self.n_local_proposals = n_local_proposals
        self.n_global_proposals = n_global_proposals
        self.merge_radius = merge_frac * self.sigma
        self.nonneg = nonneg
        self.rng = np.random.default_rng(seed)

        self._raw_coherence = coherence
        self.k_max = 8.0
        self.evals = 0

        self.centers: list[_Center] = []
        self.x = self.rng.uniform(self.lo, self.hi) if x0 is None else np.asarray(x0, dtype=float)
        self.step_count = 0
        self.best_x = self.x.copy()
        self.best_R = self._eval(self.x)

    def _eval(self, x: ArrayF) -> float:
        """Coherence query (counted, so budgets can be matched to baselines)."""
        self.evals += 1
        return self._raw_coherence(x)

    # ----- modification map -----
    def delta_R(self, x: ArrayF) -> float:
        if not self.centers:
            return 0.0
        Z = np.array([c.z for c in self.centers])
        V = np.array([c.v for c in self.centers])
        sq = np.sum((Z - x) ** 2, axis=1)
        return float(np.sum(V * np.exp(-sq / (2.0 * self.sigma ** 2))))

    def R_eff(self, x: ArrayF) -> float:
        return self._eval(x) + self.delta_R(x)

    def n_centers(self) -> int:
        return len(self.centers)

    # ----- action proposals -----
    def _proposals(self) -> ArrayF:
        """Candidate moves: stay + Gaussian local steps + uniform global jumps.

        The local step size is annealed from ``step0`` to ``step_min`` (cooling,
        like simulated annealing); the global jumps let the agent escape a basin
        once its learned modifications have consolidated.
        """
        progress = min(self.step_count / 400.0, 1.0)
        step = (1.0 - progress) * self.step0 + progress * self.step_min
        cands = [self.x.copy()]  # stay
        for _ in range(self.n_local_proposals):
            y = self.x + self.rng.normal(0.0, step, self.dim)
            cands.append(np.clip(y, self.lo, self.hi))
        for _ in range(self.n_global_proposals):
            cands.append(self.rng.uniform(self.lo, self.hi))
        return np.array(cands)

    # ----- one learning step -----
    def step(self) -> None:
        cands = self._proposals()
        # coherence of each candidate (the "stay" candidate is cands[0] == x)
        coh = np.array([self._eval(c) for c in cands])
        R_eff_cur = coh[0] + self.delta_R(self.x)
        increments = np.array([coh[j] + self.delta_R(cands[j]) - R_eff_cur
                               for j in range(len(cands))])
        probs = boltzmann_prob(self.k, increments)
        idx = int(self.rng.choice(len(cands), p=probs))
        new_x = cands[idx]

        # MODIFY via discrepancy signal (Postulate IV): external coherence at the
        # destination minus current effective coherence.
        discrepancy = coh[idx] - R_eff_cur
        pos_d = max(discrepancy, 0.0)
        driving = self.alpha * pos_d
        self._reinforce(new_x, driving)
        self._decay(exclude=new_x)

        # ADAPT responsiveness if the move improved effective coherence (capped).
        if increments[idx] > 0.0:
            self.k = min(self.k + self.k_adapt, self.k_max)

        self.x = new_x
        self.step_count += 1
        R_here = coh[idx]
        if R_here > self.best_R:
            self.best_R = R_here
            self.best_x = self.x.copy()

    def _reinforce(self, x: ArrayF, driving: float) -> None:
        # find nearest center
        nearest = None
        best_d = np.inf
        for c in self.centers:
            d = float(np.linalg.norm(c.z - x))
            if d < best_d:
                best_d, nearest = d, c
        if nearest is not None and best_d < self.merge_radius:
            nearest.v = nearest.v + driving - self.mu * nearest.v
            if self.nonneg:
                nearest.v = max(nearest.v, 0.0)
        elif driving > 0.0:
            self.centers.append(_Center(z=x.copy(), v=driving))

    def _decay(self, exclude: ArrayF) -> None:
        survivors: list[_Center] = []
        for c in self.centers:
            if np.array_equal(c.z, exclude):
                survivors.append(c)
                continue
            c.v *= (1.0 - self.mu)
            if self.nonneg:
                c.v = max(c.v, 0.0)
            if c.v > 1e-6:
                survivors.append(c)
        self.centers = survivors

    def viable_count(self, grid: ArrayF, theta: float) -> int:
        vals = np.array([self.R_eff(g) for g in grid])
        return int(np.sum(vals > theta))

    def run(self, n_steps: int, grid: ArrayF | None = None, theta: float = 0.0) -> dict:
        best_trace = []
        viable_trace = []
        for _ in range(n_steps):
            self.step()
            best_trace.append(self.best_R)
            if grid is not None:
                viable_trace.append(self.viable_count(grid, theta))
        return {
            "best_x": self.best_x.copy(),
            "best_R": self.best_R,
            "best_f": -self.best_R,
            "n_centers": self.n_centers(),
            "k": self.k,
            "evals": self.evals,
            "best_trace": np.array(best_trace),
            "viable_trace": np.array(viable_trace) if grid is not None else None,
        }

    def run_until_evals(self, eval_budget: int, trace_points: int = 0) -> dict:
        """Run until ``eval_budget`` coherence queries are spent (budget match).

        If ``trace_points > 0`` also return a best-so-far convergence trace
        sampled at that many evenly spaced evaluation checkpoints.
        """
        checkpoints = (np.linspace(eval_budget / trace_points, eval_budget, trace_points)
                       if trace_points > 0 else np.array([]))
        ci = 0
        trace_evals, trace_best = [], []
        while self.evals < eval_budget:
            self.step()
            while ci < len(checkpoints) and self.evals >= checkpoints[ci]:
                trace_evals.append(self.evals)
                trace_best.append(-self.best_R)  # best objective so far
                ci += 1
        out = {
            "best_x": self.best_x.copy(),
            "best_R": self.best_R,
            "best_f": -self.best_R,
            "n_centers": self.n_centers(),
            "k": self.k,
            "evals": self.evals,
        }
        if trace_points > 0:
            out["trace_evals"] = np.array(trace_evals)
            out["trace_best_f"] = np.array(trace_best)
        return out
