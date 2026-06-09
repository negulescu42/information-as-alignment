"""The integrated Multi-Scale Coherence Network.

This wires the three layers into one running system:

* **Layer 1** -- every node is a coherence-gradient learner with its own
  kernel modification map (:class:`mscn.learner.IBFLearner`).
* **Layer 2** -- the nodes sit on a scale-free coupling graph; each node's
  coherence gains a relational term that rewards alignment with its neighbours,
  so connected agents reach consensus (cooperation in state space) and the
  network coherence exceeds the sum of individual coherences.
* **Layer 3** -- adversarial perturbations periodically erode the learned
  modifications; a phase controller monitors the mean coherence and boosts the
  driving signal to stay super-critical (self-correction). The final population
  is coarse-grained into a hierarchy of macro-agents.

Running it prints a single report in which learning, cooperation, self-correction
and coarse-graining are all visibly active at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import network as net
from .landscapes import Landscape
from .learner import IBFLearner

ArrayF = np.ndarray


@dataclass
class MSCNConfig:
    n_agents: int = 27
    coupling_strength: float = 0.4
    proximity_scale: float = 1.5          # tau in the relational kernel
    graph_m: int = 2                       # scale-free attachment parameter
    rounds: int = 120
    warmup: int = 20                       # rounds before the threshold is fixed
    recovery_fraction: float = 0.7         # theta = fraction of peak effective coherence
    alpha_base: float = 0.3
    alpha_boost: float = 1.2
    mu: float = 0.03
    perturb_every: int = 30
    perturb_size: float = 0.75             # fraction of modification eroded
    viability_theta: float | None = None   # auto if None
    hierarchy_factor: int = 3              # coarse-grain groups of this size
    seed: int = 0


class MSCN:
    def __init__(self, landscape: Landscape, config: MSCNConfig | None = None) -> None:
        self.L = landscape
        self.cfg = config or MSCNConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        # sparse adjacency (Interface Principle, roadmap 3.5): coupling is O(n*k_eff),
        # not O(n^2) -- each agent couples only to its (boundary) graph neighbours.
        self.adj, self.adj_w = net.scale_free_adjacency(
            self.cfg.n_agents, self.cfg.graph_m, 1.0, seed=self.cfg.seed)
        # one Layer-1 learner per node
        self.agents = [
            IBFLearner(
                landscape.coherence, landscape.lo, landscape.hi,
                alpha=self.cfg.alpha_base, mu=self.cfg.mu, k=1.0, k_adapt=0.05,
                seed=self.cfg.seed * 1000 + i,
            )
            for i in range(self.cfg.n_agents)
        ]
        self.states = np.array([a.x.copy() for a in self.agents])
        # viability threshold is fixed adaptively after warmup (see run())
        self.theta = self.cfg.viability_theta if self.cfg.viability_theta is not None else -np.inf
        self._peak_effective = -np.inf

    # ----- relational coherence (Layer 2) -----
    def _r_pair(self, x: ArrayF, y: ArrayF) -> float:
        return float(np.exp(-np.sum((x - y) ** 2) / (2.0 * self.cfg.proximity_scale ** 2)))

    def _coupling_term(self, i: int, x: ArrayF) -> float:
        nb = self.adj[i]
        if len(nb) == 0:
            return 0.0
        sq = np.sum((self.states[nb] - x) ** 2, axis=1)
        rp = np.exp(-sq / (2.0 * self.cfg.proximity_scale ** 2))
        return self.cfg.coupling_strength * float(self.adj_w[i] @ rp)

    def _agent_coherence(self, i: int):
        def coh(x: ArrayF) -> float:
            return self.L.coherence(x) + self._coupling_term(i, x)
        return coh

    # ----- metrics (all O(n*k_eff) over the sparse edge set) -----
    def network_coherence(self) -> float:
        total = self.individual_total()
        for i in range(self.cfg.n_agents):
            nb = self.adj[i]
            for j, w in zip(nb, self.adj_w[i]):
                if j > i:                      # each undirected edge once
                    total += float(w) * self._r_pair(self.states[i], self.states[j])
        return total

    def individual_total(self) -> float:
        return float(sum(self.L.coherence(self.states[i]) + self.agents[i].delta_R(self.states[i])
                         for i in range(self.cfg.n_agents)))

    def consensus(self) -> float:
        """Mean distance across coupled edges (lower = more aligned)."""
        ds, w = 0.0, 0.0
        for i in range(self.cfg.n_agents):
            for j in self.adj[i]:
                if j > i:
                    ds += float(np.linalg.norm(self.states[i] - self.states[j]))
                    w += 1.0
        return ds / w if w > 0 else 0.0

    def effective_coherence(self, i: int) -> float:
        """Layer-1 effective coherence R_eff = landscape + learned modification."""
        return self.L.coherence(self.states[i]) + self.agents[i].delta_R(self.states[i])

    def n_viable(self) -> int:
        return int(sum(self.effective_coherence(i) >= self.theta for i in range(self.cfg.n_agents)))

    def mean_coherence(self) -> float:
        """Mean *effective* coherence (this is what the phase controller monitors)."""
        return float(np.mean([self.effective_coherence(i) for i in range(self.cfg.n_agents)]))

    # ----- one synchronous round -----
    def step(self, round_idx: int) -> dict:
        cfg = self.cfg
        perturbed = (round_idx > 0 and round_idx % cfg.perturb_every == 0)
        if perturbed:
            # adversarial erosion of the learned modifications (Layer 3 stressor)
            for a in self.agents:
                for c in a.centers:
                    c.v *= (1.0 - cfg.perturb_size)

        # phase control (Layer 3): boost driving if mean coherence is low
        boosting = self.mean_coherence() < self.theta
        alpha = cfg.alpha_boost if boosting else cfg.alpha_base

        new_states = self.states.copy()
        for i, a in enumerate(self.agents):
            a.alpha = alpha
            a._raw_coherence = self._agent_coherence(i)  # couple to current snapshot
            a.step()
            new_states[i] = a.x.copy()
        self.states = new_states

        return {
            "round": round_idx,
            "perturbed": perturbed,
            "boosting": boosting,
            "best_coherence": max(a.best_R for a in self.agents),
            "mean_coherence": self.mean_coherence(),
            "consensus": self.consensus(),
            "network_coherence": self.network_coherence(),
            "individual_total": self.individual_total(),
            "n_viable": self.n_viable(),
        }

    def run(self) -> dict:
        history = []
        recovered = []  # (perturb_round, rounds_to_recover)
        for t in range(self.cfg.rounds):
            rec = self.step(t)
            self._peak_effective = max(self._peak_effective, rec["mean_coherence"])
            # fix the viability threshold once warmup has settled the population
            if t == self.cfg.warmup and self.cfg.viability_theta is None:
                self.theta = self.cfg.recovery_fraction * self._peak_effective
                rec["n_viable"] = self.n_viable()
            history.append(rec)

        # measure recovery from each perturbation (rounds until mean coherence
        # climbs back above theta)
        for t, rec in enumerate(history):
            if rec["perturbed"] and rec["mean_coherence"] < self.theta:
                for dt in range(1, self.cfg.rounds - t):
                    if history[t + dt]["mean_coherence"] >= self.theta:
                        recovered.append((t, dt))
                        break

        coarse = self.coarse_grain()
        final = history[-1]
        return {
            "history": history,
            "final": final,
            "coarse_levels": coarse,
            "theta": self.theta,
            "recoveries": recovered,
            "coupling_helps": final["network_coherence"] >= final["individual_total"] - 1e-6,
        }

    # ----- coarse-graining into a macro-hierarchy (Layer 3) -----
    def coarse_grain(self) -> list[dict]:
        """Recursively average groups of states; report coherence per level."""
        levels = []
        states = self.states.copy()
        level = 0
        while True:
            mean_coh = float(np.mean([self.L.coherence(s) for s in states]))
            levels.append({"level": level, "n_units": len(states), "mean_coherence": mean_coh})
            if len(states) <= 1:
                break
            f = self.cfg.hierarchy_factor
            if len(states) <= f:                       # collapse the remainder to one macro-agent
                states = states.mean(axis=0, keepdims=True)
                level += 1
                continue
            n_groups = len(states) // f
            usable = n_groups * f
            grouped = states[:usable].reshape(n_groups, f, -1).mean(axis=1)
            if usable < len(states):                   # carry the leftover units up as a group
                grouped = np.vstack([grouped, states[usable:].mean(axis=0, keepdims=True)])
            states = grouped
            level += 1
        return levels
