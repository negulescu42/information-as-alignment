"""A *full* IBF unit for chess -- the closed acting loop, not a batch predictor.

The n-gram/kernel/VOM chess models instantiate only the IBF unit's memory +
selection substrate. This module runs the **complete Layer-1 agent loop** online
over the game stream, so the chess model is a genuine IBF unit:

    SENSE   R_eff(c, m) = R̂(m) + Σ_ℓ w_ℓ · δR(c_ℓ, m)      (baseline + kernel-localised modification)
    SELECT  Boltzmann policy  P(m | c) ∝ exp(k · R_eff)
    ACT     emit the move; the environment (the game) reveals the demonstrated move a
    MODIFY  discrepancy D = T_ext − R_eff(c, a);  δR(c_ℓ, a) ← δR + α·max(D,0) − μ·δR
            competitors decay  δR(c_ℓ, m≠a) ← (1−μ)·δR   (selective retention)
    ADAPT   if the agent already rated a coherent (R_eff(c,a) high), k ← k + k_adapt (agency)

`δR` is a coherence modification over the configuration-tree (move-context
suffixes up to `max_depth`); the readout weights `w_ℓ` are the localisation kernel
over that tree (deeper context = more specific). `R̂` is the baseline landscape
(global log-frequency). It learns in a single **continual** online pass — no batch
counts — and exhibits the proven IBF-unit properties (Boltzmann monotonicity,
selective retention `δR→α/μ`, adaptive agency).

Drop-in compatible with `evaluate_rules` (`predict`, `inv_vocab`, `unigram`).
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


class IBFChessAgent:
    def __init__(self, max_depth: int = 4, alpha: float = 0.5, mu: float = 0.04,
                 k: float = 1.0, k_adapt: float = 0.002, k_max: float = 8.0,
                 T_ext: float = 4.0, global_topk: int = 40, agency_frac: float = 0.5,
                 adaptive_mu: bool = False, mu_lambda: float = 1.0,
                 seed: int = 0) -> None:
        self.D = max_depth
        self.alpha = alpha
        self.mu = mu
        self.adaptive_mu = adaptive_mu        # per-center mu_i = mu/(1+lambda*count) (roadmap 1.3)
        self.mu_lambda = mu_lambda
        self.k = k
        self.k0 = k
        self.k_adapt = k_adapt
        self.k_max = k_max
        self.T_ext = T_ext
        self.global_topk = global_topk
        self.agency_frac = agency_frac
        self.vocab: dict[str, int] = {}
        self.inv_vocab: list[str] = []
        self.unigram = Counter()              # for evaluate_rules' unigram baseline
        self.R_hat: dict[int, float] = {}     # baseline coherence (log global count)
        self.dR: dict[tuple, dict[int, float]] = {}   # context -> {move: modification}
        self.cnt: dict[tuple, dict[int, int]] = {}    # context -> {move: reinforcement count}
        self.k_history: list[float] = []
        self._agency_hits = 0
        self._steps = 0

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.inv_vocab)
            self.inv_vocab.append(tok)
        return self.vocab[tok]

    @staticmethod
    def _w(ell: int) -> float:
        return float(ell)  # deeper (more specific) context weighted more

    def _active(self, ids: list[int]) -> list[tuple[int, tuple]]:
        D = min(self.D, len(ids))
        return [(ell, tuple(ids[len(ids) - ell:])) for ell in range(1, D + 1)]

    def _r_eff(self, active: list[tuple[int, tuple]], m: int) -> float:
        s = self.R_hat.get(m, 0.0)
        for ell, ctx in active:
            d = self.dR.get(ctx)
            if d is not None:
                v = d.get(m)
                if v is not None:
                    s += self._w(ell) * v
        return s

    def _global_top(self) -> list[int]:
        return [m for m, _ in self.unigram.most_common(self.global_topk)]

    def _candidates(self, active: list[tuple[int, tuple]]) -> set[int]:
        cands: set[int] = set(self._global_top())
        for _, ctx in active:
            d = self.dR.get(ctx)
            if d:
                cands |= set(d.keys())
        return cands

    # ----- one acting step: sense -> select -> observe -> modify -> adapt -----
    def observe_step(self, ids: list[int], a: int) -> None:
        self._steps += 1
        active = self._active(ids)
        reff_a = self._r_eff(active, a)        # effective coherence the agent already assigns to a

        # ADAPT (agency): if the agent already rated the demonstrated move coherent,
        # it is "getting it right" -> grow responsiveness.
        if reff_a >= self.agency_frac * self.T_ext:
            self.k = min(self.k + self.k_adapt, self.k_max)
            self._agency_hits += 1

        # MODIFY: discrepancy-driven reinforcement of a + competitive decay (retention).
        # Per-center adaptive mu (roadmap 1.3): mu_i = mu/(1+lambda*count_i) -- a move
        # reinforced many times (a stable rule) crystallises (mu_i -> 0); a one-off move
        # stays plastic (mu_i ~ mu) and fades. Amplitude bound holds for each center.
        disc = max(self.T_ext - reff_a, 0.0)
        for ell, ctx in active:
            d = self.dR.get(ctx)
            if d is None:
                d = self.dR[ctx] = {}
                self.cnt[ctx] = {}
            c = self.cnt[ctx]
            c[a] = c.get(a, 0) + 1
            mu_a = self.mu / (1.0 + self.mu_lambda * (c[a] - 1)) if self.adaptive_mu else self.mu
            old = d.get(a, 0.0)
            d[a] = max(old + self.alpha * disc - mu_a * old, 0.0)
            if self.mu > 0 and len(d) > 1:
                for m in list(d.keys()):
                    if m != a:
                        mu_m = self.mu / (1.0 + self.mu_lambda * c.get(m, 0)) if self.adaptive_mu else self.mu
                        d[m] *= (1.0 - mu_m)

        # baseline landscape update (global frequency)
        self.unigram[a] += 1
        self.R_hat[a] = math.log1p(self.unigram[a])

    def train(self, games: list[list[str]], passes: int = 1) -> "IBFChessAgent":
        for _ in range(passes):
            for g in games:
                ids = [self._id(t) for t in g]
                for i, a in enumerate(ids):
                    self.observe_step(ids[:i], a)
                self.k_history.append(self.k)
        return self

    # ----- prediction: Boltzmann over effective coherence -----
    def predict(self, history_tokens: list[str], top: int | None = None) -> list[tuple[str, float]]:
        ids = [self.vocab[t] for t in history_tokens if t in self.vocab]
        active = self._active(ids)
        cands = self._candidates(active)
        if not cands:
            return []
        scores = np.array([self._r_eff(active, m) for m in cands], dtype=float)
        cl = list(cands)
        logits = self.k * scores
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        order = np.argsort(-p)
        out = [(self.inv_vocab[cl[i]], float(p[i])) for i in order]
        return out[:top] if top else out

    def prob_of(self, history_tokens: list[str], token: str) -> float:
        if token not in self.vocab:
            return 0.0
        return dict(self.predict(history_tokens)).get(token, 0.0)

    def move_salience(self) -> dict[str, float]:
        return {self.inv_vocab[i]: c for i, c in self.unigram.items()}

    # ----- IBF-unit diagnostics -----
    def agency_rate(self) -> float:
        return self._agency_hits / max(self._steps, 1)

    def retention_stats(self) -> dict:
        """Distribution of learned modifications: reinforced (high) vs decayed (low)."""
        vals = [v for d in self.dR.values() for v in d.values()]
        vals = np.array(vals) if vals else np.array([0.0])
        eq = self.alpha / self.mu if self.mu else float("inf")
        ref = (0.05 * self.alpha / self.mu) if self.mu else max(0.05 * float(vals.max()), 1e-9)
        return {"equilibrium_alpha_over_mu": eq,
                "max_dR": float(vals.max()), "mean_dR": float(vals.mean()),
                "frac_near_zero": float(np.mean(vals < ref))}
