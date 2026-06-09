"""Route A -- a recurrent, variable-order sufficient-statistic chess model.

The kernel model failed because its context is a *lossy sum* (provably not
predictively sufficient -- `formal/CausalStates.lean`). Route A replaces it with a
genuine recurrent state `z_t = G(z_{t-1}, a_t)` that aims at sufficiency.

The non-neural, prior-free realization is a **variable-order Markov / prediction-
suffix machine**: the recurrent state `z_t` is the *deepest move-context suffix that
has support in the data*; `G` walks one token deeper, or backs off when the deeper
context is unseen. This is an IBF **adaptive-resolution coarse-graining** -- it
refines the history partition exactly where finer resolution sharpens the next-move
distribution, and coarsens (backs off) elsewhere. Unlike a fixed n-gram it uses
*as much history as prediction needs*, moving toward the causal-state partition that
the theorem identifies as the unique coarsest sufficient representation.

It is still a *statistic of the move history* (bounded depth), so it is an
approximation to a fully simulation-faithful state, not the final word -- but it is
the theory-indicated step, and it lets us measure predictive **sufficiency**
directly (does the recurrent state determine the legal-move set?).

Drop-in compatible with `evaluate_rules` (`predict`, `inv_vocab`, `unigram`).
"""

from __future__ import annotations

from collections import Counter

import numpy as np


class RecurrentIBFChessModel:
    """Variable-order recurrent state with interpolated backoff."""

    def __init__(self, max_depth: int = 6, backoff_K: float = 2.0, min_count: int = 2) -> None:
        self.max_depth = max_depth
        self.backoff_K = backoff_K
        self.min_count = min_count
        self.vocab: dict[str, int] = {}
        self.inv_vocab: list[str] = []
        self.counts: dict[tuple, Counter] = {}
        self.unigram = Counter()

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.inv_vocab)
            self.inv_vocab.append(tok)
        return self.vocab[tok]

    # ----- training: count every suffix up to max_depth -----
    def train(self, games: list[list[str]]) -> "RecurrentIBFChessModel":
        D = self.max_depth
        for g in games:
            ids = [self._id(t) for t in g]
            for t, nxt in enumerate(ids):
                self.unigram[nxt] += 1
                lo = max(0, t - D)
                for ell in range(0, t - lo + 1):
                    ctx = tuple(ids[t - ell:t])  # length-ell suffix (ell=0 -> global ())
                    c = self.counts.get(ctx)
                    if c is None:
                        self.counts[ctx] = c = Counter()
                    c[nxt] += 1
        return self

    # ----- recurrent state z_t = deepest supported suffix -----
    def state_id(self, ids: list[int]) -> tuple:
        D = min(self.max_depth, len(ids))
        for ell in range(D, 0, -1):
            ctx = tuple(ids[len(ids) - ell:])
            c = self.counts.get(ctx)
            if c is not None and sum(c.values()) >= self.min_count:
                return ctx
        return ()

    # ----- interpolated variable-order prediction -----
    def _dist(self, ids: list[int]) -> dict[int, float]:
        base = self.counts.get((), Counter())
        tot0 = sum(base.values())
        dist = {a: c / tot0 for a, c in base.items()} if tot0 > 0 else {}
        D = min(self.max_depth, len(ids))
        for ell in range(1, D + 1):
            ctx = tuple(ids[len(ids) - ell:])
            c = self.counts.get(ctx)
            if not c:
                continue
            tot = sum(c.values())
            lam = tot / (tot + self.backoff_K)        # trust deep context iff well-supported
            ml = {a: cnt / tot for a, cnt in c.items()}
            keys = set(ml) | set(dist)
            dist = {a: lam * ml.get(a, 0.0) + (1 - lam) * dist.get(a, 0.0) for a in keys}
        return dist

    def predict(self, history_tokens: list[str], top: int | None = None) -> list[tuple[str, float]]:
        ids = [self.vocab[t] for t in history_tokens if t in self.vocab]
        dist = self._dist(ids)
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        out = [(self.inv_vocab[i], p) for i, p in ranked]
        return out[:top] if top else out

    def prob_of(self, history_tokens: list[str], token: str) -> float:
        if token not in self.vocab:
            return 0.0
        ids = [self.vocab[t] for t in history_tokens if t in self.vocab]
        return self._dist(ids).get(self.vocab[token], 0.0)

    def move_salience(self) -> dict[str, float]:
        return {self.inv_vocab[i]: c for i, c in self.unigram.items()}

    def context_id(self, history_tokens: list[str]) -> tuple:
        """The recurrent state id for a raw token history (for the purity probe)."""
        ids = [self.vocab[t] for t in history_tokens if t in self.vocab]
        return self.state_id(ids)
