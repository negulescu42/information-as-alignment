"""Kernel-based IBF chess model (the faithful IBF mechanism).

The n-gram model in `chess_world.py` keys coherence on an *exact* context match,
so a novel position falls back to global statistics. The actual IBF mechanism is
**kernel-based** (Gaussian-kernel memory centres over a configuration space, as in
the repo's `IBFAgent`): coherence at a query is the kernel-weighted sum over
stored experience, so *similar* positions share coherence and novel positions
generalise from nearby ones.

This module implements that for chess, still under strict no-priors (moves are
opaque tokens):

1. **Token embeddings** are learned from move co-occurrence (PPMI + truncated
   SVD) — purely distributional, no board/piece knowledge.
2. A **context vector** `z(history)` is a recency-weighted sum of recent move
   embeddings (a continuous summary of the position-so-far).
3. **Coherence memory**: every observed transition stores a centre `(z, next)`.
   Prediction at a query `z` is a Gaussian-kernel-weighted vote over the nearest
   centres (Nadaraya-Watson) — `δR_a(z) = Σ_i K(z, z_i)·[next_i = a]`.

Drop-in compatible with `evaluate_rules` and `recover_board_geometry`
(`predict`, `inv_vocab`, `move_salience`). Requires `scipy`.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

try:
    from scipy.sparse.linalg import svds
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False


def operating_bandwidth(d_shell: float, n_eff: float, eps: float) -> float:
    """Optimal kernel bandwidth (Operating Resolution principle, arXiv:2604.07108):

        sigma* = d_shell / sqrt(2 * log(N_eff / eps))

    the largest bandwidth for which the aggregate bleed from ``N_eff`` interfering
    centres at distance >= ``d_shell`` stays below tolerance ``eps``. Requires
    ``n_eff > eps > 0``.
    """
    return d_shell / np.sqrt(2.0 * np.log(n_eff / eps))


class KernelIBFChessModel:
    def __init__(self, dim: int = 24, cooc_window: int = 4, ctx_len: int = 6,
                 gamma: float = 0.7, k_neighbors: int = 64, epsilon: float = 0.01,
                 sigma_scale: float = 1.0, unigram_mix: float = 0.05,
                 max_centers: int = 500_000, seed: int = 0) -> None:
        self.dim = dim
        self.cooc_window = cooc_window
        self.ctx_len = ctx_len
        self.gamma = gamma
        self.k = k_neighbors
        self.epsilon = epsilon
        self.sigma_scale = sigma_scale
        self.unigram_mix = unigram_mix
        self.max_centers = max_centers
        self.rng = np.random.default_rng(seed)
        self.vocab: dict[str, int] = {}
        self.inv_vocab: list[str] = []
        self.unigram = Counter()

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.inv_vocab)
            self.inv_vocab.append(tok)
        return self.vocab[tok]

    # ----- training -----
    def train(self, games: list[list[str]]) -> "KernelIBFChessModel":
        if not HAS_SCIPY:
            raise RuntimeError("scipy required (pip install scipy)")
        seqs = [[self._id(t) for t in g] for g in games]
        V = len(self.inv_vocab)
        for s in seqs:
            self.unigram.update(s)

        # (1) move embeddings from co-occurrence (PPMI + SVD) -- prior-free
        C = np.zeros((V, V))
        for s in seqs:
            if len(s) < 2:
                continue
            a = np.array(s)
            for off in range(1, self.cooc_window + 1):
                if len(a) > off:
                    np.add.at(C, (a[:-off], a[off:]), 1.0)
                    np.add.at(C, (a[off:], a[:-off]), 1.0)
        E = self._embeddings(C)
        self.E = E

        # (2)+(3) context centres (z -> next token), reservoir-capped for scale
        Z, A, n_seen = [], [], 0
        cap = self.max_centers
        for s in seqs:
            zacc = np.zeros(self.dim)
            for t, nxt in enumerate(s):
                if t > 0:
                    z = zacc / (np.linalg.norm(zacc) + 1e-9)
                    if len(Z) < cap:
                        Z.append(z.copy()); A.append(nxt)
                    else:  # reservoir sampling keeps it bounded at scale
                        j = self.rng.integers(0, n_seen + 1)
                        if j < cap:
                            Z[j] = z.copy(); A[j] = nxt
                    n_seen += 1
                zacc = self.gamma * zacc + E[nxt]  # recency-weighted running context
        self.Z = np.array(Z)
        self.A = np.array(A)
        self.tree = cKDTree(self.Z)

        # (4) kernel bandwidth via the Operating Resolution principle
        #     sigma* = d_shell / sqrt(2 * log(N_eff / eps))   (arXiv:2604.07108)
        n_s = min(2000, len(self.Z))
        sample = self.Z[self.rng.choice(len(self.Z), size=n_s, replace=False)]
        kk = min(self.k, len(self.Z))
        d, _ = self.tree.query(sample, k=kk)              # distances to local k-NN
        d_shell = float(np.median(d[:, -1]))              # local-neighbourhood radius (shell)
        sref = d_shell + 1e-12                            # reference scale for N_eff (Path A)
        w = np.exp(-(d ** 2) / (2 * sref ** 2))           # kernel weights at sigma_ref
        n_eff = float(np.mean(w.sum(1) ** 2 / (np.sum(w ** 2, axis=1) + 1e-12)))
        self.d_shell, self.n_eff = d_shell, max(n_eff, 1.0)
        self.sigma = self.sigma_scale * operating_bandwidth(
            d_shell, max(self.n_eff, self.epsilon * 1.001), self.epsilon) + 1e-12

        tot = sum(self.unigram.values())
        self._uni = {i: c / tot for i, c in self.unigram.items()}
        return self

    def _embeddings(self, C: np.ndarray) -> np.ndarray:
        total = C.sum() + 1e-9
        row = C.sum(1) + 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((C / total) / ((row[:, None] / total) * (row[None, :] / total)) + 1e-12)
        ppmi = np.maximum(pmi, 0.0)
        ppmi[C == 0] = 0.0
        d = min(self.dim, min(ppmi.shape) - 1)
        U, S, _ = svds(ppmi, k=d)
        E = U * np.sqrt(np.maximum(S, 0.0))
        if E.shape[1] < self.dim:  # pad if vocab tiny
            E = np.pad(E, ((0, 0), (0, self.dim - E.shape[1])))
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        return E

    # ----- context vector + prediction -----
    def _context_vec(self, ids: list[int]) -> np.ndarray | None:
        if not ids:
            return None
        recent = ids[-self.ctx_len:]
        m = len(recent)
        w = self.gamma ** np.arange(m - 1, -1, -1)  # most recent largest
        z = (w[:, None] * self.E[recent]).sum(0)
        n = np.linalg.norm(z)
        return z / n if n > 1e-9 else None

    def predict(self, history_tokens: list[str], top: int | None = None) -> list[tuple[str, float]]:
        ids = [self.vocab[t] for t in history_tokens if t in self.vocab]
        z = self._context_vec(ids)
        if z is None:
            ranked = sorted(self._uni.items(), key=lambda kv: -kv[1])
            out = [(self.inv_vocab[i], p) for i, p in ranked]
            return out[:top] if top else out
        kk = min(self.k, len(self.Z))
        dist, idx = self.tree.query(z, k=kk)
        dist = np.atleast_1d(dist); idx = np.atleast_1d(idx)
        w = np.exp(-(dist ** 2) / (2 * self.sigma ** 2))
        scores: dict[int, float] = {}
        for wi, a in zip(w, self.A[idx]):
            scores[a] = scores.get(a, 0.0) + float(wi)
        zsum = sum(scores.values())
        dist_d: dict[int, float] = {}
        if zsum > 0:
            for a, sc in scores.items():
                dist_d[a] = (1 - self.unigram_mix) * sc / zsum
        for a, p in self._uni.items():  # smoothing / backoff
            dist_d[a] = dist_d.get(a, 0.0) + self.unigram_mix * p
        ranked = sorted(dist_d.items(), key=lambda kv: -kv[1])
        out = [(self.inv_vocab[i], p) for i, p in ranked]
        return out[:top] if top else out

    def prob_of(self, history_tokens: list[str], token: str) -> float:
        if token not in self.vocab:
            return 0.0
        return dict(self.predict(history_tokens)).get(token, 0.0)

    def move_salience(self) -> dict[str, float]:
        return {self.inv_vocab[i]: c for i, c in self.unigram.items()}
