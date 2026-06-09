"""Layer 3 (part A) -- coarse-graining, renormalisation flow, hierarchy.

The meta-layer relates fine-scale configurations to coarse-scale ones through a
surjective projection (Postulate II). Two proven facts are demonstrated:

* **Renormalisation flow** (RenormalizationGroup.lean): under repeated
  coarse-graining the *relevant* (large-scale) structure is preserved while
  *irrelevant* (small-scale) fluctuations are suppressed -- the signal-to-noise
  ratio grows.
* **Coherence non-increasing under coarsening**: block-averaging cannot create
  coherence that was not present at the fine scale (the coarse peak never
  exceeds the fine peak).

The :class:`HierarchicalOptimizer` then puts this to work: by coarse-graining a
high-dimensional problem into independent blocks it optimises each scale
separately and beats a flat learner of equal evaluation budget when the problem
has genuine block (multi-scale) structure.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .landscapes import Landscape
from .learner import IBFLearner

ArrayF = np.ndarray


# ---------------------------------------------------------------------------
#  Coarse-graining projection
# ---------------------------------------------------------------------------

def block_means(x: ArrayF, n_blocks: int) -> ArrayF:
    """Surjective coarse-graining: average each contiguous block (Postulate II)."""
    x = np.asarray(x, dtype=float)
    return x.reshape(n_blocks, -1).mean(axis=1)


def coarse_grain_1d(field: ArrayF, factor: int = 2) -> ArrayF:
    """One RG step on a 1-D field: average non-overlapping windows of `factor`."""
    field = np.asarray(field, dtype=float)
    usable = (field.size // factor) * factor
    return field[:usable].reshape(-1, factor).mean(axis=1)


def rg_flow(field: ArrayF, factor: int = 2, steps: int = 3) -> list[ArrayF]:
    out = [np.asarray(field, dtype=float)]
    for _ in range(steps):
        if out[-1].size < factor:
            break
        out.append(coarse_grain_1d(out[-1], factor))
    return out


def signal_to_noise(field: ArrayF, signal: ArrayF) -> float:
    """Power of the (resampled) signal component over residual-noise power."""
    field = np.asarray(field, dtype=float)
    sig = np.interp(np.linspace(0, 1, field.size), np.linspace(0, 1, len(signal)), signal)
    noise = field - sig
    p_sig = float(np.mean(sig ** 2))
    p_noise = float(np.mean(noise ** 2)) + 1e-12
    return p_sig / p_noise


# ---------------------------------------------------------------------------
#  Hierarchical optimiser
# ---------------------------------------------------------------------------

class HierarchicalOptimizer:
    """Coarse-grain a block-structured landscape and optimise each block.

    The landscape is assumed (approximately) block-separable:
    ``f(x) = sum_b f_b(block_b)``. Each block is handed to its own Layer-1
    learner; a top-level meta step then jointly polishes the assembled solution.
    """

    def __init__(self, landscape: Landscape, n_blocks: int, seed: int = 0) -> None:
        assert landscape.dim % n_blocks == 0, "dim must divide into equal blocks"
        self.L = landscape
        self.n_blocks = n_blocks
        self.block_size = landscape.dim // n_blocks
        self.seed = seed

    def _block_coherence(self, b: int) -> Callable[[ArrayF], float]:
        lo = b * self.block_size
        hi = lo + self.block_size

        def coh(xb: ArrayF, _lo=lo, _hi=hi) -> float:
            full = self._anchor.copy()
            full[_lo:_hi] = xb
            return -self.L.f(full)

        return coh

    def run(self, eval_budget: int) -> dict:
        self._anchor = np.zeros(self.L.dim)
        per_block = max(eval_budget // self.n_blocks, 1)
        x = np.zeros(self.L.dim)
        for b in range(self.n_blocks):
            lo = b * self.block_size
            hi = lo + self.block_size
            sub = IBFLearner(
                self._block_coherence(b),
                self.L.lo[lo:hi], self.L.hi[lo:hi],
                alpha=0.4, mu=0.03, k=1.0, k_adapt=0.05,
                seed=self.seed + b,
            )
            res = sub.run_until_evals(per_block)
            x[lo:hi] = res["best_x"]
        return {"best_x": x, "best_f": float(self.L.f(x)), "n_blocks": self.n_blocks}


def flat_baseline(landscape: Landscape, eval_budget: int, seed: int = 0) -> dict:
    """A single flat Layer-1 learner over the full space (same eval budget)."""
    learner = IBFLearner(
        landscape.coherence, landscape.lo, landscape.hi,
        alpha=0.4, mu=0.03, k=1.0, k_adapt=0.05, seed=seed,
    )
    res = learner.run_until_evals(eval_budget)
    return {"best_x": res["best_x"], "best_f": res["best_f"]}


def block_landscape(base_factory: Callable[[int], Landscape], n_blocks: int,
                    block_size: int) -> Landscape:
    """Compose ``n_blocks`` independent copies of a base landscape side by side."""
    from .landscapes import Landscape as _L
    base = base_factory(block_size)
    dim = n_blocks * block_size

    def f(x: ArrayF) -> float:
        x = np.asarray(x, dtype=float)
        return float(sum(base.f(x[b * block_size:(b + 1) * block_size]) for b in range(n_blocks)))

    def f_batch(X: ArrayF) -> ArrayF:
        X = np.asarray(X, dtype=float)
        return sum(base.f_batch(X[:, b * block_size:(b + 1) * block_size]) for b in range(n_blocks))

    lo = np.tile(base.lo, n_blocks)
    hi = np.tile(base.hi, n_blocks)
    x_opt = np.tile(base.x_opt, n_blocks)
    return _L(f"block-{base.name}x{n_blocks}", dim, lo, hi, f, f_batch, x_opt, n_blocks * base.f_opt)
