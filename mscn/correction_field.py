"""Interface-Principle pruning -- O(M) -> O(M_boundary) correction-field evaluation.

A correction field is `δR(y) = Σ_i v_i · K_σ(y, z_i)` with the Gaussian kernel. The
Interface Principle (`interface_principle`, `InterfacePrinciple.lean`) proves the
aggregate tail from non-local centres is controlled by the **boundary** subset; a
centre at distance > r contributes at most `V_max · exp(-r²/(2σ²))`, which is
exponentially small. So we may evaluate the field using only centres within a shell
`r = c·σ` of the query (found in O(log M + M_boundary) via a k-d tree) and skip the
deep interior, with total error bounded by `V_max · M · exp(-c²/2)`.

This drops per-query cost from O(M) (sum over all centres) to O(M_boundary) (the
local boundary set), which is what lets a kernel `δR` memory scale to 10^4–10^6
centres. **Caveat (`no_shielding_equal_weights`):** this relies on genuine
distance-based decay (the Gaussian kernel). It does *not* apply to count-based,
distance-free memories (the n-gram / VOM chess models) — there is no metric to prune
on; all entries must be considered.
"""

from __future__ import annotations

import time

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False


class CorrectionField:
    def __init__(self, centers: np.ndarray, amplitudes: np.ndarray, sigma: float) -> None:
        self.Z = np.asarray(centers, dtype=float)
        self.V = np.asarray(amplitudes, dtype=float)
        self.sigma = float(sigma)
        self.tree = cKDTree(self.Z) if HAS_SCIPY else None

    # full O(M) evaluation
    def eval_full(self, y: np.ndarray) -> float:
        sq = np.sum((self.Z - y) ** 2, axis=1)
        return float(self.V @ np.exp(-sq / (2 * self.sigma ** 2)))

    def eval_full_batch(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, dtype=float)
        sq = (np.sum(Y * Y, axis=1)[:, None] + np.sum(self.Z * self.Z, axis=1)[None, :]
              - 2.0 * Y @ self.Z.T)
        return np.exp(-sq / (2 * self.sigma ** 2)) @ self.V

    # Interface-Principle-pruned O(M_boundary) evaluation
    def eval_pruned(self, y: np.ndarray, c: float = 3.5) -> float:
        idx = self.tree.query_ball_point(y, r=c * self.sigma)
        if not idx:
            return 0.0
        Zb = self.Z[idx]
        sq = np.sum((Zb - y) ** 2, axis=1)
        return float(self.V[idx] @ np.exp(-sq / (2 * self.sigma ** 2)))

    def tail_bound(self, c: float = 3.5) -> float:
        """Worst-case error from pruning at r = c·σ: V_max · M · exp(-c²/2)."""
        return float(np.max(np.abs(self.V))) * len(self.Z) * np.exp(-c ** 2 / 2)


def interface_pruning_benchmark(dim: int = 8, sigma: float = 0.5, density: float = 0.8,
                                Ms=(1000, 10000, 100000), n_queries: int = 300,
                                c: float = 3.5, seed: int = 0) -> None:
    """Fixed local density, growing volume: M_boundary ~ const while M grows, so the
    pruned evaluation is ~O(1)/query and the full one is O(M)."""
    if not HAS_SCIPY:
        print("scipy required"); return
    rng = np.random.default_rng(seed)
    print(f"Interface-Principle pruning  (dim={dim}, sigma={sigma}, c={c}, "
          f"fixed density={density}; r=c·σ={c*sigma:.2f})")
    print(f"  {'M':>8}{'M_bndry':>9}{'full ms/q':>11}{'pruned ms/q':>13}{'speedup':>9}"
          f"{'max err':>11}{'tail bound':>12}")
    for M in Ms:
        L = (M / density) ** (1.0 / dim)
        Z = rng.uniform(0, L, size=(M, dim))
        V = rng.uniform(-1, 1, size=M)
        F = CorrectionField(Z, V, sigma)
        Q = rng.uniform(0, L, size=(n_queries, dim))
        t = time.perf_counter()
        full = np.array([F.eval_full(y) for y in Q])
        t_full = (time.perf_counter() - t) / n_queries * 1e3
        t = time.perf_counter()
        pruned = np.array([F.eval_pruned(y, c) for y in Q])
        t_prun = (time.perf_counter() - t) / n_queries * 1e3
        mb = float(np.mean([len(F.tree.query_ball_point(y, r=c * sigma)) for y in Q]))
        err = float(np.max(np.abs(full - pruned)))
        print(f"  {M:>8}{mb:>9.0f}{t_full:>11.3f}{t_prun:>13.3f}{t_full/max(t_prun,1e-9):>8.0f}x"
              f"{err:>11.2e}{F.tail_bound(c):>12.2e}")


if __name__ == "__main__":
    interface_pruning_benchmark()
