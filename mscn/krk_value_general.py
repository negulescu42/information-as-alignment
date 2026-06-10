"""Scalable representation on KRK -- can a GENERALISING value substrate close the
technique gap the tabular agent cannot?

ARCHITECTURE 9.4's honest ceiling: the tabular closed-loop agent becomes SAFE
(win-preservation 0.999) but not FAST (DTM-optimality ~0.26, mates ~40 plies) --
262k afterstates are visited too sparsely to refine technique. This module attacks
the roadmap's open problem #2 (scalable representation) on that exact yardstick.

The prior-free move: the agent's learned legality tables ALREADY contain the board
geometry. The king-piece's acceptance graph (which piece is 'king-like' is itself
decided from the data: the tag with the smaller accepted out-degree) is the
adjacency graph of the 64 opaque squares; its Laplacian eigenmaps recover the 2-D
grid -- the agent derives COORDINATES from its own movement model (the Stage-2
result, now produced inside the acting loop). Two generalising substrates are then
built on those emergent coordinates, both trained by the same TD-as-IBF-MODIFY
updates as the table:

  * kernel-absolute -- a Gaussian-kernel delta-R memory (the Layer-1 substrate)
    over the 6-D absolute embedded state phi = (Z[wk], Z[bk], Z[wr]). The repo's
    recurring dimensionality wall predicts this struggles;
  * tile-relative   -- a linear tile-code over the three RELATIVE embedded
    displacements (bk-wk, bk-wr, wr-wk). One generic relational inductive bias
    (relative position matters), no chess knowledge -- the '8.10 structure buys
    data-efficiency' theme made testable in value learning.

Both run as HYBRIDS: the exact table where visited, the generalising substrate as
the prior for unvisited afterstates (basin expansion: the substrate only adds
information where the table has none).

PRE-REGISTERED (before the full run): at matched experience (50k episodes), the
tile-relative hybrid must beat the tabular agent on DTM-optimality AND
plies-to-mate (paired CIs over seeds, lo > 0 / hi < 0 respectively), with mate
rate not worse. The kernel-absolute arm is measured under the same criteria and
reported however it lands.

Run: ``python -m mscn.krk_value_general [--episodes 50000] [--seeds 3]``
(numpy+scipy; ~30-45 min).
"""

from __future__ import annotations

import numpy as np

from .krk_closed_loop import KRKClosedLoopAgent, run_agent
from .stats import fmt_ci, paired_ci, verdict


# ---------------------------------------------------------------------------
#  Emergent geometry: spectral embedding of the agent's OWN king-move graph
# ---------------------------------------------------------------------------

def emergent_embedding(agent: KRKClosedLoopAgent) -> tuple[np.ndarray, int]:
    """2-D Laplacian-eigenmap coordinates of the 64 squares, from the learned
    acceptance graph of the king-like tag (chosen by smaller out-degree)."""
    deg = [(agent.A[t] > 0).sum(axis=1).mean() for t in (0, 1)]
    ktag = int(np.argmin(deg))
    W = (agent.A[ktag] > 0).astype(float)
    W = np.maximum(W, W.T)                          # symmetrise
    L = np.diag(W.sum(axis=1)) - W
    vals, vecs = np.linalg.eigh(L)
    Z = vecs[:, 1:3]                                # first two non-trivial modes
    Z = (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0) + 1e-12)
    return Z.astype(np.float32), ktag


def procrustes_disparity(Z: np.ndarray) -> float:
    """Evaluation-only: how close is the emergent embedding to the true grid?"""
    from scipy.spatial import procrustes
    true = np.array([[s // 8, s % 8] for s in range(64)], dtype=float)
    _, _, disparity = procrustes(true, Z)
    return float(disparity)


def _phi(Z: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """[n, 6] absolute embedded state (wk, bk, wr) for afterstate placements."""
    wk, bk, wr = pids // 4096, (pids // 64) % 64, pids % 64
    return np.concatenate([Z[wk], Z[bk], Z[wr]], axis=1)


# ---------------------------------------------------------------------------
#  Generalising agents (hybrid: exact table where visited, substrate elsewhere)
# ---------------------------------------------------------------------------

class _HybridAgent(KRKClosedLoopAgent):
    """Table + generalising prior; geometry is built at `embed_at` episodes from
    the agent's own legality tables (before that: pure tabular)."""

    def __init__(self, embed_at: int = 8000, **kw) -> None:
        super().__init__(**kw)
        self.embed_at = embed_at
        self.Z: np.ndarray | None = None
        self.visits = np.zeros(64 * 64 * 64, dtype=np.uint8)
        self.episodes_seen = 0
        self.embed_info: dict = {}

    def maybe_embed(self) -> None:
        self.episodes_seen += 1
        if self.Z is None and self.episodes_seen >= self.embed_at:
            self.Z, ktag = emergent_embedding(self)
            self.embed_info = {"ktag": ktag,
                               "procrustes": procrustes_disparity(self.Z)}
            self._on_embed()

    def _on_embed(self) -> None:                    # substrate-specific setup
        pass

    def _general(self, pids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _general_update(self, pid: int, target: float) -> None:
        raise NotImplementedError

    def _value(self, pids: np.ndarray) -> np.ndarray:
        v = self.V[pids]
        if self.Z is None:
            return v
        # shrinkage blend: a state visited n times trusts its table value with
        # weight n against the generalising prior with weight lam -- a one-visit
        # Monte-Carlo entry should not override the substrate forever.
        n = self.visits[pids].astype(np.float32)
        g = self._general(pids)
        lam = 2.0
        return (n * v + lam * g) / (n + lam)

    def learn_value(self, afterstates, reward, truncated=False) -> None:
        if not afterstates:
            return
        # same backward sweep as the base, mirrored into the substrate
        if truncated:
            G = float(self._value(np.array([afterstates[-1]]))[0])
            rest = afterstates[:-1]
        else:
            term = afterstates[-1]
            self._td_abs.append(abs(reward - self.V[term]))
            self.V[term] = reward
            self.visits[term] = min(int(self.visits[term]) + 1, 255)   # no uint8 wrap
            if self.Z is not None:
                self._general_update(term, reward)
            G = self.gamma * reward
            rest = afterstates[:-1]
        for s in reversed(rest):
            d = G - self.V[s]
            self.V[s] += self.alpha * d
            self.visits[s] = min(int(self.visits[s]) + 1, 255)
            self._td_abs.append(abs(d))
            if self.Z is not None:
                self._general_update(s, G)
            G = self.gamma * G


class TileRelativeAgent(_HybridAgent):
    """Linear tile-code over the three RELATIVE embedded displacements, plus one
    pairwise-CONJUNCTION plane (the two displacements relative to the enemy king)
    -- additive planes cannot represent interaction patterns ('piece between
    kings'); a pairwise product can, still linear-in-features."""

    def __init__(self, n_tiles: int = 12, lr: float = 0.15, **kw) -> None:
        super().__init__(**kw)
        self.n_tiles, self.lr = n_tiles, lr
        n2 = n_tiles * n_tiles
        self.w = np.zeros((3, n2), dtype=np.float32)
        self.wc = np.zeros(n2 * n2, dtype=np.float32)           # conj(t0, t1)

    def _tiles(self, pids: np.ndarray) -> np.ndarray:
        wk, bk, wr = pids // 4096, (pids // 64) % 64, pids % 64
        rel = np.stack([self.Z[bk] - self.Z[wk],
                        self.Z[bk] - self.Z[wr],
                        self.Z[wr] - self.Z[wk]], axis=0)      # [3, n, 2]
        ij = np.clip(((rel + 1.0) * 0.5 * self.n_tiles).astype(int),
                     0, self.n_tiles - 1)
        return ij[..., 0] * self.n_tiles + ij[..., 1]           # [3, n]

    def _general(self, pids: np.ndarray) -> np.ndarray:
        t = self._tiles(pids)
        conj = t[0] * (self.n_tiles ** 2) + t[1]
        return (self.w[0, t[0]] + self.w[1, t[1]] + self.w[2, t[2]]
                + self.wc[conj]) / 4.0

    def _general_update(self, pid: int, target: float) -> None:
        t = self._tiles(np.array([pid]))[:, 0]
        conj = t[0] * (self.n_tiles ** 2) + t[1]
        pred = (self.w[0, t[0]] + self.w[1, t[1]] + self.w[2, t[2]]
                + self.wc[conj]) / 4.0
        err = self.lr * (target - pred) / 4.0
        for p in range(3):
            self.w[p, t[p]] += err
        self.wc[conj] += err


class KernelAbsoluteAgent(_HybridAgent):
    """Gaussian-kernel delta-R memory over the 6-D absolute embedded state.

    The bandwidth is NOT hand-tuned: once enough centres exist it is calibrated
    by the Operating-Bandwidth principle (the manuscript's sigma_op =
    d / sqrt(2 ln(N_eff/eps)); the chess_kernel.py implementation precedent):
    d = median nearest-neighbour radius of the stored centres (the locality
    radius of the embedded state space), N_eff = participation ratio of the
    nonlocal weights at the reference width sigma_ref = d, eps = 0.05."""

    def __init__(self, max_centers: int = 800, sigma: float = 0.22,
                 lr: float = 0.3, calibrate_at: int = 150,
                 eps_tol: float = 0.05, **kw) -> None:
        super().__init__(**kw)
        self.max_centers, self.sigma, self.klr = max_centers, sigma, lr
        self.calibrate_at, self.eps_tol = calibrate_at, eps_tol
        self.calibrated = False
        self.cz = np.zeros((0, 6), dtype=np.float32)
        self.cv = np.zeros(0, dtype=np.float32)

    def _calibrate(self) -> None:
        D = np.sqrt(((self.cz[:, None, :] - self.cz[None]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        d_shell = float(np.median(D.min(axis=1)))   # locality radius
        sref = d_shell
        neffs = []
        for q in range(min(64, self.cz.shape[0])):  # nonlocal participation ratio
            nl = D[q][np.isfinite(D[q]) & (D[q] > d_shell)]
            if nl.size:
                w = np.exp(-nl ** 2 / (2 * sref ** 2))
                neffs.append((w.sum() ** 2) / max((w ** 2).sum(), 1e-12))
        n_eff = float(np.mean(neffs)) if neffs else 1.0
        n_eff = max(n_eff, self.eps_tol * 1.01 / 1.0)
        self.sigma = d_shell / np.sqrt(2 * np.log(max(n_eff, 1.05) / self.eps_tol))
        self.calibrated = True
        self.embed_info["sigma_op"] = float(self.sigma)
        self.embed_info["d_shell"] = d_shell
        self.embed_info["n_eff"] = n_eff

    def _general(self, pids: np.ndarray) -> np.ndarray:
        if self.cz.shape[0] == 0:
            return np.zeros(pids.size, dtype=np.float32)
        f = _phi(self.Z, pids)                                  # [n, 6]
        d2 = ((f[:, None, :] - self.cz[None]) ** 2).sum(axis=2)
        K = np.exp(-d2 / (2 * self.sigma ** 2))
        s = K.sum(axis=1)
        return np.where(s > 1e-6, (K @ self.cv) / np.maximum(s, 1e-6), 0.0)

    def _general_update(self, pid: int, target: float) -> None:
        f = _phi(self.Z, np.array([pid]))[0]
        if self.cz.shape[0]:
            d2 = ((self.cz - f) ** 2).sum(axis=1)
            j = int(np.argmin(d2))
            if d2[j] < (0.5 * self.sigma) ** 2:                 # merge
                self.cv[j] += self.klr * (target - self.cv[j])
                return
        if self.cz.shape[0] < self.max_centers:
            self.cz = np.vstack([self.cz, f[None]])
            self.cv = np.append(self.cv, np.float32(target))
            if not self.calibrated and self.cz.shape[0] >= self.calibrate_at:
                self._calibrate()                   # operating-bandwidth sigma*


# ---------------------------------------------------------------------------
#  The pre-registered comparison
# ---------------------------------------------------------------------------

ARMS = {
    "tabular": lambda seed: None,                   # run_agent builds the default
    "tile-relative": lambda seed: TileRelativeAgent(seed=seed),
    "kernel-absolute": lambda seed: KernelAbsoluteAgent(seed=seed),
}


def run_arm(name: str, n_episodes: int, seed: int, window: int) -> list[dict]:
    ag = ARMS[name](seed)
    if ag is None:
        return run_agent(n_episodes, seed=seed, window=window)["windows"]
    # wrap the episode driver so the embedding fires at the right time
    orig = ag.learn_value

    def hooked(afterstates, reward, truncated=False):
        ag.maybe_embed()
        orig(afterstates, reward, truncated)
    ag.learn_value = hooked
    out = run_agent(n_episodes, seed=seed, window=window, agent=ag)
    if ag.embed_info:
        out["windows"][-1]["procrustes"] = ag.embed_info["procrustes"]
    return out["windows"]


def main(n_episodes: int = 50000, n_seeds: int = 3) -> None:
    print("\n" + "#" * 74)
    print("#  KRK GENERALISING VALUE -- emergent geometry + two substrates vs the")
    print("#  tabular ceiling (pre-registered: tile-relative must win on technique)")
    print("#" * 74)
    W = 2000
    runs: dict[str, list] = {}
    for name in ARMS:
        runs[name] = [run_arm(name, n_episodes, s, W) for s in range(n_seeds)]
        n_w = min(len(r) for r in runs[name])
        last = runs[name][0][n_w - 1]
        extra = (f"   emergent-geometry Procrustes {last['procrustes']:.3f}"
                 if "procrustes" in last else "")
        print(f"\n  [{name}] {n_seeds} seeds x {n_episodes} episodes{extra}")
        print(f"  {'window':>7}{'mate':>8}{'plies':>8}{'preserve':>10}{'optimal':>9}")
        for i in range(n_w):
            if i % max(n_w // 6, 1) and i != n_w - 1:
                continue
            m = {k: float(np.nanmean([r[i][k] for r in runs[name]]))
                 for k in ("mate", "plies", "preserve", "optimal")}
            print(f"  {(i + 1) * W:>7}{m['mate']:>8.3f}{m['plies']:>8.1f}"
                  f"{m['preserve']:>10.3f}{m['optimal']:>9.3f}")

    n_w = min(len(r) for rs in runs.values() for r in rs)

    def last_col(name, key):
        return np.array([float(np.nanmean([r[i][key] for i in range(n_w - 3, n_w)]))
                         for r in runs[name]])

    print("\n  pre-registered comparison at matched experience "
          "(mean of last 3 windows, paired over seeds):")
    cis = {}
    for name in ("tile-relative", "kernel-absolute"):
        for key, better in (("optimal", "hi"), ("plies", "lo"), ("mate", "hi")):
            ci = paired_ci(last_col(name, key), last_col("tabular", key))
            cis[(name, key)] = ci
            print(f"   {name:<16} {key:<8} vs tabular: {fmt_ci(ci)} {verdict(ci)}")

    ok_opt = cis[("tile-relative", "optimal")]["lo"] > 0
    ok_plies = cis[("tile-relative", "plies")]["hi"] < 0
    ok_mate = cis[("tile-relative", "mate")]["hi"] > -0.02
    print(f"\n  PRE-REGISTERED CRITERION (tile-relative beats tabular on technique):"
          f" {'MET' if ok_opt and ok_plies and ok_mate else 'NOT MET'}")
    print(f"   optimality sig-up: {ok_opt}; plies sig-down: {ok_plies}; "
          f"mate-rate not worse: {ok_mate}")
    kern = cis[("kernel-absolute", "optimal")]
    print(f"   kernel-absolute (reported however it lands): optimality "
          f"{fmt_ci(kern)} {verdict(kern)} -- the dimensionality wall "
          f"{'holds' if kern['lo'] <= 0 else 'is crossed'} on the 6-D absolute state.")
    if n_episodes >= 30000:
        assert ok_mate, "the generalising prior must not cost mate rate"
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="KRK generalising value substrates")
    p.add_argument("--episodes", type=int, default=50000)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    main(n_episodes=a.episodes, n_seeds=a.seeds)
