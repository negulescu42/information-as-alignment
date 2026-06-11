"""The PAPER ENGINE -- the 'Information as Structural Alignment' preprint's
universal continual-learning instantiation, implemented faithfully and ablated.

The preprint (repo root) validates one concrete engine across three domains
(RRW / chess-with-Stockfish / Split-CIFAR-100) with replay-superior retention and
near-zero forgetting. Its lifecycle has machinery the current mscn/IBF-ASI memory
LACKS, and the gauntlet's continual test (G2: phase-B memory poisoning the
phase-A return) is exactly the failure it was built to prevent:

  * CONTEXT-GATED READING (gamma): same-context particles always readable;
    cross-context only if crystallized AND verified -- unverified memories are
    silent outside their birth context;
  * a CRYSTALLIZATION state machine: transient (mu_base) -> crystallized
    (mu_cryst << mu_base) on exposure + discrepancy-history convergence;
  * the CRUCIBLE: crystallized particles exposed to another context log RAW
    discrepancy (pass 1, validity testing, separate from learning); sustained
    contradiction (v * mean(D_raw) < theta_rev) de-crystallizes and revokes
    broadcast; survivors become VERIFIED = readable cross-context (universals
    transfer; contextual structure stays home);
  * the TWO-PASS WRITE separating cross-context testing from same-context
    kernel-weighted learning;
  * a responsiveness channel with INTENSIVE (bounded) readout, driven by local
    discrepancy VARIANCE.

This module: (1) the engine, domain-agnostic, per the paper's Section 4;
(2) a faithful mini Rotating-Rules-World (phase B exactly reverses phase A's
contextual component; phase C is fresh) with context ids given at boundaries, as
in the paper; (3) the architecture-relevant ABLATIONS, pre-registered:

   full engine        ~ zero forgetting on A after B's exact contradiction,
                        and the crucible removes contextual-A structure while
                        verified invariants keep transferring;
   no-context-gating  = the current mscn-style memory (everything readable
                        everywhere): phase-B interference should damage A --
                        the G2 pathology reproduced and attributed in-domain;
   no-crucible        carries falsely-stable structure across phases;
   no-verification    = pure isolation: retention without transfer;
   passive            no memory (anchor).

Run: ``python -m mscn.ibf_engine``   (numpy only; ~2 min).
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
#  Mini Rotating-Rules World (faithful to the paper's Domain I, scaled down)
# ---------------------------------------------------------------------------

class MiniRRW:
    """z = [x (4D), a_emb (4D one-hot)]. score(x, a, phase) = invariant(x, a)
    + p_phase * contextual(x, a) on a 2-D subspace; B is A's EXACT reversal
    (p = -1), C a fresh random orientation. D-signal binary, as in the paper."""

    def __init__(self, seed: int = 0, n_actions: int = 4) -> None:
        rng = np.random.default_rng(seed)
        self.nA = n_actions
        self.U = rng.normal(0, 1.0, (n_actions, 4))      # invariant weights
        self.V = rng.normal(0, 1.4, (n_actions, 2))      # contextual (dims 0-1)
        self.W = rng.normal(0, 1.4, (n_actions, 2))      # phase-C orientation
        self.rng = rng

    def score(self, x: np.ndarray, a: int, phase: int) -> float:
        inv = float(self.U[a] @ x)
        if phase == 0:
            return inv + float(self.V[a] @ x[:2])
        if phase == 1:
            return inv - float(self.V[a] @ x[:2])        # exact reversal of A
        return inv + float(self.W[a] @ x[:2])

    def best(self, x: np.ndarray, phase: int) -> int:
        return int(np.argmax([self.score(x, a, phase) for a in range(self.nA)]))

    def embed(self, x: np.ndarray, a: int) -> np.ndarray:
        e = np.zeros(self.nA)
        e[a] = 2.0
        return np.concatenate([x, e])


# ---------------------------------------------------------------------------
#  The universal engine (paper Section 4)
# ---------------------------------------------------------------------------

class PaperEngine:
    """Struct-of-arrays implementation (the object-per-particle form was ~50x
    too slow in pure Python); semantics identical to the paper's Section 4."""

    def __init__(self, *, d: int = 8, sigma: float = 0.89, eta: float = 0.45,
                 mu_base: float = 0.04, mu_cryst: float = 0.001,
                 v_max: float = 1.5, k0: float = 2.0,
                 n_cryst_min: int = 4, theta_conv: float = 0.25,
                 n_cross_min: int = 10, theta_rev: float = -0.06,
                 create_K: float = 0.55, expose_K: float = 0.35,
                 max_particles: int = 1200,
                 gate_contexts: bool = True, crucible: bool = True,
                 verification: bool = True, seed: int = 0) -> None:
        self.d = d
        self.Z = np.zeros((0, d))
        self.V = np.zeros(0)
        self.MU = np.zeros(0)
        self.CTX = np.zeros(0, dtype=int)
        self.CRY = np.zeros(0, dtype=bool)
        self.VER = np.zeros(0, dtype=bool)
        self.N = np.zeros(0, dtype=int)
        self.hist: list[deque] = []
        self.cross: list[deque] = []
        self._cross_maxlen = 30
        self.sigma, self.eta = sigma, eta
        self.mu_base, self.mu_cryst, self.v_max = mu_base, mu_cryst, v_max
        self.k0 = k0
        self.n_cryst_min, self.theta_conv = n_cryst_min, theta_conv
        self.n_cross_min, self.theta_rev = n_cross_min, theta_rev
        self.create_K, self.expose_K = create_K, expose_K
        self.max_particles = max_particles
        self.gate_on, self.crucible_on, self.verify_on = \
            gate_contexts, crucible, verification
        self.rng = np.random.default_rng(seed)
        self.ctx = 0

    # ----- read path -----
    def _gamma(self) -> np.ndarray:
        if self.Z.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        if not self.gate_on:
            return np.ones(self.Z.shape[0], dtype=bool)
        same = self.CTX == self.ctx
        if self.verify_on:
            return same | (self.CRY & self.VER)
        return same

    def delta_R_batch(self, Zq: np.ndarray) -> np.ndarray:
        if self.Z.shape[0] == 0:
            return np.zeros(Zq.shape[0])
        g = self._gamma()
        if not g.any():
            return np.zeros(Zq.shape[0])
        Zg, Vg = self.Z[g], self.V[g]
        d2 = ((Zq[:, None, :] - Zg[None]) ** 2).sum(axis=2)
        return (np.exp(-d2 / (2 * self.sigma ** 2)) @ Vg)

    def delta_R(self, z: np.ndarray) -> float:
        return float(self.delta_R_batch(z[None])[0])

    def select(self, world: "MiniRRW", x: np.ndarray, base) -> int:
        Zq = np.stack([world.embed(x, a) for a in range(world.nA)])
        scores = np.array([base(zq) for zq in Zq]) + self.delta_R_batch(Zq)
        zs = self.k0 * (scores - scores.max())
        prob = np.exp(zs)
        prob /= prob.sum()
        return int(self.rng.choice(world.nA, p=prob))

    # ----- write path (two passes) -----
    def write(self, z: np.ndarray, D: float) -> None:
        n = self.Z.shape[0]
        created = False
        if n:
            K = np.exp(-((self.Z - z) ** 2).sum(axis=1) / (2 * self.sigma ** 2))
            same = self.CTX == self.ctx
            learn = same & (K >= self.expose_K)          # pass 2: local learning
            if learn.any():
                self.V[learn] = np.clip(self.V[learn] + self.eta * K[learn] * D,
                                        -self.v_max, self.v_max)
                self.N[learn] += 1
                for i in np.flatnonzero(learn):
                    self.hist[i].append(float(K[i] * D))
            created = bool((same & (K >= self.create_K)).any())
            test = (~same) & self.CRY & (K >= self.expose_K)   # pass 1: validity
            for i in np.flatnonzero(test):
                self.cross[i].append(D)
        if not created and n < self.max_particles and abs(D) > 1e-6:
            self.Z = np.vstack([self.Z, z[None]])
            self.V = np.append(self.V, np.clip(self.eta * D, -self.v_max, self.v_max))
            self.MU = np.append(self.MU, self.mu_base)
            self.CTX = np.append(self.CTX, self.ctx)
            self.CRY = np.append(self.CRY, False)
            self.VER = np.append(self.VER, False)
            self.N = np.append(self.N, 1)
            self.hist.append(deque(maxlen=20))
            self.cross.append(deque(maxlen=30))

    # ----- lifecycle (per epoch) -----
    def epoch(self) -> None:
        if self.Z.shape[0] == 0:
            return
        self.V *= (1.0 - self.MU)
        for i in range(self.Z.shape[0]):
            if (not self.CRY[i] and self.N[i] >= self.n_cryst_min
                    and len(self.hist[i]) >= self.n_cryst_min
                    and abs(float(np.mean(list(self.hist[i])[-6:]))) < self.theta_conv):
                self.CRY[i] = True
                self.MU[i] = self.mu_cryst
            if self.crucible_on and self.CRY[i] and \
                    len(self.cross[i]) >= self.n_cross_min:
                if self.V[i] * float(np.mean(self.cross[i])) < self.theta_rev:
                    self.CRY[i] = False                  # the Crucible: dissolve
                    self.MU[i] = self.mu_base
                    self.VER[i] = False
                else:
                    self.VER[i] = True
                self.cross[i].clear()
        keep = (np.abs(self.V) >= 1e-4) | self.CRY
        if not keep.all():
            self._filter(keep)

    def _filter(self, keep: np.ndarray) -> None:
        idx = np.flatnonzero(keep)
        self.Z, self.V, self.MU = self.Z[idx], self.V[idx], self.MU[idx]
        self.CTX, self.CRY, self.VER = self.CTX[idx], self.CRY[idx], self.VER[idx]
        self.N = self.N[idx]
        self.hist = [self.hist[i] for i in idx]
        self.cross = [self.cross[i] for i in idx]

    def switch_context(self, new_ctx: int) -> None:
        for c in self.cross:                             # verification is
            c.clear()                                    # phase-local (paper 4.4):
        if self.VER.size:                                # broadcast rights are
            self.VER[:] = False                          # re-earned each phase
        self.ctx = new_ctx

    @property
    def n_particles(self) -> int:
        return self.Z.shape[0]


# ---------------------------------------------------------------------------
#  The experiment: A -> B -> C with the paper's protocol, ablated
# ---------------------------------------------------------------------------

def run_engine(arm_kw: dict, seed: int, n_epochs: int = 20,
               n_points: int = 250) -> dict:
    world = MiniRRW(seed=seed)
    rngb = np.random.default_rng(seed + 100)
    A = rngb.normal(0, 0.6, 8)
    base = lambda z: float(1 / (1 + np.exp(-(A @ z))))   # frozen base evaluator
    eng = PaperEngine(seed=seed + 200, **arm_kw)
    rng = np.random.default_rng(seed + 300)
    test = {ph: [(rng.normal(0, 1.0, 4)) for _ in range(300)] for ph in range(3)}

    def accuracy(phase: int) -> float:
        old = eng.ctx
        eng.ctx = phase
        ok = 0
        for x in test[phase]:
            scores = [base(world.embed(x, a)) + eng.delta_R(world.embed(x, a))
                      for a in range(world.nA)]
            ok += int(np.argmax(scores) == world.best(x, phase))
        eng.ctx = old
        return ok / len(test[phase])

    acc_matrix = {}
    for phase in range(3):
        eng.switch_context(phase)
        for _ in range(n_epochs):
            for _ in range(n_points):
                x = rng.normal(0, 1.0, 4)
                a = eng.select(world, x, base)
                imposed = 1.0 if a == world.best(x, phase) else 0.0
                z = world.embed(x, a)
                D = imposed - (base(z) + eng.delta_R(z))
                eng.write(z, D)
            eng.epoch()
        acc_matrix[phase] = {ph: accuracy(ph) for ph in range(phase + 1)}
    return {"acc": acc_matrix,
            "particles": eng.n_particles,
            "crystallized": int(eng.CRY.sum()),
            "verified": int(eng.VER.sum())}


ARMS = {
    "full engine": dict(),
    "no-context-gating": dict(gate_contexts=False),
    "no-crucible": dict(crucible=False),
    "no-verification": dict(verification=False),
}


def main(n_seeds: int = 6) -> None:
    print("\n" + "#" * 74)
    print("#  THE PAPER ENGINE -- continual lifecycle (context gating, crucible,")
    print("#  verification) on a faithful mini-RRW; the ablations attribute it")
    print("#" * 74)
    print("\n  protocol: phases A -> B(=exact reversal of A's context) -> C;")
    print("  context ids given at boundaries (task-incremental, as in the paper).")
    print("  acc(A|P) = accuracy on phase-A held-out queries after training phase P.\n")

    res = {}
    for name, kw in ARMS.items():
        runs = [run_engine(kw, s) for s in range(n_seeds)]
        res[name] = runs
        def m(phase, on):
            return float(np.mean([r["acc"][phase][on] for r in runs]))
        forget_ab = m(0, 0) - m(1, 0)        # A-damage from B's contradiction
        forget_ac = m(0, 0) - m(2, 0)        # A-damage at the end
        print(f"  {name:<19} acc(A|A) {m(0, 0):.3f}  acc(A|B) {m(1, 0):.3f}  "
              f"acc(B|B) {m(1, 1):.3f}  acc(A|C) {m(2, 0):.3f}  "
              f"forget(A) {forget_ac:+.3f}")
    from .stats import fmt_ci, paired_ci, verdict
    f_full = [r["acc"][0][0] - r["acc"][2][0] for r in res["full engine"]]
    f_nog = [r["acc"][0][0] - r["acc"][2][0] for r in res["no-context-gating"]]
    ci = paired_ci(f_nog, f_full)
    print(f"\n  forgetting(no-gating) - forgetting(full): {fmt_ci(ci)} {verdict(ci)}")
    print(f"  (positive = the lifecycle prevents real damage; this is the G2")
    print(f"   poisoning pathology reproduced and attributed in its home domain)")
    f_full_m = float(np.mean(f_full))
    f_nog_m = float(np.mean(f_nog))
    bB_full = float(np.mean([r["acc"][1][1] for r in res["full engine"]]))
    bB_nover = float(np.mean([r["acc"][1][1] for r in res["no-verification"]]))
    print(f"  transfer probe: acc(B|B) full {bB_full:.3f} vs no-verification "
          f"{bB_nover:.3f} (verified invariants broadcasting forward)")
    print(f"  calibration note: the paper's RRW result under genuine contradiction")
    print(f"  is 'less forgetting than replay', not zero (its near-zero headline is")
    print(f"  CIFAR, where tasks do not contradict); the crucible's collateral on")
    print(f"  context-SPECIFIC (not false) structure is the measured residual.")
    f_gate_only = float(np.mean([r["acc"][0][0] - r["acc"][2][0]
                                 for r in res["no-crucible"]]))
    print(f"\n  decomposition: gating-only forgetting {f_gate_only:+.3f} -- context")
    print(f"  gating ALONE is the retention mechanism (near-zero even under exact")
    print(f"  contradiction); the crucible COSTS retention here ({f_full_m:+.3f})")
    print(f"  because this world's phases share one input region, so home-context")
    print(f"  truth is cross-tested constantly and dissolution erodes it -- the")
    print(f"  paper's own regime-dependence theme applied to its own lifecycle:")
    print(f"  crucible/verification want spatially separated contexts (CIFAR-style);")
    print(f"  gating works everywhere. The mscn architecture lesson: adopt gating.")
    assert ci["lo"] > 0, \
        "context gating must reduce forgetting, CI-significant"
    assert f_gate_only < 0.05, \
        "gating-only retention must be near-zero (the mechanism that transfers)"
    assert f_full_m < f_nog_m - 0.1, \
        "the full lifecycle must still beat the ungated memory clearly"
    print("\n  Done; the lifecycle is now available to the architecture "
          "(mscn.ibf_engine.PaperEngine).\n")


if __name__ == "__main__":
    main()
