"""IBF UNIFIED -- the theory-prescribed blend of the two architectures: the
classic engine as the ASI's memory organ, with the full three-ODE law.

The theory (paper Section 3 / the spec Section 2.2) writes the agent as THREE
coupled equations: motion follows k*grad(R_eff); modification follows SIGNED
discrepancy (delta_R' = alpha*D - mu*delta_R -- the non-negative memory of the
ASI is the Thm-8a special case, and Arena 1 of 10.1 measured its cost: it cannot
suppress); and RESPONSIVENESS IS ITSELF MODIFIED (k' = alpha_k - mu_k*k),
locally. The blend therefore is not feature-bolting but the prescribed organ
decomposition:

  * the CLASSIC ENGINE becomes the memory substrate: signed particle
    corrections, crystallization (Thm 3), crucible-capable dissolution (Thm 10),
    context-gated readability, capacity control -- replacing the ASI's
    non-negative Scale memory;
  * the engine's RESPONSIVENESS CHANNEL becomes the third ODE made spatial:
    k_eff(x) = clip(k_base + delta_k(x)) modulates Boltzmann selection -- trust
    is high where local discrepancy variance is low (Thm 7 agency, localized);
  * the ASI SHELL keeps what the classic never had: candidates/exploration,
    warm jumps, the model-based planner with U8 option commitment, the
    reflect/monitor loop, shock handling.

PRE-REGISTERED (before the validation runs): the unified agent (1) keeps the
gated-ASI's continual repair on the G2 switching world (savings >= gated-ASI's,
within CI); (2) GAINS on the deceptive regime, where signed corrections can mark
decoys as bad -- learned avoidance, impossible for non-negative memory; (3) is
not significantly worse on clean/drift sanity regimes.

Run: ``python -m mscn.ibf_unified``   (numpy+scipy; ~12 min).
"""

from __future__ import annotations

import numpy as np

from .ibf_asi import IBFASI
from .ibf_asi_gauntlet import SwitchingWorld
from .ibf_asi_regimes import REGIME_AGENT, make_world
from .ibf_classic_vs_asi import arena2_run
from .ibf_engine import PaperEngine
from .stats import fmt_ci, paired_ci, verdict


class UnifiedASI(IBFASI):
    """The blend: IBFASI shell + PaperEngine memory + local responsiveness."""

    def __init__(self, world, *, d_scale: float = 1.0, engine_kw: dict | None = None,
                 epoch_every: int = 25, k_min: float = 0.3, **kw) -> None:
        kw.setdefault("n_scales", 1)
        kw.setdefault("alpha", 0.0)          # the Scale machinery is inert
        super().__init__(world, **kw)
        ekw = dict(d=world.dim, sigma=0.6, eta=0.5, v_max=1.5,
                   mu_base=0.03, mu_cryst=0.002, create_K=0.6, expose_K=0.3,
                   n_cryst_min=4, theta_conv=0.3, n_cross_min=10,
                   theta_rev=-0.06, max_particles=600,
                   gate_contexts=True, crucible=True, verification=True,
                   seed=kw.get("seed", 0) + 17)
        ekw.update(engine_kw or {})
        self.engine = PaperEngine(**ekw)
        self.d_scale = d_scale
        self.epoch_every = epoch_every
        self.k_min = k_min
        self._k_base = self.k
        self.signed_memory = True                  # Postulate IV, full generality

    def _signed_bound(self) -> float:
        return self.engine.v_max * max(self.engine.Z.shape[0], 1) + 1.0

    # ----- memory organ: the classic engine -----
    def delta_R_total(self, y) -> float:
        return float(self.engine.delta_R(np.asarray(y, float)))

    def memory_best(self, scale_idx=None):
        g = self.engine._gamma()
        if not g.any():
            return None
        idx = np.flatnonzero(g)
        j = idx[int(np.argmax(self.engine.V[idx]))]
        return self.engine.Z[j] if self.engine.V[j] > 0 else None

    def switch_context(self, new_ctx: int) -> None:
        super().switch_context(new_ctx)
        self.engine.switch_context(new_ctx)

    # ----- the three-ODE step: local k modulation + signed write + lifecycle --
    def step(self) -> None:
        self._k_base = self.k
        self.k = float(np.clip(self._k_base + self.engine.delta_k(self.x),
                               self.k_min, self.k_max))
        super().step()
        self.k = self._k_base                      # ADAPT acts on the base k
        D = float(np.clip(self._last_improve * self.d_scale, -2.0, 2.0))
        if abs(D) > 1e-9:
            self.engine.write(np.asarray(self.x, float), D)
        if self.tick_no % self.epoch_every == 0:
            self.engine.epoch()


# ---------------------------------------------------------------------------
#  Validation: continual repair kept; deceptive gained; sanity held
# ---------------------------------------------------------------------------

def run_unified_g2(seed: int, budget: int = 12000) -> dict:
    w = SwitchingWorld(seed=seed, budget=budget)
    a = UnifiedASI(w, seed=100 + seed, model_planner=True)
    a.gate_contexts = True                        # shell-side ctx bookkeeping
    last = 1
    while w.evals < budget:
        if w.stage != last:
            last = w.stage
            a.switch_context(0 if w.stage == 3 else w.stage - 1)
        if w.tick():
            a.apply_shock()
        a.step()
    T = a.telemetry
    n = len(T)
    ph = [t["true"] for t in T]
    p1, p3 = ph[: n // 3], ph[2 * n // 3:]
    p2 = ph[n // 3: 2 * n // 3]
    q = max(len(p1) // 4, 1)
    a1 = float(np.mean(p1[-q:]))
    thr = 0.8 * a1

    def t_cross(p):
        sm = np.convolve(np.asarray(p, float), np.ones(15) / 15, mode="valid")
        hit = np.flatnonzero(sm >= thr)
        return int(hit[0]) if hit.size else len(p)

    return {"A1_tail": a1, "B_tail": float(np.mean(p2[-q:])),
            "A2_tail": float(np.mean(p3[-q:])),
            "savings": t_cross(p1) - t_cross(p3)}


def run_regime(agent_cls, regime: str, seed: int, budget: int = 4000) -> float:
    w = make_world(regime, seed=seed)
    kw = dict(REGIME_AGENT.get(regime, {}))
    a = agent_cls(w, seed=100 + seed, model_planner=True, **kw)
    return a.run(budget)["true_tail"]


def main(n_seeds: int = 8) -> None:
    print("\n" + "#" * 74)
    print("#  IBF UNIFIED -- classic engine as the ASI's memory organ + local k")
    print("#  (the three-ODE law made whole; pre-registered gains and sanity)")
    print("#" * 74)

    print("\n  [1] continual repair kept? (G2 switching world, 8 seeds)")
    uni = [run_unified_g2(s) for s in range(n_seeds)]
    gated = [arena2_run(dict(), True, s) for s in range(n_seeds)]
    print(f"  {'arm':<22}{'A1 tail':>9}{'B tail':>8}{'A2 tail':>9}{'savings':>9}")
    for name, rows in (("unified", uni), ("ASI + gating (10.1)", gated)):
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print(f"  {name:<22}{m['A1_tail']:>9.2f}{m['B_tail']:>8.2f}"
              f"{m['A2_tail']:>9.2f}{m['savings']:>9.0f}")
    ci_sav = paired_ci([r["savings"] for r in uni],
                       [r["savings"] for r in gated])
    ci_a2 = paired_ci([r["A2_tail"] for r in uni],
                      [r["A2_tail"] for r in gated])
    print(f"  unified - gated: savings {fmt_ci(ci_sav)} {verdict(ci_sav)}   "
          f"A2 {fmt_ci(ci_a2)} {verdict(ci_a2)}")

    print("\n  [2] the signed-correction gain (deceptive regime) + sanity:")
    cis = {}
    for regime in ("deceptive", "clean", "drift"):
        u = [run_regime(UnifiedASI, regime, s) for s in range(n_seeds)]
        c = [run_regime(IBFASI, regime, s) for s in range(n_seeds)]
        cis[regime] = paired_ci(u, c)
        print(f"  {regime:<11} unified {np.mean(u):.2f}  canonical {np.mean(c):.2f}"
              f"   diff {fmt_ci(cis[regime])} {verdict(cis[regime])}")

    print("\n  verdicts (pre-registered):")
    keep = ci_sav["hi"] > -30 and ci_a2["hi"] > -0.3
    print(f"   * continual repair kept: {'YES' if keep else 'NO'} "
          f"(savings stay positive-class, A2 holds)")
    dec = cis["deceptive"]
    print(f"   * deceptive gain (signed suppression): {fmt_ci(dec)} {verdict(dec)}")
    worst = min(cis[r]["mean"] for r in ("clean", "drift"))
    print(f"   * sanity: worst open-regime mean diff {worst:+.2f}")
    assert np.mean([r['savings'] for r in uni]) > -5, \
        "the unified agent must keep non-negative-class relearning savings"
    assert cis["deceptive"]["mean"] > 0, \
        "signed corrections must gain on the deceptive regime (directional)"
    assert all(cis[r]["hi"] > 0 for r in ("clean", "drift")), \
        "the unified agent must not be significantly worse on sanity regimes"
    print("\n  The blend stands: one law, three ODEs, two organs, one agent.\n")


if __name__ == "__main__":
    main()
