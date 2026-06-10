"""The regimes x mechanisms matrix -- WHERE does each IBF-ASI stage earn its keep?

The integrated-agent validations produced honest nulls for several stages (plan,
scale-count, reflect) *in one world*. The repo's recurring meta-lesson is that
every mechanism's win is regime-dependent; this module makes that systematic: a
grid of six regimes (each isolating one stressor) x six mechanism ablations, every
cell an eval-budget-matched PAIRED per-seed difference (full minus ablation) with
a 95% t-interval. The deliverable is the MAP -- including its nulls -- not a
single headline number.

Regimes (world specs):
  clean      no noise, no drift, no shocks (control: most stages should be ~free)
  noisy      heavy observation noise (memory's de-noising regime)
  drift      volatile fine structure + slow basin drift (dissolution's regime)
  shocked    displacement + fast-mode memory damage (consolidation/reflect regime)
  deceptive  near-tall decoys, narrow optimum, low ripple (exploration's regime)
  moat       a low-coherence ring around the optimum (lookahead's putative regime)
  moat-local the moat under LOCOMOTION constraints (no uniform-jump candidates, no
             restart teleports): the only way into the unvisited interior is to
             WALK through the low-coherence ring -- the regime where a real
             model-based planner must earn its keep.

Mechanisms (ablation pairs, same seed in both arms). The base agent carries the
model-based planner (learned discrete value map + optimism + BFS, `model_planner`);
`model-plan` ablates it; `rollout-plan` ablates the old stochastic-rollout depth.

PRE-REGISTERED criterion (stated before the first full run of the new planner):
the (model-plan, moat-local) cell must be CI-significantly positive; secondary,
model-plan must not be significantly harmful in any other regime.

OUTCOME (reported, per the pre-registration contract): **NOT MET.** Four planner
designs were measured (stochastic rollout; raw-scale optimistic BFS; R_eff-scale
frontier BFS; + U3 ascent-arbitration and calibrated satiation), then the
mechanism was ISOLATED (no memory, no warm jumps, both arms locomotion-only) and
the moat swept shallow-to-deep. The boundary result: in a 2-D continuous field
there is NO regime where H-step planning over a learned cell model adds
CI-measurable value -- shallow barriers are crossed by Boltzmann diffusion anyway
(planning unnecessary: reactive crosses 3/10), and deep barriers hide what is
behind them, making the crossing a needle-in-a-haystack exploration problem that
frontier optimism does not solve at these budgets (planning insufficient: planner
crosses 1/10 isolated). U7's 0->100% lived in a 1-D corridor where 'beyond the
trap' is the only unexplored direction; planning binds in DISCRETE, LOW-BRANCHING
state spaces (exactly where the chess search gains of ARCHITECTURE 3.18 live),
not in continuous fields where exploration channels dominate. The planner stays
in the agent (fuzz-covered, eval-parity, never significantly harmful) and the
named follow-up is to bind it on the discrete substrate (corridor/KRK).

Run: ``python -m mscn.ibf_asi_regimes [--seeds 8]``   (numpy+scipy, ~10-15 min).
"""

from __future__ import annotations

import numpy as np

from .ibf_asi import ASIWorld, IBFASI
from .stats import fmt_ci, paired_ci

EVALS = 4000


class MoatWorld(ASIWorld):
    """ASIWorld plus a negative-coherence ring (moat) around the global basin:
    reaching the optimum requires crossing a region that LOWERS coherence --
    reactive ascent stalls at the rim; lookahead can see across. Decoys are
    displaced OUTSIDE a clearance radius: a decoy sitting on the ring fills the
    dip and bridges it for myopic agents (measured leak, fixed here)."""

    def __init__(self, moat_amp: float = 2.5, moat_sigma: float = 0.6,
                 clearance: float = 4.0, **kw) -> None:
        super().__init__(**kw)
        self.moat_amp, self.moat_sigma = moat_amp, moat_sigma
        for j in range(1, len(self.c)):
            d = self.c[j] - self.c[0]
            dist = float(np.linalg.norm(d))
            if dist < clearance:
                unit = d / dist if dist > 1e-9 else \
                    self.rng.normal(size=self.dim) / np.sqrt(self.dim)
                self.c[j] = np.clip(self.c[0] + unit * (clearance + 1.0),
                                    self.lo, self.hi)

    def true_coherence(self, x):
        v = super().true_coherence(x)
        d = np.sqrt(np.sum((np.asarray(x, float) - self.c[0]) ** 2))
        ring = np.exp(-((d - 1.8) ** 2) / (2 * self.moat_sigma ** 2))
        return float(v - self.moat_amp * ring)


def make_world(regime: str, seed: int) -> ASIWorld:
    if regime == "clean":
        return ASIWorld(seed=seed, noise=0.0, drift=0.0, phase_drift=0.0)
    if regime == "noisy":
        return ASIWorld(seed=seed, noise=0.8, drift=0.0, phase_drift=0.0)
    if regime == "drift":
        return ASIWorld(seed=seed, noise=0.35, drift=0.005, phase_drift=0.3)
    if regime == "shocked":
        return ASIWorld(seed=seed, noise=0.35, shock_every=60)
    if regime == "deceptive":
        w = ASIWorld(seed=seed, deceptive=True, noise=0.15, drift=0.0,
                     phase_drift=0.02, ripple=0.1)
        w.S = np.array([0.55] + [1.3] * (len(w.c) - 1))
        return w
    if regime in ("moat", "moat-local"):
        w = MoatWorld(seed=seed, noise=0.15, drift=0.0, phase_drift=0.02,
                      ripple=0.1, n_decoys=3)
        w.S = np.array([0.8] + [1.2] * (len(w.c) - 1))
        return w
    raise ValueError(regime)


REGIMES = ("clean", "noisy", "drift", "shocked", "deceptive", "moat", "moat-local")

# locomotion constraint: candidates cannot teleport (no uniform jumps, no
# restart-teleports) -- crossing the ring must be walked.
REGIME_AGENT = {"moat-local": dict(n_jumps=0, two_sided_k=False)}

BASE = dict(model_planner=True)      # the full agent now carries the real planner

ABLATIONS = {
    "memory":       dict(alpha=0.0),
    "multiscale":   dict(n_scales=1),
    "model-plan":   dict(model_planner=False),
    "rollout-plan": dict(horizon=1),
    "dissolve":     dict(dissolve=False),
    "reflect":      dict(reflect=False),
    "two-sided-k":  dict(two_sided_k=False),
}


def run_cell(regime: str, agent_kw: dict, n_seeds: int) -> list[float]:
    out = []
    for s in range(n_seeds):
        w = make_world(regime, seed=s)
        kw = {**BASE, **REGIME_AGENT.get(regime, {}), **agent_kw}
        a = IBFASI(w, seed=100 + s, **kw)
        if regime == "moat-local":
            # the task is defined as: reach the optimum from OUTSIDE the moat
            ang = 2 * np.pi * (s % 8) / 8.0
            a.x = np.clip(w.c[0] + 4.0 * np.array([np.cos(ang), np.sin(ang)]),
                          w.lo, w.hi)
        out.append(a.run(EVALS)["true_tail"])
    return out


def main(n_seeds: int = 8) -> None:
    print("\n" + "#" * 74)
    print("#  IBF-ASI REGIME MATRIX -- where does each stage earn its keep?")
    print(f"#  cell = paired (full - ablation) true-coherence, {n_seeds} seeds, "
          f"{EVALS} evals")
    print("#" * 74)

    fulls = {r: run_cell(r, dict(), n_seeds) for r in REGIMES}
    cells: dict[tuple[str, str], dict] = {}
    for mech, kw in ABLATIONS.items():
        for r in REGIMES:
            cells[(mech, r)] = paired_ci(fulls[r], run_cell(r, kw, n_seeds))

    print(f"\n  full-agent mean true coherence per regime:")
    print("  " + "".join(f"{r:>11}" for r in REGIMES))
    print("  " + "".join(f"{np.mean(fulls[r]):>11.2f}" for r in REGIMES))

    print(f"\n  mechanism advantage (+ = stage helps; * = 95% CI excludes 0):\n")
    print(f"  {'mechanism':<13}" + "".join(f"{r:>11}" for r in REGIMES))
    for mech in ABLATIONS:
        row = ""
        for r in REGIMES:
            ci = cells[(mech, r)]
            mark = "*" if (ci["lo"] > 0 or ci["hi"] < 0) else " "
            row += f"{ci['mean']:>+10.2f}{mark}"
        print(f"  {mech:<13}" + row)

    # ----- reading the map (assert only the robust diagonal) -----
    mem_sig = [r for r in REGIMES if cells[("memory", r)]["lo"] > 0]
    neg_sig = [(m, r) for m in ABLATIONS for r in REGIMES
               if cells[(m, r)]["hi"] < 0]
    mp_moat = cells[("model-plan", "moat-local")]
    print(f"\n  reading the map:")
    print(f"   * memory is CI-significant in: {', '.join(mem_sig) or 'none'}")
    print(f"   * MODEL-PLANNER, pre-registered cell (moat-local): {fmt_ci(mp_moat)}")
    verdict_met = mp_moat["lo"] > 0
    print(f"     PRE-REGISTERED CRITERION: {'MET' if verdict_met else 'NOT MET'} -- "
          f"{'the planner binds here' if verdict_met else 'see module docstring:'}")
    if not verdict_met:
        print(f"     continuous 2-D offers no niche between 'diffusion crosses anyway'")
        print(f"     and 'needle-in-a-haystack'; planning binds in discrete low-")
        print(f"     branching spaces (the 3.18 chess search gains), not here.")
    print(f"   * rollout-plan in moat-local (the old operator): "
          f"{fmt_ci(cells[('rollout-plan', 'moat-local')])}")
    print(f"   * dissolve in its regime (drift): "
          f"{fmt_ci(cells[('dissolve', 'drift')])}")
    print(f"   * two-sided k in its regime (deceptive): "
          f"{fmt_ci(cells[('two-sided-k', 'deceptive')])}")
    print(f"   * significantly HARMFUL cells: {neg_sig or 'none'}")

    assert len(mem_sig) >= 4, \
        "memory must be CI-significant in most regimes (measured: all six)"
    assert all(cells[("dissolve", r)]["hi"] > 0 for r in REGIMES), \
        "dissolution must not be significantly harmful in any regime"
    assert cells[("memory", "clean")]["hi"] > -0.2, \
        "memory must not be strongly harmful even where unneeded"
    assert all(cells[("model-plan", r)]["hi"] > 0 for r in REGIMES), \
        "the model-based planner must not be significantly harmful in any regime"

    print("\n  The matrix -- including its nulls -- is the architecture's claim map:")
    print("  MEMORY is the one universally significant stage; dissolution is")
    print("  redundant with base decay at these timescales (the 3.14 module-level")
    print("  finding, reproduced at system level); NEITHER plan operator binds in")
    print("  continuous 2-D (the boundary result in the docstring -- planning needs")
    print("  discrete low-branching structure); and the two-sided-k restart is the")
    print("  only stage with significantly harmful cells (it abandons position,")
    print("  which is sometimes the asset -- see the harmful-cells line). A stage's")
    print("  value is a property of (mechanism x regime); the honest spec for the")
    print("  integrated agent is exactly this table.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="IBF-ASI regime matrix")
    p.add_argument("--seeds", type=int, default=8)
    a = p.parse_args()
    main(n_seeds=a.seeds)
