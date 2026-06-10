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

Mechanisms (ablation pairs, same seed in both arms):
  memory, multiscale, planning, dissolve, reflect, two-sided-k

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
    reactive ascent stalls at the rim; lookahead can see across."""

    def __init__(self, moat_amp: float = 2.0, moat_sigma: float = 1.6, **kw) -> None:
        super().__init__(**kw)
        self.moat_amp, self.moat_sigma = moat_amp, moat_sigma

    def true_coherence(self, x):
        v = super().true_coherence(x)
        d = np.sqrt(np.sum((np.asarray(x, float) - self.c[0]) ** 2))
        ring = np.exp(-((d - 1.8) ** 2) / (2 * 0.6 ** 2))   # ring at radius 1.8
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
    if regime == "moat":
        w = MoatWorld(seed=seed, noise=0.15, drift=0.0, phase_drift=0.02,
                      ripple=0.1, n_decoys=3)
        w.S = np.array([0.8] + [1.2] * (len(w.c) - 1))
        return w
    raise ValueError(regime)


REGIMES = ("clean", "noisy", "drift", "shocked", "deceptive", "moat")

ABLATIONS = {
    "memory":      dict(alpha=0.0),
    "multiscale":  dict(n_scales=1),
    "planning":    dict(horizon=1),
    "dissolve":    dict(dissolve=False),
    "reflect":     dict(reflect=False),
    "two-sided-k": dict(two_sided_k=False),
}


def run_cell(regime: str, agent_kw: dict, n_seeds: int) -> list[float]:
    out = []
    for s in range(n_seeds):
        w = make_world(regime, seed=s)
        a = IBFASI(w, seed=100 + s, **agent_kw)
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
    print(f"\n  reading the map:")
    print(f"   * memory is CI-significant in: {', '.join(mem_sig) or 'none'}")
    print(f"   * dissolve in its regime (drift): "
          f"{fmt_ci(cells[('dissolve', 'drift')])}")
    print(f"   * planning in its putative regime (moat): "
          f"{fmt_ci(cells[('planning', 'moat')])}")
    print(f"   * two-sided k in its regime (deceptive): "
          f"{fmt_ci(cells[('two-sided-k', 'deceptive')])}")
    print(f"   * significantly HARMFUL cells: {neg_sig or 'none'}")

    assert len(mem_sig) >= 4, \
        "memory must be CI-significant in most regimes (measured: all six)"
    assert all(cells[("dissolve", r)]["hi"] > 0 for r in REGIMES), \
        "dissolution must not be significantly harmful in any regime"
    assert cells[("memory", "clean")]["hi"] > -0.2, \
        "memory must not be strongly harmful even where unneeded"

    print("\n  The matrix -- including its nulls -- is the architecture's claim map:")
    print("  MEMORY is the one universally significant stage; dissolution is")
    print("  redundant with base decay at these timescales (the 3.14 module-level")
    print("  finding, reproduced at system level); this plan operator (stochastic")
    print("  rollout on a noisy sensed field) buys nothing even in the moat regime")
    print("  (U7's win used BFS over a learned DISCRETE simulation); and the")
    print("  two-sided-k restart is actively harmful under shocks and in the moat")
    print("  (it abandons position exactly when position is the asset). A stage's")
    print("  value is a property of (mechanism x regime); the honest spec for the")
    print("  integrated agent is exactly this table.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="IBF-ASI regime matrix")
    p.add_argument("--seeds", type=int, default=8)
    a = p.parse_args()
    main(n_seeds=a.seeds)
