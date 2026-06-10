"""THE GAUNTLET -- compound, continual, and adversarial benchmarking of the
integrated IBF-ASI (the tests the single-stressor matrix cannot see).

Pre-registered questions (answers reported however they land, paired CIs):

  G1 COMPOUND-STRESS LADDER. Stressors stack: L0 clean -> L1 +noise -> L2
     +fine-drift -> L3 +shocks -> L4 +basin-drift -> L5 +deception +moat, all at
     once. Single stressors are solved ground (the 8x7 matrix); does the full
     agent's advantage COMPOSE, and where does each contestant break? Contestants:
     full ASI, no-memory ablation, the Layer-1 IBFLearner, CMA-ES (static-world
     assumptions, honest caveat), random anchor.

  G2 CONTINUAL SWITCHING (A -> B -> A). The world changes character mid-life and
     then RETURNS (same layout restored). Measured per agent: adaptation (phase-B
     tail), retention/savings (early return-A vs early first-A), interference
     (return-A tail vs first-A tail). The consolidation hypothesis finally faces
     its real exam: no-memory should adapt fast and retain nothing; no-dissolve
     should retain but adapt worse; the full multiscale agent's consolidated
     coarse memory should buy BOTH. Measured either way.

  G3 ADVERSARIAL TARGETED SHOCKS. The world reads the agent: displacement away
     from -- and memory damage focused on -- its own best-known basin (vs the
     standard untargeted shock). Does the q-record/warm-jump machinery survive an
     aimed attack, and what do reflect/multiscale buy in recovery?

Run: ``python -m mscn.ibf_asi_gauntlet [--seeds 8]``  (numpy+scipy(+cma), ~45 min).
"""

from __future__ import annotations

import numpy as np

from .ibf_asi import ASIWorld, IBFASI
from .ibf_asi_regimes import MoatWorld
from .stats import fmt_ci, paired_ci, verdict

EVALS = 4000
TICK_EVERY = 16


# ---------------------------------------------------------------------------
#  G1: the compound-stress ladder
# ---------------------------------------------------------------------------

def ladder_world(level: int, seed: int) -> ASIWorld:
    kw = dict(seed=seed, noise=0.0, drift=0.0, phase_drift=0.0, shock_every=0)
    if level >= 1:
        kw["noise"] = 0.5
    if level >= 2:
        kw["phase_drift"] = 0.3
    if level >= 3:
        kw["shock_every"] = 60
    if level >= 4:
        kw["drift"] = 0.02
    if level >= 5:
        kw["deceptive"] = True
        w = MoatWorld(n_decoys=4, moat_amp=2.5, **kw)
        w.S = np.array([0.8] + [1.2] * (len(w.c) - 1))
        return w
    return ASIWorld(**kw)


def run_asi_world(w: ASIWorld, seed: int, agent_kw: dict,
                  budget: int = EVALS) -> float:
    kw = {"model_planner": True, **agent_kw}
    a = IBFASI(w, seed=100 + seed, **kw)
    return a.run(budget)["true_tail"]


def run_layer1_world(w: ASIWorld, seed: int, budget: int = EVALS) -> float:
    from .learner import IBFLearner
    holder = type("H", (), {})()
    senses = {"n": 0}

    def sense(x):
        senses["n"] += 1
        if senses["n"] % TICK_EVERY == 0:
            if w.tick():
                holder.x = w.shock_point()
                ag.centers = [c for c in ag.centers if w.rng.random() < 0.3]
        return w.sense(x)

    ag = IBFLearner(sense, w.lo, w.hi, seed=100 + seed)
    holder.x = ag.x
    trace = []
    while w.evals < budget:
        ag.x = holder.x
        ag.step()
        holder.x = ag.x
        trace.append(w.true_coherence(ag.x))
    return float(np.mean(trace[-max(len(trace) // 4, 1):]))


def run_cma_world(w: ASIWorld, seed: int, budget: int = EVALS) -> float | None:
    try:
        import cma
    except Exception:                                # pragma: no cover
        return None
    senses = {"n": 0}

    def sense(x):
        senses["n"] += 1
        if senses["n"] % TICK_EVERY == 0:
            w.tick()                                 # population: shock-exempt
        return w.sense(np.asarray(x))

    trace = []
    restart = 0
    while w.evals < budget:
        es = cma.CMAEvolutionStrategy(
            list(w.rng.uniform(w.lo, w.hi)), 2.0,
            {"popsize": 8, "verbose": -9, "seed": 1 + seed + 97 * restart,
             "bounds": [float(w.lo[0]), float(w.hi[0])]})
        while w.evals < budget and not es.stop():
            X = es.ask()
            es.tell(X, [-sense(xi) for xi in X])
            trace.append(w.true_coherence(np.asarray(es.mean)))
        restart += 1
    return float(np.mean(trace[-max(len(trace) // 4, 1):]))


G1_AGENTS = {
    "full ASI": lambda w, s: run_asi_world(w, s, dict()),
    "no-memory": lambda w, s: run_asi_world(w, s, dict(alpha=0.0)),
    "layer-1": run_layer1_world,
    "CMA-ES": run_cma_world,
    "random": lambda w, s: run_asi_world(
        w, s, dict(alpha=0.0, k0=1e-9, k_adapt=0.0, k_max=1e-9,
                   model_planner=False, horizon=1, n_scales=1)),
}


def g1_ladder(n_seeds: int) -> dict:
    out: dict = {}
    for name, fn in G1_AGENTS.items():
        out[name] = {}
        for level in range(6):
            vals = [fn(ladder_world(level, s), s) for s in range(n_seeds)]
            out[name][level] = [v for v in vals if v is not None]
    return out


# ---------------------------------------------------------------------------
#  G2: continual switching A -> B -> A
# ---------------------------------------------------------------------------

class SwitchingWorld(ASIWorld):
    """Three equal phases on the eval clock: layout A (calm) -> layout B
    (different basins + heavy noise) -> layout A restored exactly."""

    def __init__(self, seed: int, budget: int = 12000) -> None:
        super().__init__(seed=seed, noise=0.25, drift=0.0, phase_drift=0.05)
        self.budget = budget
        self._A = [c.copy() for c in self.c]
        rng_b = np.random.default_rng(seed + 5000)
        self._B = [rng_b.uniform(self.lo * 0.7, self.hi * 0.7)
                   for _ in range(len(self.c))]
        self.stage = 1

    def sense(self, x) -> float:
        third = self.budget // 3
        if self.evals >= third and self.stage == 1:
            self.c = [c.copy() for c in self._B]
            self.noise = 0.6
            self.stage = 2
        elif self.evals >= 2 * third and self.stage == 2:
            self.c = [c.copy() for c in self._A]      # exact return to A
            self.noise = 0.25
            self.stage = 3
        return super().sense(x)


G2_AGENTS = {
    "full ASI": dict(),
    "no-dissolve": dict(dissolve=False),
    "single-scale": dict(n_scales=1),
    "no-memory": dict(alpha=0.0),
}


def g2_continual(n_seeds: int, budget: int = 12000) -> dict:
    out: dict = {}
    for name, kw in G2_AGENTS.items():
        rows = []
        for s in range(n_seeds):
            w = SwitchingWorld(seed=s, budget=budget)
            a = IBFASI(w, seed=100 + s, model_planner=True, **kw)
            a.run(budget)
            T = a.telemetry
            n = len(T)
            ph = [t["true"] for t in T]
            p1, p2, p3 = ph[: n // 3], ph[n // 3: 2 * n // 3], ph[2 * n // 3:]
            q = max(len(p1) // 4, 1)

            def smooth(v, k=15):
                v = np.asarray(v, float)
                c = np.convolve(v, np.ones(k) / k, mode="valid")
                return c

            a1_tail = float(np.mean(p1[-q:]))
            thr = 0.8 * a1_tail
            # classic SAVINGS: ticks to re-reach 80% of own phase-1 tail level
            # (absolute early means measure the position-hole at the switch, not
            # memory -- both phases must CLIMB to the threshold)
            def t_cross(p):
                sm = smooth(p)
                hit = np.flatnonzero(sm >= thr)
                return int(hit[0]) if hit.size else len(p)

            rows.append({
                "A1_tail": a1_tail,
                "B_tail": float(np.mean(p2[-q:])),
                "A2_tail": float(np.mean(p3[-q:])),
                "t1": t_cross(p1), "t3": t_cross(p3),
                "savings": t_cross(p1) - t_cross(p3),
            })
        out[name] = rows
    return out


# ---------------------------------------------------------------------------
#  G3: adversarial, memory-targeted shocks
# ---------------------------------------------------------------------------

def g3_run(seed: int, agent_kw: dict, targeted: bool,
           budget: int = EVALS, shock_every: int = 60) -> dict:
    w = ASIWorld(seed=seed, noise=0.35, shock_every=0)   # shocks run by hand
    a = IBFASI(w, seed=100 + seed, model_planner=True, **agent_kw)
    n_since = 0
    while w.evals < budget:
        w.tick()
        n_since += 1
        if n_since >= shock_every // 4:                  # ~per agent-ticks
            n_since = 0
            if targeted and a.memory_best() is not None:
                best = a.memory_best()
                # displace AWAY from the best-known basin, damage memory NEAR it
                away = a.x + (a.x - best)
                a.x = np.clip(best + 4.0 * (away - best) /
                              (np.linalg.norm(away - best) + 1e-9), w.lo, w.hi)
                for s_ in a.scales[:1]:
                    s_.centers = [c for c in s_.centers
                                  if np.linalg.norm(c.z - best) > 1.5
                                  or a.rng.random() < 0.15]
            else:
                a.apply_shock()
        a.step()
    r = a.run(0) if False else None
    T = a.telemetry
    tail = T[-max(len(T) // 4, 1):]
    return {"true_tail": float(np.mean([t["true"] for t in tail])),
            "ok": float(np.mean([t["ok"] for t in tail]))}


G3_AGENTS = {"full ASI": dict(), "no-reflect": dict(reflect=False),
             "single-scale": dict(n_scales=1)}


def g3_adversarial(n_seeds: int) -> dict:
    out: dict = {}
    for name, kw in G3_AGENTS.items():
        out[name] = {
            "random": [g3_run(s, kw, targeted=False)["true_tail"]
                       for s in range(n_seeds)],
            "targeted": [g3_run(s, kw, targeted=True)["true_tail"]
                         for s in range(n_seeds)],
        }
    return out


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

def main(n_seeds: int = 8) -> None:
    print("\n" + "#" * 74)
    print("#  THE GAUNTLET -- compound, continual, adversarial (paired CIs)")
    print("#" * 74)

    print("\n  [G1] compound-stress ladder (tail TRUE coherence; levels stack):")
    print(f"  {'agent':<12}" + "".join(f"{'L' + str(l):>8}" for l in range(6)))
    g1 = g1_ladder(n_seeds)
    for name in G1_AGENTS:
        row = "".join(f"{np.mean(g1[name][l]):>8.2f}" if g1[name][l] else f"{'--':>8}"
                      for l in range(6))
        print(f"  {name:<12}" + row)
    print("  paired vs full ASI at L5 (the everything-at-once world):")
    for name in ("layer-1", "no-memory", "CMA-ES"):
        if g1[name][5] and len(g1[name][5]) == len(g1["full ASI"][5]):
            ci = paired_ci(g1["full ASI"][5], g1[name][5])
            print(f"   full - {name:<10}: {fmt_ci(ci)} {verdict(ci)}")

    print("\n  [G2] continual switching A->B->A (means over seeds; t1/t3 = ticks")
    print("  to reach 80% of own phase-1 tail; savings = t1 - t3, positive good):")
    g2 = g2_continual(n_seeds)
    print(f"  {'agent':<14}{'A1 tail':>9}{'B tail':>8}{'A2 tail':>9}"
          f"{'t1':>6}{'t3':>6}{'savings':>9}")
    sav = {}
    for name, rows in g2.items():
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        sav[name] = [r["savings"] for r in rows]
        print(f"  {name:<14}{m['A1_tail']:>9.2f}{m['B_tail']:>8.2f}"
              f"{m['A2_tail']:>9.2f}{m['t1']:>6.0f}{m['t3']:>6.0f}"
              f"{m['savings']:>9.0f}")
    ci_sav = paired_ci(sav["full ASI"], sav["no-memory"])
    print(f"  RETENTION/SAVINGS, full vs no-memory: {fmt_ci(ci_sav)} {verdict(ci_sav)}")
    ci_int = paired_ci([r["A2_tail"] for r in g2["full ASI"]],
                       [r["A2_tail"] for r in g2["no-dissolve"]])
    print(f"  RETURN-PHASE INTERFERENCE (A2 tail), full vs no-dissolve: "
          f"{fmt_ci(ci_int)} {verdict(ci_int)}")
    ci_b = paired_ci([r["B_tail"] for r in g2["full ASI"]],
                     [r["B_tail"] for r in g2["no-dissolve"]])
    print(f"  ADAPTATION (B tail), full vs no-dissolve: {fmt_ci(ci_b)} {verdict(ci_b)}")

    print("\n  [G3] adversarial memory-targeted shocks vs random shocks:")
    g3 = g3_adversarial(n_seeds)
    print(f"  {'agent':<14}{'random':>9}{'targeted':>10}{'damage':>9}")
    for name, r in g3.items():
        dmg = np.mean(r["random"]) - np.mean(r["targeted"])
        print(f"  {name:<14}{np.mean(r['random']):>9.2f}"
              f"{np.mean(r['targeted']):>10.2f}{dmg:>9.2f}")
    ci_t = paired_ci(g3["full ASI"]["random"], g3["full ASI"]["targeted"])
    print(f"  full ASI, random - targeted: {fmt_ci(ci_t)} {verdict(ci_t)} "
          f"(positive = the aimed attack genuinely hurts more)")

    # ----- minimal sanity asserts; the table is the deliverable -----
    assert np.mean(g1["full ASI"][5]) > np.mean(g1["random"][5]) + 0.3, \
        "the full agent must beat random even in the everything-at-once world"
    assert np.mean(sav["full ASI"]) >= np.mean(sav["no-memory"]) - 5, \
        "memory must not make relearning slower on return to A"
    print("\n  Gauntlet complete; every number above is the honest measurement.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="IBF-ASI gauntlet")
    p.add_argument("--seeds", type=int, default=8)
    a = p.parse_args()
    main(n_seeds=a.seeds)
