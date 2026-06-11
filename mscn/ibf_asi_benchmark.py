"""FULL-CAPACITY BENCHMARK -- the integrated IBF-ASI vs external baselines, plus
capacity scaling.

Two questions, answered with the repo's norms (eval-budget matched, paired per-seed
CIs, honest losses reported):

(1) STRENGTH: across all eight regimes of the claim map, how does the FULL agent
    (all nine stages + the model-based planner with U8 option commitment) compare
    to agents that are not it?
      * random   -- uniform selection among the same candidate set (k = 0), no
                    memory: the lower anchor.
      * reactive -- Boltzmann hill-walker, no memory, no planning, single scale:
                    the stage-zero ancestor.
      * layer-1  -- the original `learner.py` IBFLearner (the validated MSCN
                    Layer-1 optimiser), budget-matched through the same world.
      * CMA-ES   -- the strongest classical optimiser from the repo's benchmarks
                    (`strong_baselines`), where applicable. Honest caveats: CMA is
                    designed for static objectives (drift/shocked worlds violate
                    its assumptions; shocks cannot displace a population method,
                    which is an exemption in its favour) and needs dim >= 2 (the
                    corridor is skipped).
    The world clock is COMMON: every agent senses through a wrapper that advances
    world time (drift/shocks) once per TICK_EVERY senses, so time exposure is
    identical at equal eval budgets.

(2) CAPACITY: how does the full agent scale with its own budgets -- memory
    capacity Gamma, number of scales, experience (eval budget) -- and what does a
    tick cost in wall time as memory fills (the O(M) kernel concern; the
    Interface-Principle pruning of `correction_field.py` is the named remedy at
    10^5+ centres)?

Run: ``python -m mscn.ibf_asi_benchmark [--seeds 8]``   (numpy+scipy(+cma), ~15-25 min).
"""

from __future__ import annotations

import time

import numpy as np

from .ibf_asi import IBFASI
from .ibf_asi_regimes import BASE, REGIMES, REGIME_AGENT, make_world
from .stats import fmt_ci, paired_ci, verdict

try:
    import cma
    HAS_CMA = True
except Exception:                                   # pragma: no cover
    HAS_CMA = False

EVALS = 4000
TICK_EVERY = 16            # senses per world tick (the ASI's own average rate)


class TickingWorld:
    """Common clock: wraps a world so that time (drift/shocks) advances once per
    TICK_EVERY senses for EVERY agent type. Shocks displace `holder.x` when the
    agent has a position (population methods are exempt -- noted honestly)."""

    def __init__(self, world, holder=None) -> None:
        self.w = world
        self.holder = holder
        self._n = 0

    def sense(self, x) -> float:
        self._n += 1
        if self._n % TICK_EVERY == 0:
            shocked = self.w.tick()
            if shocked and self.holder is not None and hasattr(self.holder, "x"):
                self.holder.x = self.w.shock_point()
                if hasattr(self.holder, "apply_shock"):
                    self.holder.apply_shock()
        return self.w.sense(x)

    @property
    def evals(self) -> int:
        return self.w.evals


def _spawn(regime: str, w, s: int):
    if regime == "moat-local":
        ang = 2 * np.pi * (s % 8) / 8.0
        return np.clip(w.c[0] + 4.0 * np.array([np.cos(ang), np.sin(ang)]),
                       w.lo, w.hi)
    if regime == "corridor":
        return np.array([-5.0])
    return None


# ---------------------------------------------------------------------------
#  Contestants (each returns the tail-mean TRUE coherence; eval-budget matched)
# ---------------------------------------------------------------------------

def run_asi(regime: str, s: int, agent_kw: dict, budget: int = EVALS) -> float:
    w = make_world(regime, seed=s)
    kw = {**BASE, **REGIME_AGENT.get(regime, {}), **agent_kw}
    a = IBFASI(w, seed=100 + s, **kw)
    x0 = _spawn(regime, w, s)
    if x0 is not None:
        a.x = x0
    return a.run(budget)["true_tail"]


def contestant_full(regime: str, s: int) -> float:
    return run_asi(regime, s, dict())


def contestant_lean(regime: str, s: int) -> float:
    """The 'lean' operating mode found by the gap diagnosis: the open-landscape
    deficit to layer-1 was EVAL OVERHEAD per decision (the H=2 rollout burns 9
    senses/tick for measured-nothing on open ground), not capability. horizon=1
    + annealed candidate steps closes the gap to ns while keeping the corridor
    win fully (the planner candidate is rollout-independent)."""
    return run_asi(regime, s, dict(horizon=1, anneal_steps=True))


def contestant_reactive(regime: str, s: int) -> float:
    return run_asi(regime, s, dict(alpha=0.0, horizon=1, n_scales=1,
                                   model_planner=False, reflect=False,
                                   dissolve=False))


def contestant_random(regime: str, s: int) -> float:
    return run_asi(regime, s, dict(alpha=0.0, horizon=1, n_scales=1,
                                   model_planner=False, reflect=False,
                                   dissolve=False, k0=1e-9, k_adapt=0.0,
                                   k_max=1e-9))


def contestant_layer1(regime: str, s: int, budget: int = EVALS) -> float:
    from .learner import IBFLearner
    w = make_world(regime, seed=s)
    holder = type("H", (), {})()                    # mutable .x holder for shocks
    tw = TickingWorld(w, holder)
    # locomotion regimes forbid teleports for everyone: no global proposals, and
    # the step scale is bounded to the common locomotion scale (the learner's
    # default step0 is 0.2*range = 2.4 here -- it leaps the valley in one
    # 'local' proposal otherwise; measured).
    loco = regime in ("moat-local", "corridor")
    ag = IBFLearner(lambda x: tw.sense(x), w.lo, w.hi, seed=100 + s,
                    n_global_proposals=0 if loco else 2,
                    step0=0.5 if loco else None)
    x0 = _spawn(regime, w, s)
    if x0 is not None:
        ag.x = x0.copy()
    holder.x = ag.x
    # shocks hit fast modes for every memory agent: same damage as the ASI's
    holder.apply_shock = lambda: setattr(
        ag, "centers", [c for c in ag.centers if w.rng.random() < 0.3])
    trace = []
    while tw.evals < budget:
        ag.x = holder.x                             # apply any shock displacement
        ag.step()
        holder.x = ag.x
        trace.append(w.true_coherence(ag.x))
    tail = trace[-max(len(trace) // 4, 1):]
    return float(np.mean(tail))


def contestant_cma(regime: str, s: int, budget: int = EVALS) -> float | None:
    if not HAS_CMA:
        return None
    w = make_world(regime, seed=s)
    if w.dim < 2:                                   # CMA needs dim >= 2
        return None
    tw = TickingWorld(w)                            # population: shock-exempt
    trace = []
    restart = 0
    while tw.evals < budget:                        # standard restart strategy
        x0 = w.rng.uniform(w.lo, w.hi)
        es = cma.CMAEvolutionStrategy(
            list(x0), 2.0, {"popsize": 8, "verbose": -9, "seed": 1 + s + 97 * restart,
                            "bounds": [float(w.lo[0]), float(w.hi[0])]})
        while tw.evals < budget and not es.stop():
            X = es.ask()
            es.tell(X, [-tw.sense(np.asarray(xi)) for xi in X])
            trace.append(w.true_coherence(np.asarray(es.mean)))
        restart += 1
    tail = trace[-max(len(trace) // 4, 1):]
    return float(np.mean(tail))


CONTESTANTS = {
    "full IBF-ASI": contestant_full,
    "full ASI (lean)": contestant_lean,
    "layer-1 IBFLearner": contestant_layer1,
    "CMA-ES": contestant_cma,
    "reactive": contestant_reactive,
    "random": contestant_random,
}


# ---------------------------------------------------------------------------
#  Part 1: strength table
# ---------------------------------------------------------------------------

def strength_table(n_seeds: int = 8) -> dict:
    table: dict[str, dict[str, list]] = {n: {} for n in CONTESTANTS}
    for name, fn in CONTESTANTS.items():
        for r in REGIMES:
            vals = [fn(r, s) for s in range(n_seeds)]
            table[name][r] = vals
    return table


# ---------------------------------------------------------------------------
#  Part 2: capacity scaling (the 'full capacity' question)
# ---------------------------------------------------------------------------

def capacity_sweeps(n_seeds: int = 6) -> dict:
    out = {"Gamma": {}, "scales": {}, "experience": {}, "cost": {}}
    for g in (2.0, 8.0, 60.0, 240.0):
        out["Gamma"][g] = [run_asi("drift", s, dict(Gamma=g)) for s in range(n_seeds)]
    for ns in (1, 2, 3, 4):
        out["scales"][ns] = [run_asi("drift", s, dict(n_scales=ns))
                             for s in range(n_seeds)]
    for b in (2000, 4000, 9000, 18000):
        out["experience"][b] = [run_asi("drift", s, dict(), budget=b)
                                for s in range(n_seeds)]
    # wall-time cost per tick as memory fills (the O(M) kernel concern)
    w = make_world("drift", seed=0)
    a = IBFASI(w, seed=100, **BASE)
    marks = {}
    t0, e0, n0 = time.perf_counter(), 0, 0
    while w.evals < 18000:
        w.tick()
        a.step()
        if w.evals // 4500 != e0:
            e0 = w.evals // 4500
            dt = time.perf_counter() - t0
            marks[w.evals] = (1000 * dt / max(a.tick_no - n0, 1),
                              sum(len(s.centers) for s in a.scales))
            t0, n0 = time.perf_counter(), a.tick_no
    out["cost"] = marks
    return out


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

def main(n_seeds: int = 8) -> None:
    print("\n" + "#" * 74)
    print("#  IBF-ASI FULL-CAPACITY BENCHMARK -- vs external baselines + scaling")
    print(f"#  eval budget {EVALS}, {n_seeds} seeds, common world clock "
          f"(tick per {TICK_EVERY} senses)")
    print("#" * 74)

    tbl = strength_table(n_seeds)
    print("\n  [1] strength: tail-mean TRUE coherence (mean over seeds; '--' = n/a):\n")
    print(f"  {'agent':<20}" + "".join(f"{r[:9]:>10}" for r in REGIMES) + f"{'AGG':>8}")
    aggs = {}
    for name in CONTESTANTS:
        row, vals_all = "", []
        for r in REGIMES:
            vals = [v for v in tbl[name][r] if v is not None]
            if vals:
                row += f"{np.mean(vals):>10.2f}"
                vals_all.append(np.mean(vals))
            else:
                row += f"{'--':>10}"
        aggs[name] = float(np.mean(vals_all))
        print(f"  {name:<20}" + row + f"{aggs[name]:>8.2f}")

    print("\n  paired CIs vs the full agent (positive = full wins; per regime):")
    full = tbl["full IBF-ASI"]
    for name in CONTESTANTS:
        if name == "full IBF-ASI":
            continue
        cells = []
        for r in REGIMES:
            ours, theirs = full[r], tbl[name][r]
            if any(v is None for v in theirs):
                cells.append("    --  ")
                continue
            ci = paired_ci(ours, theirs)
            mark = "*" if ci["lo"] > 0 else ("!" if ci["hi"] < 0 else " ")
            cells.append(f"{ci['mean']:>+7.2f}{mark}")
        print(f"   vs {name:<19}" + "".join(f"{c:>10}" for c in cells)
              + "   (* full sig-better, ! sig-worse)")

    print("\n  [2] capacity scaling (drift regime, full agent):")
    cs = capacity_sweeps()
    print("      Gamma budget : " + "  ".join(
        f"G={g:g}: {np.mean(v):.2f}" for g, v in cs["Gamma"].items()))
    print("      scale count  : " + "  ".join(
        f"S={k}: {np.mean(v):.2f}" for k, v in cs["scales"].items()))
    print("      experience   : " + "  ".join(
        f"{b//1000}k: {np.mean(v):.2f}" for b, v in cs["experience"].items()))
    print("      cost/tick    : " + "  ".join(
        f"@{e//1000}k evals: {ms:.1f}ms ({nc} ctr)"
        for e, (ms, nc) in cs["cost"].items()))

    # ----- honest verdicts -----
    agg_full = aggs["full IBF-ASI"]
    ci_rand = paired_ci([np.mean(full[r]) for r in REGIMES],
                        [np.mean(tbl["random"][r]) for r in REGIMES])
    print(f"\n  verdict:")
    print(f"   * aggregate: full {agg_full:.2f} | " + " | ".join(
        f"{n} {aggs[n]:.2f}" for n in CONTESTANTS if n != "full IBF-ASI"))
    print(f"   * full vs random across regimes: {fmt_ci(ci_rand)} {verdict(ci_rand)}")
    losses = []
    for name in CONTESTANTS:
        if name == "full IBF-ASI":
            continue
        for r in REGIMES:
            theirs = tbl[name][r]
            if any(v is None for v in theirs):
                continue
            ci = paired_ci(full[r], theirs)
            if ci["hi"] < 0:
                losses.append((name, r, ci["mean"]))
    print(f"   * regimes where a baseline SIG-beats the full agent (honest): "
          f"{losses or 'none'}")
    assert aggs["full IBF-ASI"] > aggs["random"] + 0.5, \
        "the full agent must decisively beat the random anchor in aggregate"
    assert aggs["full IBF-ASI"] > aggs["reactive"], \
        "the full agent must beat its stage-zero ancestor in aggregate"
    print("\n  Done. Every number above is eval-budget matched on a common world")
    print("  clock; losses are reported, not hidden.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="IBF-ASI full-capacity benchmark")
    p.add_argument("--seeds", type=int, default=8)
    a = p.parse_args()
    main(n_seeds=a.seeds)
