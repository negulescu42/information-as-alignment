"""Guided tour of the MSCN toy model.

Run everything::

    python -m mscn.demo

or a single stage::

    python -m mscn.demo pilot1   # individual coherence-gradient learning
    python -m mscn.demo pilot2   # emergent cooperation
    python -m mscn.demo pilot3   # coarse-graining / hierarchy
    python -m mscn.demo pilot4   # self-knowledge & the consciousness phase transition
    python -m mscn.demo mscn     # the full integrated network

Add ``--figures`` to also write plots into ``mscn_outputs/`` (needs matplotlib).
"""

from __future__ import annotations

import argparse

import numpy as np

from . import baselines, games, hierarchy, landscapes, network, phase
from . import selfmodel as sm
from .learner import (DiscreteIBFLearner, IBFLearner, boltzmann_prob_best,
                      euler_step, modification_decay, modification_equilibrium)
from .mscn import MSCN, MSCNConfig


def _rule(title: str) -> None:
    print("\n" + "=" * 74)
    print("  " + title)
    print("=" * 74)


# ===========================================================================
#  Pilot 1 -- individual coherence-gradient learning
# ===========================================================================

def pilot1(seeds: int = 10) -> None:
    _rule("PILOT 1  -- Individual Coherence-Gradient Learner (Layer 1)")

    # (a) Boltzmann monotonicity / agency advantage (Thm 7, 8c)
    print("\n[Agency, Thm 7/8c]  P(best action) = exp(k.d)/(exp(k.d)+1) rises with k:")
    ks = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    ps = [boltzmann_prob_best(k, 0.5) for k in ks]
    print("   k       : " + "  ".join(f"{k:5.1f}" for k in ks))
    print("   P(best) : " + "  ".join(f"{p:5.3f}" for p in ps))
    print(f"   strictly increasing in k: {all(a < b for a, b in zip(ps, ps[1:]))}"
          f"   (k=0 -> 0.5 random ; k->inf -> 1.0 greedy)")

    # (b) Selective retention + Euler->ODE convergence (Thm 8b, 11)
    alpha, mu = 0.5, 0.1
    f = 0.0
    for _ in range(200):
        f = euler_step(f, alpha, mu)
    print("\n[Selective retention, Thm 8b]  reinforced -> alpha/mu ; unreinforced -> 0:")
    print(f"   reinforced equilibrium  : iterated Euler = {f:.3f}  (alpha/mu = {modification_equilibrium(alpha, mu):.3f})")
    print(f"   unreinforced decay f0=3 : t=0:{modification_decay(3, mu, 0):.2f} "
          f"t=20:{modification_decay(3, mu, 20):.2f} t=60:{modification_decay(3, mu, 60):.3f} -> 0")

    # (c) Basin expansion (Thm 8a): viable(R_eff) superset of viable(baseline)
    L1 = landscapes.rastrigin(1)
    grid = np.linspace(L1.lo[0], L1.hi[0], 400).reshape(-1, 1)
    theta = -3.0
    base_coh = np.array([L1.coherence(g) for g in grid])
    ag = IBFLearner(L1.coherence, L1.lo, L1.hi, alpha=0.5, mu=0.02, k=1.0, seed=3)
    ag.run(200)
    reff = np.array([ag.R_eff(g) for g in grid])
    base_v = int(np.sum(base_coh > theta))
    learn_v = int(np.sum(reff > theta))
    superset = bool(np.all((base_coh > theta) <= (reff > theta)))
    print("\n[Basin expansion, Thm 8a]  non-negative modification only grows the viable set:")
    print(f"   baseline viable grid pts = {base_v}   learned viable grid pts = {learn_v}")
    print(f"   every baseline-viable point stays viable: {superset}   deltaR >= 0 everywhere: "
          f"{bool(np.all(reff >= base_coh - 1e-9))}")

    # (d) The learner as an optimiser vs classical baselines (budget-matched)
    print("\n[Optimisation]  best objective f (lower is better), eval-budget matched,"
          f" mean over {seeds} seeds:")
    print(f"   {'landscape':<16}{'IBF':>10}{'random':>10}{'hill':>10}{'SA':>10}   winner")
    specs = [("rastrigin", 2, 2000), ("rastrigin", 5, 4000),
             ("ackley", 2, 2000), ("ackley", 5, 4000), ("schwefel", 2, 2000)]
    for name, dim, budget in specs:
        res = {m: [] for m in ("IBF", "random", "hill", "SA")}
        for s in range(seeds):
            L = landscapes.make(name, dim)
            res["IBF"].append(IBFLearner(L.coherence, L.lo, L.hi, alpha=0.4, mu=0.03,
                                         k=1.0, k_adapt=0.05, seed=s).run_until_evals(budget)["best_f"])
            res["random"].append(baselines.random_search(L, budget, seed=s)["best_f"])
            res["hill"].append(baselines.hill_climbing(L, budget, seed=s)["best_f"])
            res["SA"].append(baselines.simulated_annealing(L, budget, seed=s)["best_f"])
        means = {m: float(np.mean(v)) for m, v in res.items()}
        winner = min(means, key=means.get)
        tag = f"{name}-{dim}d"
        print(f"   {tag:<16}" + "".join(f"{means[m]:>10.3f}" for m in ("IBF", "random", "hill", "SA"))
              + f"   {winner}")
    print("   -> IBF beats random search in nearly every regime and is best of all four")
    print("      on the higher-dimensional multimodal landscapes (memory + basin expansion).")


# ===========================================================================
#  Pilot 2 -- emergent cooperation
# ===========================================================================

def pilot2(seeds: int = 24) -> None:
    _rule("PILOT 2  -- Emergent Cooperation in the Iterated Prisoner's Dilemma (Layer 2)")

    def mem(s):
        return games.IBFGameAgent(seed=s)

    def none(s):
        return games.IBFGameAgent(alpha=0.0, k=5.0, init_coop=0.0, seed=s)

    rounds = 300
    print("\n[Cooperation emerges only with modification memory]")
    m = games.play_match(mem(1), mem(2), rounds)
    n = games.play_match(none(1), none(2), rounds)
    print(f"   IBF-memory vs IBF-memory : mutual cooperation (late) = {m['mutual_coop_late']:.2f}"
          f"   score = {m['score_a']:.2f}")
    print(f"   memoryless vs memoryless : mutual cooperation (late) = {n['mutual_coop_late']:.2f}"
          f"   score = {n['score_a']:.2f}   (defection trap)")

    print("\n[Behaviour against fixed strategies -- cooperates with reciprocators,"
          " defends vs defectors]")
    for nm, opp in [("AllD", games.AlwaysDefect), ("TitForTat", games.TitForTat),
                    ("AllC", games.AlwaysCooperate), ("Pavlov", games.Pavlov)]:
        r = games.play_match(mem(1), opp(), rounds)
        print(f"   IBF-memory vs {nm:<9}: my cooperation (late) = {r['coop_a_late']:.2f}"
              f"   my score = {r['score_a']:.2f}")

    print("\n[Causal test, EC-4]  relational coupling J -- not the optimism -- sustains it:")
    for J in (0.0, 4.0):
        vals = [games.play_match(games.IBFGameAgent(coupling=J, seed=s),
                                 games.IBFGameAgent(coupling=J, seed=s + 100), rounds)["mutual_coop_late"]
                for s in range(seeds)]
        label = "defection trap" if J == 0 else "cooperation"
        print(f"   J = {J:.1f} -> mutual cooperation = {np.mean(vals):.2f}   ({label})")

    print("\n[Round-robin tournament]  mean score per round / cooperation rate:")
    players = {
        "IBF-mem": lambda: mem(7), "IBF-none": lambda: none(7),
        "TitForTat": games.TitForTat, "Pavlov": games.Pavlov,
        "AllD": games.AlwaysDefect, "AllC": games.AlwaysCooperate,
        "Random": lambda: games.RandomPlayer(seed=7),
    }
    t = games.round_robin(players, rounds=200)
    for l in sorted(t["summary"], key=lambda x: -t["summary"][x]["mean_score"]):
        s = t["summary"][l]
        print(f"   {l:<10} score = {s['mean_score']:.3f}   cooperation = {s['mean_coop']:.2f}")

    print("\n[Evolutionary selection by differential persistence -- Thm 6 corollary]")
    dp = games.differential_persistence(np.array([1.0, 2.0, 4.0, 8.0]), mu=0.05, threshold=0.5)
    for f0, surv in zip(dp["strengths"], dp["survival_time"]):
        print(f"   modification strength {f0:.1f} -> survives {surv:.0f} steps above threshold")
    print("   -> stronger memory persists longer under uniform decay (selection, no fitness fn)")


# ===========================================================================
#  Pilot 3 -- coarse-graining and hierarchy
# ===========================================================================

def pilot3(seeds: int = 8) -> None:
    _rule("PILOT 3  -- Coarse-Graining, Renormalisation Flow & Hierarchy (Layer 3)")

    rng = np.random.default_rng(0)
    n = 64
    xs = np.linspace(0, 1, n)
    signal = np.sin(2 * np.pi * xs)
    field = signal + rng.normal(0, 0.7, n)
    flow = hierarchy.rg_flow(field, factor=2, steps=4)
    print("\n[RG flow]  irrelevant small-scale noise is suppressed; the relevant scale emerges:")
    for i, f in enumerate(flow):
        print(f"   level {i}: size = {f.size:2d}   signal/noise = {hierarchy.signal_to_noise(f, signal):.2f}")
    peaks = [float(np.max(np.abs(f))) for f in flow]
    print(f"   peak |field| per level = {[round(p, 2) for p in peaks]}  -> "
          f"non-increasing under coarsening: {all(a >= b - 1e-9 for a, b in zip(peaks, peaks[1:]))}")

    print(f"\n[Hierarchy vs flat]  block-structured Rosenbrock, eval-budget matched,"
          f" mean over {seeds} seeds:")
    for n_blocks, block_size, budget in [(6, 2, 6000), (4, 3, 8000)]:
        hv, fv, wins = [], [], 0
        for s in range(seeds):
            L = hierarchy.block_landscape(landscapes.rosenbrock, n_blocks, block_size)
            h = hierarchy.HierarchicalOptimizer(L, n_blocks, seed=s).run(budget)
            fl = hierarchy.flat_baseline(L, budget, seed=s)
            hv.append(h["best_f"])
            fv.append(fl["best_f"])
            wins += int(h["best_f"] < fl["best_f"])
        dim = n_blocks * block_size
        print(f"   {n_blocks} blocks x {block_size}d = {dim}d :  hierarchical f = {np.mean(hv):6.3f}"
              f"   flat f = {np.mean(fv):6.3f}   hierarchy wins {wins}/{seeds}")
    print("   -> coarse-graining into independent sub-problems beats flat high-dim search.")


# ===========================================================================
#  Pilot 4 -- self-knowledge & the consciousness phase transition
# ===========================================================================

def pilot4() -> None:
    _rule("PILOT 4  -- Bounded Self-Knowledge & the Consciousness Phase Transition (Layer 3)")

    print("\n[Lawvere obstruction]  no self-model is surjective -- perfect self-knowledge is impossible:")
    rng = np.random.default_rng(0)
    for n in (3, 5, 10, 50):
        model = rng.integers(0, 2, size=(n, n))
        res = sm.lawvere_obstruction(model, sm.bool_not)
        print(f"   |S| = {n:2d}: the diagonal response escapes every self-model -> "
              f"perfect self-knowledge possible: {res['perfect_self_knowledge_possible']}")

    print("\n[Self-knowledge ceiling]  achievable introspection depth = floor(Gamma / base_cost):")
    for G, bc in [(10, 3), (10, 2.5), (7, 4)]:
        print(f"   Gamma = {G}, base_cost = {bc} -> max depth = {sm.achievable_depth(G, bc)}")

    print("\n[Consciousness-competence tradeoff]  capacity spent on introspection is lost to the task:")
    tc = sm.tradeoff_curve(12, 2)
    for d, s, dom in zip(tc["depth"], tc["self_budget"], tc["domain_capacity"]):
        bar = "#" * int(dom)
        print(f"   depth {d}: self-budget {int(s):2d}  domain capacity {int(dom):2d}  {bar}")

    print("\n[Phase transition]  E* = c + alpha/mu vs threshold theta  (c=0.5, theta=1.0, mu=0.2):")
    c, theta, mu = 0.5, 1.0, 0.2
    ac = phase.critical_alpha(mu, theta, c)
    print(f"   critical alpha = mu*(theta-c) = {ac:.3f}")
    for a in (0.05, ac, 0.30):
        print(f"   alpha = {a:.3f} -> E* = {phase.equilibrium_coherence(c, a, mu):.3f}"
              f"   [{phase.phase_of(a, mu, theta, c)}]")
    print(f"   consciousness is dissipative: undriven lifetime (f0=2, gap=0.5) = "
          f"{phase.dissipative_lifetime(2, mu, 0.5):.2f} steps")

    print("\n[Zombie-Twin]  a self-monitoring agent outlasts a non-monitoring twin"
          " under identical shocks:")
    st = phase.zombie_twin_stats(150)
    print(f"   conscious (self-correcting) mean survival = {st['mean_survival_conscious']:.0f}")
    print(f"   zombie    (no monitoring)   mean survival = {st['mean_survival_zombie']:.0f}")
    print(f"   conscious outlasts the zombie in {100 * st['conscious_outlasts_fraction']:.0f}% of episodes")


# ===========================================================================
#  Full MSCN
# ===========================================================================

def mscn_demo() -> None:
    _rule("FULL MSCN  -- all three layers running at once")
    L = landscapes.rastrigin(2)
    cfg = MSCNConfig(n_agents=27, rounds=120, coupling_strength=0.4, perturb_every=30, seed=1)
    res = MSCN(L, cfg).run()
    h = res["history"]
    print(f"\n   {cfg.n_agents} coupled learners on Rastrigin-2D, scale-free graph,"
          f" {cfg.rounds} rounds")
    print(f"   viability threshold theta = {res['theta']:.3f}\n")
    print("   round | best | mean_eff | consensus | viable | event")
    marks = [0, cfg.warmup, 29, 30, 33, 59, 60, 63, 89, 90, 93, cfg.rounds - 1]
    for t in marks:
        r = h[t]
        ev = "PERTURB" if r["perturbed"] else ("self-correct" if r["boosting"] else "")
        print(f"    {r['round']:4d} | {r['best_coherence']:4.2f} | {r['mean_coherence']:8.3f} |"
              f"  {r['consensus']:6.3f}   | {r['n_viable']:2d}/{cfg.n_agents:<2d} | {ev}")
    print()
    print(f"   learning      : best coherence {h[0]['best_coherence']:.2f} -> {h[-1]['best_coherence']:.2f}")
    print(f"   cooperation   : edge consensus {h[0]['consensus']:.2f} -> {h[-1]['consensus']:.2f}"
          f" (coupled agents align)")
    print(f"   coupling helps: network coherence >= sum of individuals: {res['coupling_helps']}")
    print(f"   self-correct  : recovered after each perturbation in "
          f"{[d for _, d in res['recoveries']]} rounds")
    print(f"   coarse-grain  : macro-hierarchy "
          + " -> ".join(str(lv["n_units"]) for lv in res["coarse_levels"]))


# ===========================================================================
#  Optional figures
# ===========================================================================

def make_figures(outdir: str = "mscn_outputs") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"   (figures skipped: matplotlib unavailable: {e})")
        return
    import os
    os.makedirs(outdir, exist_ok=True)

    # Boltzmann agency curves
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ks = np.linspace(0, 20, 200)
    for d in (0.1, 0.3, 0.5, 1.0):
        ax.plot(ks, [boltzmann_prob_best(k, d) for k in ks], label=f"gap={d}")
    ax.set_xlabel("responsiveness k"); ax.set_ylabel("P(best action)")
    ax.set_title("Agency advantage (Thm 7/8c)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/agency.png", dpi=120); plt.close(fig)

    # MSCN trajectories
    L = landscapes.rastrigin(2)
    res = MSCN(L, MSCNConfig(seed=1)).run()
    h = res["history"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    rounds = [r["round"] for r in h]
    ax[0].plot(rounds, [r["best_coherence"] for r in h], label="best")
    ax[0].plot(rounds, [r["mean_coherence"] for r in h], label="mean eff")
    ax[0].axhline(res["theta"], ls="--", c="grey", label="theta")
    for r in h:
        if r["perturbed"]:
            ax[0].axvline(r["round"], c="red", alpha=0.3)
    ax[0].set_title("MSCN learning + self-correction"); ax[0].legend(fontsize=8)
    ax[0].set_xlabel("round"); ax[0].set_ylabel("coherence")
    ax[1].plot(rounds, [r["consensus"] for r in h], c="green")
    ax[1].set_title("Consensus (cooperation)"); ax[1].set_xlabel("round")
    ax[1].set_ylabel("mean edge distance")
    fig.tight_layout(); fig.savefig(f"{outdir}/mscn.png", dpi=120); plt.close(fig)
    print(f"   figures written to {outdir}/agency.png and {outdir}/mscn.png")


STAGES = {"pilot1": pilot1, "pilot2": pilot2, "pilot3": pilot3, "pilot4": pilot4, "mscn": mscn_demo}


def main() -> None:
    p = argparse.ArgumentParser(description="MSCN toy-model demo")
    p.add_argument("stage", nargs="?", default="all", choices=list(STAGES) + ["all"])
    p.add_argument("--figures", action="store_true", help="also write plots (needs matplotlib)")
    args = p.parse_args()

    print("\n" + "#" * 74)
    print("#  MULTI-SCALE COHERENCE NETWORK (MSCN) -- IBF toy model")
    print("#  Knowledge is stored as coherence modifications, not weights.")
    print("#" * 74)

    if args.stage == "all":
        for fn in STAGES.values():
            fn()
    else:
        STAGES[args.stage]()

    if args.figures:
        _rule("FIGURES")
        make_figures()
    print()


if __name__ == "__main__":
    main()
