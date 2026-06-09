"""Benchmark harness for the MSCN toy model.

Runs statistically-grounded, evaluation-budget-matched comparisons and prints a
report. Sections:

* **optimiser**   -- Layer-1 IBF learner vs random search / hill climbing /
  simulated annealing across several functions and dimensions (mean +/- std,
  win-rate, mean rank, wall-clock).
* **convergence** -- best-so-far vs evaluation budget (figure with --figures).
* **cooperation** -- round-robin tournament statistics over many seeds.
* **hierarchy**   -- hierarchical vs flat learner across dimensions.
* **scaling**     -- integrated-MSCN wall-clock vs agent count and dimension.

Run::

    python -m mscn.benchmark             # default (20 seeds)
    python -m mscn.benchmark --quick     # fast (8 seeds)
    python -m mscn.benchmark optimiser --figures
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import baselines, games, hierarchy, landscapes, strong_baselines
from .learner import IBFLearner
from .mscn import MSCN, MSCNConfig

ArrayF = np.ndarray

FUNCTIONS = ["sphere", "rastrigin", "ackley", "schwefel", "rosenbrock"]
DIMS = [2, 5, 10]


def _budget(dim: int) -> int:
    return max(500 * dim, 1500)


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print("  " + title)
    print("=" * 78)


# ===========================================================================
#  Optimiser benchmark
# ===========================================================================

def _run_methods(name: str, dim: int, budget: int, seed: int) -> dict:
    L = landscapes.make(name, dim)
    out = {}
    t = time.perf_counter()
    out["IBF"] = IBFLearner(L.coherence, L.lo, L.hi, alpha=0.4, mu=0.03, k=1.0,
                            k_adapt=0.05, seed=seed).run_until_evals(budget)["best_f"]
    out["_t_IBF"] = time.perf_counter() - t
    t = time.perf_counter(); out["random"] = baselines.random_search(L, budget, seed)["best_f"]
    out["_t_random"] = time.perf_counter() - t
    t = time.perf_counter(); out["hill"] = baselines.hill_climbing(L, budget, seed)["best_f"]
    out["_t_hill"] = time.perf_counter() - t
    t = time.perf_counter(); out["SA"] = baselines.simulated_annealing(L, budget, seed)["best_f"]
    out["_t_SA"] = time.perf_counter() - t
    return out


def optimiser_benchmark(seeds: int = 20, functions=None, dims=None) -> dict:
    functions = functions or FUNCTIONS
    dims = dims or DIMS
    methods = ["IBF", "random", "hill", "SA"]
    _rule(f"OPTIMISER BENCHMARK  (eval-budget matched, {seeds} seeds)")
    print(f"\n  {'function':<11}{'dim':>4} {'budget':>7} | "
          + "".join(f"{m:>16}" for m in methods)
          + " | IBF win vs rnd/hill/SA")
    print("  " + "-" * 110)

    rank_acc = {m: [] for m in methods}
    time_acc = {m: [] for m in methods}
    results = {}
    for name in functions:
        for dim in dims:
            budget = _budget(dim)
            vals = {m: [] for m in methods}
            for s in range(seeds):
                r = _run_methods(name, dim, budget, s)
                for m in methods:
                    vals[m].append(r[m])
                    time_acc[m].append(r["_t_" + m])
                # rank per seed (1 = best/lowest f)
                order = sorted(methods, key=lambda m: r[m])
                for rk, m in enumerate(order):
                    rank_acc[m].append(rk + 1)
            results[(name, dim)] = {m: np.array(vals[m]) for m in methods}
            cell = ""
            for m in methods:
                v = np.array(vals[m])
                cell += f"{np.mean(v):>8.2f}±{np.std(v):<6.2f}"
            wins = {b: float(np.mean(np.array(vals["IBF"]) < np.array(vals[b]))) for b in ("random", "hill", "SA")}
            print(f"  {name:<11}{dim:>4} {budget:>7} | {cell} | "
                  f"{wins['random']:.2f} / {wins['hill']:.2f} / {wins['SA']:.2f}")

    print("\n  mean rank (1 = best over all function x dim x seed):")
    for m in methods:
        print(f"    {m:<8} {np.mean(rank_acc[m]):.2f}")
    print("\n  mean wall-clock per run (ms):")
    for m in methods:
        print(f"    {m:<8} {1000 * np.mean(time_acc[m]):7.1f}")
    return results


# ===========================================================================
#  Strong-baseline benchmark (CMA-ES, differential evolution, dual annealing)
# ===========================================================================

def strong_benchmark(seeds: int = 15, functions=None, dims=None) -> dict:
    """IBF vs SOTA black-box optimisers, evaluation-budget matched."""
    functions = functions or FUNCTIONS
    dims = dims or DIMS
    strong = [m for m, ok in strong_baselines.AVAILABLE.items() if ok]
    if not strong:
        _rule("STRONG-BASELINE BENCHMARK")
        print("\n  (install `cma` and `scipy` to enable: pip install cma scipy)")
        return {}
    methods = ["IBF", "SA"] + strong
    _rule(f"STRONG-BASELINE BENCHMARK  (vs {', '.join(strong)}; {seeds} seeds, budget-matched)")
    print(f"\n  {'function':<11}{'dim':>4} {'budget':>7} | "
          + "".join(f"{m:>12}" for m in methods)
          + " | IBF win vs " + "/".join(m.split('-')[0] for m in strong))
    print("  " + "-" * (30 + 12 * len(methods) + 24))

    rank_acc = {m: [] for m in methods}
    time_acc = {m: [] for m in methods}
    results = {}
    for name in functions:
        for dim in dims:
            budget = _budget(dim)
            vals = {m: [] for m in methods}
            for s in range(seeds):
                L = landscapes.make(name, dim)
                runs = {}
                t = time.perf_counter()
                runs["IBF"] = IBFLearner(L.coherence, L.lo, L.hi, alpha=0.4, mu=0.03,
                                         k=1.0, k_adapt=0.05, seed=s).run_until_evals(budget)["best_f"]
                time_acc["IBF"].append(time.perf_counter() - t)
                t = time.perf_counter(); runs["SA"] = baselines.simulated_annealing(L, budget, s)["best_f"]
                time_acc["SA"].append(time.perf_counter() - t)
                for m in strong:
                    t = time.perf_counter()
                    runs[m] = strong_baselines.STRONG[m](L, budget, s)["best_f"]
                    time_acc[m].append(time.perf_counter() - t)
                for m in methods:
                    vals[m].append(runs[m])
                order = sorted(methods, key=lambda m: runs[m])
                for rk, m in enumerate(order):
                    rank_acc[m].append(rk + 1)
            results[(name, dim)] = {m: np.array(vals[m]) for m in methods}
            cell = "".join(f"{np.mean(vals[m]):>12.2f}" for m in methods)
            wins = "/".join(f"{np.mean(np.array(vals['IBF']) < np.array(vals[m])):.2f}" for m in strong)
            print(f"  {name:<11}{dim:>4} {budget:>7} | {cell} | {wins}")

    print("\n  mean rank (1 = best of these methods, over all function x dim x seed):")
    for m in sorted(methods, key=lambda m: np.mean(rank_acc[m])):
        print(f"    {m:<12} {np.mean(rank_acc[m]):.2f}")
    print("\n  mean wall-clock per run (ms):")
    for m in methods:
        print(f"    {m:<12} {1000 * np.mean(time_acc[m]):8.1f}")
    return results


# ===========================================================================
#  Convergence (best-so-far vs evaluations)
# ===========================================================================

def _random_trace(L, budget, seed, points):
    rng = np.random.default_rng(seed)
    X = L.random_points(budget, rng)
    f = L.f_batch(X)
    best = np.minimum.accumulate(f)
    idx = np.linspace(budget // points, budget, points).astype(int) - 1
    return idx + 1, best[idx]


def _sa_trace(L, budget, seed, points, step_frac=0.1, t0=5.0, t_min=1e-3):
    rng = np.random.default_rng(seed)
    step = step_frac * L.range
    x = L.random_point(rng); fx = L.f(x); best = fx
    cool = (t_min / t0) ** (1.0 / max(budget, 1)); t = t0
    traj = []
    for i in range(budget):
        y = L.clip(x + rng.normal(0, 1, L.dim) * step); fy = L.f(y)
        if fy < fx or rng.random() < np.exp(-(fy - fx) / max(t, 1e-12)):
            x, fx = y, fy; best = min(best, fx)
        t *= cool
        traj.append(best)
    idx = np.linspace(budget // points, budget, points).astype(int) - 1
    return idx + 1, np.array(traj)[idx]


def convergence_benchmark(seeds: int = 20, specs=None, points: int = 40, figpath: str | None = None) -> dict:
    specs = specs or [("rastrigin", 5), ("ackley", 5)]
    _rule(f"CONVERGENCE  (best-so-far vs evaluations, mean of {seeds} seeds)")
    data = {}
    for name, dim in specs:
        budget = _budget(dim)
        ibf, rnd, sa = [], [], []
        ev = None
        for s in range(seeds):
            L = landscapes.make(name, dim)
            r = IBFLearner(L.coherence, L.lo, L.hi, alpha=0.4, mu=0.03, k=1.0,
                           k_adapt=0.05, seed=s).run_until_evals(budget, trace_points=points)
            ibf.append(r["trace_best_f"]); ev = r["trace_evals"]
            _, rt = _random_trace(L, budget, s, points); rnd.append(rt)
            _, st = _sa_trace(L, budget, s, points); sa.append(st)
        data[(name, dim)] = {
            "evals": ev,
            "IBF": np.mean(ibf, axis=0), "random": np.mean(rnd, axis=0), "SA": np.mean(sa, axis=0),
        }
        d = data[(name, dim)]
        print(f"\n  {name}-{dim}d (budget {budget}):  best-so-far at 25% / 50% / 100% of budget")
        for m in ("IBF", "random", "SA"):
            q = len(d[m])
            print(f"    {m:<8} {d[m][q//4]:8.3f}  {d[m][q//2]:8.3f}  {d[m][-1]:8.3f}")
    if figpath:
        _convergence_figure(data, figpath)
    return data


def _convergence_figure(data, figpath):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"   (figure skipped: {e})"); return
    import os
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    specs = list(data)
    fig, axes = plt.subplots(1, len(specs), figsize=(5.2 * len(specs), 3.6), squeeze=False)
    for ax, key in zip(axes[0], specs):
        d = data[key]
        for m, c in (("IBF", "C0"), ("random", "C1"), ("SA", "C2")):
            ax.plot(d["evals"], d[m], label=m, color=c)
        ax.set_yscale("log")
        ax.set(title=f"{key[0]}-{key[1]}d convergence", xlabel="evaluations", ylabel="best f (log)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figpath, dpi=120); plt.close(fig)
    print(f"\n  convergence figure -> {figpath}")


# ===========================================================================
#  Cooperation, hierarchy, scaling
# ===========================================================================

def cooperation_benchmark(seeds: int = 20) -> dict:
    _rule(f"COOPERATION BENCHMARK  (round-robin, {seeds} tournaments)")
    factories = {
        "IBF-mem": lambda s: games.IBFGameAgent(seed=s),
        "IBF-none": lambda s: games.IBFGameAgent(alpha=0.0, k=5.0, init_coop=0.0, seed=s),
        "TitForTat": lambda s: games.TitForTat(),
        "Pavlov": lambda s: games.Pavlov(),
        "AllD": lambda s: games.AlwaysDefect(),
        "AllC": lambda s: games.AlwaysCooperate(),
        "Random": lambda s: games.RandomPlayer(seed=s),
    }
    scores = {l: [] for l in factories}
    coops = {l: [] for l in factories}
    for s in range(seeds):
        players = {l: (lambda f=f, s=s: f(s)) for l, f in factories.items()}
        t = games.round_robin(players, rounds=200)
        for l in factories:
            scores[l].append(t["summary"][l]["mean_score"])
            coops[l].append(t["summary"][l]["mean_coop"])
    print(f"\n  {'strategy':<11}{'score (mean±std)':>22}{'cooperation':>16}")
    for l in sorted(factories, key=lambda x: -np.mean(scores[x])):
        print(f"  {l:<11}{np.mean(scores[l]):>10.3f} ± {np.std(scores[l]):<7.3f}"
              f"{np.mean(coops[l]):>12.2f}")
    return {"scores": scores, "coops": coops}


def hierarchy_benchmark(seeds: int = 20) -> dict:
    _rule(f"HIERARCHY BENCHMARK  (hierarchical vs flat, {seeds} seeds)")
    configs = [(4, 2), (6, 2), (8, 2), (4, 3), (5, 3)]
    print(f"\n  {'blocks x dim':<14}{'total dim':>10}{'hierarchical':>16}{'flat':>14}{'win-rate':>10}")
    for n_blocks, bsize in configs:
        budget = _budget(n_blocks * bsize)
        hv, fv, wins = [], [], 0
        for s in range(seeds):
            L = hierarchy.block_landscape(landscapes.rosenbrock, n_blocks, bsize)
            h = hierarchy.HierarchicalOptimizer(L, n_blocks, seed=s).run(budget)["best_f"]
            f = hierarchy.flat_baseline(L, budget, seed=s)["best_f"]
            hv.append(h); fv.append(f); wins += int(h < f)
        print(f"  {f'{n_blocks} x {bsize}':<14}{n_blocks*bsize:>10}{np.mean(hv):>16.3f}"
              f"{np.mean(fv):>14.3f}{wins/seeds:>10.2f}")
    return {}


def scaling_benchmark() -> dict:
    _rule("SCALING BENCHMARK  (integrated MSCN wall-clock)")
    print(f"\n  {'agents':>8}{'rounds':>8}{'time (s)':>12}{'best coh':>12}{'consensus':>12}")
    for n in [9, 27, 64, 125]:
        L = landscapes.rastrigin(2)
        cfg = MSCNConfig(n_agents=n, rounds=60, perturb_every=30, seed=1)
        t = time.perf_counter()
        res = MSCN(L, cfg).run()
        dt = time.perf_counter() - t
        f = res["final"]
        print(f"  {n:>8}{cfg.rounds:>8}{dt:>12.2f}{f['best_coherence']:>12.3f}{f['consensus']:>12.3f}")
    return {}


# ===========================================================================
#  Main
# ===========================================================================

def run_all(seeds: int, figures: bool) -> None:
    optimiser_benchmark(seeds)
    strong_benchmark(seeds)
    convergence_benchmark(seeds, figpath="mscn_outputs/convergence.png" if figures else None)
    cooperation_benchmark(seeds)
    hierarchy_benchmark(seeds)
    scaling_benchmark()


SECTIONS = {
    "optimiser": lambda seeds, figures: optimiser_benchmark(seeds),
    "strong": lambda seeds, figures: strong_benchmark(seeds),
    "convergence": lambda seeds, figures: convergence_benchmark(
        seeds, figpath="mscn_outputs/convergence.png" if figures else None),
    "cooperation": lambda seeds, figures: cooperation_benchmark(seeds),
    "hierarchy": lambda seeds, figures: hierarchy_benchmark(seeds),
    "scaling": lambda seeds, figures: scaling_benchmark(),
    "all": run_all,
}


def main() -> None:
    p = argparse.ArgumentParser(description="MSCN benchmark harness")
    p.add_argument("section", nargs="?", default="all", choices=list(SECTIONS))
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="fewer seeds (8)")
    p.add_argument("--figures", action="store_true")
    args = p.parse_args()
    seeds = args.seeds if args.seeds is not None else (8 if args.quick else 20)

    print("\n" + "#" * 78)
    print(f"#  MSCN BENCHMARK   seeds={seeds}")
    print("#" * 78)
    t = time.perf_counter()
    SECTIONS[args.section](seeds, args.figures)
    print(f"\n(total benchmark time: {time.perf_counter() - t:.1f}s)")


if __name__ == "__main__":
    main()
