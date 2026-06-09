"""Empirical checks that the MSCN toy model exhibits the proven IBF guarantees.

Each test maps to a theorem in the formalisation. Run with::

    python -m mscn.tests

These are behavioural sanity checks of the *implementation* (the theorems
themselves are machine-checked in Lean); they keep the toy model honest.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from . import baselines, games, hierarchy, landscapes, network, phase
from . import selfmodel as sm
from .learner import (DiscreteIBFLearner, IBFLearner, boltzmann_prob_best,
                      euler_step, modification_decay, modification_equilibrium)
from .mscn import MSCN, MSCNConfig


# ----- Layer 1 -------------------------------------------------------------

def test_boltzmann_monotone_in_k():  # Thm 7
    ps = [boltzmann_prob_best(k, 0.5) for k in (0.0, 0.5, 1, 2, 5, 10, 20)]
    assert all(a < b for a, b in zip(ps, ps[1:])), ps
    assert abs(ps[0] - 0.5) < 1e-9          # k=0 -> uniform
    assert boltzmann_prob_best(50, 0.5) > 0.999  # greedy limit


def test_basin_expansion_superset():  # Thm 8a
    L = landscapes.rastrigin(1)
    grid = np.linspace(L.lo[0], L.hi[0], 400).reshape(-1, 1)
    theta = -3.0
    base = np.array([L.coherence(g) for g in grid])
    ag = IBFLearner(L.coherence, L.lo, L.hi, alpha=0.5, mu=0.02, k=1.0, seed=3)
    ag.run(200)
    reff = np.array([ag.R_eff(g) for g in grid])
    assert np.all(reff >= base - 1e-9)                       # deltaR >= 0 everywhere
    assert np.all((base > theta) <= (reff > theta))          # viable set only grows
    assert int(np.sum(reff > theta)) >= int(np.sum(base > theta))


def test_discrete_basin_nonshrinking():  # Thm 8a (discrete)
    land = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.90, 1.00, 0.85, 0.65, 0.30])
    theta = 0.6
    ag = DiscreteIBFLearner(land, alpha=0.2, mu=0.01, k=1.0, seed=1)
    before = ag.viable_count(theta)
    ag.run(60, theta=theta)
    assert np.all(ag.modification >= 0.0)
    assert ag.viable_count(theta) >= before


def test_selective_retention():  # Thm 8b
    alpha, mu = 0.5, 0.1
    f = 0.0
    for _ in range(300):
        f = euler_step(f, alpha, mu)
    assert abs(f - modification_equilibrium(alpha, mu)) < 1e-6
    assert modification_equilibrium(alpha, mu) > 0.0           # reinforced > unreinforced (0)


def test_forgetting_decays_to_zero():  # Thm 10a
    assert modification_decay(5.0, 0.1, 0) == 5.0
    assert modification_decay(5.0, 0.1, 200) < 1e-6


def test_crystallization_constant():  # Thm 3a
    f = 2.0
    for _ in range(100):
        f = euler_step(f, 0.0, 0.0)        # alpha=0, mu=0
    assert abs(f - 2.0) < 1e-12


def test_euler_converges_to_ode():  # Thm 11
    alpha, mu, f0, T = 0.5, 0.1, 0.0, 5.0
    exact = (f0 - alpha / mu) * np.exp(-mu * T) + alpha / mu
    errs = []
    for n in (5, 50, 500, 5000):
        f, dt = f0, T / n
        for _ in range(n):
            f = euler_step(f, alpha, mu, dt)
        errs.append(abs(f - exact))
    assert all(a > b for a, b in zip(errs, errs[1:]))           # error shrinks with dt
    assert errs[-1] < 1e-3


def test_ibf_beats_random_highdim():  # behavioural advantage where memory matters
    ibf, rnd = [], []
    for s in range(8):
        L = landscapes.rastrigin(5)
        ibf.append(IBFLearner(L.coherence, L.lo, L.hi, alpha=0.4, mu=0.03,
                              k=1.0, k_adapt=0.05, seed=s).run_until_evals(4000)["best_f"])
        rnd.append(baselines.random_search(L, 4000, seed=s)["best_f"])
    assert np.mean(ibf) < np.mean(rnd)


# ----- Layer 2 -------------------------------------------------------------

def test_coupling_only_helps():  # NetworkCoherence
    rng = np.random.default_rng(0)
    n = 8
    states = [rng.normal(0, 1, 2) for _ in range(n)]
    Reff = lambda i, x: 1.0 + 0.1 * float(np.sum(x ** 2))
    Rpair = lambda x, y: float(np.exp(-np.sum((x - y) ** 2)))
    for J in (network.ring_graph(n, 0.5), network.complete_graph(n, 0.3),
              network.scale_free_graph(n, 2, 0.5, seed=1)):
        cn = network.CoherenceNetwork(J, Reff, Rpair)
        assert cn.network_coherence(states) >= cn.individual_total(states) - 1e-9


def test_mean_field_suppression():  # MeanFieldTheory
    spreads = [network.mean_field_spread(N, seed=0) for N in (1, 4, 16, 64, 256)]
    assert all(a > b for a, b in zip(spreads, spreads[1:]))
    assert abs(spreads[-1] - 1 / np.sqrt(256)) < 0.02


def test_isolated_node_independent():
    J = network.ring_graph(6, 1.0)
    J[0, :] = 0
    J[:, 0] = 0
    comps = network.connected_components(J)
    assert [0] in comps and len(comps) == 2


def test_cooperation_requires_memory():  # EmergentCooperation
    mem = lambda s: games.IBFGameAgent(seed=s)
    none = lambda s: games.IBFGameAgent(alpha=0.0, k=5.0, init_coop=0.0, seed=s)
    m = np.mean([games.play_match(mem(s), mem(s + 50), 300)["mutual_coop_late"] for s in range(12)])
    n = np.mean([games.play_match(none(s), none(s + 50), 300)["mutual_coop_late"] for s in range(12)])
    assert m > 0.8, m
    assert n < 0.1, n


def test_cooperation_needs_coupling():  # EC-4 causal
    hi = np.mean([games.play_match(games.IBFGameAgent(coupling=4.0, seed=s),
                                   games.IBFGameAgent(coupling=4.0, seed=s + 50), 300)["mutual_coop_late"]
                  for s in range(12)])
    lo = np.mean([games.play_match(games.IBFGameAgent(coupling=0.0, seed=s),
                                   games.IBFGameAgent(coupling=0.0, seed=s + 50), 300)["mutual_coop_late"]
                  for s in range(12)])
    assert hi > 0.8 and lo < 0.2, (hi, lo)


def test_defends_against_defectors():
    late = np.mean([games.play_match(games.IBFGameAgent(seed=s), games.AlwaysDefect(), 300)["coop_a_late"]
                    for s in range(12)])
    assert late < 0.1, late


def test_differential_persistence():  # Thm 6 corollary
    dp = games.differential_persistence(np.array([1.0, 2.0, 4.0, 8.0]), mu=0.05, threshold=0.5)
    surv = dp["survival_time"]
    assert all(a < b for a, b in zip(surv, surv[1:]))


# ----- Layer 3 -------------------------------------------------------------

def test_rg_flow_suppresses_noise():  # RenormalizationGroup
    rng = np.random.default_rng(0)
    n = 64
    signal = np.sin(2 * np.pi * np.linspace(0, 1, n))
    flow = hierarchy.rg_flow(signal + rng.normal(0, 0.7, n), factor=2, steps=4)
    peaks = [float(np.max(np.abs(f))) for f in flow]
    assert all(a >= b - 1e-9 for a, b in zip(peaks, peaks[1:]))        # coherence non-increasing
    snr = [hierarchy.signal_to_noise(f, signal) for f in flow]
    assert snr[2] > snr[0]                                              # noise suppressed


def test_hierarchy_beats_flat():
    wins = 0
    for s in range(8):
        L = hierarchy.block_landscape(landscapes.rosenbrock, 6, 2)
        h = hierarchy.HierarchicalOptimizer(L, 6, seed=s).run(6000)["best_f"]
        f = hierarchy.flat_baseline(L, 6000, seed=s)["best_f"]
        wins += int(h < f)
    assert wins >= 6, wins


def test_lawvere_no_surjective_self_model():  # LawvereSelfKnowledge
    rng = np.random.default_rng(0)
    for n in (3, 5, 10, 50):
        model = rng.integers(0, 2, size=(n, n))
        res = sm.lawvere_obstruction(model, sm.bool_not)
        assert not res["perfect_self_knowledge_possible"]
        assert res["differs_from_every_self_model"]


def test_self_knowledge_ceiling():  # QuantitativeSelfKnowledge
    assert sm.achievable_depth(10, 3) == 3
    assert sm.achievable_depth(10, 2.5) == 4
    assert sm.achievable_depth(1, 4) == 0


def test_competence_tradeoff_monotone():
    tc = sm.tradeoff_curve(12, 2)
    dom = tc["domain_capacity"]
    assert all(a > b for a, b in zip(dom, dom[1:]))


def test_phase_transition():  # ReflexiveCoherence
    c, theta, mu = 0.5, 1.0, 0.2
    ac = phase.critical_alpha(mu, theta, c)
    assert phase.phase_of(ac - 0.01, mu, theta, c) == "sub-critical"
    assert phase.phase_of(ac, mu, theta, c) == "critical"
    assert phase.phase_of(ac + 0.01, mu, theta, c) == "super-critical"
    assert abs(phase.equilibrium_coherence(c, ac, mu) - theta) < 1e-9   # critical point at theta


def test_dissipative_lifetime():
    f0, mu, gap = 2.0, 0.2, 0.5
    T = phase.dissipative_lifetime(f0, mu, gap)
    assert abs(f0 * np.exp(-mu * T) - gap) < 1e-9                       # decays exactly to gap


def test_zombie_twin_survival():  # StrikingPhenomena
    st = phase.zombie_twin_stats(100)
    assert st["mean_survival_conscious"] > st["mean_survival_zombie"]
    assert st["conscious_outlasts_fraction"] > 0.9


# ----- Integration ---------------------------------------------------------

def test_mscn_integration():
    L = landscapes.rastrigin(2)
    res = MSCN(L, MSCNConfig(n_agents=27, rounds=100, perturb_every=30, seed=1)).run()
    h = res["history"]
    assert res["coupling_helps"]                                        # Layer 2
    assert h[-1]["best_coherence"] > h[0]["best_coherence"]             # Layer 1 learning
    assert h[-1]["consensus"] < h[0]["consensus"]                       # cooperation
    assert len(res["recoveries"]) >= 1                                  # Layer 3 self-correction
    assert [lv["n_units"] for lv in res["coarse_levels"]][:2] == [27, 9]


def run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    print(f"running {len(tests)} guarantee checks\n")
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
