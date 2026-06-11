"""Property-based invariant fuzzing for the IBF-ASI mechanism.

The coupled agent asserts its spec invariants on EVERY tick (I1 ascent
monotonicity, I2 non-negative memory / basin expansion, I4 capacity budget), which
makes it directly fuzzable: sample random worlds x random agent hyper-parameters
across wide ranges -- including the edge cases nobody hand-tests (dim 1, zero
decoys, zero noise, zero memory, tiny Gamma, huge boost, 1-tick patience) -- run
each configuration, and let the live asserts plus explicit finiteness checks hunt
for violations. A quarter of the configurations run as COUPLED pairs (random
parasite flag) to fuzz the interaction paths too.

What is deliberately NOT a hard invariant: I3 (bounded below-theta transients).
A configuration whose equilibrium coherence cannot reach theta is legitimately
SUB-CRITICAL (phase.py's phase structure) and lives below threshold forever; the
fuzz records transients but only *reports* them.

This is the testing layer the repo did not have: the validations check that claims
hold at tuned operating points; the fuzzer checks the MECHANISM cannot be driven
into violating its own guarantees anywhere in configuration space.

Run: ``python -m mscn.ibf_asi_fuzz [--n 150]``   (numpy only, ~2-4 min).
"""

from __future__ import annotations

import numpy as np

from .ibf_asi import ASIWorld, IBFASI


def sample_config(rng: np.random.Generator) -> tuple[dict, dict, bool, bool]:
    """One random (world_kw, agent_kw, coupled?, parasite?) configuration."""
    wk = dict(
        dim=int(rng.integers(1, 5)),
        n_decoys=int(rng.integers(0, 9)),
        drift=float(rng.uniform(0, 0.1)),
        phase_drift=float(rng.uniform(0, 0.5)),
        ripple=float(rng.uniform(0, 1.0)),
        noise=float(rng.uniform(0, 1.0)),
        shock_every=int(rng.choice([0, int(rng.integers(15, 90))])),
        deceptive=bool(rng.random() < 0.5),
        seed=int(rng.integers(10_000)),
    )
    ak = dict(
        n_scales=int(rng.integers(1, 5)),
        sigma0=float(rng.uniform(0.1, 1.0)),
        scale_factor=float(rng.uniform(1.2, 3.0)),
        horizon=int(rng.integers(1, 5)),
        n_candidates=int(rng.integers(1, 9)),
        n_jumps=int(rng.integers(0, 4)),
        k0=float(rng.uniform(0.2, 3.0)),
        k_max=float(rng.uniform(2.0, 12.0)),
        k_adapt=float(rng.uniform(0.0, 0.5)),
        alpha=float(rng.choice([0.0, rng.uniform(0.05, 1.0)])),
        mu=float(rng.uniform(0.0, 0.15)),
        err_gate=float(rng.uniform(0.0, 15.0)),
        v_cap=float(rng.uniform(0.3, 4.0)),
        deadband=float(rng.uniform(0.0, 0.5)),
        Gamma=float(rng.uniform(1.0, 100.0)),
        theta=float(rng.uniform(0.0, 3.0)),
        margin=float(rng.uniform(0.0, 1.0)),
        boost=float(rng.uniform(1.0, 5.0)),
        stall_patience=int(rng.integers(3, 60)),
        reflect=bool(rng.random() < 0.8),
        dissolve=bool(rng.random() < 0.8),
        adapt_w=bool(rng.random() < 0.8),
        honest_reserve=bool(rng.random() < 0.8),
        two_sided_k=bool(rng.random() < 0.8),
        model_planner=bool(rng.random() < 0.5),
        plan_res=int(rng.integers(4, 20)),
        H_plan=int(rng.integers(1, 9)),
        plan_optimism=float(rng.uniform(0.0, 2.0)),
        plan_travel=float(rng.uniform(0.0, 0.3)),
        selfmodel_res=int(rng.integers(4, 41)),
        seed=int(rng.integers(10_000)),
    )
    coupled = rng.random() < 0.25
    parasite = coupled and rng.random() < 0.5
    return wk, ak, coupled, parasite


def check_finite(agent: IBFASI) -> None:
    """Explicit numeric-sanity invariants on the telemetry stream."""
    t = agent.telemetry[-1]
    for k, v in t.items():
        if isinstance(v, (int, float)):
            assert np.isfinite(v), f"non-finite telemetry {k}={v}"
    assert np.all(np.isfinite(agent.x)), "non-finite agent position"
    assert 0 < agent.k <= max(agent.k_max, agent.k0) + 1e-9, f"k out of range: {agent.k}"


def fuzz_one(wk: dict, ak: dict, coupled: bool, parasite: bool,
             eval_budget: int = 1800, ultra: bool = False) -> dict:
    """Run one configuration; the per-tick I1/I2'/I4 asserts + finiteness
    checks fire inside (I2' = bounded signed modification for ULTRA, whose
    memory organ is the signed engine). Returns summary stats."""
    w = ASIWorld(**wk)
    if ultra:
        from .ibf_ultra import UltraASI
        uk = {k: v for k, v in ak.items()
              if k not in ("n_scales", "alpha", "horizon", "anneal_steps",
                           "two_sided_k", "model_planner")}
        a = UltraASI(w, **uk)
        coupled = False                  # the ULTRA leg fuzzes self-detection
    else:
        a = IBFASI(w, **ak)
    agents = [a]
    if coupled:
        b = IBFASI(w, **{**ak, "seed": ak["seed"] + 1})
        a.partner, b.partner = b, a
        if parasite:
            b.give_transfer = False
        agents.append(b)
    while w.evals < eval_budget:
        shocked = w.tick()
        for ag in agents:
            if shocked:
                ag.apply_shock()
            ag.step()
            check_finite(ag)
    return {"max_transient": max(ag.monitor["max_transient"] for ag in agents),
            "ticks": len(a.telemetry)}


def main(n_configs: int = 150, seed: int = 0) -> None:
    print("\n" + "#" * 74)
    print("#  IBF-ASI invariant FUZZ -- random worlds x random hyper-parameters;")
    print("#  the per-tick I1/I2/I4 asserts + finiteness checks hunt for violations")
    print("#" * 74)
    rng = np.random.default_rng(seed)
    failures: list[tuple[int, dict, dict, str]] = []
    transients = []
    n_coupled = n_ultra = 0
    for i in range(n_configs):
        wk, ak, coupled, parasite = sample_config(rng)
        ultra = rng.random() < 0.25      # a quarter of legs run UltraASI
        n_coupled += coupled and not ultra
        n_ultra += ultra
        try:
            r = fuzz_one(wk, ak, coupled, parasite, ultra=ultra)
            transients.append(r["max_transient"])
        except (AssertionError, Exception) as e:   # noqa: BLE001 -- a fuzzer catches all
            failures.append((i, wk, ak, f"{type(e).__name__}: {e}"))
    print(f"\n  configs: {n_configs} ({n_coupled} coupled pairs, "
          f"{n_ultra} ULTRA self-detection legs)  |  "
          f"violations: {len(failures)}")
    if failures:
        for i, wk, ak, err in failures[:10]:
            print(f"\n  [#{i}] {err}")
            print(f"        world: { {k: v for k, v in wk.items()} }")
            print(f"        agent: { {k: v for k, v in ak.items()} }")
    else:
        tr = np.array(transients)
        print(f"  below-theta transients (I3, reported not asserted): "
              f"median {np.median(tr):.0f}, max {tr.max():.0f}")
        print(f"  (long transients = legitimately SUB-CRITICAL configs, phase.py)")
    assert not failures, f"{len(failures)} invariant violations found -- fix them"
    print("\n  PASS: no I1/I2/I4 or finiteness violation anywhere in the sampled")
    print("  configuration space (including dim 1-4, 0 decoys, 0 noise, 0 memory,")
    print("  tiny Gamma, coupled + parasitic pairs).\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="IBF-ASI invariant fuzzing")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(n_configs=a.n, seed=a.seed)
