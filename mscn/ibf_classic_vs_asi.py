"""IBF CLASSIC vs IBF-ASI -- the two architectures, head to head, each in the
other's arena.

IBF CLASSIC = the preprint's engine (`ibf_engine.PaperEngine`): an
evaluator-corrector over a frozen base -- signed kernel particles, context-gated
reads, crystallization/crucible lifecycle. Its home arena: contextual evaluation
under contradiction (mini-RRW).

IBF-ASI = the nine-stage embodied agent (`ibf_asi.IBFASI`): multiscale
non-negative memory, warm jumps, planner/options, reflect, capacity budget. Its
home arena: continual navigation (the gauntlet's switching world).

ARENA 1 (classic's home, mini-RRW A -> B=reversal -> C): the ASI's MEMORY
SUBSTRATE is ported as an evaluator-corrector (multiscale Scales, raw-positive
reinforcement, error-gated dissolution, capacity projection, NO contexts -- its
actual laws) and faces the classic engine. Pre-registered: the classic's context
gating wins retention; the ASI substrate, being ungated, forgets under B's
contradiction -- measuring how much its OTHER machinery (multiscale, dissolve)
recovers.

ARENA 2 (ASI's home, the G2 switching world A -> B -> A): the classic's proven
mechanism (context gating) is TRANSPLANTED into the ASI
(`IBFASI.gate_contexts`, phases signalled at boundaries, task-incremental as in
the paper) and faces the canonical ASI. Pre-registered: gating repairs the
measured G2 pathologies -- the return-phase asymptote at least holds and the
negative relearning savings improve -- without costing phase-B adaptation.

Run: ``python -m mscn.ibf_classic_vs_asi``   (numpy+scipy; ~8 min).
"""

from __future__ import annotations

import numpy as np

from .ibf_asi import IBFASI, Scale
from .ibf_asi_gauntlet import SwitchingWorld
from .ibf_engine import MiniRRW, PaperEngine
from .stats import fmt_ci, paired_ci, verdict


# ---------------------------------------------------------------------------
#  Arena 1: mini-RRW (classic's home). Memory substrates behind one interface.
# ---------------------------------------------------------------------------

class ASIMemoryAdapter:
    """The IBF-ASI memory substrate as an evaluator-corrector: multiscale
    non-negative kernel memory with merge/v_cap, error-gated dissolution,
    capacity projection -- and NO context concept (its actual law set)."""

    def __init__(self, seed: int = 0, sigma0: float = 0.89, n_scales: int = 3,
                 alpha: float = 0.45, mu: float = 0.02, err_gate: float = 8.0,
                 v_cap: float = 1.5, Gamma: float = 200.0) -> None:
        self.scales = [Scale(sigma=sigma0 * 2.0 ** s, alpha=alpha)
                       for s in range(n_scales)]
        self.mu, self.err_gate, self.v_cap, self.Gamma = mu, err_gate, v_cap, Gamma
        self.rng = np.random.default_rng(seed)

    def switch(self, ctx: int) -> None:               # no context concept
        pass

    def delta(self, Zq: np.ndarray) -> np.ndarray:
        return np.array([sum(s.w * s.delta_R(z) for s in self.scales)
                         for z in Zq])

    def write(self, z: np.ndarray, D: float) -> None:
        posD, negD = max(D, 0.0), max(-D, 0.0)
        s0 = self.scales[0]
        if posD > 0:
            for c in s0.centers:
                if np.sum((c.z - z) ** 2) < (0.5 * s0.sigma) ** 2:
                    c.v = min(c.v + s0.alpha * posD, self.v_cap)
                    break
            else:
                s0.centers.append(type("C", (), {})())
                c = s0.centers[-1]
                c.z, c.v, c.q, c.err, c.age, c.transferred, c.ctx = \
                    z.copy(), min(s0.alpha * posD, self.v_cap), posD, 0.0, 0, False, 0
        if negD > 0:                                  # error-gating evidence
            for s in self.scales:
                for c in s.centers:
                    if np.sum((c.z - z) ** 2) < s.sigma ** 2:
                        c.err = 0.85 * c.err + 0.15 * negD

    def epoch(self) -> None:
        for si, s in enumerate(self.scales):
            survivors = []
            for c in s.centers:
                c.age += 1
                c.v *= max(1.0 - self.mu * (1.0 + self.err_gate * c.err), 0.0)
                if c.v > 1e-4:
                    survivors.append(c)
            s.centers = survivors
            if si + 1 < len(self.scales) and s.centers:    # consolidation
                vmax = max(c.v for c in s.centers)
                for c in s.centers:
                    if not c.transferred and c.age > 3 and c.v > 0.6 * vmax \
                            and c.err < 0.05:
                        coarse = self.scales[si + 1]
                        coarse.centers.append(type("C", (), {})())
                        cc = coarse.centers[-1]
                        cc.z, cc.v, cc.q, cc.err, cc.age = \
                            c.z.copy(), 0.7 * c.v, c.q, 0.0, 0
                        cc.transferred, cc.ctx = True, 0
                        c.transferred = True
        total = sum(c.v for s in self.scales for c in s.centers)
        if total > self.Gamma:                        # capacity projection
            f = self.Gamma / total
            for s in self.scales:
                for c in s.centers:
                    c.v *= f


class ClassicAdapter:
    def __init__(self, seed: int = 0, **kw) -> None:
        self.eng = PaperEngine(seed=seed, **kw)

    def switch(self, ctx: int) -> None:
        self.eng.switch_context(ctx)

    def delta(self, Zq: np.ndarray) -> np.ndarray:
        return self.eng.delta_R_batch(Zq)

    def write(self, z: np.ndarray, D: float) -> None:
        self.eng.write(z, D)

    def epoch(self) -> None:
        self.eng.epoch()


def arena1_run(memory, seed: int, n_epochs: int = 20, n_points: int = 250,
               k0: float = 2.0) -> dict:
    world = MiniRRW(seed=seed)
    rngb = np.random.default_rng(seed + 100)
    A = rngb.normal(0, 0.6, 8)
    base = lambda z: float(1 / (1 + np.exp(-(A @ z))))
    rng = np.random.default_rng(seed + 300)
    test = {ph: [rng.normal(0, 1.0, 4) for _ in range(300)] for ph in range(3)}

    def accuracy(phase: int) -> float:
        memory.switch(phase)
        ok = 0
        for x in test[phase]:
            Zq = np.stack([world.embed(x, a) for a in range(world.nA)])
            scores = np.array([base(z) for z in Zq]) + memory.delta(Zq)
            ok += int(np.argmax(scores) == world.best(x, phase))
        return ok / len(test[phase])

    acc = {}
    for phase in range(3):
        memory.switch(phase)
        for _ in range(n_epochs):
            for _ in range(n_points):
                x = rng.normal(0, 1.0, 4)
                Zq = np.stack([world.embed(x, a) for a in range(world.nA)])
                scores = np.array([base(z) for z in Zq]) + memory.delta(Zq)
                zs = k0 * (scores - scores.max())
                p = np.exp(zs)
                p /= p.sum()
                a = int(rng.choice(world.nA, p=p))
                imposed = 1.0 if a == world.best(x, phase) else 0.0
                z = world.embed(x, a)
                D = imposed - (base(z) + float(memory.delta(z[None])[0]))
                memory.write(z, D)
            memory.epoch()
        acc[phase] = {ph: accuracy(ph) for ph in range(phase + 1)}
        memory.switch(phase)
    return acc


# ---------------------------------------------------------------------------
#  Arena 2: the G2 switching world (ASI's home), gating transplanted
# ---------------------------------------------------------------------------

def arena2_run(agent_kw: dict, gated: bool, seed: int,
               budget: int = 12000) -> dict:
    w = SwitchingWorld(seed=seed, budget=budget)
    a = IBFASI(w, seed=100 + seed, model_planner=True, **agent_kw)
    a.gate_contexts = gated
    last_stage = 1
    while w.evals < budget:
        if w.stage != last_stage:
            last_stage = w.stage
            a.switch_context(0 if w.stage == 3 else w.stage - 1)
        shocked = w.tick()
        if shocked:
            a.apply_shock()
        a.step()
    T = a.telemetry
    n = len(T)
    ph = [t["true"] for t in T]
    p1, p2, p3 = ph[: n // 3], ph[n // 3: 2 * n // 3], ph[2 * n // 3:]
    q = max(len(p1) // 4, 1)
    a1_tail = float(np.mean(p1[-q:]))
    thr = 0.8 * a1_tail

    def t_cross(p):
        sm = np.convolve(np.asarray(p, float), np.ones(15) / 15, mode="valid")
        hit = np.flatnonzero(sm >= thr)
        return int(hit[0]) if hit.size else len(p)

    return {"A1_tail": a1_tail, "B_tail": float(np.mean(p2[-q:])),
            "A2_tail": float(np.mean(p3[-q:])),
            "savings": t_cross(p1) - t_cross(p3)}


def main(n_seeds: int = 6) -> None:
    print("\n" + "#" * 74)
    print("#  IBF CLASSIC vs IBF-ASI -- each architecture in the other's arena")
    print("#" * 74)

    print("\n  [ARENA 1] mini-RRW, classic's home (A -> B=exact reversal -> C):")
    arms1 = {
        "classic (full lifecycle)": lambda s: ClassicAdapter(seed=s),
        "classic (gating only)": lambda s: ClassicAdapter(seed=s, crucible=False),
        "ASI memory substrate": lambda s: ASIMemoryAdapter(seed=s),
    }
    r1 = {}
    print(f"  {'arm':<26}{'acc(A|A)':>9}{'acc(A|B)':>9}{'acc(B|B)':>9}"
          f"{'forget(A)':>10}")
    for name, mk in arms1.items():
        runs = [arena1_run(mk(s), s) for s in range(n_seeds)]
        r1[name] = runs
        aa = float(np.mean([r[0][0] for r in runs]))
        ab = float(np.mean([r[1][0] for r in runs]))
        bb = float(np.mean([r[1][1] for r in runs]))
        f = float(np.mean([r[0][0] - r[2][0] for r in runs]))
        print(f"  {name:<26}{aa:>9.3f}{ab:>9.3f}{bb:>9.3f}{f:>10.3f}")
    f_asi = [r[0][0] - r[2][0] for r in r1["ASI memory substrate"]]
    f_cls = [r[0][0] - r[2][0] for r in r1["classic (gating only)"]]
    ci1 = paired_ci(f_asi, f_cls)
    print(f"  forgetting: ASI-substrate - classic(gating): {fmt_ci(ci1)} {verdict(ci1)}")

    print("\n  [ARENA 2] G2 switching world, ASI's home (A -> B -> A, 8 seeds):")
    arms2 = {
        "ASI canonical": (dict(), False),
        "ASI + classic gating": (dict(), True),
        "ASI no-memory": (dict(alpha=0.0), False),
    }
    r2 = {}
    print(f"  {'arm':<26}{'A1 tail':>9}{'B tail':>8}{'A2 tail':>9}{'savings':>9}")
    for name, (kw, gated) in arms2.items():
        runs = [arena2_run(kw, gated, s) for s in range(8)]
        r2[name] = runs
        m = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        print(f"  {name:<26}{m['A1_tail']:>9.2f}{m['B_tail']:>8.2f}"
              f"{m['A2_tail']:>9.2f}{m['savings']:>9.0f}")
    ci_a2 = paired_ci([r["A2_tail"] for r in r2["ASI + classic gating"]],
                      [r["A2_tail"] for r in r2["ASI canonical"]])
    ci_sav = paired_ci([r["savings"] for r in r2["ASI + classic gating"]],
                       [r["savings"] for r in r2["ASI canonical"]])
    ci_b = paired_ci([r["B_tail"] for r in r2["ASI + classic gating"]],
                     [r["B_tail"] for r in r2["ASI canonical"]])
    print(f"  gating - canonical: A2 {fmt_ci(ci_a2)} {verdict(ci_a2)}")
    print(f"                      savings {fmt_ci(ci_sav)} {verdict(ci_sav)}")
    print(f"                      B-tail {fmt_ci(ci_b)} {verdict(ci_b)}")

    print("\n  verdicts:")
    print("   * Arena 1 (classic's home): the classic's context gating is the")
    print("     retention mechanism; the ASI substrate, ungated by design, pays")
    print(f"     the measured forgetting difference above.")
    print("   * Arena 2 (ASI's home): the transplant verdict is printed above --")
    print("     what the classic mechanism buys the ASI agent, with CIs.")
    assert ci1["mean"] > 0, \
        "the ungated ASI substrate must forget more than the gated classic"
    assert ci_b["hi"] > -0.15, \
        "the gating transplant must not wreck phase-B adaptation"
    print()


if __name__ == "__main__":
    main()
