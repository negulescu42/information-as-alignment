"""Adaptive decay (roadmap 1.3) on non-stationary streams -- an honest evaluation.

Roadmap 1.3 proposes per-center adaptive decay `μ_i` for non-stationary
environments: "centers repeatedly reinforced get low μ (long-term memory); centers
deposited once get high μ (working memory)". This module tests that claim on a
controlled associative-memory stream with mixed timescales (stable / slow-drift /
fast-drift contexts), comparing four decay policies:

* ``cryst``  -- μ = 0 (crystallise everything)
* ``fixed``  -- a single tuned μ
* ``count``  -- μ_i = μ/(1 + λ·count_i)  (the roadmap's reinforcement-count scheme)
* ``error``  -- μ_i = μ·err_i, gated by each context's recent prediction-error EWMA

Findings (see `ARCHITECTURE.md` §3.14):
- On **stationary** data, count-based adaptive μ helps (crystallises stable rules) —
  confirmed separately on chess (`IBFChessAgent(adaptive_mu=True)`: legal@1 0.10→0.15).
- On **drift**, the count scheme is *counterproductive*: it crystallises now-stale
  patterns (low μ) and cannot unlearn them, performing like μ=0.
- **error-gating** (forget centers that became wrong) is the correct signal, but a
  *well-tuned fixed μ* matched to the change timescale matches or edges it on these
  tasks. So μ>0 is required for non-stationarity, but the count-based μ_i is the wrong
  choice and adaptive μ is not a clear win over a tuned fixed μ.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def run_stream(mode: str, mu: float = 0.3, lam: float = 1.0, beta: float = 0.9,
               n_contexts: int = 60, steps: int = 80000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    grp = {c: ("stable" if c < n_contexts // 3 else
               "slow" if c < 2 * n_contexts // 3 else "fast") for c in range(n_contexts)}
    period = {"stable": 10 ** 9, "slow": 3000, "fast": 150}

    def target(c, t):
        return c if grp[c] == "stable" else (t // period[grp[c]]) % 4

    dR = defaultdict(lambda: defaultdict(float))
    err = defaultdict(float)
    cnt = defaultdict(lambda: defaultdict(int))
    corr = {g: [0, 0] for g in ("stable", "slow", "fast")}
    for t in range(steps):
        c = int(rng.integers(n_contexts))
        tgt = target(c, t)
        d = dR[c]
        wrong = 1.0
        if d:
            pred = max(d, key=d.get)
            wrong = float(pred != tgt)
            if t > steps // 2:
                corr[grp[c]][0] += (pred == tgt)
                corr[grp[c]][1] += 1
        err[c] = beta * err[c] + (1 - beta) * wrong
        cnt[c][tgt] += 1
        d[tgt] = d.get(tgt, 0.0) + 0.5
        for m in list(d):
            if m == tgt:
                continue
            if mode == "cryst":
                mm = 0.0
            elif mode == "fixed":
                mm = mu
            elif mode == "count":
                mm = mu / (1 + lam * cnt[c][m])
            else:  # error-gated
                mm = mu * err[c]
            d[m] *= (1 - mm)
    return {g: corr[g][0] / max(corr[g][1], 1) for g in corr}


def demo() -> None:
    print("Adaptive decay on a mixed-timescale stream (stable / slow-drift / fast-drift):")
    print(f"  {'policy':<24}{'stable':>8}{'slow':>8}{'fast':>8}{'mean':>9}")
    for mode, lbl, mu in [("cryst", "fixed mu=0", 0.0), ("fixed", "fixed mu=0.4", 0.4),
                          ("count", "adaptive (count)", 0.4), ("error", "adaptive (error-gated)", 0.6)]:
        r = run_stream(mode, mu=mu)
        print(f"  {lbl:<24}{r['stable']:>8.3f}{r['slow']:>8.3f}{r['fast']:>8.3f}"
              f"{np.mean(list(r.values())):>9.3f}")
    print("  -> count-based adaptive mu crystallises stale patterns (fails drift, ~ mu=0);")
    print("     a tuned fixed mu and error-gating handle drift; adaptive is not a clear win.")


if __name__ == "__main__":
    demo()
