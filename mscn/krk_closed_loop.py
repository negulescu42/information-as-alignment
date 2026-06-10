"""D1+D4 -- the CLOSED ONLINE LOOP on K+R-vs-K: one agent, one stream, everything
learned while acting, with an EXACT oracle watching.

The original handover names this the unifying move: a single IBF agent that, while
acting, runs sense -> select -> act -> MODIFY(discrepancy) -> adapt(k), learning its
representation of the rules AND its value landscape AND its policy in one
discrepancy-driven loop -- with strength rising and gap-to-oracle falling measured
ONLINE, not in separate batch phases. KRK is the tractable world where this is
fully measurable: `krk_world.py` provides an exact referee (cross-validated
move-for-move against python-chess) and an exact retrograde DTM tablebase (max DTM
reproduces the known 16-move bound) as the oracle.

STRICT NO-PRIORS for the agent. It observes the three piece locations (its two
pieces carry opaque persistent tags, it knows which side is its own) and the
terminal signals (win=1, else 0). It is given NO movement rules, no notion of
check/mate/stalemate, no geometry. What it learns, all online, all from one stream:

  * MOVEMENT LEGALITY -- the referee protocol of the chess arc (3.9): the agent
    submits its full ranked proposal list (tag, destination); the referee plays the
    highest-ranked legal one; every proposal ranked above it is thereby revealed
    illegal. Acceptance coherence acc[tag, from, to] accumulates -- the learned
    movement model (rook lines and king steps must EMERGE in these tables).
  * VALUE -- coherence over afterstates, learned from terminal outcomes ONLY by
    the TD/Monte-Carlo update  V(s) += alpha * (G - V(s)), which is literally the
    IBF modification step (formal seed: `tdError'`, `td_is_modification_step` in
    AGIFoundations 11): discrepancy = (return - current coherence), MODIFY.
  * POLICY -- Boltzmann-k selection over (legality coherence + k * value), realised
    as Plackett-Luce ranking (Gumbel perturbation at temperature 1/k), with k
    adapted on improvement (Thm 8c agency at the episode level).

Measured ONLINE per window: mate rate, plies-to-mate, first-proposal legality, TD
discrepancy, and -- against the exact tablebase, evaluation-only -- the fraction of
moves that preserve the win and the fraction that are DTM-optimal. Baselines:
random ranking, and legality-only (learned movement, zero value) to isolate what
the VALUE loop adds.

Run: ``python -m mscn.krk_closed_loop [--episodes 30000] [--seeds 3]``
(numpy only; ~10-20 min at defaults).
"""

from __future__ import annotations

import numpy as np

from .krk_world import INF, tables
from .stats import fmt_ci, paired_ci, verdict


class KRKClosedLoopAgent:
    """The Layer-1 IBF unit instantiated on the KRK stream (tabular-vectorised)."""

    def __init__(self, *, alpha: float = 0.25, gamma: float = 0.97,
                 k0: float = 2.0, k_adapt: float = 0.5, k_max: float = 12.0,
                 value_on: bool = True, seed: int = 0) -> None:
        self.T = tables()
        self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma = alpha, gamma
        self.k, self.k_adapt, self.k_max = k0, k_adapt, k_max
        self.value_on = value_on
        # legality coherence: acceptance/rejection counts per (tag, from, to)
        self.A = np.zeros((2, 64, 64), dtype=np.float32)   # tag 0 = piece "K-like"
        self.R = np.zeros((2, 64, 64), dtype=np.float32)
        # value coherence over afterstate placements (black to move): delta-R
        self.V = np.zeros(64 * 64 * 64, dtype=np.float32)
        # referee ground truth: legal (tag, dest) per placement, built lazily rows
        self._td_abs: list[float] = []

    def _value(self, pids: np.ndarray) -> np.ndarray:
        """Value of afterstates -- overridable by generalising substrates."""
        return self.V[pids]

    # ----- proposal ranking (Plackett-Luce at temperature 1/k) -----
    def rank_moves(self, wk: int, bk: int, wr: int) -> tuple[np.ndarray, ...]:
        dests = np.arange(64)
        # tag 0 moves the piece that started as the king-like one (opaque to agent)
        after_k = (dests * 64 + bk) * 64 + wr          # tag 0 (wk piece) -> dest
        after_r = (wk * 64 + bk) * 64 + dests          # tag 1 (wr piece) -> dest
        logp_k = np.log((self.A[0, wk] + 1.0) / (self.A[0, wk] + self.R[0, wk] + 2.0))
        logp_r = np.log((self.A[1, wr] + 1.0) / (self.A[1, wr] + self.R[1, wr] + 2.0))
        vals = self._value(np.concatenate([after_k, after_r])) * self.value_on
        score = np.concatenate([logp_k, logp_r]) + self.k * vals
        gumbel = -np.log(-np.log(self.rng.uniform(1e-12, 1.0, 128)))
        order = np.argsort(-(score + gumbel / max(self.k, 1e-6)))
        tags = (order >= 64).astype(np.int32)
        tos = order % 64
        froms = np.where(tags == 0, wk, wr)
        afters = np.where(tags == 0, after_k[tos], after_r[tos])
        return tags, froms, tos, afters

    def learn_legality(self, tags, froms, tos, accepted_i: int) -> None:
        for i in range(accepted_i):
            self.R[tags[i], froms[i], tos[i]] += 1.0
        self.A[tags[accepted_i], froms[accepted_i], tos[accepted_i]] += 1.0

    def learn_value(self, afterstates: list[int], reward: float,
                    truncated: bool = False) -> None:
        """Backward sweep: V += alpha*(G - V) -- TD as the IBF modification step.
        The TERMINAL afterstate's return is known exactly (zero variance), so its
        step size is 1; a truncated episode bootstraps from V instead of
        pretending the return was 0."""
        if not afterstates:
            return
        if truncated:
            G = float(self.V[afterstates[-1]])
            rest = afterstates[:-1]
        else:
            term = afterstates[-1]
            self._td_abs.append(abs(reward - self.V[term]))
            self.V[term] = reward                  # exact terminal value
            G = self.gamma * reward
            rest = afterstates[:-1]
        for s in reversed(rest):
            d = G - self.V[s]
            self.V[s] += self.alpha * d
            self._td_abs.append(abs(d))
            G = self.gamma * G


def run_agent(n_episodes: int, *, seed: int = 0, value_on: bool = True,
              random_rank: bool = False, max_plies: int = 80,
              window: int = 2000, agent: "KRKClosedLoopAgent | None" = None) -> dict:
    T = tables()
    ag = agent if agent is not None else KRKClosedLoopAgent(seed=seed, value_on=value_on)
    rng = np.random.default_rng(1000 + seed)
    starts = np.flatnonzero(T.legal_w)
    # referee legality lookup: (tag, dest) legal per placement, derived from w_succ
    stats_keys = ("mate", "plies", "legal1", "td", "preserve", "optimal", "k")
    windows: list[dict] = []
    cur = {k: [] for k in stats_keys}

    for ep in range(n_episodes):
        p = int(starts[rng.integers(starts.size)])
        wk, bk, wr = p // 4096, (p // 64) % 64, p % 64
        afterstates: list[int] = []
        reward, plies = 0.0, 0
        terminal = False
        first_legal_flags = []
        pres, opt, won_moves = 0, 0, 0
        while plies < max_plies:
            # ----- white (the agent) -----
            if random_rank:
                tags = rng.integers(0, 2, 128).astype(np.int32)
                tos = rng.integers(0, 64, 128).astype(np.int32)
                froms = np.where(tags == 0, wk, wr)
                afters = np.where(tags == 0, (tos * 64 + bk) * 64 + wr,
                                  (wk * 64 + bk) * 64 + tos)
            else:
                tags, froms, tos, afters = ag.rank_moves(wk, bk, wr)
            pid = (wk * 64 + bk) * 64 + wr
            d0 = int(T.dtm_w[pid])                  # oracle, evaluation-only
            legal_kd = _legal_lookup(T, pid)
            li = next((i for i in range(128)
                       if legal_kd[tags[i], tos[i]]), None)
            if li is None:                          # no legal white move (rare)
                reward = 0.0
                break
            if not random_rank:
                ag.learn_legality(tags, froms, tos, li)
            first_legal_flags.append(1.0 if li == 0 else 0.0)
            after = int(afters[li])
            afterstates.append(after)
            if d0 < INF:                            # oracle move-quality bookkeeping
                won_moves += 1
                d1 = int(T.dtm_b[after])
                pres += (d1 < INF)
                opt += (d1 == d0 - 1)
            wk, wr = after // 4096, after % 64      # bk unchanged by white
            plies += 1
            if T.b_mate[after]:
                reward, terminal = 1.0, True
                break
            if T.b_stale[after]:
                reward, terminal = 0.0, True
                break
            # ----- black (uniform random legal, the environment) -----
            row = T.b_succ[after]
            opts = row[row != -1]
            mv = int(opts[rng.integers(opts.size)])
            if mv == -2:                            # rook captured: dead draw
                reward, terminal = 0.0, True
                break
            wk, bk, wr = mv // 4096, (mv // 64) % 64, mv % 64
            plies += 1
        if not random_rank and value_on:
            ag.learn_value(afterstates, reward, truncated=not terminal)
        cur["mate"].append(reward)
        if reward > 0:
            cur["plies"].append(plies)
        cur["legal1"].append(float(np.mean(first_legal_flags)) if first_legal_flags else 0.0)
        cur["td"].append(float(np.mean(ag._td_abs[-len(afterstates):]))
                         if ag._td_abs and value_on else 0.0)
        if won_moves:
            cur["preserve"].append(pres / won_moves)
            cur["optimal"].append(opt / won_moves)
        cur["k"].append(ag.k)
        if (ep + 1) % window == 0:
            w = {k: float(np.mean(v)) if v else float("nan") for k, v in cur.items()}
            windows.append(w)
            # ADAPT k on improvement (episode-level agency, Thm 8c)
            if not random_rank and len(windows) >= 2 and \
                    windows[-1]["mate"] > windows[-2]["mate"]:
                ag.k = min(ag.k + ag.k_adapt, ag.k_max)
            cur = {k: [] for k in stats_keys}
    return {"windows": windows, "agent": ag}


_LEGAL_CACHE: dict = {}


def _legal_lookup(T, pid: int) -> np.ndarray:
    """[2, 64] bool: is (tag, dest) legal at this placement (referee-side)."""
    hit = _LEGAL_CACHE.get(pid)
    if hit is not None:
        return hit
    wk, bk, wr = pid // 4096, (pid // 64) % 64, pid % 64
    m = np.zeros((2, 64), dtype=bool)
    for piece, _f, t in T.white_moves(wk, bk, wr):
        m[0 if piece == "K" else 1, t] = True
    if len(_LEGAL_CACHE) < 200_000:
        _LEGAL_CACHE[pid] = m
    return m


def main(n_episodes: int = 30000, n_seeds: int = 3) -> None:
    print("\n" + "#" * 74)
    print("#  KRK CLOSED LOOP -- representation + value + policy learned ONLINE in")
    print("#  one stream; exact tablebase oracle watching (evaluation-only)")
    print("#" * 74)
    W = 2000

    print(f"\n  [baseline] random ranking, {n_episodes // 3} episodes:")
    rb = run_agent(n_episodes // 3, seed=99, random_rank=True,
                   window=max(n_episodes // 3, 1))
    rb_mate = float(np.mean([w["mate"] for w in rb["windows"]]))
    rb_pres = float(np.nanmean([w["preserve"] for w in rb["windows"]]))
    print(f"     mate rate {rb_mate:.3f}   win-preserving moves {rb_pres:.3f}")

    print(f"\n  [agents] {n_seeds} seeds x {n_episodes} episodes "
          f"(windows of {W}):")
    runs, runs_lo = [], []
    for s in range(n_seeds):
        runs.append(run_agent(n_episodes, seed=s, window=W)["windows"])
        runs_lo.append(run_agent(n_episodes, seed=s, value_on=False,
                                 window=W)["windows"])
    n_w = min(len(r) for r in runs)

    def col(rs, key, i):
        return [r[i][key] for r in rs]

    print(f"\n  {'window':>7}{'mate':>8}{'plies':>8}{'legal@1':>9}{'TD-err':>8}"
          f"{'preserve':>10}{'optimal':>9}{'k':>6}   (means over seeds)")
    for i in range(n_w):
        if i % max(n_w // 10, 1) and i != n_w - 1:
            continue
        m = {k: float(np.nanmean(col(runs, k, i)))
             for k in ("mate", "plies", "legal1", "td", "preserve", "optimal", "k")}
        print(f"  {(i + 1) * W:>7}{m['mate']:>8.3f}{m['plies']:>8.1f}"
              f"{m['legal1']:>9.3f}{m['td']:>8.3f}{m['preserve']:>10.3f}"
              f"{m['optimal']:>9.3f}{m['k']:>6.1f}")

    first = {k: np.array(col(runs, k, 0)) for k in ("mate", "legal1", "preserve", "td")}
    last = {k: np.array(col(runs, k, n_w - 1)) for k in ("mate", "legal1", "preserve", "td")}
    td_curve = [float(np.nanmean(col(runs, "td", i))) for i in range(n_w)]
    td_peak = max(td_curve)
    last_lo = np.array(col(runs_lo, "mate", n_w - 1))
    ci_m = paired_ci(last["mate"], first["mate"])
    ci_v = paired_ci(last["mate"], last_lo)
    print(f"\n  online learning (last window vs first, paired over seeds):")
    print(f"   * mate rate     : {np.mean(first['mate']):.3f} -> "
          f"{np.mean(last['mate']):.3f}   diff {fmt_ci(ci_m)} {verdict(ci_m)}")
    print(f"   * legal@1       : {np.mean(first['legal1']):.3f} -> "
          f"{np.mean(last['legal1']):.3f}   (movement rules emerged from rejections)")
    print(f"   * TD discrepancy: rises to {td_peak:.3f} (reward signal arriving), "
          f"then falls to {td_curve[-1]:.3f}")
    print(f"   * win-preserving: {np.mean(first['preserve']):.3f} -> "
          f"{np.mean(last['preserve']):.3f}   vs random {rb_pres:.3f}  "
          f"(gap to the EXACT oracle, online)")
    print(f"   * value's contribution: mate rate with value {np.mean(last['mate']):.3f}"
          f" vs legality-only {np.mean(last_lo):.3f}   diff {fmt_ci(ci_v)} {verdict(ci_v)}")

    if n_episodes < 20000:
        print("\n  [smoke run] asserts are calibrated for full-scale runs "
              "(>= 20k episodes); skipped.\n")
        return
    assert np.mean(last["legal1"]) > 0.85, \
        "movement legality must be learned online from rejections"
    assert ci_m["lo"] > 0, \
        "mate rate must rise online, CI-significant over seeds"
    assert np.mean(last["mate"]) > rb_mate + 0.1, \
        "the closed loop must clearly beat the random baseline"
    assert td_curve[-1] < 0.9 * td_peak, \
        "TD discrepancy (the MODIFY driver) must fall from its peak as V converges"
    assert ci_v["lo"] > 0, \
        "the value loop's contribution over legality-only must be CI-significant"

    print("\n  verdict: ONE agent, ONE stream -- movement legality, value and policy")
    print("  all learned online, discrepancy-driven, with the gap to an EXACT oracle")
    print("  measured live. Honest ceilings are printed above, not hidden: the")
    print("  tabular state space (262k afterstates) is visited sparsely, so the")
    print("  remaining gap to DTM-optimal play is the SCALABLE-REPRESENTATION open")
    print("  problem (roadmap 8.2) made quantitative.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="KRK closed online loop")
    p.add_argument("--episodes", type=int, default=30000)
    p.add_argument("--seeds", type=int, default=3)
    a = p.parse_args()
    main(n_episodes=a.episodes, n_seeds=a.seeds)
