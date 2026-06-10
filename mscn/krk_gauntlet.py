"""G4 -- KRK under PERFECT DEFENSE: the trained closed-loop agents, frozen, vs
the tablebase-optimal defender (maximum distribution shift).

The 9.4/9.5 agents trained and were measured against a RANDOM black king. Here
the trained policies are FROZEN (no learning, near-greedy ranking) and evaluated
against the cruellest opponent the oracle can build: black plays, at every move,
the reply that maximises white's remaining distance-to-mate -- and takes any
drawing escape (stalemate trap, rook capture, fortress) the instant white's play
allows one. Pre-registered questions:
  * how much of the trained strength survives the shift random -> optimal?
  * does the generalising (tile-relative) agent's technique advantage GROW under
    pressure (it mates faster, so it should leak fewer escapes)?
Reported: mate rate within the ply cap, win-preservation per move, mean plies
over optimal DTM (the efficiency gap), all on tablebase-WON random starts.

Run: ``python -m mscn.krk_gauntlet [--episodes 30000]``  (numpy; ~15 min).
"""

from __future__ import annotations

import numpy as np

from .krk_closed_loop import KRKClosedLoopAgent, _legal_lookup, run_agent
from .krk_value_general import TileRelativeAgent
from .krk_world import INF, tables
from .stats import fmt_ci, paired_ci, verdict


def black_optimal(T, after_pid: int, rng) -> int | None:
    """Perfect defence: take a drawing escape if one exists (rook capture or a
    successor where white is no longer winning); else maximise white's DTM."""
    row = T.b_succ[after_pid]
    opts = row[row != -1]
    if opts.size == 0:
        return None                                  # mate or stalemate
    best, best_d = None, -1
    for mv in opts:
        if mv == -2:
            return -2                                # capture the rook: draw
        d = int(T.dtm_w[mv])
        if d >= INF:
            return int(mv)                           # drawn position: escape
        if d > best_d:
            best_d, best = d, int(mv)
    return best


def evaluate_frozen(ag: KRKClosedLoopAgent, optimal: bool, n_episodes: int,
                    seed: int, max_plies: int = 80) -> dict:
    """Frozen-policy evaluation: no learning, tiny ranking noise (tie-breaks)."""
    T = tables()
    rng = np.random.default_rng(7000 + seed)
    starts = np.flatnonzero(T.legal_w & (T.dtm_w < INF))
    k_save, ag.k = ag.k, 25.0                        # near-greedy ranking
    mates, plies_over, pres, moves = 0, [], 0, 0
    for _ in range(n_episodes):
        p = int(starts[rng.integers(starts.size)])
        wk, bk, wr = p // 4096, (p // 64) % 64, p % 64
        plies = 0
        d_start = int(T.dtm_w[p])
        while plies < max_plies:
            pid = (wk * 64 + bk) * 64 + wr
            d0 = int(T.dtm_w[pid])
            tags, froms, tos, afters = ag.rank_moves(wk, bk, wr)
            legal_kd = _legal_lookup(T, pid)
            li = next((i for i in range(128) if legal_kd[tags[i], tos[i]]), None)
            if li is None:
                break
            after = int(afters[li])
            moves += 1
            if d0 < INF:
                pres += (int(T.dtm_b[after]) < INF)
            wk, wr = after // 4096, after % 64
            plies += 1
            if T.b_mate[after]:
                mates += 1
                plies_over.append(plies - d_start)
                break
            if T.b_stale[after]:
                break
            mv = black_optimal(T, after, rng) if optimal else None
            if not optimal:
                row = T.b_succ[after]
                opts = row[row != -1]
                mv = int(opts[rng.integers(opts.size)])
            if mv is None or mv == -2:
                break
            wk, bk, wr = mv // 4096, (mv // 64) % 64, mv % 64
            plies += 1
    ag.k = k_save
    return {"mate": mates / n_episodes,
            "preserve": pres / max(moves, 1),
            "plies_over": float(np.mean(plies_over)) if plies_over else float("nan")}


def main(n_train: int = 30000, n_eval: int = 600, n_seeds: int = 3) -> None:
    print("\n" + "#" * 74)
    print("#  G4 -- KRK vs the TABLEBASE-OPTIMAL defender (frozen trained policies)")
    print("#" * 74)
    res: dict = {}
    for name, builder in (("tabular", lambda s: None),
                          ("tile-relative", lambda s: TileRelativeAgent(seed=s))):
        rows = {"random": [], "optimal": []}
        for s in range(n_seeds):
            ag = builder(s)
            if ag is None:
                out = run_agent(n_train, seed=s, window=n_train)
                ag = out["agent"]
            else:
                orig = ag.learn_value

                def hooked(afterstates, reward, truncated=False, _a=ag, _o=orig):
                    _a.maybe_embed()
                    _o(afterstates, reward, truncated)
                ag.learn_value = hooked
                ag = run_agent(n_train, seed=s, window=n_train, agent=ag)["agent"]
            for mode in ("random", "optimal"):
                rows[mode].append(evaluate_frozen(ag, mode == "optimal",
                                                  n_eval, seed=s))
        res[name] = rows
        for mode in ("random", "optimal"):
            m = {k: float(np.nanmean([r[k] for r in rows[mode]]))
                 for k in rows[mode][0]}
            print(f"  {name:<14} vs {mode:<8} mate {m['mate']:.3f}  "
                  f"win-preserving {m['preserve']:.3f}  "
                  f"plies-over-optimal {m['plies_over']:.1f}")

    print("\n  paired comparisons (3 seeds):")
    for key, label in (("mate", "mate rate"), ("preserve", "win-preservation")):
        ci = paired_ci([r[key] for r in res["tile-relative"]["optimal"]],
                       [r[key] for r in res["tabular"]["optimal"]])
        print(f"   tile - tabular vs OPTIMAL defence, {label:<16}: "
              f"{fmt_ci(ci)} {verdict(ci)}")
    drop_tab = (np.mean([r["mate"] for r in res["tabular"]["random"]])
                - np.mean([r["mate"] for r in res["tabular"]["optimal"]]))
    drop_til = (np.mean([r["mate"] for r in res["tile-relative"]["random"]])
                - np.mean([r["mate"] for r in res["tile-relative"]["optimal"]]))
    print(f"   distribution-shift cost (mate drop random->optimal): "
          f"tabular {drop_tab:.3f}, tile {drop_til:.3f}")
    assert np.mean([r["mate"] for r in res["tile-relative"]["optimal"]]) > \
        np.mean([r["mate"] for r in res["tabular"]["optimal"]]) - 0.02, \
        "the generalising agent must hold its advantage under perfect defence"
    print("\n  Done; the shift cost and the surviving margin are the findings.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="KRK vs optimal defender")
    p.add_argument("--episodes", type=int, default=30000)
    a = p.parse_args()
    main(n_train=a.episodes)
