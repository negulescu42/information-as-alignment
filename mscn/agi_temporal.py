"""AGI Upgrade 8 -- Temporal Abstraction (macro-actions as coherence basins).

A **macro-action** is a coherence basin: entering it and following the coherence
gradient converges to a high-coherence state, and by basin invariance (Thm 2) coherence
is non-decreasing along the way, so the macro is *safe* -- it never drops below where it
started (`temporal_abstraction_safety`). Composing safe in-basin macros keeps the whole
trajectory safe (`macro_action_composition`). Planning over *basins* rather than
primitive moves is a coarser, cheaper search.

Setup: a multi-basin corridor (a chain of coherence bumps of increasing height, the
last being the goal). Two measured claims:

1. **Efficiency.** A hierarchical planner that plans over macro-actions (ascend the
   current basin; transit to the next basin toward the goal) reaches the goal in a
   handful of **high-level decisions**, vs a primitive step-by-step planner that needs
   one (deep-lookahead) decision per cell.
2. **Safety.** Every in-basin **ascent macro** is coherence-non-decreasing (basin
   invariance / `temporal_abstraction_safety`): coherence along it never falls below its
   start -- so the composed plan's ascent segments are all safe.

Run: ``python -m mscn.agi_temporal``   (numpy only).
"""

from __future__ import annotations

import numpy as np

ArrayF = np.ndarray


class BasinCorridor:
    """A length-M corridor whose coherence field is a chain of bumps (basins) of
    increasing height; the last bump is the goal. Valleys separate the basins."""

    def __init__(self, peaks=(4, 12, 20, 28, 36), heights=(0.5, 0.6, 0.72, 0.85, 1.0),
                 width: float = 2.0, M: int = 40) -> None:
        self.M = M
        self.peaks = list(peaks)
        self.goal = peaks[-1]
        i = np.arange(M)
        self.R = sum(h * np.exp(-((i - p) ** 2) / (2 * width ** 2))
                     for p, h in zip(peaks, heights))

    def step(self, s: int, a: int) -> int:
        return int(min(max(s + (1 if a == 1 else -1), 0), self.M - 1))

    def coh(self, s: int) -> float:
        return float(self.R[s])


# ---- macro-actions ----

def ascend_macro(world: BasinCorridor, s: int) -> list[int]:
    """Greedy coherence ascent to the local basin peak. Returns the trajectory.
    Coherence is non-decreasing along it by construction (basin invariance)."""
    traj = [s]
    while True:
        nbrs = [world.step(s, a) for a in range(2)]
        best = max(nbrs, key=world.coh)
        if world.coh(best) <= world.coh(s) + 1e-12:
            break                                  # at the basin peak
        s = best
        traj.append(s)
    return traj


def transit_macro(world: BasinCorridor, s: int, direction: int) -> list[int]:
    """Cross the valley toward `direction` until coherence starts rising again (entered
    the next basin). Then an ascent macro takes over. (The risky inter-basin segment.)"""
    traj = [s]
    a = 1 if direction > 0 else 0
    rising = False
    for _ in range(world.M):
        ns = world.step(s, a)
        if ns == s:
            break
        if world.coh(ns) > world.coh(s):          # started climbing the next basin
            rising = True
        elif rising:                              # was rising, now not -> at next peak region
            break
        traj.append(ns)
        s = ns
        if rising and world.coh(world.step(s, a)) <= world.coh(s):
            break
    return traj


def hierarchical_plan(world: BasinCorridor, start: int) -> dict:
    """Plan over macro-actions: ascend the current basin; if not at goal, transit toward
    the goal and ascend again. Count high-level decisions and check macro safety."""
    s = start
    decisions = 0
    full_traj = [s]
    ascent_safe = True
    for _ in range(20):
        macro = ascend_macro(world, s)            # safe in-basin ascent
        ascent_safe &= all(world.coh(macro[j]) >= world.coh(macro[0]) - 1e-12
                           for j in range(len(macro)))
        full_traj += macro[1:]
        s = macro[-1]
        decisions += 1
        if s == world.goal:
            break
        t = transit_macro(world, s, np.sign(world.goal - s))   # cross to next basin
        full_traj += t[1:]
        s = t[-1]
        decisions += 1
        if s == world.goal:
            # one more ascent may be needed
            macro = ascend_macro(world, s)
            full_traj += macro[1:]; s = macro[-1]; decisions += 1
            break
    reached = s == world.goal
    return {"decisions": decisions, "reached": reached, "traj": full_traj,
            "ascent_safe": ascent_safe}


def primitive_plan(world: BasinCorridor, start: int, depth: int = 10) -> dict:
    """Primitive baseline: one BFS-lookahead decision per cell (Upgrade-7 style)."""
    from collections import deque

    def act(s):
        best_R, best_first, best_dist = world.coh(s), None, 0
        seen = {s: 0}; dq = deque([(s, None, 0)])
        while dq:
            st, first, dist = dq.popleft()
            r = world.coh(st)
            if r > best_R + 1e-12 or (abs(r - best_R) <= 1e-12 and dist < best_dist):
                best_R, best_first, best_dist = r, first, dist
            if dist < depth:
                for a in range(2):
                    ns = world.step(st, a)
                    if ns not in seen:
                        seen[ns] = dist + 1
                        dq.append((ns, a if first is None else first, dist + 1))
        return best_first if best_first is not None else 0

    s = start; decisions = 0
    for _ in range(4 * world.M):
        if s == world.goal:
            break
        s = world.step(s, act(s)); decisions += 1
    return {"decisions": decisions, "reached": s == world.goal}


def main() -> None:
    print("\n" + "#" * 72)
    print("#  UPGRADE 8 -- TEMPORAL ABSTRACTION (macro-actions as basins)")
    print("#  plan over coherence basins; in-basin ascent is safe (basin invariance)")
    print("#" * 72)

    world = BasinCorridor()
    print(f"\n  basin-chain corridor M={world.M}: peaks {world.peaks} (goal {world.goal})\n")

    starts = list(range(0, 8))
    macro = [hierarchical_plan(world, s) for s in starts]
    prim = [primitive_plan(world, s, depth=10) for s in starts]

    macro_dec = float(np.mean([m["decisions"] for m in macro if m["reached"]]))
    prim_dec = float(np.mean([p["decisions"] for p in prim if p["reached"]]))
    macro_reached = float(np.mean([m["reached"] for m in macro]))
    prim_reached = float(np.mean([p["reached"] for p in prim]))
    all_safe = all(m["ascent_safe"] for m in macro)

    print(f"  (1) efficiency -- decisions to reach the goal (mean over {len(starts)} starts):")
    print(f"      primitive (per-cell lookahead) : {prim_dec:.1f} decisions  "
          f"(reached {prim_reached:.0%})")
    print(f"      macro (per-basin)              : {macro_dec:.1f} decisions  "
          f"(reached {macro_reached:.0%})")
    print(f"      -> temporal abstraction reaches the goal in {prim_dec/macro_dec:.1f}x "
          f"fewer high-level decisions")

    print(f"\n  (2) safety -- in-basin ascent macros are coherence-non-decreasing")
    print(f"      (temporal_abstraction_safety / basin invariance, Thm 2):")
    print(f"      all ascent macros monotone-safe: {all_safe}")

    assert macro_reached > 0.99 and prim_dec / macro_dec > 2.0, \
        "macro-planning must reach the goal in clearly fewer decisions"
    assert all_safe, "in-basin ascent macros must be coherence-non-decreasing (basin invariance)"
    print(f"\n  Upgrade 8 mechanism is functional: planning over macro-actions (coherence")
    print(f"  basins) reaches the goal in {prim_dec/macro_dec:.1f}x fewer high-level decisions than")
    print(f"  primitive planning, and every in-basin ascent macro is safe (coherence non-")
    print(f"  decreasing) -- temporal abstraction with the basin-invariance guarantee.\n")


if __name__ == "__main__":
    main()
