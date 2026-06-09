"""AGI Upgrade 7 -- Coherence-Based Planning via Imagined Trajectories.

With a learned simulation ``G`` (Upgrade 1) the agent can *imagine* action sequences
and evaluate them on the coherence landscape, instead of acting reactively. The
gradient-flow convergence theorem (Thm 1) makes an imagined rollout meaningful;
`planning_horizon_value_bound` bounds the value of depth-``d`` planning and
`planning_diminishing_returns` says the gains taper.

Setup: a 1-D corridor whose coherence field has a **deceptive trap bump** near the
start and a higher **goal peak** at the far end, with a low-coherence **valley**
between. A reactive (depth-1, greedy-coherence) agent climbs onto the trap bump and
oscillates there -- every immediate neighbour is lower. A depth-``d`` planner imagines
rollouts through the (learned) transition model with a discount (so reaching high
coherence *sooner* wins), sees *across* the valley to the goal once ``d`` spans it, and
commits to crossing.

The transition model is **learned** from random exploration (tabular ``G_hat``, exact
for a deterministic world -- the Upgrade-1 mechanism), so planning runs on an imagined
model. Measured advantage: goal-reaching **success rises with planning depth** and then
plateaus (diminishing returns) once the horizon spans the valley.

Run: ``python -m mscn.agi_planning``   (numpy only).
"""

from __future__ import annotations

import numpy as np

ArrayF = np.ndarray


class Corridor:
    """A length-M corridor; 2 actions (left=-1, right=+1), clamped at the ends. The
    coherence field: a trap bump at `trap`, a goal peak at `M-1`, a valley between."""

    def __init__(self, M: int = 22, trap: int = 5, trap_h: float = 0.6,
                 goal_h: float = 1.0, st: float = 1.8, sg: float = 4.5) -> None:
        self.M = M
        self.trap = trap
        self.goal = M - 1
        i = np.arange(M)
        self.R = (trap_h * np.exp(-((i - trap) ** 2) / (2 * st ** 2))
                  + goal_h * np.exp(-((i - (M - 1)) ** 2) / (2 * sg ** 2)))

    def step(self, s: int, a: int) -> int:
        return int(min(max(s + (1 if a == 1 else -1), 0), self.M - 1))

    def coh(self, s: int) -> float:
        return float(self.R[s])


def learn_transition(world: Corridor, n_steps: int = 4000, seed: int = 0) -> dict:
    """Learn G_hat tabularly from random exploration (exact for a deterministic world
    -- the Upgrade-1 simulation), and report faithfulness vs the truth."""
    rng = np.random.default_rng(seed)
    model: dict[tuple, int] = {}
    s = int(rng.integers(world.M))
    for _ in range(n_steps):
        a = int(rng.integers(2))
        model[(s, a)] = world.step(s, a)
        s = model[(s, a)]
        if rng.random() < 0.03:
            s = int(rng.integers(world.M))
    ok = tot = 0
    for x in range(world.M):
        for a in range(2):
            if (x, a) in model:
                ok += model[(x, a)] == world.step(x, a); tot += 1
    return {"model": model, "faithful": ok / max(tot, 1), "coverage": tot / (world.M * 2)}


def _sim(model: dict, world: Corridor, s: int, a: int) -> int:
    return model.get((s, a), world.step(s, a))


def plan_action(model: dict, world: Corridor, s: int, depth: int) -> int:
    """Imagine rollouts through the learned model: BFS to `depth`, find the highest-
    coherence reachable state (nearest on a tie), and return the first action toward it.
    A reactive agent (depth 1) sees only its neighbours and stays on the trap bump; a
    deep planner sees across the valley to the goal."""
    from collections import deque

    best_R, best_first, best_dist = world.coh(s), None, 0
    seen = {s: 0}
    dq = deque([(s, None, 0)])
    while dq:
        st, first, dist = dq.popleft()
        r = world.coh(st)
        if r > best_R + 1e-12 or (abs(r - best_R) <= 1e-12 and dist < best_dist):
            best_R, best_first, best_dist = r, first, dist
        if dist < depth:
            for a in range(2):
                ns = _sim(model, world, st, a)
                if ns not in seen:
                    seen[ns] = dist + 1
                    dq.append((ns, a if first is None else first, dist + 1))
    return best_first if best_first is not None else 0


def run_episode(world: Corridor, model: dict, start: int, depth: int, max_steps: int = 80) -> bool:
    s = start
    for _ in range(max_steps):
        if s == world.goal:
            return True
        s = world.step(s, plan_action(model, world, s, depth))
    return s == world.goal


def success_rate(world: Corridor, model: dict, depth: int) -> float:
    # start left of the goal basin, where the trap is the nearer attractor
    starts = list(range(0, world.trap + 3))
    return float(np.mean([run_episode(world, model, st, depth) for st in starts]))


def main() -> None:
    print("\n" + "#" * 72)
    print("#  UPGRADE 7 -- COHERENCE-BASED PLANNING (imagined trajectories)")
    print("#  imagine rollouts through the LEARNED simulation; pick the best by coherence")
    print("#" * 72)

    world = Corridor()
    lm = learn_transition(world)
    model = lm["model"]
    valley = world.goal - world.trap
    print(f"\n  learned transition model G_hat: faithful {lm['faithful']:.3f}, "
          f"coverage {lm['coverage']:.0%}")
    print(f"  corridor M={world.M}: trap bump at {world.trap}, goal at {world.goal} "
          f"(valley span ~{valley})\n")

    print(f"  goal-reaching success rate vs planning depth (starts near the trap):")
    print(f"      {'depth':>6}{'success':>10}")
    rates = {}
    depths = (1, 6, 12, 16, 20, 24, 28)
    for d in depths:
        rates[d] = success_rate(world, model, d)
        tag = "  (reactive)" if d == 1 else ""
        print(f"      {d:>6}{rates[d]:>9.0%}{tag}")
    reactive = rates[1]
    best = max(rates.values())
    early = rates[16] - rates[1]
    late = rates[28] - rates[24]
    print(f"\n  reading:")
    print(f"   * reactive (depth 1) success {reactive:.0%}; deep planning lifts it to {best:.0%}")
    print(f"     -> imagined rollouts see across the valley and commit to crossing it")
    print(f"   * diminishing returns: gain depth1->16 = {early:+.0%} >> gain depth24->28 = {late:+.0%}")
    print(f"     (planning_diminishing_returns: once the horizon spans the valley, more depth")
    print(f"      adds nothing -- planning_horizon_value_bound)")

    assert best - reactive > 0.4, f"planning must clearly beat reactive: {reactive:.0%}->{best:.0%}"
    assert lm["faithful"] > 0.99, "the learned simulation must be faithful for planning to be valid"
    assert late <= early + 1e-9, "gains must taper with depth (diminishing returns)"
    print(f"\n  Upgrade 7 mechanism is functional: planning over the learned simulation")
    print(f"  lifts goal-reaching from {reactive:.0%} (reactive, trapped) to {best:.0%}, with")
    print(f"  clear diminishing returns in depth -- imagined trajectories convert a trapped")
    print(f"  reactive agent into one that solves the task.\n")


if __name__ == "__main__":
    main()
