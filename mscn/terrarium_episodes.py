"""TERRARIUM EPISODES -- the showcase benchmark: scripts, scoring, report.

Every episode is BOTH a story a novice can follow in one sentence AND a real
scored benchmark. The two faces share one truth: the front-stage numbers are
EXACTLY the backstage means (8+ seed paired CIs via ``mscn.stats``), never
prettified. Each episode is a *re-demonstration* of a measured ARCHITECTURE
section 9-10 result, so its expectation is PRE-REGISTERED by citation: the
ancestor claim is named in the episode docstring, and if the demo contradicts
the ancestor, the DEMO is wrong until proven otherwise.

Rendering is OPTIONAL (``--render``): without it every episode is a headless
scoring run (used by the test gate); with it the episode also writes PNG/GIF
assets and a story panel into the report directory.

Episodes (the cast):
  E1 two-twins    seasons + fog; memory keeper vs amnesiac     (ancestor 9.6 G2)
  E2 earthquake   shock displacement + fast-memory damage      (ancestor 9/9.1 shocked)
  E3 mirage-field signed memory paints X on checked decoys     (ancestor 10.1/10.2)
  E4 canyon       planner + option commitment crosses          (ancestor 9.2 corridor)
  E5 trade        reciprocity-gated exchange, scarce info      (ancestor 9 V5 / 6.4)
  E6 chess-garden KRK rules + value learned online             (ancestor 9.4)
  E7 grand-tour   ULTRA through everything, no season bell     (ancestor 11.x)

Run: ``python -m mscn.terrarium_episodes --episode e1 [--render] [--quick]``
     ``python -m mscn.terrarium_episodes --episode all --render``  (full show)
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .ibf_asi import IBFASI
from .ibf_asi_gauntlet import SwitchingWorld
from .stats import fmt_ci, mean_ci, paired_ci, verdict
from .terrarium import (Recorder, hstack_images, render_frame, save_gif,
                        save_png)

REPORT_DIR = "terrarium_report"
ASSET_DIR = os.path.join(REPORT_DIR, "assets")

EPISODES: dict = {}


def episode(name: str):
    def deco(fn):
        EPISODES[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
#  Shared machinery: the G2 protocol (phase metrics exactly as the gauntlet's)
# ---------------------------------------------------------------------------

def g2_phase_stats(true_trace: list[float]) -> dict:
    """A1/B/A2 tails + time-to-threshold savings, identical to 9.6 G2."""
    ph = list(true_trace)
    n = len(ph)
    p1, p2, p3 = ph[: n // 3], ph[n // 3: 2 * n // 3], ph[2 * n // 3:]
    q = max(len(p1) // 4, 1)
    a1 = float(np.mean(p1[-q:]))
    thr = 0.8 * a1

    def t_cross(p):
        sm = np.convolve(np.asarray(p, float), np.ones(15) / 15, mode="valid")
        hit = np.flatnonzero(sm >= thr)
        return int(hit[0]) if hit.size else len(p)

    return {"A1_tail": a1, "B_tail": float(np.mean(p2[-q:])),
            "A2_tail": float(np.mean(p3[-q:])),
            "t1": t_cross(p1), "t3": t_cross(p3),
            "savings": t_cross(p1) - t_cross(p3)}


def run_g2_arm(agent_kw: dict, seed: int, budget: int = 12000) -> dict:
    """One unsignalled G2 run (no context bell -- the 9.6 protocol)."""
    w = SwitchingWorld(seed=seed, budget=budget)
    a = IBFASI(w, seed=100 + seed, model_planner=True, **agent_kw)
    basin = []
    while w.evals < budget:
        if w.tick():
            a.apply_shock()
        a.step()
        basin.append(float(w.in_global_basin(a.x)))
    st = g2_phase_stats([t["true"] for t in a.telemetry])
    q = max(len(basin) // 12, 1)              # final quarter of phase 3
    st["home_frac_A2"] = float(np.mean(basin[-q:]))
    return st


SEASON = {1: "~ SUMMER ~", 2: "* WINTER *", 3: "~ SUMMER AGAIN ~"}


# ---------------------------------------------------------------------------
#  E1 -- THE TWO TWINS
# ---------------------------------------------------------------------------

@episode("e1")
def e1_two_twins(out: str = REPORT_DIR, n_seeds: int = 8, budget: int = 12000,
                 render: bool = False, render_seed: int = 0,
                 quick: bool = False) -> dict:
    """E1 -- "Winter came. The forgetful twin lost the map; ours kept it."

    Two identical agents live through seasons (the G2 switching world: summer
    layout -> a different, foggier winter layout -> the exact summer restored).
    One keeps the full memory apparatus; the amnesiac twin has alpha=0.

    PRE-REGISTERED (re-demonstration of 9.6 G2 + 10.1): the keeper's recovered
    asymptote (A2 tail) beats the amnesiac's, CI-significant (ancestor: full
    3.33 vs no-memory 2.11, sig). HONEST NOTE carried with the episode: memory
    does NOT buy relearning *speed* here (ancestor savings are negative; the
    phase-B residue slows the re-climb) -- it buys the recovered asymptote.
    """
    if quick:
        n_seeds, budget = 3, 6000
    arms = {"keeper": dict(), "amnesiac": dict(alpha=0.0)}
    rows = {name: [run_g2_arm(kw, s, budget) for s in range(n_seeds)]
            for name, kw in arms.items()}
    ci_a2 = paired_ci([r["A2_tail"] for r in rows["keeper"]],
                      [r["A2_tail"] for r in rows["amnesiac"]])
    ci_home = paired_ci([r["home_frac_A2"] for r in rows["keeper"]],
                        [r["home_frac_A2"] for r in rows["amnesiac"]])
    sav = mean_ci([r["savings"] for r in rows["keeper"]])
    m = {n: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
         for n, rs in rows.items()}
    print(f"  [E1] keeper A2 {m['keeper']['A2_tail']:.2f} vs amnesiac "
          f"{m['amnesiac']['A2_tail']:.2f}   paired {fmt_ci(ci_a2)} {verdict(ci_a2)}")
    print(f"       home-frac A2 {m['keeper']['home_frac_A2']:.2f} vs "
          f"{m['amnesiac']['home_frac_A2']:.2f}  {fmt_ci(ci_home)} {verdict(ci_home)}")
    print(f"       keeper savings (honest note) {sav['mean']:+.0f} "
          f"[{sav['lo']:+.0f}, {sav['hi']:+.0f}] ticks")
    if quick:
        assert ci_a2["mean"] > 0, "E1 quick: keeper must lead the asymptote (dir.)"
    else:
        assert ci_a2["lo"] > 0, \
            "E1 PRE-REG: recovered asymptote must be CI-significant (ancestor 9.6)"

    assets = []
    if render:
        assets = _render_e1(out, render_seed, budget)
    panel = {
        "title": "Episode 1 — The Two Twins",
        "caption": "Winter came. The forgetful twin lost the map; ours kept it.",
        "novice": [
            f"home quality after winter: keeper {m['keeper']['A2_tail']:.2f} "
            f"vs amnesiac {m['amnesiac']['A2_tail']:.2f} (coherence, higher better)",
            f"time living in the best meadow after winter: keeper "
            f"{100 * m['keeper']['home_frac_A2']:.0f}% vs amnesiac "
            f"{100 * m['amnesiac']['home_frac_A2']:.0f}%",
        ],
        "backstage": [
            f"recovered asymptote (A2 tail), paired over {n_seeds} seeds: "
            f"{fmt_ci(ci_a2)} {verdict(ci_a2)} — ancestor §9.6 G2 (+sig)",
            f"home-basin occupancy diff: {fmt_ci(ci_home)} {verdict(ci_home)}",
            f"honest note (ancestor kept): keeper's own relearning savings "
            f"{sav['mean']:+.0f} [{sav['lo']:+.0f}, {sav['hi']:+.0f}] ticks — "
            f"memory buys the recovered asymptote, NOT relearning speed; the "
            f"winter residue slows the first re-climb.",
        ],
        "assets": assets,
        "mechanism": "continual memory (the one universally significant stage)",
    }
    return {"panel": panel, "ci": {"A2": ci_a2, "home": ci_home},
            "means": m}


def _render_e1(out: str, seed: int, budget: int) -> list[str]:
    os.makedirs(ASSET_DIR, exist_ok=True)
    twins = {
        "keeper": IBFASI, "amnesiac": IBFASI}
    kws = {"keeper": dict(), "amnesiac": dict(alpha=0.0)}
    W, A, REC, TRAIL, STAGE = {}, {}, {}, {}, {}
    for name in twins:
        W[name] = SwitchingWorld(seed=seed, budget=budget)
        A[name] = IBFASI(W[name], seed=100 + seed, model_planner=True, **kws[name])
        REC[name] = Recorder(W[name])
        REC[name].refresh_terrain()
        TRAIL[name] = []
        STAGE[name] = 1
    t0 = REC["keeper"]._terrain
    vlim = (float(t0.min()) - 0.5, float(t0.max()) + 0.5)
    frames, tick, every = [], 0, 5
    while any(W[n].evals < budget for n in twins):
        events = {n: [] for n in twins}
        for n in twins:
            if W[n].evals >= budget:
                continue
            if W[n].tick():
                A[n].apply_shock()
            A[n].step()
            TRAIL[n].append(np.asarray(A[n].x, float).copy())
            if W[n].stage != STAGE[n]:
                STAGE[n] = W[n].stage
                REC[n].refresh_terrain()
                events[n].append(SEASON[STAGE[n]])
        tick += 1
        if tick % every:
            continue
        panels = []
        for n in twins:
            REC[n].snap(A[n], events=events[n], meters={
                "season": SEASON[STAGE[n]].strip("~* "),
                "coherence": f"{A[n].telemetry[-1]['true']:.2f}" if A[n].telemetry
                else "--"})
            panels.append(render_frame(
                REC[n].snaps[-1], W[n],
                title=f"{'THE KEEPER' if n == 'keeper' else 'THE AMNESIAC'}",
                trail=TRAIL[n][-70:], vlim=vlim))
        frames.append(hstack_images(panels))
    gif = save_gif(frames, os.path.join(ASSET_DIR, "e1_two_twins.gif"))
    png = save_png(frames[-1], os.path.join(ASSET_DIR, "e1_two_twins_final.png"))
    print(f"       assets: {gif['path']} ({gif['bytes'] / 1e6:.2f} MB, "
          f"{gif['frames']} frames), {png}")
    return [gif["path"], png]


# ---------------------------------------------------------------------------
#  Report generation (assembled from episode panels)
# ---------------------------------------------------------------------------

def write_report(panels: list[dict], extra_sections: list[str] | None = None,
                 out: str = REPORT_DIR) -> str:
    os.makedirs(out, exist_ok=True)
    lines = [
        "# THE TERRARIUM — the IBF/MSCN apparatus, visible",
        "",
        "One living world; every validated capability an episode a novice can",
        "follow in one sentence. **The numbers on the front stage are exactly",
        "the backstage paired-CI means** (8+ seeds, `mscn/stats.py`) — never",
        "prettified. Each episode re-demonstrates a measured result from",
        "`mscn/ARCHITECTURE.md` §9–§11; where the science says *null*, the",
        "showcase says so too (see the “what doesn’t help” panel).",
        "",
        "Reproduce: `python -m mscn.terrarium_episodes --episode all --render`",
        "",
    ]
    for p in panels:
        lines += [f"## {p['title']}", "", f"> *“{p['caption']}”*", ""]
        for a in p.get("assets", []):
            rel = os.path.relpath(a, out)
            lines.append(f"![{p['title']}]({rel})")
        lines += ["", f"**Mechanism made visible:** {p['mechanism']}", "",
                  "**Novice scoreboard** (these ARE the measured means):", ""]
        lines += [f"- {s}" for s in p["novice"]]
        lines += ["", "**Backstage (the real benchmark):**", ""]
        lines += [f"- {s}" for s in p["backstage"]]
        lines.append("")
    for s in (extra_sections or []):
        lines += [s, ""]
    path = os.path.join(out, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  report written: {path}")
    return path


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Terrarium episodes")
    p.add_argument("--episode", default="all",
                   help=f"one of {sorted(EPISODES)} or 'all'")
    p.add_argument("--render", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    names = sorted(EPISODES) if args.episode == "all" else [args.episode]
    panels = []
    for n in names:
        print(f"\n== episode {n} ==")
        r = EPISODES[n](render=args.render, quick=args.quick,
                        render_seed=args.seed)
        panels.append(r["panel"])
    if args.render:
        write_report(panels)


if __name__ == "__main__":
    main()
