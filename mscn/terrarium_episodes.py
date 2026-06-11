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
#  E2 -- THE EARTHQUAKE
# ---------------------------------------------------------------------------

def _run_e2_arm(agent_kw: dict, seed: int, budget: int) -> dict:
    from .ibf_asi import ASIWorld, IBFASI as _A
    w = ASIWorld(seed=seed, noise=0.35, shock_every=80)
    a = _A(w, seed=100 + seed, **agent_kw)
    shock_ticks: list[int] = []
    while w.evals < budget:
        if w.tick():
            a.apply_shock()
            shock_ticks.append(a.tick_no)
        a.step()
    true = np.array([t["true"] for t in a.telemetry])
    q = max(len(true) // 4, 1)
    recov = []
    for i, st in enumerate(shock_ticks):
        if st < 25 or st + 5 >= len(true):
            continue
        base = float(np.mean(true[st - 20: st]))
        end = shock_ticks[i + 1] if i + 1 < len(shock_ticks) else len(true)
        win = true[st:end]
        sm = np.convolve(win, np.ones(5) / 5, mode="valid")
        hit = np.flatnonzero(sm >= 0.8 * base)
        recov.append(int(hit[0]) if hit.size else len(win))
    return {"true_tail": float(np.mean(true[-q:])),
            "recovery": float(np.mean(recov)) if recov else float("nan"),
            "n_shocks": len(shock_ticks)}


@episode("e2")
def e2_earthquake(out: str = REPORT_DIR, n_seeds: int = 8, budget: int = 9000,
                  render: bool = False, render_seed: int = 1,
                  quick: bool = False) -> dict:
    """E2 -- "Knocked across the map, it walks straight home."

    Earthquakes displace the agent AND damage its fast (fine-scale) memory --
    the V1 shocked regime. The keeper's consolidated memory + warm jumps bring
    it home; the amnesiac restarts from scratch every time.

    PRE-REGISTERED (re-demonstration of 9 V1 / the shocked matrix cell):
    keeper tail coherence beats amnesiac, CI-significant (ancestor: memory
    +1.18 [+0.88, +1.47] at 16 seeds; shocked cell +0.7..0.9*). Recovery time
    is a derived presentation metric, reported with its CI, no ancestor claim.
    """
    if quick:
        n_seeds, budget = 3, 4000
    arms = {"keeper": dict(), "amnesiac": dict(alpha=0.0)}
    rows = {n: [_run_e2_arm(kw, s, budget) for s in range(n_seeds)]
            for n, kw in arms.items()}
    ci = paired_ci([r["true_tail"] for r in rows["keeper"]],
                   [r["true_tail"] for r in rows["amnesiac"]])
    rec_k = [r["recovery"] for r in rows["keeper"] if np.isfinite(r["recovery"])]
    rec_a = [r["recovery"] for r in rows["amnesiac"] if np.isfinite(r["recovery"])]
    ci_rec = paired_ci([r["recovery"] for r in rows["keeper"]],
                       [r["recovery"] for r in rows["amnesiac"]])
    m = {n: float(np.mean([r["true_tail"] for r in rs])) for n, rs in rows.items()}
    print(f"  [E2] keeper tail {m['keeper']:.2f} vs amnesiac {m['amnesiac']:.2f}"
          f"   paired {fmt_ci(ci)} {verdict(ci)}")
    print(f"       recovery ticks after a quake: keeper {np.mean(rec_k):.0f} vs "
          f"amnesiac {np.mean(rec_a):.0f}  diff {fmt_ci(ci_rec)} {verdict(ci_rec)}")
    if quick:
        assert ci["mean"] > 0, "E2 quick: keeper must lead (directional)"
    else:
        assert ci["lo"] > 0, \
            "E2 PRE-REG: memory must be CI-significant under shocks (ancestor V1)"

    assets = []
    if render:
        assets = _render_e2(out, render_seed, budget=5000)
    panel = {
        "title": "Episode 2 — The Earthquake",
        "caption": "Knocked across the map, it walks straight home.",
        "novice": [
            f"life quality in quake country: keeper {m['keeper']:.2f} vs "
            f"amnesiac {m['amnesiac']:.2f}",
            f"ticks to walk home after a quake: keeper {np.mean(rec_k):.0f} vs "
            f"amnesiac {np.mean(rec_a):.0f}",
        ],
        "backstage": [
            f"tail coherence, paired over {n_seeds} seeds: {fmt_ci(ci)} "
            f"{verdict(ci)} — ancestor §9 V1 (+1.18 sig) / shocked matrix cell",
            f"recovery-time diff (presentation metric, no ancestor): "
            f"{fmt_ci(ci_rec)} {verdict(ci_rec)}",
        ],
        "assets": assets,
        "mechanism": "consolidation + warm-jump homing (memory under shocks)",
    }
    return {"panel": panel, "ci": {"tail": ci, "recovery": ci_rec}, "means": m}


def _render_e2(out: str, seed: int, budget: int) -> list[str]:
    from .ibf_asi import ASIWorld, IBFASI as _A
    os.makedirs(ASSET_DIR, exist_ok=True)
    kws = {"keeper": dict(), "amnesiac": dict(alpha=0.0)}
    W, A, REC, TRAIL = {}, {}, {}, {}
    for n, kw in kws.items():
        W[n] = ASIWorld(seed=seed, noise=0.35, shock_every=80)
        A[n] = _A(W[n], seed=100 + seed, **kw)
        REC[n] = Recorder(W[n])
        REC[n].refresh_terrain()
        TRAIL[n] = []
    t0 = REC["keeper"]._terrain
    vlim = (float(t0.min()) - 0.5, float(t0.max()) + 0.5)
    frames, tick, every = [], 0, 4
    flash = {n: 0 for n in kws}
    while any(W[n].evals < budget for n in kws):
        events = {n: [] for n in kws}
        for n in kws:
            if W[n].evals >= budget:
                continue
            if W[n].tick():
                A[n].apply_shock()
                flash[n] = 3
                REC[n].refresh_terrain()
            A[n].step()
            TRAIL[n].append(np.asarray(A[n].x, float).copy())
            if flash[n] > 0:
                events[n].append("*** EARTHQUAKE ***")
                flash[n] -= 1
        tick += 1
        if tick % every:
            continue
        panels = []
        for n in kws:
            REC[n].snap(A[n], events=events[n], meters={
                "coherence": f"{A[n].telemetry[-1]['true']:.2f}"})
            panels.append(render_frame(
                REC[n].snaps[-1], W[n],
                title="THE KEEPER" if n == "keeper" else "THE AMNESIAC",
                trail=TRAIL[n][-50:], vlim=vlim))
        frames.append(hstack_images(panels))
    gif = save_gif(frames, os.path.join(ASSET_DIR, "e2_earthquake.gif"))
    png = save_png(frames[-1], os.path.join(ASSET_DIR, "e2_earthquake_final.png"))
    print(f"       assets: {gif['path']} ({gif['bytes'] / 1e6:.2f} MB)")
    return [gif["path"], png]


# ---------------------------------------------------------------------------
#  E3 -- THE MIRAGE FIELD
# ---------------------------------------------------------------------------

def _decoy_dwell(world, xs: list, grace: int = 20) -> float:
    """Fraction of an agent's life spent at mirages it has ALREADY checked:
    for each decoy, ticks within its radius after (first visit + grace),
    summed, over total post-grace life."""
    xs = np.asarray(xs)
    total, dwell = 0, 0
    for c in world.c[1:]:
        near = np.flatnonzero(np.sum((xs - c) ** 2, axis=1) < 1.2 ** 2)
        if near.size == 0:
            continue
        t0 = int(near[0]) + grace
        dwell += int(np.sum(near >= t0))
    total = len(xs)
    return dwell / max(total, 1)


def _run_e3_arm(signed: bool, seed: int, budget: int) -> dict:
    from .ibf_asi import IBFASI as _A
    from .ibf_asi_regimes import make_world
    w = make_world("deceptive", seed=seed)
    if signed:
        from .ibf_ultra import UltraASI
        a = UltraASI(w, seed=100 + seed)
    else:
        # the matched twin: identical policy economics (lean, no restart
        # teleports, planner on) -- ONLY the memory law differs. The first
        # version of this arm left two_sided_k at its default (True), which
        # confounded the comparison with restart policy; caught and fixed
        # before the full run.
        a = _A(w, seed=100 + seed, model_planner=True, horizon=1,
               anneal_steps=True, two_sided_k=False)
    xs = []
    while w.evals < budget:
        if w.tick():
            a.apply_shock()
        a.step()
        xs.append(np.asarray(a.x, float).copy())
    q = max(len(xs) // 4, 1)
    out = {"true_tail": float(np.mean([t["true"] for t in a.telemetry[-q:]])),
           "decoy_dwell": _decoy_dwell(w, xs)}
    if signed:
        eng = a.engine
        neg = eng.V < -1e-3
        neg_mass_decoys = 0.0
        if neg.any():
            Zn, Vn = eng.Z[neg], np.abs(eng.V[neg])
            at_decoy = np.zeros(len(Zn), dtype=bool)
            for c in w.c[1:]:
                at_decoy |= np.sum((Zn - c) ** 2, axis=1) < 1.2 ** 2
            neg_mass_decoys = float(Vn[at_decoy].sum() / max(Vn.sum(), 1e-9))
        out["neg_at_decoys"] = neg_mass_decoys
        out["n_neg"] = int(neg.sum())
    return out


@episode("e3")
def e3_mirage_field(out: str = REPORT_DIR, n_seeds: int = 8, budget: int = 4000,
                    render: bool = False, render_seed: int = 0,
                    quick: bool = False) -> dict:
    """E3 -- "It paints an X on every lie it has personally checked."

    The deceptive world: four mirages nearly as tall as the real oasis. The
    signed agent (ULTRA: the classic engine as memory organ) can write
    NEGATIVE corrections where experience disappointed; non-negative memory
    (Thm-8a special case) structurally cannot.

    PRE-REGISTERED, two-level honesty (ancestors 10.1 / 10.2):
      * navigator level (ancestor 10.2): net coherence difference signed vs
        matched non-negative twin expected to be a NULL (+0.09 [-0.27, +0.45]
        ns) -- this episode is the showcase's flagship "what doesn't help
        here, and why" panel. Checked-mirage dwell registered directional
        (signed lower); reported MET / NOT MET with its CI either way.
      * evaluator level (ancestor 10.1 Arena 1, re-demonstrated live): where
        suppression is structurally REQUIRED, signed memory is load-bearing:
        the non-negative substrate forgets significantly more under exact
        contradiction (ancestor +0.32 [+0.13, +0.51] sig). Asserted.
    """
    if quick:
        n_seeds, budget = 3, 2500
    sig = [_run_e3_arm(True, s, budget) for s in range(n_seeds)]
    non = [_run_e3_arm(False, s, budget) for s in range(n_seeds)]
    # the evaluator half (10.1 Arena 1, reduced protocol)
    from .ibf_classic_vs_asi import ASIMemoryAdapter, ClassicAdapter, arena1_run
    n1 = 3 if quick else 6
    a1_cls = [arena1_run(ClassicAdapter(seed=s, crucible=False), s)
              for s in range(n1)]
    a1_asi = [arena1_run(ASIMemoryAdapter(seed=s), s) for s in range(n1)]
    f_cls = [r[0][0] - r[2][0] for r in a1_cls]
    f_asi = [r[0][0] - r[2][0] for r in a1_asi]
    ci_forget = paired_ci(f_asi, f_cls)
    ci_dwell = paired_ci([r["decoy_dwell"] for r in sig],
                         [r["decoy_dwell"] for r in non])
    ci_true = paired_ci([r["true_tail"] for r in sig],
                        [r["true_tail"] for r in non])
    neg_at = float(np.mean([r["neg_at_decoys"] for r in sig]))
    n_neg = float(np.mean([r["n_neg"] for r in sig]))
    print(f"  [E3] checked-mirage dwell: signed {np.mean([r['decoy_dwell'] for r in sig]):.3f}"
          f" vs nonneg {np.mean([r['decoy_dwell'] for r in non]):.3f}"
          f"   diff {fmt_ci(ci_dwell)} {verdict(ci_dwell)}")
    print(f"       net coherence: {fmt_ci(ci_true)} {verdict(ci_true)} "
          f"(ancestor expects ns)")
    print(f"       X-marks: {n_neg:.0f} negative particles/run, "
          f"{100 * neg_at:.0f}% of negative mass sits on mirages")
    dwell_met = ci_dwell["mean"] < 0
    print(f"       dwell pre-registration (signed lower, directional): "
          f"{'MET' if dwell_met else 'NOT MET'}")
    print(f"       evaluator arena (10.1): extra forgetting of nonneg substrate "
          f"{fmt_ci(ci_forget)} {verdict(ci_forget)}")
    if ci_true["lo"] > 0 or ci_true["hi"] < 0:
        print("       !! navigator net coherence came out SIGNIFICANT -- "
              "contradicts the 10.2 ancestor null; investigate before showing")
    if not quick:
        assert ci_forget["lo"] > 0, \
            "E3 PRE-REG: evaluator arena must re-demonstrate 10.1 (sig)"

    assets = []
    if render:
        assets = _render_e3(out, render_seed, budget)
    panel = {
        "title": "Episode 3 — The Mirage Field",
        "caption": "It paints an X on every lie it has personally checked.",
        "novice": [
            f"time at already-checked mirages: signed "
            f"{100 * np.mean([r['decoy_dwell'] for r in sig]):.0f}% vs "
            f"forgiving twin {100 * np.mean([r['decoy_dwell'] for r in non]):.0f}% "
            f"({'the ink helps' if dwell_met else 'the ink does NOT buy escape here'})",
            f"where the red ink saves lives: the judge's arena — the forgiving "
            f"memory forgets {np.mean(f_asi):.2f} of what it knew under "
            f"contradiction; the signed one {np.mean(f_cls):.2f}",
        ],
        "backstage": [
            f"NAVIGATOR (the honest null, ancestor §10.2 kept): net coherence "
            f"{fmt_ci(ci_true)} {verdict(ci_true)}; checked-mirage dwell "
            f"{fmt_ci(ci_dwell)} {verdict(ci_dwell)} — directional registration "
            f"{'MET' if dwell_met else 'NOT MET'}. In open navigation, marking "
            f"lies is decoration: Boltzmann exploration already leaves them.",
            f"EVALUATOR (ancestor §10.1, re-demonstrated): the non-negative "
            f"substrate's EXTRA forgetting under exact contradiction "
            f"{fmt_ci(ci_forget)} {verdict(ci_forget)} — suppression is "
            f"structurally impossible for non-negative memory (Thm-8a special "
            f"case); signed corrections are load-bearing exactly there.",
        ],
        "assets": assets,
        "mechanism": "SIGNED corrections (Postulate IV) — and their habitat map",
    }
    return {"panel": panel, "ci": {"dwell": ci_dwell, "true": ci_true,
                                   "forget": ci_forget}}


def _render_e3(out: str, seed: int, budget: int) -> list[str]:
    from .ibf_asi_regimes import make_world
    from .ibf_ultra import UltraASI
    os.makedirs(ASSET_DIR, exist_ok=True)
    w = make_world("deceptive", seed=seed)
    a = UltraASI(w, seed=100 + seed)
    rec = Recorder(w)
    rec.refresh_terrain()
    trail: list = []
    frames = []
    t0 = rec._terrain
    vlim = (float(t0.min()) - 0.5, float(t0.max()) + 0.5)
    tick = 0
    while w.evals < budget:
        if w.tick():
            a.apply_shock()
        a.step()
        trail.append(np.asarray(a.x, float).copy())
        tick += 1
        if tick % 4:
            continue
        n_neg = int((a.engine.V < -1e-3).sum())
        rec.snap(a, meters={"X marks": n_neg,
                            "coherence": f"{a.telemetry[-1]['true']:.2f}"})
        frames.append(render_frame(rec.snaps[-1], w, title="THE MIRAGE FIELD",
                                   trail=trail[-60:], vlim=vlim))
    gif = save_gif(frames, os.path.join(ASSET_DIR, "e3_mirage_field.gif"))
    png = save_png(frames[-1], os.path.join(ASSET_DIR, "e3_mirage_final.png"))
    print(f"       assets: {gif['path']} ({gif['bytes'] / 1e6:.2f} MB)")
    return [gif["path"], png]


# ---------------------------------------------------------------------------
#  E4 -- THE CANYON
# ---------------------------------------------------------------------------

def _run_e4_arm(planner: bool, seed: int, budget: int = 4000) -> dict:
    from .ibf_asi import IBFASI as _A
    from .ibf_asi_regimes import make_world
    w = make_world("corridor", seed=seed)
    a = _A(w, seed=100 + seed, model_planner=planner, n_jumps=0,
           two_sided_k=False)
    a.x = np.array([-5.0])
    r = a.run(budget)
    # CorridorWorld has a single centre, so in_global_basin is vacuous; the
    # goal basin is x > 2.0 (the peak at 3.5, past the wide valley)
    return {"true_tail": r["true_tail"], "goal": float(a.x[0] > 2.0)}


@episode("e4")
def e4_canyon(out: str = REPORT_DIR, n_seeds: int = 10, budget: int = 4000,
              render: bool = False, render_seed: int = 0,
              quick: bool = False) -> dict:
    """E4 -- "It walks DOWN into the canyon because it knows what's beyond."

    The corridor (9.2): a start mound, a tempting trap hill, a wide low valley,
    and a taller goal beyond it. No teleports -- crossing must be WALKED. The
    planner + U8 option commitment is the only equipment that crosses; without
    it the agent is trap-locked by its own memory homing.

    PRE-REGISTERED (re-demonstration of 9.2 / the starred corridor cell):
    planner-vs-no-planner paired tail coherence CI-significantly positive with
    lo > 0.5 (ancestor +1.40*, goal 10/10 vs trap-locked 10/10).
    """
    if quick:
        n_seeds, budget = 3, 3000
    plan = [_run_e4_arm(True, s, budget) for s in range(n_seeds)]
    nopl = [_run_e4_arm(False, s, budget) for s in range(n_seeds)]
    ci = paired_ci([r["true_tail"] for r in plan], [r["true_tail"] for r in nopl])
    g_p = sum(r["goal"] for r in plan)
    g_n = sum(r["goal"] for r in nopl)
    print(f"  [E4] goal reached: planner {g_p:.0f}/{n_seeds} vs "
          f"trap-locked {g_n:.0f}/{n_seeds}")
    print(f"       tail coherence diff {fmt_ci(ci)} {verdict(ci)} "
          f"(ancestor +1.40*)")
    if quick:
        assert ci["mean"] > 0, "E4 quick: planner must lead (directional)"
    else:
        assert ci["lo"] > 0.5, \
            "E4 PRE-REG: corridor must stay decisively bound (ancestor 9.2)"

    assets = []
    if render:
        assets = _render_e4(out, render_seed, budget=2500)
    panel = {
        "title": "Episode 4 — The Canyon",
        "caption": "It walks DOWN into the canyon because it knows what's beyond.",
        "novice": [
            f"reached the far side: with a plan {g_p:.0f}/{n_seeds} journeys; "
            f"without {g_n:.0f}/{n_seeds} (stuck on the tempting little hill)",
        ],
        "backstage": [
            f"planner-vs-none tail coherence, paired: {fmt_ci(ci)} {verdict(ci)} "
            f"— ancestor §9.2 corridor cell (+1.40*, the only starred positive "
            f"planning cell in the matrix)",
            "honest scope (ancestor kept): planning binds ONLY in this discrete "
            "low-branching structure — in open 2-D terrain both pre-registered "
            "planning criteria came back null (§9.2); see the “what doesn’t "
            "help” panel.",
        ],
        "assets": assets,
        "mechanism": "model-based planning + U8 option commitment",
    }
    return {"panel": panel, "ci": {"tail": ci}, "goal": (g_p, g_n)}


def _render_e4(out: str, seed: int, budget: int) -> list[str]:
    from .ibf_asi import IBFASI as _A
    from .ibf_asi_regimes import make_world
    from .terrarium import render_profile_frame, vstack_images
    os.makedirs(ASSET_DIR, exist_ok=True)
    arms = {"WITH A PLAN": True, "NO PLAN (trap-locked)": False}
    W, A, REC, TRAIL = {}, {}, {}, {}
    for n, planner in arms.items():
        W[n] = make_world("corridor", seed=seed)
        A[n] = _A(W[n], seed=100 + seed, model_planner=planner, n_jumps=0,
                  two_sided_k=False)
        A[n].x = np.array([-5.0])
        REC[n] = Recorder(W[n])
        TRAIL[n] = []
    frames, tick = [], 0
    while any(W[n].evals < budget for n in arms):
        for n in arms:
            if W[n].evals >= budget:
                continue
            W[n].tick()
            A[n].step()
            TRAIL[n].append(np.asarray(A[n].x, float).copy())
        tick += 1
        if tick % 3:
            continue
        panels = []
        for n in arms:
            REC[n].snap(A[n], meters={
                "position": f"{float(A[n].x[0]):+.1f}",
                "committed": "YES" if A[n].option_target is not None else "no"})
            panels.append(render_profile_frame(
                REC[n].snaps[-1], W[n], title=n, trail=TRAIL[n][-80:]))
        frames.append(vstack_images(panels))
    gif = save_gif(frames, os.path.join(ASSET_DIR, "e4_canyon.gif"))
    png = save_png(frames[-1], os.path.join(ASSET_DIR, "e4_canyon_final.png"))
    print(f"       assets: {gif['path']} ({gif['bytes'] / 1e6:.2f} MB)")
    return [gif["path"], png]


# ---------------------------------------------------------------------------
#  E5 -- THE TRADE
# ---------------------------------------------------------------------------

def _run_e5_mode(mode: str, seed: int, ticks: int = 400) -> tuple:
    from .ibf_asi import ASIWorld, IBFASI as _A
    w = ASIWorld(seed=seed, dim=3, n_decoys=8, noise=0.35, drift=0.03)
    w.A = np.array([3.0] + [1.5] * 8)
    w.S = np.array([0.7] + [1.1] * 8)
    A = _A(w, seed=300 + seed)
    B = _A(w, seed=400 + seed)
    if mode != "solo":
        A.partner, B.partner = B, A
    if mode == "parasitic":
        B.give_transfer = False
    for _ in range(ticks):
        w.tick()
        A.step()
        B.step()
    ta = float(np.mean([t["true"] for t in A.telemetry[-100:]]))
    tb = float(np.mean([t["true"] for t in B.telemetry[-100:]]))
    return ta, tb, A.given, B.given


@episode("e5")
def e5_trade(out: str = REPORT_DIR, n_seeds: int = 24,
             render: bool = False, render_seed: int = 2,
             quick: bool = False) -> dict:
    """E5 -- "They trade maps — until one stops giving."

    Two agents in the same scarce-information world (3-D, narrow optimum,
    drifting) exchange crystallised discoveries under a RECIPROCITY LEDGER
    (credit-gated: A keeps giving only while B reciprocates).

    PRE-REGISTERED (re-demonstration of 9 V5 / 6.4 at ancestor power, 24
    seeds): cooperation beats solo for the pair (ancestor +0.54 [+0.01, +1.08]
    sig — assert mean > 0.2 as the suite does); defection must not pay
    (directional). HONEST SCOPE carried: in 2-D the same claim is a NULL
    (solo discovery is cheap; sharing is worthless) — both readings kept.
    """
    if quick:
        n_seeds = 4
    res = {m: [_run_e5_mode(m, s) for s in range(n_seeds)]
           for m in ("cooperative", "parasitic", "solo")}
    J = {m: [r[0] + r[1] for r in rs] for m, rs in res.items()}
    B_ = {m: [r[1] for r in rs] for m, rs in res.items()}
    ci_js = paired_ci(J["cooperative"], J["solo"])
    ci_bb = paired_ci(B_["cooperative"], B_["parasitic"])
    gifts = float(np.mean([r[2] + r[3] for r in res["cooperative"]]))
    gifts_para = float(np.mean([r[2] + r[3] for r in res["parasitic"]]))
    print(f"  [E5] joint coherence: coop {np.mean(J['cooperative']):.2f}  "
          f"parasitic {np.mean(J['parasitic']):.2f}  solo {np.mean(J['solo']):.2f}")
    print(f"       coop-solo (joint) {fmt_ci(ci_js)} {verdict(ci_js)}   "
          f"coopB-paraB {fmt_ci(ci_bb)} {verdict(ci_bb)}")
    print(f"       gifts exchanged: cooperative {gifts:.1f}/pair vs parasitic "
          f"{gifts_para:.1f} (the ledger cuts the parasite off)")
    if quick:
        print("       [quick] indicative only: the ancestor effect needs 24 "
              "seeds in this regime; no assert")
    else:
        assert ci_js["mean"] > 0.2, \
            "E5 PRE-REG: cooperation must beat solo for the pair (suite level)"
        assert ci_bb["mean"] >= -0.05, "E5 PRE-REG: defection must not pay"

    assets = []
    if render:
        assets = _render_e5(out, render_seed)
    panel = {
        "title": "Episode 5 — The Trade",
        "caption": "They trade maps — until one stops giving.",
        "novice": [
            f"a trading pair lives better than two loners: joint coherence "
            f"{np.mean(J['cooperative']):.2f} vs {np.mean(J['solo']):.2f}",
            f"free-riding does not pay: the moocher ends at "
            f"{np.mean(B_['parasitic']):.2f} vs {np.mean(B_['cooperative']):.2f} "
            f"as an honest trader — the ledger cuts gifts to "
            f"{gifts_para:.1f} (vs {gifts:.1f})",
        ],
        "backstage": [
            f"coop − solo (joint), paired over {n_seeds} seeds: {fmt_ci(ci_js)} "
            f"{verdict(ci_js)} — ancestor §9 V5 / 6.4 (+0.54 [+0.01, +1.08] "
            f"sig in this scarce-information regime)",
            f"defector's payoff: {fmt_ci(ci_bb)} {verdict(ci_bb)} — defection "
            f"neither pays nor costs (ancestor kept)",
            "honest scope (ancestor kept): in 2-D the cooperation claim is a "
            "NULL — solo discovery is cheap there and sharing is worthless. "
            "Cooperation earns its keep only where information is scarce.",
        ],
        "assets": assets,
        "mechanism": "reciprocity-ledgered transfer (6.4, EC-4)",
    }
    return {"panel": panel, "ci": {"joint": ci_js, "defect": ci_bb}}


class _Slice2D:
    """A 2-D presentation slice of a 3-D world (terrain at the optimum's z)."""

    def __init__(self, w3) -> None:
        self.w3 = w3
        self.dim = 2
        self.lo, self.hi = w3.lo[:2], w3.hi[:2]
        self.noise = w3.noise

    def true_coherence(self, xy) -> float:
        z = self.w3.c[0][2]
        return self.w3.true_coherence(np.array([xy[0], xy[1], z]))


def _render_e5(out: str, seed: int, ticks: int = 400) -> list[str]:
    from .ibf_asi import ASIWorld, IBFASI as _A
    os.makedirs(ASSET_DIR, exist_ok=True)
    w = ASIWorld(seed=seed, dim=3, n_decoys=8, noise=0.35, drift=0.03)
    w.A = np.array([3.0] + [1.5] * 8)
    w.S = np.array([0.7] + [1.1] * 8)
    A = _A(w, seed=300 + seed)
    B = _A(w, seed=400 + seed)
    A.partner, B.partner = B, A
    slice2d = _Slice2D(w)
    rec = Recorder(slice2d)
    rec.refresh_terrain()
    trailA, trailB, frames = [], [], []
    given_last = (0, 0)
    for t in range(ticks):
        w.tick()
        A.step()
        B.step()
        trailA.append(np.asarray(A.x[:2], float).copy())
        trailB.append(np.asarray(B.x[:2], float).copy())
        if t % 3:
            continue
        if t % 24 == 0:
            rec.refresh_terrain()                  # the world drifts
        events = []
        if (A.given, B.given) != given_last:
            events.append(">> MAP TRADED <<")
            given_last = (A.given, B.given)
        rec.snap(A, events=events, meters={
            "ledger": f"A gave {A.given} | B gave {B.given}",
            "joint": f"{A.telemetry[-1]['true'] + B.telemetry[-1]['true']:.2f}"})
        snap = rec.snaps[-1]
        snap.x = np.asarray(A.x[:2], float).copy()
        img = render_frame(snap, slice2d, title="THE TRADE (slice view)",
                           trail=trailA[-60:])
        frames.append(img)
    gif = save_gif(frames, os.path.join(ASSET_DIR, "e5_trade.gif"))
    png = save_png(frames[-1], os.path.join(ASSET_DIR, "e5_trade_final.png"))
    print(f"       assets: {gif['path']} ({gif['bytes'] / 1e6:.2f} MB)")
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
