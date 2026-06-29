"""
run_gate2_k8.py
===============

Gate 2 RETRY with a richer task: k=8 actions instead of k=2. The Interface
Principle was proved for DENSE fields; a 2-action toy produces correction fields
that are geometrically all-surface (Gate 2 k=2 FAILED: ~15% interior, not
shielded). Eight actions partition the (u1, u_c*u2) plane into 8 angular sectors,
so Scale 2 deposits corrections for 8 candidates per position -> a denser field
that should grow genuine interior.

ONLY THE TASK CHANGES. Same 20D observations, f=3.0 generator, interactive
encoder (Candidate A signature, now 8-D -> q2 = 2+8 = 10-D), same Scale 1 with
splitting, same Scale 2 engine, and the SAME Gate 2 analysis pipeline
(`analyze_trained_system`) with the same thresholds.

Run 1: validate the richer task trains (interactive accuracy_gap < 0.15). 3 seeds.
Run 2: Gate 2 analysis on the trained 8-action system. 5 seeds.

Usage:
    python run_gate2_k8.py [n_probes]      # default 5
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import ibf_v1_engine as E
E.C.k = 8                                     # engine now selects among 8 actions
E.ACTION_EMB = np.eye(8)
from ibf_v1_engine import C

from gate1_environment import TwoScaleToyEnvironment, Gate1Config
from gate1_encoders import PassThroughEncoder, build_interactive_coords
from scale1_representation import Scale1RepresentationLearner
from run_gate1 import run_scale2_v1
import run_gate2 as G2

OUT_DIR = "gate2_k8_outputs"
FREQ = 3.0
N_ACTIONS = 8
ACC_GAP_THRESHOLD = 0.15
_CACHE = {}


def fit_k8(seed, n_probes):
    key = (seed, n_probes)
    if key in _CACHE:
        return _CACHE[key]
    # Moderate config: k=8 Scale 2 deposits ~4x the centers of k=2, so N_train /
    # E_scale2 are trimmed to keep the (n_centers x n_actions) inner loop
    # tractable. The crystallized-center count (what Gate 2 analyses) stays well
    # above the k=2 baseline.
    cfg = Gate1Config(generator="1B", u2_freq=FREQ, k=N_ACTIONS, n_contexts=2,
                      N_repr_pool=800, N_train_pool=400, N_test=600,
                      E_scale1=20, E_scale2=15, sigma_x_scale=0.7,
                      enable_splitting=True, split_threshold=0.12)
    env = TwoScaleToyEnvironment(seed, cfg)
    s1 = Scale1RepresentationLearner(cfg, seed, enable_crystallization=True,
                                     graph_mode="multiplicative")
    s1.fit(env, verbose=False)
    parts = s1.get_crystallized_particles()
    geom = s1.make_encoder(np.eye(N_ACTIONS), parts)

    # interactive coords: single-context 8-slot probe signature -> q2 = 10-D
    ctx = ('A',)
    cP = build_interactive_coords(geom, env.train_x20, env.train_u2, env, N_ACTIONS, n_probes, seed, ctx)
    cA = build_interactive_coords(geom, env.test_A_x20, env.test_A_u2, env, N_ACTIONS, n_probes, seed, ctx)
    cB = build_interactive_coords(geom, env.test_B_x20, env.test_B_u2, env, N_ACTIONS, n_probes, seed, ctx)
    enc = PassThroughEncoder(np.eye(N_ACTIONS))
    m, _, agent = run_scale2_v1(enc, cP, env.train_u2, cA, env.test_A_u2,
                                cB, env.test_B_u2, env, seed, cfg.E_scale2, want_log=True)

    # oracle baseline (gap reference)
    om, _, _ = run_scale2_v1(PassThroughEncoder(np.eye(N_ACTIONS)),
                             env.train_u2, env.train_u2, env.test_A_u2, env.test_A_u2,
                             env.test_B_u2, env.test_B_u2, env, seed, cfg.E_scale2)
    res = dict(cfg=cfg, env=env, agent=agent, enc=enc, cA=cA, cB=cB,
               coord_dim=int(cA.shape[1]), ACC_interactive=m["ACC_gate"],
               ACC_oracle=om["ACC_gate"], n_cryst=m["n_scale2_cryst"],
               n_centers=len(agent.centers))
    _CACHE[key] = res
    return res


def main():
    # 8 actions need more probes than k=2's 5 to cover the action set: with 5
    # probes the winner is sampled only ~49% of the time, the signature is too
    # noisy to crystallize, and the gap fails (empirically +0.165). 15 probes
    # gives reliable coverage. (Inference cost scales with the action count.)
    n_probes = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    fig_dir = os.path.join(OUT_DIR, "basin_detection")
    for sub in ["basin_detection", "fidelity", "behavioral"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    out = os.path.join(OUT_DIR, "gate2_k8_results.json")

    # ---- Run 1: validate the richer task ----
    print("=== Run 1: validate k=8 interactive task (n_probes=%d) ===" % n_probes)
    run1 = []
    for seed in range(3):
        r = fit_k8(seed, n_probes)
        gap = r["ACC_oracle"] - r["ACC_interactive"]
        run1.append(dict(seed=seed, coord_dim=r["coord_dim"],
                         ACC_interactive=r["ACC_interactive"], ACC_oracle=r["ACC_oracle"],
                         accuracy_gap=gap, n_centers=r["n_centers"], n_cryst=r["n_cryst"]))
        print("  s%d dim=%d ACC_int=%.3f ACC_oracle=%.3f gap=%+.3f | centers=%d cryst=%d"
              % (seed, r["coord_dim"], r["ACC_interactive"], r["ACC_oracle"], gap,
                 r["n_centers"], r["n_cryst"]))
    med_gap = float(np.median([r["accuracy_gap"] for r in run1]))
    run1_ok = med_gap < ACC_GAP_THRESHOLD
    print("  Run1 median gap = %+.3f -> %s" % (med_gap, "OK" if run1_ok else "FAIL"))

    # ---- Run 2: Gate 2 analysis (same pipeline) ----
    print("\n=== Run 2: Gate 2 analysis on trained k=8 system (same code/thresholds) ===")
    G2.OUT_DIR = OUT_DIR
    rows = []
    for seed in range(5):
        r = fit_k8(seed, n_probes)
        res = G2.analyze_trained_system(r["agent"], r["enc"], r["cA"], r["cB"],
                                        r["env"], seed, r["cfg"].E_scale2, fig_dir)
        rows.append(res)
        if res.get("degenerate"):
            print("  s%d DEGENERATE (%s)" % (seed, res["degenerate"]))
        else:
            print("  s%d: cryst=%d basins=%d sizes=%s interior=%d (%.0f%%) | "
                  "ACC full=%.3f comp=%.3f delta=%+.3f"
                  % (seed, res["n_cryst"], res["n_basins"], res["basin_sizes"],
                     res["n_interior"], 100 * res["compression_ratio"],
                     res["ACC_full"], res["ACC_compressed"], res["ACC_delta"]))
        with open(out, "w") as f:
            json.dump(dict(freq=FREQ, n_actions=N_ACTIONS, n_probes=n_probes,
                           tau_basin=G2.TAU_BASIN, tau_face=G2.TAU_FACE,
                           rel_err_threshold=G2.REL_ERR_THRESHOLD,
                           acc_delta_threshold=G2.ACC_DELTA_THRESHOLD,
                           compression_threshold=G2.COMPRESSION_THRESHOLD,
                           run1=run1, run1_median_gap=med_gap, rows=rows), f, indent=2)
    print("\nSaved %s" % out)


if __name__ == "__main__":
    main()
