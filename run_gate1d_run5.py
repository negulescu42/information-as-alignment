"""
run_gate1d_run5.py
==================

Gate 1D -- Run 5 (official pass) + informed-vs-random probe control.

Pass criterion (supervisor): accuracy_gap < 0.15 at f=3.0, interactive encoder,
across 10 seeds (median AND every individual seed). rho_struct is reported but
NOT gating -- the operational content of Postulate 2 is whether the emergent
configuration space SUPPORTS the higher-scale correction dynamics.

Conditions per seed (f=3.0, n_probes=5):
    oracle                 -- true [u1,u2]            (gap reference)
    static                 -- Gate 1C emergent coords (also the scout agent)
    interactive_random     -- probes via uniform action selection
    interactive_informed   -- probes via Boltzmann over the scout's Scale 2
                              corrections (cross-scale agency)

Usage:
    python run_gate1d_run5.py [n_seeds]      # default 10
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

from ibf_v1_engine import C
from gate1_encoders import (PassThroughEncoder, build_interactive_coords,
                            build_interactive_coords_informed)
from run_gate1 import run_scale2_v1, manifold_rho
from run_gate1b_variants import coord_recovery
from run_gate1d import fit_scale1, N_PROBES_DEFAULT

OUT_DIR = "gate1d_outputs"
FREQ = 3.0
ACC_GAP_THRESHOLD = 0.15
RHO_U2_GATE = 0.5


def rho_block(coords_tA, u_tA, seed):
    sub = np.random.RandomState(seed + 5).choice(len(coords_tA), 300, replace=False)
    return dict(
        rho_struct=float(spearmanr(pdist(coords_tA[sub]), pdist(u_tA[sub])).correlation),
        rho_u1=coord_recovery(coords_tA[sub], u_tA[sub, 0], seed),
        rho_u2=coord_recovery(coords_tA[sub], u_tA[sub, 1], seed))


def scale2(coords_pool, coords_tA, coords_tB, env, seed, E2, want_agent=False):
    enc = PassThroughEncoder(np.eye(C.k))
    m, _, agent = run_scale2_v1(enc, coords_pool, env.train_u2, coords_tA, env.test_A_u2,
                                coords_tB, env.test_B_u2, env, seed, E2, want_log=want_agent)
    return m, enc, agent


def run_seed(seed, n_probes):
    cfg, env, s1 = fit_scale1(FREQ, seed)
    E2 = cfg.E_scale2
    parts = s1.get_crystallized_particles()
    geom = s1.make_encoder(np.eye(C.k), parts)

    qg_p = geom.encode_observation_batch(env.train_x20)
    qg_A = geom.encode_observation_batch(env.test_A_x20)
    qg_B = geom.encode_observation_batch(env.test_B_x20)

    # oracle reference
    om, _, _ = scale2(env.train_u2, env.test_A_u2, env.test_B_u2, env, seed, E2)
    acc_oracle = om["ACC_gate"]

    # static (the scout)
    sm, static_enc, scout = scale2(qg_p, qg_A, qg_B, env, seed, E2, want_agent=True)
    scout_aemb = static_enc.action_embedding

    # interactive -- random probes
    rp = build_interactive_coords(geom, env.train_x20, env.train_u2, env, C.k, n_probes, seed)
    rA = build_interactive_coords(geom, env.test_A_x20, env.test_A_u2, env, C.k, n_probes, seed)
    rB = build_interactive_coords(geom, env.test_B_x20, env.test_B_u2, env, C.k, n_probes, seed)
    rm, _, _ = scale2(rp, rA, rB, env, seed, E2)

    # interactive -- informed (agency-guided) probes
    ip = build_interactive_coords_informed(geom, scout, scout_aemb,
                                           env.train_x20, env.train_u2, env, C.k, n_probes, seed)
    iA = build_interactive_coords_informed(geom, scout, scout_aemb,
                                           env.test_A_x20, env.test_A_u2, env, C.k, n_probes, seed)
    iB = build_interactive_coords_informed(geom, scout, scout_aemb,
                                           env.test_B_x20, env.test_B_u2, env, C.k, n_probes, seed)
    im, _, _ = scale2(ip, iA, iB, env, seed, E2)

    rows = []
    for name, coords_tA, m in [("static", qg_A, sm),
                               ("interactive_random", rA, rm),
                               ("interactive_informed", iA, im)]:
        rb = rho_block(coords_tA, env.test_A_u2, seed)
        rows.append(dict(seed=seed, encoder=name, n_probes=n_probes,
                         coord_dim=int(coords_tA.shape[1]), ACC=m["ACC_gate"],
                         ACC_oracle=acc_oracle,
                         accuracy_gap=acc_oracle - m["ACC_gate"],
                         n_scale2_cryst=m["n_scale2_cryst"], **rb))
    rows.append(dict(seed=seed, encoder="oracle", n_probes=n_probes, coord_dim=2,
                     ACC=acc_oracle, ACC_oracle=acc_oracle, accuracy_gap=0.0,
                     n_scale2_cryst=om["n_scale2_cryst"],
                     rho_struct=1.0, rho_u1=1.0, rho_u2=1.0))
    return rows


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    n_probes = N_PROBES_DEFAULT
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate1d_run5_results.json")
    print("Gate 1D Run 5 | f=%.1f | n_probes=%d | seeds=%d | pass: accuracy_gap < %.2f"
          % (FREQ, n_probes, n_seeds, ACC_GAP_THRESHOLD))
    rows = []
    for seed in range(n_seeds):
        seed_rows = run_seed(seed, n_probes)
        rows.extend(seed_rows)
        d = {r["encoder"]: r for r in seed_rows}
        print("  s%d | random: gap=%+.3f rho_u2=%.3f ACC=%.3f | informed: gap=%+.3f "
              "rho_u2=%.3f ACC=%.3f | static gap=%+.3f"
              % (seed, d["interactive_random"]["accuracy_gap"], d["interactive_random"]["rho_u2"],
                 d["interactive_random"]["ACC"], d["interactive_informed"]["accuracy_gap"],
                 d["interactive_informed"]["rho_u2"], d["interactive_informed"]["ACC"],
                 d["static"]["accuracy_gap"]))
        with open(out, "w") as f:
            json.dump(dict(freq=FREQ, n_probes=n_probes,
                           acc_gap_threshold=ACC_GAP_THRESHOLD, rows=rows), f, indent=2)
    print("\nSaved %s" % out)


if __name__ == "__main__":
    main()
