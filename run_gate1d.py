"""
run_gate1d.py
=============

Gate 1D -- interactive encoding.

Gate 1C proved discrepancy-driven splitting makes crystallized signatures encode
the aliased coordinate (the information is INSIDE the system), but a static
observation->representation map cannot get it OUT at test time: supervised
x20->u2 has ceiling ~0.04 at u2_freq=3.0.

Gate 1D replaces the static x20->q projection with an INTERACTIVE encoder that
takes n_probes exploratory actions, observes rewards, and folds the resulting
behavioral signature into the representation:

    q2 = [ static geometric projection(x20) , interactive probe signature ]

Conditions per (freq, seed):
    static       -- Gate 1C emergent encoder (baseline control)
    interactive  -- q2 above, n_probes probes over both contexts (Candidate A)
    oracle       -- true [u1, u2] (upper bound control)

Reports rho_struct, rho_u1, rho_u2 and Scale 2 ACC / accuracy_gap.
Scale 1 (splitting), Scale 2 engine, and the 1B generator are unchanged.

Run plan:
    Run 1  sanity   : f=1.5, {static, interactive, oracle}, 5 seeds
    Run 2  headline : f=3.0, {static, interactive, oracle}, 5 seeds
    Run 3  sweep    : f=3.0, interactive, n_probes in {1,3,5,10,20}, 3 seeds (rho only)

Usage:
    python run_gate1d.py [n_seeds]      # default 5 (sweep uses min(n_seeds,3))
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
from gate1_environment import TwoScaleToyEnvironment, Gate1Config
from gate1_encoders import PassThroughEncoder, build_interactive_coords
from scale1_representation import Scale1RepresentationLearner
from run_gate1 import run_scale2_v1, manifold_rho
from run_gate1b_variants import coord_recovery

OUT_DIR = "gate1d_outputs"
RHO_THRESHOLD = 0.8
RHO_U2_GATE = 0.5
ACC_GAP_THRESHOLD = 0.15
N_PROBES_DEFAULT = 5
SWEEP_PROBES = [1, 3, 5, 10, 20]

_FIT_CACHE = {}


def fit_scale1(freq, seed):
    key = (freq, seed)
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    cfg = Gate1Config(generator="1B", u2_freq=freq, N_repr_pool=1500,
                      N_train_pool=600, N_test=800, E_scale1=30, E_scale2=20,
                      sigma_x_scale=0.7, enable_splitting=True, split_threshold=0.12)
    env = TwoScaleToyEnvironment(seed, cfg)
    s1 = Scale1RepresentationLearner(cfg, seed, enable_crystallization=True,
                                     graph_mode="multiplicative")
    s1.fit(env, verbose=False)
    _FIT_CACHE[key] = (cfg, env, s1)
    return _FIT_CACHE[key]


def make_coords(encoder_type, env, s1, n_probes, seed, X20, U):
    if encoder_type == "oracle":
        return U.copy()
    parts = s1.get_crystallized_particles()
    static_enc = s1.make_encoder(np.eye(C.k), parts)
    if encoder_type == "static":
        return static_enc.encode_observation_batch(X20)
    if encoder_type == "interactive":
        return build_interactive_coords(static_enc, X20, U, env, C.k, n_probes, seed)
    raise ValueError(encoder_type)


def eval_rho(encoder_type, env, s1, n_probes, seed):
    cA = make_coords(encoder_type, env, s1, n_probes, seed, env.test_A_x20, env.test_A_u2)
    sub = np.random.RandomState(seed + 5).choice(len(cA), 300, replace=False)
    if encoder_type == "oracle":
        return dict(rho_struct=1.0, rho_u1=1.0, rho_u2=1.0)
    return dict(
        rho_struct=float(spearmanr(pdist(cA[sub]), pdist(env.test_A_u2[sub])).correlation),
        rho_u1=coord_recovery(cA[sub], env.test_A_u2[sub, 0], seed),
        rho_u2=coord_recovery(cA[sub], env.test_A_u2[sub, 1], seed))


def run_full(encoder_type, freq, seed, n_probes):
    cfg, env, s1 = fit_scale1(freq, seed)
    rho = eval_rho(encoder_type, env, s1, n_probes, seed)
    cP = make_coords(encoder_type, env, s1, n_probes, seed, env.train_x20, env.train_u2)
    cA = make_coords(encoder_type, env, s1, n_probes, seed, env.test_A_x20, env.test_A_u2)
    cB = make_coords(encoder_type, env, s1, n_probes, seed, env.test_B_x20, env.test_B_u2)
    enc = PassThroughEncoder(np.eye(C.k))
    m, _, _ = run_scale2_v1(enc, cP, env.train_u2, cA, env.test_A_u2,
                            cB, env.test_B_u2, env, seed, cfg.E_scale2)
    return dict(encoder=encoder_type, freq=freq, seed=seed, n_probes=n_probes,
                coord_dim=int(cA.shape[1]), ACC=m['ACC_gate'],
                n_scale2_cryst=m['n_scale2_cryst'], **rho)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sweep_seeds = min(n_seeds, 3)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate1d_results.json")
    rows = []

    def save():
        with open(out, "w") as f:
            json.dump(dict(rho_threshold=RHO_THRESHOLD, rho_u2_gate=RHO_U2_GATE,
                           acc_gap_threshold=ACC_GAP_THRESHOLD,
                           n_probes_default=N_PROBES_DEFAULT,
                           sweep_probes=SWEEP_PROBES, rows=rows), f, indent=2)

    for tag, freq in [("Run1-sanity", 1.5), ("Run2-headline", 3.0)]:
        print("\n=== %s (f=%.1f) ===" % (tag, freq))
        for seed in range(n_seeds):
            for enc in ["static", "interactive", "oracle"]:
                r = run_full(enc, freq, seed, N_PROBES_DEFAULT)
                r["run"] = tag
                rows.append(r)
                print("  f=%.1f s%d %-11s dim=%2d rho_struct=%.3f rho_u1=%.3f "
                      "rho_u2=%.3f ACC=%.3f" % (freq, seed, enc, r['coord_dim'],
                      r['rho_struct'], r['rho_u1'], r['rho_u2'], r['ACC']))
            save()

    print("\n=== Run3-sweep (f=3.0, interactive, probe budget) ===")
    for seed in range(sweep_seeds):
        for nprobes in SWEEP_PROBES:
            cfg, env, s1 = fit_scale1(3.0, seed)
            rho = eval_rho("interactive", env, s1, nprobes, seed)
            r = dict(run="Run3-sweep", encoder="interactive", freq=3.0, seed=seed,
                     n_probes=nprobes, coord_dim=2 + C.k * 2, ACC=None,
                     n_scale2_cryst=None, **rho)
            rows.append(r)
            print("  s%d n_probes=%2d rho_struct=%.3f rho_u1=%.3f rho_u2=%.3f"
                  % (seed, nprobes, r['rho_struct'], r['rho_u1'], r['rho_u2']))
        save()

    save()
    print("\nSaved %s" % out)


if __name__ == "__main__":
    main()
