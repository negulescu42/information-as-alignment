"""
run_gate1c.py
=============

Gate 1C -- Scale 1 behavioral localization via discrepancy-driven splitting.

Blocker carried from Gate 1B: Scale 1 particles are too broad in observation
space; each particle's activation ball spans most of the u2 range, so its
behavioral signature averages over contradictory rewards and collapses to a
function of u1. This experiment tests whether discrepancy-driven splitting
(`enable_splitting`) lets particles localize -- and signatures encode -- the
behaviorally aliased coordinate u2.

It sweeps the splitting ablation (off / on) across aliasing frequencies and
reports, per seed:

    -- Scale 1 localization (the leading indicator):
       signature->u2 (regression and pairwise), per-particle u2 variance,
       n_particles, n_crystallized, n_splits
    -- end-to-end emergent recovery: rho_struct, rho_u1, rho_u2 (best graph mode)
    -- generator fairness: supervised x20->u2 ceiling (can ANY static encoder
       recover u2 at test time?)
    -- Scale 2 (headline frequency only): ACC_oracle, ACC_emergent, accuracy_gap

Decision context (supervisor):
    leading indicator: signature->u2 > 0.3  -> Scale 1 blocker broken
    pass:              emergent rho_u2 > 0.5 then rho_struct > 0.8 & gap < 0.15

Usage:
    python run_gate1c.py [n_seeds]
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, cdist
from sklearn.neighbors import KNeighborsRegressor

from ibf_v1_engine import C
from gate1_environment import TwoScaleToyEnvironment, Gate1Config
from gate1_encoders import Oracle2DEncoder
from scale1_representation import Scale1RepresentationLearner
from run_gate1 import run_scale2_v1, manifold_rho
from run_gate1b_variants import coord_recovery

OUT_DIR = "gate1c_outputs"
FREQS = [3.0, 2.0, 1.5]
GRAPH_MODES = ["multiplicative", "union"]
SCALE2_FREQ = 1.5          # headline frequency for ACC / gap
RHO_THRESHOLD = 0.8
RHO_U2_GATE = 0.5
ACC_GAP_THRESHOLD = 0.15


def supervised_u2_ceiling(env):
    """Can any static encoder recover u2 from x20? kNN regression with truth."""
    r = KNeighborsRegressor(10).fit(env.pool_x20, env.pool_u2[:, 1]).predict(env.test_A_x20)
    return float(spearmanr(r, env.test_A_u2[:, 1]).correlation)


def scale1_diagnostics(s1, env):
    parts = s1.get_crystallized_particles()
    if len(parts) < 5:
        return dict(sig_u1_reg=float("nan"), sig_u2_reg=float("nan"),
                    sig_u2_pair=float("nan"), per_particle_u2_var=float("nan"))
    X = np.array([p.x for p in parts])
    B = np.array([p.signature for p in parts])
    nn = np.argmin(cdist(X, env.pool_x20), axis=1)
    Up = env.pool_u2[nn]
    vars = []
    for p in parts[:80]:
        d = np.sum((env.pool_x20 - p.x) ** 2, 1)
        m = np.exp(-d / (2 * p.sigma ** 2)) > 0.15
        if m.sum() > 3:
            vars.append(float(np.var(env.pool_u2[m, 1])))
    return dict(
        sig_u1_reg=coord_recovery(B, Up[:, 0], 0),
        sig_u2_reg=coord_recovery(B, Up[:, 1], 0),
        sig_u2_pair=float(spearmanr(pdist(B), pdist(Up[:, 1:2])).correlation),
        per_particle_u2_var=float(np.median(vars)) if vars else float("nan"))


def run_cell(freq, split, seed, with_scale2=False):
    cfg = Gate1Config(generator="1B", u2_freq=freq, N_repr_pool=1500,
                      N_train_pool=600, N_test=800, E_scale1=30, E_scale2=20,
                      sigma_x_scale=0.7, enable_splitting=split, split_threshold=0.12)
    env = TwoScaleToyEnvironment(seed, cfg)
    s1 = Scale1RepresentationLearner(cfg, seed, enable_crystallization=True)
    s1.fit(env, verbose=False)
    parts = s1.get_crystallized_particles()
    diag = scale1_diagnostics(s1, env)

    sub = np.random.RandomState(seed + 5).choice(cfg.N_test, 300, replace=False)
    Xt, Ut = env.test_A_x20[sub], env.test_A_u2[sub]
    emergent = {}
    for mode in GRAPH_MODES:
        s1.graph_mode = mode
        qt = s1.make_encoder(np.eye(C.k), parts).encode_observation_batch(Xt)
        emergent[mode] = dict(
            rho_struct=float(spearmanr(pdist(qt), pdist(Ut)).correlation),
            rho_u1=coord_recovery(qt, Ut[:, 0], 0),
            rho_u2=coord_recovery(qt, Ut[:, 1], 0))
    best_mode = max(emergent, key=lambda m: emergent[m]["rho_struct"])

    row = dict(freq=freq, split=split, seed=seed,
               n_particles=len(s1.particles), n_crystallized=len(parts),
               n_splits=s1.n_splits, x_u2_ceiling=supervised_u2_ceiling(env),
               diagnostic=diag, emergent=emergent, best_mode=best_mode,
               rho_struct=emergent[best_mode]["rho_struct"],
               rho_u1=emergent[best_mode]["rho_u1"],
               rho_u2=emergent[best_mode]["rho_u2"])

    if with_scale2:
        s1.graph_mode = best_mode
        enc = s1.make_encoder(np.eye(C.k), parts)
        em, _, _ = run_scale2_v1(enc, env.train_x20, env.train_u2,
                                 env.test_A_x20, env.test_A_u2,
                                 env.test_B_x20, env.test_B_u2, env, seed, cfg.E_scale2)
        oracle_enc = Oracle2DEncoder(np.eye(C.k))
        om, _, _ = run_scale2_v1(oracle_enc, env.train_u2, env.train_u2,
                                 env.test_A_u2, env.test_A_u2, env.test_B_u2, env.test_B_u2,
                                 env, seed, cfg.E_scale2)
        row["ACC_oracle"] = om["ACC_gate"]
        row["ACC_emergent"] = em["ACC_gate"]
        row["accuracy_gap"] = om["ACC_gate"] - em["ACC_gate"]
    return row


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate1c_results.json")
    print("Gate 1C | freqs=%s | split=[off,on] | seeds=%d" % (FREQS, n_seeds))
    rows = []
    for freq in FREQS:
        for split in (False, True):
            for seed in range(n_seeds):
                with_s2 = (abs(freq - SCALE2_FREQ) < 1e-9)
                r = run_cell(freq, split, seed, with_scale2=with_s2)
                rows.append(r)
                d = r["diagnostic"]
                print("  f=%.1f split=%-5s s%d: parts=%d cryst=%d splits=%d | "
                      "sig_u2(reg)=%.3f per_u2var=%.3f | emergent[%s] rho=%.3f u1=%.3f u2=%.3f | x->u2=%.3f%s"
                      % (freq, split, seed, r["n_particles"], r["n_crystallized"],
                         r["n_splits"], d["sig_u2_reg"], d["per_particle_u2_var"],
                         r["best_mode"], r["rho_struct"], r["rho_u1"], r["rho_u2"],
                         r["x_u2_ceiling"],
                         (" gap=%+.3f" % r["accuracy_gap"]) if "accuracy_gap" in r else ""))
            with open(out, "w") as f:
                json.dump(dict(freqs=FREQS, graph_modes=GRAPH_MODES,
                               rho_threshold=RHO_THRESHOLD, rho_u2_gate=RHO_U2_GATE,
                               acc_gap_threshold=ACC_GAP_THRESHOLD, rows=rows), f, indent=2)
            print("  [checkpoint] freq=%.1f split=%s saved" % (freq, split))
    print("\nSaved %s" % out)


if __name__ == "__main__":
    main()
