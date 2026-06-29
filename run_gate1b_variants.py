"""
run_gate1b_variants.py
======================

Gate 1B behavioral-geometry branch.

Question: when ambient geometry is insufficient (Gate 1B generator), can a
behavior-capable graph construction let crystallized behavioral signatures
supply the representational neighborhood structure that geometry cannot?

Everything is held fixed versus Gate 1B except the **graph construction /
representation extraction** layer:

    same Gate 1B generator, same task, same thresholds, same v1 Scale 2 engine,
    same oracle baseline, same pass/fail criteria, same reporting.

Graph variants compared (all on the identical crystallized particles per seed):

    multiplicative   W = W_geometry * W_behavior * W_stability   (Gate 1A BASELINE,
                     kept for comparison -- do not delete)
    additive         W = (alpha*W_geometry + beta*W_behavior) * W_stability
    union            neighbors = kNN_geometry  UNION  kNN_behavior
    behavior_first   kNN from behavioral signatures; geometry only a regularizer

For each variant (per seed) we report:
    rho_struct, rho_u1, rho_u2, ACC_oracle, ACC_emergent, accuracy_gap,
    n_repr_crystallized, gate1_pass (both thresholds)

plus geometry-only controls (raw20D / PCA2D / spectral / random2D rho) and the
no-crystallization / shuffled-signature controls, and an upstream diagnostic:
does the crystallized signature distance encode u2 at all?

Usage:
    python run_gate1b_variants.py [n_seeds]      # default 5 seeds
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, cdist
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

from ibf_v1_engine import C
from gate1_environment import TwoScaleToyEnvironment, Gate1Config
from gate1_encoders import Oracle2DEncoder
from scale1_representation import Scale1RepresentationLearner
from run_gate1 import run_scale2_v1, manifold_rho

OUT_DIR = "gate1b_variants_outputs"
RHO_THRESHOLD = 0.8
ACC_GAP_THRESHOLD = 0.15
MODES = ["multiplicative", "additive", "union", "behavior_first"]


def coord_recovery(q, u_coord, seed):
    """5-fold kNN-regression recoverability of a single hidden coord from q."""
    if len(q) < 10:
        return float("nan")
    kf = KFold(5, shuffle=True, random_state=seed)
    preds = np.zeros(len(u_coord))
    for tr, te in kf.split(q):
        reg = KNeighborsRegressor(10, weights="distance").fit(q[tr], u_coord[tr])
        preds[te] = reg.predict(q[te])
    return float(spearmanr(preds, u_coord).correlation)


def geometry_controls(env, seed, n_sub=300):
    sub = np.random.RandomState(seed + 5).choice(env.cfg.N_test, n_sub, replace=False)
    X = env.test_A_x20[sub]
    U = env.test_A_u2[sub]
    Du = pdist(U)
    raw = spearmanr(pdist(X), Du).correlation
    pca = spearmanr(pdist(PCA(2).fit_transform(X)), Du).correlation
    spec = spearmanr(pdist(SpectralEmbedding(
        n_components=2, n_neighbors=15, random_state=seed).fit_transform(X)), Du).correlation
    rp = np.random.RandomState(seed + 1).randn(X.shape[1], 2)
    rnd = spearmanr(pdist((X - X.mean(0)) @ rp), Du).correlation
    return dict(raw20D=float(raw), PCA2D=float(pca), spectral=float(spec), random2D=float(rnd))


def signature_encodes_u2(s1, env):
    """Upstream diagnostic: do crystallized signature DISTANCES track u1 / u2?"""
    parts = s1.get_crystallized_particles()
    if len(parts) < 5:
        return dict(sig_vs_u1=float("nan"), sig_vs_u2=float("nan"),
                    per_particle_u2_var=float("nan"))
    X = np.array([p.x for p in parts])
    B = np.array([p.signature for p in parts])
    nn = np.argmin(cdist(X, env.pool_x20), axis=1)
    Up = env.pool_u2[nn]
    sig_u1 = spearmanr(pdist(B), pdist(Up[:, :1])).correlation
    sig_u2 = spearmanr(pdist(B), pdist(Up[:, 1:2])).correlation
    # how well does each particle localize u2 within its activation ball?
    vars = []
    for p in parts[:80]:
        d = np.sum((env.pool_x20 - p.x) ** 2, 1)
        w = np.exp(-d / (2 * p.sigma ** 2))
        m = w > 0.15
        if m.sum() > 3:
            vars.append(float(np.var(env.pool_u2[m, 1])))
    return dict(sig_vs_u1=float(sig_u1), sig_vs_u2=float(sig_u2),
                per_particle_u2_var=float(np.median(vars)) if vars else float("nan"))


def run_seed(seed, cfg, verbose=True):
    if verbose:
        print("\n" + "=" * 64)
        print("  GATE 1B variants -- seed %d" % seed)
        print("=" * 64)
    env = TwoScaleToyEnvironment(seed, cfg)

    # Scale 1 fit (crystallization on) -- shared across all graph variants
    s1 = Scale1RepresentationLearner(cfg, seed, enable_crystallization=True)
    s1.fit(env, verbose=False)
    parts = s1.get_particles_for_embedding()
    n_cryst = len(s1.get_crystallized_particles())

    geom = geometry_controls(env, seed)
    diag = signature_encodes_u2(s1, env)
    if verbose:
        print("  particles=%d crystallized=%d | geometry-only rho: raw=%.3f pca=%.3f "
              "spectral=%.3f random=%.3f" % (len(parts), n_cryst, geom['raw20D'],
              geom['PCA2D'], geom['spectral'], geom['random2D']))
        print("  signature-distance encodes: u1=%.3f  u2=%.3f  (per-particle u2 var=%.3f)"
              % (diag['sig_vs_u1'], diag['sig_vs_u2'], diag['per_particle_u2_var']))

    sub = np.random.RandomState(seed + 5).choice(cfg.N_test, 300, replace=False)
    Xt = env.test_A_x20[sub]
    Ut = env.test_A_u2[sub]

    # oracle Scale 2 (shared)
    oracle_enc = Oracle2DEncoder(np.eye(C.k))
    om, _, _ = run_scale2_v1(oracle_enc, env.train_u2, env.train_u2,
                             env.test_A_u2, env.test_A_u2, env.test_B_u2, env.test_B_u2,
                             env, seed, cfg.E_scale2)
    acc_oracle = om['ACC_gate']

    variants = {}
    for mode in MODES:
        s1.graph_mode = mode
        q_parts = s1.extract_embedding_2d(parts)
        enc = s1.make_encoder(np.eye(C.k), parts, q_parts)
        q_test = enc.encode_observation_batch(Xt)
        rho_struct = manifold_rho(enc.encode_observation_batch(env.test_A_x20),
                                  env.test_A_u2, seed=seed)
        rho_u1 = coord_recovery(q_test, Ut[:, 0], seed)
        rho_u2 = coord_recovery(q_test, Ut[:, 1], seed)
        em, _, _ = run_scale2_v1(enc, env.train_x20, env.train_u2,
                                 env.test_A_x20, env.test_A_u2,
                                 env.test_B_x20, env.test_B_u2, env, seed, cfg.E_scale2)
        acc_em = em['ACC_gate']
        gap = acc_oracle - acc_em
        passed = (rho_struct > RHO_THRESHOLD) and (acc_em >= acc_oracle - ACC_GAP_THRESHOLD)
        variants[mode] = dict(
            rho_struct=rho_struct, rho_u1=rho_u1, rho_u2=rho_u2,
            ACC_oracle=acc_oracle, ACC_emergent=acc_em, accuracy_gap=gap,
            n_repr_crystallized=n_cryst, gate1_pass=bool(passed))
        if verbose:
            print("  %-15s rho=%.3f (u1=%.3f u2=%.3f) ACC_em=%.3f gap=%+.3f pass=%s"
                  % (mode, rho_struct, rho_u1, rho_u2, acc_em, gap, passed))

    # controls reusing the multiplicative-mode pipeline
    s1.graph_mode = "multiplicative"
    rho_shuffled = manifold_rho(
        s1.make_encoder(np.eye(C.k), parts, shuffle_signatures=True)
          .encode_observation_batch(env.test_A_x20), env.test_A_u2, seed=seed)
    s1nc = Scale1RepresentationLearner(cfg, seed, enable_crystallization=False)
    s1nc.fit(env, verbose=False)
    pnc = s1nc.get_particles_for_embedding()
    rho_nocryst = float("nan")
    if len(pnc) >= 3:
        rho_nocryst = manifold_rho(
            s1nc.make_encoder(np.eye(C.k), pnc).encode_observation_batch(env.test_A_x20),
            env.test_A_u2, seed=seed)

    return dict(seed=seed, n_repr_particles=len(parts), n_repr_crystallized=n_cryst,
                geometry_controls=geom, diagnostic=diag, variants=variants,
                rho_shuffled=rho_shuffled, rho_nocryst=float(rho_nocryst),
                ACC_oracle=acc_oracle)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cfg = Gate1Config(generator="1B", N_repr_pool=1500, N_train_pool=600,
                      N_test=800, E_scale1=30, E_scale2=20, sigma_x_scale=0.7)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate1b_variants_results.json")
    print("Gate 1B behavioral-geometry variants | generator=1B | seeds=%d" % n_seeds)
    print("  N_repr=%d N_train=%d N_test=%d E1=%d E2=%d"
          % (cfg.N_repr_pool, cfg.N_train_pool, cfg.N_test, cfg.E_scale1, cfg.E_scale2))
    rows = []
    for s in range(n_seeds):
        rows.append(run_seed(s, cfg))
        with open(out, "w") as f:
            json.dump(dict(cfg=cfg.__dict__, rho_threshold=RHO_THRESHOLD,
                           acc_gap_threshold=ACC_GAP_THRESHOLD, modes=MODES,
                           rows=rows), f, indent=2)
        print("  [checkpoint] %d/%d seeds -> %s" % (len(rows), n_seeds, out))
    print("\nSaved %s" % out)


if __name__ == "__main__":
    main()
