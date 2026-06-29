"""
run_gate1.py
============

Phase 4 of the Gate 1 branch: the end-to-end Gate 1 experiment runner.

For each seed it:
    1. generates the two-scale environment;
    2. trains the Scale 1 representation learner (20D) and extracts the emergent
       2D encoder;
    3. computes the manifold-recovery metric rho_struct;
    4. runs the existing v1 IBF correction engine (Scale 2) on top of every
       encoder condition (oracle / emergent / raw20D / pca / random, plus the
       no-crystallization ablation and shuffled-signature control);
    5. records all per-seed metrics and (for one seed) the diagnostic figures.

The v1 engine (IBFAgent, MemoryCenter, Config C) is reused unchanged from
ibf_v1_engine.py. Only the encoder and the (dimension-aware) base evaluator /
sigma calibration are generalized so the same correction dynamics can run on
arbitrary 2D/20D representations.

Usage:
    python run_gate1.py --config smoke
    python run_gate1.py --config dev      # seeds 0..9
    python run_gate1.py --config final    # seeds 0..19
    python run_gate1.py --seeds 0 1 2 --n-repr 1500 --n-train 800
"""

import os
import json
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

import ibf_v1_engine as E
from ibf_v1_engine import IBFAgent, C
from gate1_environment import TwoScaleToyEnvironment, Gate1Config
from gate1_encoders import (Oracle2DEncoder, Raw20DEncoder, PCA2DEncoder,
                            Random2DEncoder)
from scale1_representation import Scale1RepresentationLearner

OUT_DIR = "gate1_outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")

RHO_THRESHOLD = 0.8
ACC_GAP_THRESHOLD = 0.15


# ============================================================================
#  generalized Scale 2 machinery (dimension-aware, reuses IBFAgent unchanged)
# ============================================================================

class Scale2BaseEvaluator:
    """Fixed random sigmoid prior over the encoder's z-space (any dimension)."""

    def __init__(self, encoder, obs_sample, seed):
        rng = np.random.RandomState(seed)
        z_dim = encoder.encode(obs_sample[0], 0).shape[0]
        self.z_dim = z_dim
        spread = 0.0
        for var in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
            self.w = rng.randn(z_dim) * var
            self.b = 0.0
            spread = self._spread(encoder, obs_sample, rng)
            if spread >= 0.02:
                break

    def _spread(self, encoder, obs_sample, rng):
        idx = rng.choice(len(obs_sample), min(250, len(obs_sample)), replace=False)
        sp = []
        for i in idx:
            vals = np.array([self(encoder.encode(obs_sample[i], j)) for j in range(C.k)])
            sp.append(vals.max() - vals.min())
        return float(np.mean(sp))

    def __call__(self, z):
        return 1.0 / (1.0 + np.exp(-(np.dot(self.w, z) + self.b)))

    def batch(self, Z):
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))


def calibrate_sigma_obs(encoder, obs_sample, seed):
    rng = np.random.RandomState(seed + 999)
    idx = rng.choice(len(obs_sample), min(500, len(obs_sample)), replace=False)
    Z = encoder.encode_batch(obs_sample[idx], np.zeros(len(idx), dtype=int))
    Zc = Z - Z.mean(axis=0)
    _, s, _ = np.linalg.svd(Zc, full_matrices=False)
    ve = np.cumsum(s**2) / np.sum(s**2)
    eff = int(np.searchsorted(ve, 0.95)) + 1
    dists = []
    for i in range(min(500, len(Z))):
        for j in range(i + 1, min(i + 20, len(Z))):
            dists.append(np.linalg.norm(Z[i] - Z[j]))
    med = float(np.median(dists))
    return med / np.sqrt(2 * max(eff, 1)), eff


def calibrate_action_embedding_obs(encoder, obs_sample, seed):
    """Scale the encoder's action embedding to hit the v1 action-separation target."""
    scale = 1.0
    encoder.action_embedding = np.eye(C.k) * scale
    sigma, _ = calibrate_sigma_obs(encoder, obs_sample, seed)
    sub = obs_sample[:200]
    for _ in range(C.action_sep_max_iter):
        seps = []
        for o in sub:
            zs = [encoder.encode(o, j) for j in range(C.k)]
            for a in range(C.k):
                for b in range(a + 1, C.k):
                    seps.append(np.linalg.norm(zs[a] - zs[b]))
        mean_sep = float(np.mean(seps))
        ratio = mean_sep / sigma if sigma > 0 else 0.0
        if ratio >= C.action_sep_target:
            break
        scale *= C.action_sep_margin * sigma / mean_sep
        encoder.action_embedding = np.eye(C.k) * scale
        sigma, _ = calibrate_sigma_obs(encoder, obs_sample, seed)
    return sigma, scale


def evaluate_scale2(agent, encoder, obs_test, u_test, env, ctx):
    N = len(obs_test)
    Z_all = np.stack([encoder.encode_batch(obs_test, np.full(N, j, dtype=int))
                      for j in range(C.k)], axis=1)
    Z_flat = Z_all.reshape(N * C.k, -1)
    R = agent.R_eff_batch(Z_flat).reshape(N, C.k)
    return float(np.mean(np.argmax(R, axis=1) == env.correct_actions_batch(u_test, ctx)))


def run_scale2_v1(encoder, obs_pool, u_pool, obs_testA, u_testA,
                  obs_testB, u_testB, env, seed, E2, want_log=False):
    """The existing v1 A->B correction process on an arbitrary encoder."""
    np.random.seed(seed)
    sigma, scale = calibrate_action_embedding_obs(encoder, obs_pool, seed)
    base_eval = Scale2BaseEvaluator(encoder, obs_pool, seed + 2)
    ibf = IBFAgent(sigma, sigma, encoder, base_eval, True, True, True)

    ctx_map = {'A': 0, 'B': 1}
    log = {'acc_A': [], 'acc_B': []}
    N = len(obs_pool)

    for phase in ['A', 'B']:
        ibf.set_context(ctx_map[phase])
        truth = env.correct_actions_batch(u_pool, phase)
        for _epoch in range(E2):
            order = np.random.permutation(N)
            for idx in order:
                o = obs_pool[idx]
                z_all = [encoder.encode(o, j) for j in range(C.k)]
                gt = int(truth[idx])
                ch, _Rv, _kv, _ent = ibf.select_action(z_all, deterministic=False)
                Ri = 1.0 if ch == gt else 0.0
                Rc = ibf.R_eff(z_all[ch])
                ibf.update(z_all[ch], Ri - Rc, x=o, j_chosen=ch)
            ibf.end_epoch()

            saved = ibf.current_context
            ibf.current_context = 0
            aA = evaluate_scale2(ibf, encoder, obs_testA, u_testA, env, 'A')
            ibf.current_context = 1
            aB = evaluate_scale2(ibf, encoder, obs_testB, u_testB, env, 'B')
            ibf.current_context = saved
            log['acc_A'].append(aA)
            log['acc_B'].append(aB)

    m = {
        'acc_A_end_A': log['acc_A'][E2 - 1],
        'acc_A_after_B': log['acc_A'][-1],
        'acc_B_after_B': log['acc_B'][-1],
        'n_scale2_centers': len(ibf.centers),
        'n_scale2_cryst': ibf.count_crystallized(),
    }
    m['BT_A'] = m['acc_A_after_B'] - m['acc_A_end_A']
    m['ACC_gate'] = 0.5 * (m['acc_A_after_B'] + m['acc_B_after_B'])
    if want_log:
        return m, log, ibf
    return m, log, None


# ============================================================================
#  manifold recovery metric
# ============================================================================

def manifold_rho(coords_test, u_test, n_sub=300, seed=0):
    n = len(coords_test)
    sub = np.random.RandomState(seed + 5).choice(n, min(n_sub, n), replace=False)
    return float(spearmanr(pdist(coords_test[sub]), pdist(u_test[sub])).correlation)


# ============================================================================
#  per-seed Gate 1 run
# ============================================================================

def run_seed(seed, cfg: Gate1Config, make_figures=False, verbose=True):
    if verbose:
        print("\n" + "=" * 64)
        print("  GATE 1 -- seed %d" % seed)
        print("=" * 64)

    env = TwoScaleToyEnvironment(seed, cfg)

    # ---- Scale 1: representation formation (with crystallization) ----
    s1 = Scale1RepresentationLearner(cfg, seed, enable_crystallization=True)
    s1.fit(env, verbose=verbose)
    parts = s1.get_particles_for_embedding()
    q_parts = s1.extract_embedding_2d(parts)
    n_repr = len(s1.particles)
    n_cryst = len(s1.get_crystallized_particles())

    emergent_enc = s1.make_encoder(np.eye(C.k), parts, q_parts)
    rho_struct = manifold_rho(emergent_enc.encode_observation_batch(env.test_A_x20),
                              env.test_A_u2, seed=seed)

    # ---- Scale 1 ablation: no crystallization ----
    s1_nc = Scale1RepresentationLearner(cfg, seed, enable_crystallization=False)
    s1_nc.fit(env, verbose=False)
    parts_nc = s1_nc.get_particles_for_embedding()
    rho_nocryst = np.nan
    if len(parts_nc) >= 3:
        q_nc = s1_nc.extract_embedding_2d(parts_nc)
        enc_nc = s1_nc.make_encoder(np.eye(C.k), parts_nc, q_nc)
        rho_nocryst = manifold_rho(enc_nc.encode_observation_batch(env.test_A_x20),
                                   env.test_A_u2, seed=seed)

    # ---- shuffled-signature control ----
    shuf_enc = s1.make_encoder(np.eye(C.k), parts, shuffle_signatures=True)
    rho_shuffled = manifold_rho(shuf_enc.encode_observation_batch(env.test_A_x20),
                                env.test_A_u2, seed=seed)

    if verbose:
        print("  rho_struct (emergent)=%.3f | no-cryst=%.3f | shuffled-sig=%.3f"
              % (rho_struct, rho_nocryst, rho_shuffled))

    # ---- Scale 2 conditions ----
    E2 = cfg.E_scale2
    conditions = {}

    # oracle: observation IS the true 2D coord
    oracle_enc = Oracle2DEncoder(np.eye(C.k))
    conditions['oracle'] = (oracle_enc, env.train_u2, env.test_A_u2, env.test_B_u2)
    # emergent
    conditions['emergent'] = (emergent_enc, env.train_x20, env.test_A_x20, env.test_B_x20)
    # raw 20D control
    conditions['raw20D'] = (Raw20DEncoder(np.eye(C.k)),
                            env.train_x20, env.test_A_x20, env.test_B_x20)
    # PCA 2D control
    conditions['pca2D'] = (PCA2DEncoder(np.eye(C.k), env.train_x20),
                          env.train_x20, env.test_A_x20, env.test_B_x20)
    # random 2D control
    conditions['random2D'] = (Random2DEncoder(np.eye(C.k), env.train_x20, seed),
                             env.train_x20, env.test_A_x20, env.test_B_x20)

    s2 = {}
    s2_logs = {}
    for name, (enc, obs_pool, obs_tA, obs_tB) in conditions.items():
        want_log = make_figures
        m, log, _ = run_scale2_v1(
            enc, obs_pool, env.train_u2, obs_tA, env.test_A_u2,
            obs_tB, env.test_B_u2, env, seed, E2, want_log=want_log)
        s2[name] = m
        s2_logs[name] = log
        if verbose:
            print("  Scale2 %-9s ACC_gate=%.3f (A_end=%.3f A_afterB=%.3f "
                  "B_afterB=%.3f) BT_A=%+.3f centers=%d/%d"
                  % (name, m['ACC_gate'], m['acc_A_end_A'], m['acc_A_after_B'],
                     m['acc_B_after_B'], m['BT_A'], m['n_scale2_cryst'],
                     m['n_scale2_centers']))

    acc_oracle = s2['oracle']['ACC_gate']
    acc_emergent = s2['emergent']['ACC_gate']
    acc_gap = acc_oracle - acc_emergent

    passed = (rho_struct > RHO_THRESHOLD) and (acc_emergent >= acc_oracle - ACC_GAP_THRESHOLD)

    row = {
        'seed': seed,
        'rho_struct': rho_struct,
        'rho_nocryst': float(rho_nocryst),
        'rho_shuffled': rho_shuffled,
        'ACC_oracle': acc_oracle,
        'ACC_emergent': acc_emergent,
        'accuracy_gap': acc_gap,
        'Acc_A_end_A': s2['emergent']['acc_A_end_A'],
        'Acc_A_after_B': s2['emergent']['acc_A_after_B'],
        'Acc_B_after_B': s2['emergent']['acc_B_after_B'],
        'BT_A_emergent': s2['emergent']['BT_A'],
        'n_repr_particles': n_repr,
        'n_repr_crystallized': n_cryst,
        'n_scale2_centers': s2['emergent']['n_scale2_centers'],
        'n_scale2_crystallized': s2['emergent']['n_scale2_cryst'],
        'rho_pass': bool(rho_struct > RHO_THRESHOLD),
        'acc_pass': bool(acc_emergent >= acc_oracle - ACC_GAP_THRESHOLD),
        'gate1_pass': bool(passed),
        # controls (Scale 2 ACC_gate)
        'ACC_raw20D': s2['raw20D']['ACC_gate'],
        'ACC_pca2D': s2['pca2D']['ACC_gate'],
        'ACC_random2D': s2['random2D']['ACC_gate'],
    }

    if make_figures:
        from gate1_plots import make_seed_figures
        make_seed_figures(seed, env, s1, parts, q_parts, emergent_enc,
                          s2_logs, row, FIG_DIR)

    if verbose:
        print("  --> rho=%.3f (pass=%s)  acc_gap=%.3f (pass=%s)  GATE1=%s"
              % (rho_struct, row['rho_pass'], acc_gap, row['acc_pass'],
                 "PASS" if passed else "FAIL"))
    return row


# ============================================================================
#  config presets
# ============================================================================

def build_cfg(args):
    if args.config == 'smoke':
        cfg = Gate1Config(N_repr_pool=300, N_train_pool=200, N_test=500,
                          E_scale1=20, E_scale2=15, sigma_x_scale=0.7)
        seeds = [0, 1, 2]
    elif args.config == 'dev':
        cfg = Gate1Config(N_repr_pool=2000, N_train_pool=1000, N_test=1000,
                          E_scale1=30, E_scale2=25, sigma_x_scale=0.7)
        seeds = list(range(10))
    elif args.config == 'final':
        cfg = Gate1Config(N_repr_pool=2000, N_train_pool=1000, N_test=1000,
                          E_scale1=30, E_scale2=25, sigma_x_scale=0.7)
        seeds = list(range(20))
    else:  # custom
        cfg = Gate1Config(N_repr_pool=args.n_repr, N_train_pool=args.n_train,
                          N_test=args.n_test, E_scale1=args.e1, E_scale2=args.e2,
                          sigma_x_scale=0.7)
        seeds = args.seeds
    if args.seeds is not None:
        seeds = args.seeds
    if args.n_repr is not None:
        cfg.N_repr_pool = args.n_repr
    if args.n_train is not None:
        cfg.N_train_pool = args.n_train
    if args.n_test is not None:
        cfg.N_test = args.n_test
    if args.e1 is not None:
        cfg.E_scale1 = args.e1
    if args.e2 is not None:
        cfg.E_scale2 = args.e2
    return cfg, seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='custom',
                    choices=['smoke', 'dev', 'final', 'custom'])
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--n-repr', dest='n_repr', type=int, default=None)
    ap.add_argument('--n-train', dest='n_train', type=int, default=None)
    ap.add_argument('--n-test', dest='n_test', type=int, default=None)
    ap.add_argument('--e1', type=int, default=None)
    ap.add_argument('--e2', type=int, default=None)
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    cfg, seeds = build_cfg(args)

    print("Gate 1 runner | config=%s | seeds=%s" % (args.config, seeds))
    print("  N_repr=%d N_train=%d N_test=%d E_scale1=%d E_scale2=%d"
          % (cfg.N_repr_pool, cfg.N_train_pool, cfg.N_test,
             cfg.E_scale1, cfg.E_scale2))

    out = os.path.join(OUT_DIR, "gate1_results.json")
    rows = []
    for i, seed in enumerate(seeds):
        make_fig = (not args.no_figures) and (i == 0)
        rows.append(run_seed(seed, cfg, make_figures=make_fig))
        # checkpoint after every seed so partial runs survive interruption
        payload = {
            'config': args.config,
            'cfg': cfg.__dict__,
            'rho_threshold': RHO_THRESHOLD,
            'acc_gap_threshold': ACC_GAP_THRESHOLD,
            'rows': rows,
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print("  [checkpoint] %d/%d seeds saved to %s" % (len(rows), len(seeds), out))
    print("\nSaved %s (%d seeds)" % (out, len(rows)))


if __name__ == "__main__":
    main()
