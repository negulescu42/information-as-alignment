"""
postulate2.scale2  --  run the Postulate-1 correction engine on ANY representation.

The Scale-2 layer is the ORIGINAL engine (engine.IBFAgent) run unchanged on a
representation z produced by an encoder. This module only provides the
dimension-agnostic scaffolding around it:

  * Scale2BaseEvaluator       -- fixed random sigmoid prior over the z-space
  * calibrate_sigma_obs       -- kernel bandwidth from the encoded data
  * calibrate_action_embedding_obs -- action-separation calibration
  * evaluate_scale2           -- accuracy on a test set
  * run_scale2_v1             -- the standard A->B continual-learning run
  * train_phase / select_fast -- batched single-phase training (multi-context)
  * manifold_rho / coord_recovery -- representation-quality metrics

The number of actions is engine.C.k; set engine.C.k before use (2 by default,
8 for the dense-field / continual-learning experiments).
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

from engine import IBFAgent, C


# ---------------------------------------------------------------- base evaluator
class Scale2BaseEvaluator:
    """Fixed random sigmoid prior over the encoder's z-space (any dimension)."""

    def __init__(self, encoder, obs_sample, seed):
        rng = np.random.RandomState(seed)
        z_dim = encoder.encode(obs_sample[0], 0).shape[0]
        self.z_dim = z_dim
        for var in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
            self.w = rng.randn(z_dim) * var
            self.b = 0.0
            if self._spread(encoder, obs_sample, rng) >= 0.02:
                break

    def _spread(self, encoder, obs_sample, rng):
        idx = rng.choice(len(obs_sample), min(250, len(obs_sample)), replace=False)
        sp = [np.ptp([self(encoder.encode(obs_sample[i], j)) for j in range(C.k)])
              for i in idx]
        return float(np.mean(sp))

    def __call__(self, z):
        return 1.0 / (1.0 + np.exp(-(np.dot(self.w, z) + self.b)))

    def batch(self, Z):
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))


# ---------------------------------------------------------------- calibration
def calibrate_sigma_obs(encoder, obs_sample, seed):
    rng = np.random.RandomState(seed + 999)
    idx = rng.choice(len(obs_sample), min(500, len(obs_sample)), replace=False)
    Z = encoder.encode_batch(obs_sample[idx], np.zeros(len(idx), dtype=int))
    Zc = Z - Z.mean(axis=0)
    _, s, _ = np.linalg.svd(Zc, full_matrices=False)
    eff = int(np.searchsorted(np.cumsum(s**2) / np.sum(s**2), 0.95)) + 1
    dists = [np.linalg.norm(Z[i] - Z[j])
             for i in range(min(500, len(Z))) for j in range(i + 1, min(i + 20, len(Z)))]
    return float(np.median(dists)) / np.sqrt(2 * max(eff, 1)), eff


def calibrate_action_embedding_obs(encoder, obs_sample, seed):
    """Scale encoder.action_embedding to hit the engine's action-separation target."""
    scale = 1.0
    encoder.action_embedding = np.eye(C.k) * scale
    sigma, _ = calibrate_sigma_obs(encoder, obs_sample, seed)
    sub = obs_sample[:200]
    for _ in range(C.action_sep_max_iter):
        seps = [np.linalg.norm(encoder.encode(o, a) - encoder.encode(o, b))
                for o in sub for a in range(C.k) for b in range(a + 1, C.k)]
        mean_sep = float(np.mean(seps))
        if mean_sep / sigma >= C.action_sep_target:
            break
        scale *= C.action_sep_margin * sigma / mean_sep
        encoder.action_embedding = np.eye(C.k) * scale
        sigma, _ = calibrate_sigma_obs(encoder, obs_sample, seed)
    return sigma, scale


# ---------------------------------------------------------------- evaluation
def evaluate_scale2(agent, encoder, obs_test, u_test, env, ctx):
    N = len(obs_test)
    Z_all = np.stack([encoder.encode_batch(obs_test, np.full(N, j, dtype=int))
                      for j in range(C.k)], axis=1)
    R = agent.R_eff_batch(Z_all.reshape(N * C.k, -1)).reshape(N, C.k)
    return float(np.mean(np.argmax(R, axis=1) == env.correct_actions_batch(u_test, ctx)))


# ---------------------------------------------------------------- A -> B run
def run_scale2_v1(encoder, obs_pool, u_pool, obs_testA, u_testA,
                  obs_testB, u_testB, env, seed, E2, want_agent=False):
    """Standard two-context (A then B) continual-learning run of the engine."""
    np.random.seed(seed)
    sigma, _ = calibrate_action_embedding_obs(encoder, obs_pool, seed)
    base = Scale2BaseEvaluator(encoder, obs_pool, seed + 2)
    ibf = IBFAgent(sigma, sigma, encoder, base, True, True, True)
    log = {'acc_A': [], 'acc_B': []}
    N = len(obs_pool)
    for phase, cid in [('A', 0), ('B', 1)]:
        ibf.set_context(cid)
        truth = env.correct_actions_batch(u_pool, phase)
        for _ in range(E2):
            for idx in np.random.permutation(N):
                z_all = [encoder.encode(obs_pool[idx], j) for j in range(C.k)]
                ch, _, _, _ = ibf.select_action(z_all, deterministic=False)
                Ri = 1.0 if ch == int(truth[idx]) else 0.0
                ibf.update(z_all[ch], Ri - ibf.R_eff(z_all[ch]), x=obs_pool[idx], j_chosen=ch)
            ibf.end_epoch()
            sv = ibf.current_context
            ibf.current_context = 0
            log['acc_A'].append(evaluate_scale2(ibf, encoder, obs_testA, u_testA, env, 'A'))
            ibf.current_context = 1
            log['acc_B'].append(evaluate_scale2(ibf, encoder, obs_testB, u_testB, env, 'B'))
            ibf.current_context = sv
    m = {'acc_A_after_B': log['acc_A'][-1], 'acc_B_after_B': log['acc_B'][-1],
         'ACC_gate': 0.5 * (log['acc_A'][-1] + log['acc_B'][-1]),
         'n_scale2_cryst': ibf.count_crystallized()}
    return (m, log, ibf) if want_agent else (m, log, None)


# ---------------------------------------------------------------- batched training
def _Z_for(encoder, o):
    return np.concatenate([np.tile(o, (C.k, 1)), encoder.action_embedding], axis=1)


def select_fast(agent, Z):
    R = agent.R_eff_batch(Z)
    kk = np.array([agent.k_eff(Z[j]) for j in range(len(Z))])
    logits = kk * R
    logits -= logits.max()
    p = np.exp(logits); p /= p.sum()
    return int(np.random.choice(len(Z), p=p)), R


def train_phase(agent, encoder, coords_pool, u_pool, env, ctx_name, ctx_id, epochs, seed):
    """Train the engine for one context/phase on precomputed coords (batched readout)."""
    rng = np.random.RandomState(seed * 131 + ctx_id)
    np.random.seed(seed * 977 + ctx_id)
    agent.set_context(ctx_id)
    truth = env.correct_actions_batch(u_pool, ctx_name)
    for _ in range(epochs):
        for idx in rng.permutation(len(coords_pool)):
            Z = _Z_for(encoder, coords_pool[idx])
            ch, R = select_fast(agent, Z)
            Ri = 1.0 if ch == int(truth[idx]) else 0.0
            agent.update(Z[ch], Ri - R[ch], x=coords_pool[idx], j_chosen=ch)
        agent.end_epoch()


def make_agent(encoder, coords_pool, seed):
    sigma, _ = calibrate_action_embedding_obs(encoder, coords_pool, seed)
    base = Scale2BaseEvaluator(encoder, coords_pool, seed + 2)
    return IBFAgent(sigma, sigma, encoder, base, True, True, True)


# ---------------------------------------------------------------- metrics
def manifold_rho(coords_test, u_test, n_sub=300, seed=0):
    sub = np.random.RandomState(seed + 5).choice(len(coords_test),
                                                 min(n_sub, len(coords_test)), replace=False)
    return float(spearmanr(pdist(coords_test[sub]), pdist(u_test[sub])).correlation)


def coord_recovery(q, u_coord, seed):
    """5-fold kNN-regression recoverability of a hidden coordinate from q."""
    if len(q) < 10:
        return float("nan")
    preds = np.zeros(len(u_coord))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(q):
        preds[te] = KNeighborsRegressor(10, weights="distance").fit(q[tr], u_coord[tr]).predict(q[te])
    return float(spearmanr(preds, u_coord).correlation)
