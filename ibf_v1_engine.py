"""
ibf_v1_engine.py
================

Phase 0 refactor for the Gate 1 branch.

This module contains the v1 IBF continual-learning engine, extracted VERBATIM
from the original `(IBF)Toy-Model.ipynb` notebook (cell 1, Parts 1-6). The only
changes versus the notebook are mechanical:

  * matplotlib / figure-generation code has been removed (lives in the notebook);
  * top-level experiment execution has been removed (importing this module has
    no side effects);
  * nothing else in the engine logic is altered.

The classes / functions exported here are the canonical Scale 2 correction
engine for Gate 1:

    Config, MemoryCenter, IBFAgent,
    ToyEnvironment, ToyEncoder, ToyBaseEvaluator, PassiveAgent,
    calibrate_sigma, calibrate_action_embedding, evaluate, classify_center,
    run_toy_experiment

`run_toy_experiment` reproduces the original 2D toy model unchanged and is used
as the Phase-0 regression test (see run_regression.py).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict
import copy
import time


# ============================================================================
#  PART 1 -- CONFIGURATION
# ============================================================================

@dataclass
class Config:
    d: int = 2
    k: int = 2
    N_pool: int = 200
    E: int = 25
    N_test: int = 500
    eval_every: int = 50
    eta: float = 0.10
    eta_cryst: float = 0.005
    mu_base: float = 0.06
    mu_cryst: float = 0.001
    v_max: float = 0.50
    cryst_n_min: int = 15
    activation_thresh: float = 0.18
    creation_thresh: float = 0.30
    convergence_threshold: float = 0.075
    capacity: int = 2000
    alpha_shrink: float = 1.0
    sigma_floor: float = 0.15
    min_samples_shrink: int = 50
    k_0: float = 5.0
    k_min: float = 1.0
    eta_k: float = 0.05
    w_max: float = 5.0
    w_dvar_threshold: float = 0.045
    n_cross_min: int = 4
    reversal_threshold: float = -0.125
    action_sep_target: float = 3.5
    action_sep_margin: float = 4.0
    action_sep_max_iter: int = 5


C = Config()
ACTION_EMB = np.eye(C.k)

p_vec = np.array([+1.0, -1.0])
r_vec = np.array([+1.0, -1.0])


# ============================================================================
#  PART 2 -- IBF ENGINE (Cell 6 logic, VERBATIM from RRW v4.17)
# ============================================================================

@dataclass
class MemoryCenter:
    z: np.ndarray
    v: float = 0.0
    w: float = 0.0
    n_updates: int = 0
    D_sum: float = 0.0
    D_sq_sum: float = 0.0
    mu_eff: float = 0.06
    context_id: int = -1
    birth_step: int = 0
    context_update_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    sigma: float = 0.58
    D_history: List[float] = field(default_factory=list)
    D_step_history: List[int] = field(default_factory=list)
    xj_history: List[Tuple] = field(default_factory=list)
    was_ever_crystallized: bool = False
    crucible_verified: bool = False
    dissolution_log: List[Dict] = field(default_factory=list)

    def D_var(self):
        if self.n_updates < 20:
            return 0.0
        rec = self.D_history[20:][-50:]
        if len(rec) < 2:
            return 0.0
        return float(np.var(rec))

    def is_crystallized(self):
        return self.mu_eff < C.mu_base - 1e-6

    def n_cross_updates(self):
        return sum(cnt for ctx, cnt in self.context_update_counts.items()
                   if ctx != self.context_id)


class IBFAgent:
    def __init__(self, sigma, merge_threshold, encoder, base_eval,
                 enable_crystallization=True, enable_crucible=True,
                 enable_agency=True):
        self.centers = []
        self.sigma_base = sigma
        self.merge_thresh = merge_threshold
        self.encoder = encoder
        self.base_eval = base_eval
        self.current_context = -1
        self.global_step = 0
        self.enable_crystallization = enable_crystallization
        self.enable_crucible = enable_crucible
        self.enable_agency = enable_agency
        self._merge_ratio = merge_threshold / sigma
        self._dynamic_sigma_floor = min(C.sigma_floor, sigma * 0.25)
        self._needle_threshold = sigma * 0.50

    def set_context(self, ctx_id):
        if ctx_id != self.current_context:
            for c in self.centers:
                c.crucible_verified = False
        self.current_context = ctx_id

    def kernel_batch(self, z, centers=None):
        cs = centers if centers is not None else self.centers
        if not cs:
            return np.array([])
        Z = np.array([c.z for c in cs])
        sq = np.sum((Z - z[np.newaxis, :])**2, axis=1)
        sigmas = np.array([c.sigma for c in cs])
        return np.exp(-sq / (2.0 * sigmas**2))

    def _read_gate(self, c):
        if c.context_id == self.current_context:
            return 1.0
        if c.is_crystallized() and c.crucible_verified:
            return 1.0
        return 0.0

    # ════════════════════════════════════════════
    #  READ PATH (§4.2): Kernel readout, gating, action selection
    # ════════════════════════════════════════════

    def delta_R(self, z):
        if not self.centers:
            return 0.0
        K = self.kernel_batch(z)
        total = 0.0
        for i, c in enumerate(self.centers):
            g = self._read_gate(c)
            if g > 0 and K[i] > C.activation_thresh:
                total += g * c.v * K[i]
        return total

    def delta_k(self, z):
        if not self.centers or not self.enable_agency:
            return 0.0
        K = self.kernel_batch(z)
        total_w, sum_K = 0.0, 0.0
        for i, c in enumerate(self.centers):
            if not c.is_crystallized():
                continue
            g = self._read_gate(c)
            if g > 0 and K[i] > C.activation_thresh:
                total_w += g * c.w * K[i]
                sum_K += g * K[i]
        return total_w / sum_K if sum_K > 1e-6 else 0.0

    def R_eff(self, z):
        return float(np.clip(self.base_eval(z) + self.delta_R(z), 0.0, 1.0))

    def R_eff_batch(self, Z_flat):
        R_base = self.base_eval.batch(Z_flat)
        if not self.centers:
            return np.clip(R_base, 0.0, 1.0)
        Z_c = np.array([c.z for c in self.centers])
        sigmas = np.array([c.sigma for c in self.centers])
        vs = np.array([c.v for c in self.centers])
        gate = np.array([1.0 if (c.context_id == self.current_context
                                  or (c.is_crystallized() and c.crucible_verified))
                         else 0.0 for c in self.centers])
        Z_sq = np.sum(Z_flat**2, axis=1, keepdims=True)
        C_sq = np.sum(Z_c**2, axis=1, keepdims=True)
        sq_d = Z_sq + C_sq.T - 2.0 * (Z_flat @ Z_c.T)
        K = np.exp(-sq_d / (2.0 * sigmas[np.newaxis, :]**2))
        K[K < C.activation_thresh] = 0.0
        return np.clip(R_base + K @ (gate * vs), 0.0, 1.0)

    def k_eff(self, z):
        return max(C.k_min, C.k_0 + self.delta_k(z))

    def select_action(self, z_all, deterministic=False):
        R = np.array([self.R_eff(z) for z in z_all])
        if deterministic:
            return int(np.argmax(R)), R, np.full(C.k, C.k_0), 0.0
        k = np.array([self.k_eff(z) for z in z_all])
        logits = k * R
        logits -= logits.max()
        probs = np.exp(logits) / np.sum(np.exp(logits))
        chosen = int(np.random.choice(C.k, p=probs))
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        return chosen, R, k, entropy

    def _thermodynamic_shrink(self, center):
        if center.n_updates >= C.min_samples_shrink:
            dvar = center.D_var()
            calc = max(self._dynamic_sigma_floor,
                       self.sigma_base / (1.0 + C.alpha_shrink * dvar))
            center.sigma = min(center.sigma, calc)

    # ════════════════════════════════════════════
    #  WRITE PATH (§4.3): Discrepancy-driven local modification
    # ════════════════════════════════════════════

    def _update_agency(self, center, k_weight):
        if center.is_crystallized():
            dv = center.D_var()
            tw = np.clip(C.w_max * (1.0 - dv / C.w_dvar_threshold),
                         -C.w_max, C.w_max)
            center.w += C.eta_k * k_weight * (tw - center.w)
            center.w = np.clip(center.w, -C.w_max, C.w_max)

    def update(self, z_chosen, D, x=None, j_chosen=None):
        self.global_step += 1
        K_all = self.kernel_batch(z_chosen) if self.centers else np.array([])

        if self.enable_crucible:
            for i, c in enumerate(self.centers):
                if c.is_crystallized() and c.context_id != self.current_context:
                    kw = float(K_all[i])
                    if kw >= C.activation_thresh:
                        juris_D = D * kw
                        c.v = np.clip(c.v + C.eta_cryst * juris_D, -C.v_max, C.v_max)
                        c.n_updates += 1
                        c.D_sum += juris_D
                        c.D_sq_sum += juris_D * juris_D
                        c.context_update_counts[self.current_context] += 1
                        c.D_history.append(D)
                        c.D_step_history.append(self.global_step)
                        if x is not None:
                            c.xj_history.append((x.copy(), j_chosen))

        li = [i for i, c in enumerate(self.centers)
              if c.context_id == self.current_context]
        max_K = float(np.max(K_all[li])) if li else 0.0

        if max_K < C.creation_thresh:
            if len(self.centers) < C.capacity:
                juris_D = D * 1.0
                nc = MemoryCenter(
                    z=z_chosen.copy(),
                    v=np.clip(C.eta * juris_D, -C.v_max, C.v_max),
                    w=0.0, n_updates=1,
                    D_sum=juris_D, D_sq_sum=juris_D*juris_D,
                    mu_eff=C.mu_base,
                    context_id=self.current_context,
                    birth_step=self.global_step,
                    sigma=self.sigma_base)
                nc.context_update_counts[self.current_context] = 1
                nc.D_history.append(juris_D)
                nc.D_step_history.append(self.global_step)
                if x is not None:
                    nc.xj_history.append((x.copy(), j_chosen))
                self.centers.append(nc)
            return

        for i in li:
            c = self.centers[i]
            kw = float(K_all[i])
            if kw < C.activation_thresh:
                continue
            juris_D = D * kw
            effective_eta = C.eta_cryst if c.is_crystallized() else C.eta
            c.v = np.clip(c.v + effective_eta * juris_D, -C.v_max, C.v_max)
            c.n_updates += 1
            c.D_sum += juris_D
            c.D_sq_sum += juris_D * juris_D
            c.context_update_counts[self.current_context] += 1
            c.D_history.append(juris_D)
            c.D_step_history.append(self.global_step)
            if x is not None:
                c.xj_history.append((x.copy(), j_chosen))
            if self.enable_agency:
                self._update_agency(c, kw)
            self._thermodynamic_shrink(c)

    def end_epoch(self):
        for c in self.centers:
            c.v *= (1.0 - c.mu_eff)
            c.w *= (1.0 - c.mu_eff)

        for c in self.centers:
            if self.enable_crystallization:
                hist_len = len(c.D_history)
                cryst_grad = c.D_history[-50:] if hist_len > 0 else [0.0]
                cryst_mu = float(np.mean(cryst_grad))
                is_converged = abs(cryst_mu) < C.convergence_threshold

                if (not c.is_crystallized()
                        and c.n_updates >= C.cryst_n_min
                        and is_converged):
                    c.mu_eff = C.mu_cryst
                    c.was_ever_crystallized = True

                elif c.is_crystallized():
                    nc = c.n_cross_updates()
                    if nc >= C.n_cross_min:
                        crucible_grad = c.D_history[-C.n_cross_min:]
                        crucible_mu = float(np.mean(crucible_grad))
                        if (c.v * crucible_mu) < C.reversal_threshold:
                            c.dissolution_log.append({
                                'step': self.global_step,
                                'v': float(c.v),
                                'mu_D_recent': float(crucible_mu),
                                'product': float(c.v * crucible_mu),
                                'n_updates': c.n_updates,
                                'n_cross': nc,
                                'context': self.current_context,
                            })
                            c.mu_eff = C.mu_base
                            c.crucible_verified = False
                        else:
                            c.crucible_verified = True
            else:
                c.mu_eff = C.mu_base

        self._merge()

    def _merge(self):
        if len(self.centers) < 2:
            return
        merged = set()
        new = []
        for i in range(len(self.centers)):
            if i in merged:
                continue
            best = self.centers[i]
            for j in range(i+1, len(self.centers)):
                if j in merged:
                    continue
                if self.centers[i].context_id != self.centers[j].context_id:
                    continue
                dist = np.linalg.norm(self.centers[i].z - self.centers[j].z)
                ni = self.centers[i].sigma < self._needle_threshold
                nj = self.centers[j].sigma < self._needle_threshold
                if ni and nj:
                    th = self._merge_ratio * max(
                        self.centers[i].sigma, self.centers[j].sigma) * 1.5
                else:
                    th = self._merge_ratio * min(
                        self.centers[i].sigma, self.centers[j].sigma)
                if dist < th:
                    other = self.centers[j]
                    if other.n_updates > best.n_updates:
                        best, other = other, best
                    best.v = np.clip(best.v + other.v, -C.v_max, C.v_max)
                    best.w = np.clip(best.w + other.w, -C.w_max, C.w_max)
                    best.n_updates += other.n_updates
                    best.D_sum += other.D_sum
                    best.D_sq_sum += other.D_sq_sum
                    for ctx, cnt in other.context_update_counts.items():
                        best.context_update_counts[ctx] += cnt
                    best.D_history.extend(other.D_history)
                    best.D_step_history.extend(other.D_step_history)
                    best.xj_history.extend(other.xj_history)
                    best.sigma = min(best.sigma, other.sigma)
                    merged.add(j)
            new.append(best)
        if len(new) > C.capacity:
            cryst = [c for c in new if c.is_crystallized()]
            trans = sorted([c for c in new if not c.is_crystallized()],
                           key=lambda c: abs(c.v) * c.n_updates)
            keep = C.capacity - len(cryst)
            new = cryst + trans[-keep:] if keep > 0 else cryst[:C.capacity]
        self.centers = new

    def count_crystallized(self):
        return sum(1 for c in self.centers if c.is_crystallized())

    def count_verified(self):
        return sum(1 for c in self.centers
                   if c.is_crystallized() and c.crucible_verified)


# ============================================================================
#  PART 3 -- ENVIRONMENT, ENCODER, BASE EVALUATOR
# ============================================================================

class ToyEnvironment:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        self.pool = rng.randn(C.N_pool, C.d)
        self.test_A = rng.randn(C.N_test, C.d)
        self.test_B = rng.randn(C.N_test, C.d)
        self.u = {'A': +1.0, 'B': -1.0}

    def score_clean(self, x, ctx):
        u_c = self.u[ctx]
        s = np.zeros(C.k)
        for j in range(C.k):
            s[j] = x[0] * p_vec[j] + u_c * x[1] * r_vec[j]
        return s

    def correct_action(self, x, ctx):
        return int(np.argmax(self.score_clean(x, ctx)))

    def correct_actions_batch(self, X, ctx):
        u_c = self.u[ctx]
        N = len(X)
        S = np.zeros((N, C.k))
        for j in range(C.k):
            S[:, j] = X[:, 0] * p_vec[j] + u_c * X[:, 1] * r_vec[j]
        return np.argmax(S, axis=1)

    def get_test(self, ctx):
        return self.test_A if ctx == 'A' else self.test_B


class ToyEncoder:
    def encode(self, x_np, action_idx):
        return np.concatenate([x_np, ACTION_EMB[action_idx]])

    def encode_batch(self, x_np, action_indices):
        return np.concatenate([x_np, ACTION_EMB[action_indices]], axis=1)


class ToyBaseEvaluator:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        z_dim = C.d + C.k
        for var in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
            self.w = rng.randn(z_dim) * var
            self.b = 0.0
            spread = self._check_spread(rng)
            if spread >= 0.02:
                print("  Base evaluator spread: %.4f (variance=%.2f)" % (spread, var))
                return
        print("  Base evaluator spread: %.4f (max variance)" % spread)

    def _check_spread(self, rng):
        xs = rng.randn(250, C.d)
        acts = rng.randint(0, C.k, size=250)
        enc = ToyEncoder()
        Z = enc.encode_batch(xs, acts)
        spreads = []
        for i in range(0, 250 - C.k + 1, C.k):
            vals = self.batch(Z[i:i+C.k])
            spreads.append(vals.max() - vals.min())
        return float(np.mean(spreads))

    def __call__(self, z):
        return 1.0 / (1.0 + np.exp(-(np.dot(self.w, z) + self.b)))

    def batch(self, Z):
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))


class PassiveAgent:
    def __init__(self, encoder, base_eval):
        self.encoder = encoder
        self.base_eval = base_eval

    def R_eff_batch(self, Z_flat):
        return self.base_eval.batch(Z_flat)


# ============================================================================
#  PART 4 -- SIGMA CALIBRATION
#  Uses sigma = median_dist / sqrt(2*d_eff) ~ 0.8, NOT GRP P10/3 ~ 0.2
# ============================================================================

def calibrate_sigma(encoder, seed):
    rng = np.random.RandomState(seed + 999)
    xs = rng.randn(min(500, C.N_pool * 2), C.d)
    Z = encoder.encode_batch(xs, np.zeros(len(xs), dtype=int))
    Z_c = Z - Z.mean(axis=0)
    _, s, _ = np.linalg.svd(Z_c, full_matrices=False)
    var_exp = np.cumsum(s**2) / np.sum(s**2)
    eff_rank = int(np.searchsorted(var_exp, 0.95)) + 1
    dists = []
    for i in range(min(500, len(Z))):
        for j in range(i+1, min(i+20, len(Z))):
            dists.append(np.linalg.norm(Z[i] - Z[j]))
    dists = np.array(dists)
    p10 = np.percentile(dists, 10)
    med_dist = np.median(dists)
    sigma_grp = p10 / 3.0
    sigma_alt = med_dist / np.sqrt(2 * max(eff_rank, 1))
    print("  sigma cal: P10=%.3f -> sigma_grp=%.3f (too tight at d=2)" % (p10, sigma_grp))
    print("  sigma cal: med=%.3f, d_eff=%d -> sigma_alt=%.3f <- USING THIS" % (med_dist, eff_rank, sigma_alt))
    return sigma_alt, eff_rank


def calibrate_action_embedding(encoder, seed):
    global ACTION_EMB
    ACTION_EMB = np.eye(C.k)
    emb_scale = 1.0
    sigma_est, _ = calibrate_sigma(encoder, seed)
    for attempt in range(C.action_sep_max_iter):
        rng = np.random.RandomState(seed + 500)
        xs = rng.randn(200, C.d)
        seps = []
        for x in xs:
            zs = [encoder.encode(x, j) for j in range(C.k)]
            for a in range(C.k):
                for b in range(a+1, C.k):
                    seps.append(np.linalg.norm(zs[a] - zs[b]))
        mean_sep = np.mean(seps)
        ratio = mean_sep / sigma_est if sigma_est > 0 else 0.0
        print("  [Sep iter %d] sep=%.4f, sigma=%.4f, ratio=%.2f" % (attempt+1, mean_sep, sigma_est, ratio))
        if ratio >= C.action_sep_target:
            print("  Action separation OK (ratio=%.2f)" % ratio)
            break
        t = C.action_sep_margin * sigma_est / mean_sep
        emb_scale *= t
        ACTION_EMB = np.eye(C.k) * emb_scale
        sigma_est, _ = calibrate_sigma(encoder, seed)
    return sigma_est, emb_scale


# ============================================================================
#  PART 5 -- EVALUATION
# ============================================================================

def evaluate(agent, encoder, env, ctx):
    X = env.get_test(ctx)
    N = len(X)
    Z_all = np.stack([encoder.encode_batch(X, np.full(N, j, dtype=int))
                      for j in range(C.k)], axis=1)
    Z_flat = Z_all.reshape(N * C.k, -1)
    R = agent.R_eff_batch(Z_flat).reshape(N, C.k)
    return float(np.mean(np.argmax(R, axis=1) == env.correct_actions_batch(X, ctx)))


def classify_center(center, env):
    x_test = center.z[:2].copy()
    act_A = env.correct_action(x_test, 'A')
    act_B = env.correct_action(x_test, 'B')
    if act_A == act_B:
        return 'invariant'
    else:
        return 'context_specific'


# ============================================================================
#  PART 6 -- TRAINING LOOP WITH SNAPSHOTS
# ============================================================================

def run_toy_experiment(seed, tag="full",
                       enable_crystallization=True,
                       enable_crucible=True,
                       enable_agency=True,
                       collect_snapshots=False,
                       quiet=False):
    global ACTION_EMB
    np.random.seed(seed)
    lp = (lambda *a, **kw: None) if quiet else print

    lp("\n" + "="*60)
    lp("  Toy Model -- %s (seed=%d)" % (tag, seed))
    lp("="*60)

    env = ToyEnvironment(seed)
    encoder = ToyEncoder()
    sigma_est, emb_scale = calibrate_action_embedding(encoder, seed)
    base_eval = ToyBaseEvaluator(seed + 2)

    sigma_final, eff_rank = calibrate_sigma(encoder, seed)
    merge_thresh = sigma_final
    lp("  Final: sigma=%.4f, merge=%.4f, eff_rank=%d" % (sigma_final, merge_thresh, eff_rank))

    ibf = IBFAgent(sigma_final, merge_thresh, encoder, base_eval,
                   enable_crystallization, enable_crucible, enable_agency)

    snapshots = {}
    log = {'acc_A': [], 'acc_B': [], 'steps': [],
           'n_centers': [], 'n_cryst': [], 'n_verified': [],
           'k_eff_mean': [], 'entropy_mean': []}

    ctx_map = {'A': 0, 'B': 1}
    total_step = 0
    t0 = time.time()

    if collect_snapshots:
        snapshots['before'] = copy.deepcopy(ibf)

    for phase_name in ['A', 'B']:
        ctx_id = ctx_map[phase_name]
        lp("\n-- Phase %s --" % phase_name)
        ibf.set_context(ctx_id)

        for epoch in range(C.E):
            order = np.random.permutation(C.N_pool)
            ek, eH = [], []

            for idx in order:
                x = env.pool[idx]
                z_all = [encoder.encode(x, j) for j in range(C.k)]
                gt = env.correct_action(x, phase_name)

                ch, Rv, kv, ent = ibf.select_action(z_all, deterministic=False)
                ek.extend(kv.tolist())
                eH.append(ent)
                Ri = 1.0 if ch == gt else 0.0
                Rc = ibf.R_eff(z_all[ch])
                ibf.update(z_all[ch], Ri - Rc, x=x, j_chosen=ch)
                total_step += 1

            ibf.end_epoch()

            # Evaluate -- bypass set_context to preserve crucible_verified flags
            saved_ctx = ibf.current_context
            ibf.current_context = 0
            acc_a = evaluate(ibf, encoder, env, 'A')
            ibf.current_context = 1
            acc_b = evaluate(ibf, encoder, env, 'B')
            ibf.current_context = saved_ctx

            log['steps'].append(total_step)
            log['acc_A'].append(acc_a)
            log['acc_B'].append(acc_b)
            log['n_centers'].append(len(ibf.centers))
            log['n_cryst'].append(ibf.count_crystallized())
            log['n_verified'].append(ibf.count_verified())
            log['k_eff_mean'].append(float(np.mean(ek)) if ek else C.k_0)
            log['entropy_mean'].append(float(np.mean(eH)) if eH else 0.0)

            nc = len(ibf.centers)
            nx = ibf.count_crystallized()
            nv = ibf.count_verified()
            lp("  Ep %2d/%d  %d ctr (%d cryst, %d vrf)  "
               "k=%.2f H=%.3f  Acc_A=%.3f Acc_B=%.3f" %
               (epoch+1, C.E, nc, nx, nv,
                log['k_eff_mean'][-1], log['entropy_mean'][-1], acc_a, acc_b))

            if collect_snapshots:
                if phase_name == 'A' and epoch == 4:
                    snapshots['after_5_epochs'] = copy.deepcopy(ibf)
                if phase_name == 'A' and epoch == C.E - 1:
                    snapshots['end_phase_A'] = copy.deepcopy(ibf)
                if phase_name == 'B' and epoch == 0:
                    snapshots['early_phase_B'] = copy.deepcopy(ibf)
                if phase_name == 'B' and epoch == C.E - 1:
                    snapshots['end_phase_B'] = copy.deepcopy(ibf)

    elapsed = time.time() - t0
    lp("\n  Total time: %.1fs" % elapsed)

    acc_A_end_A = log['acc_A'][C.E - 1]
    acc_A_end_B = log['acc_A'][-1]
    acc_B_end_B = log['acc_B'][-1]
    BT_A = acc_A_end_B - acc_A_end_A

    lp("\n  -- Summary --")
    lp("  Centers: %d (%d cryst, %d verified)" %
       (len(ibf.centers), ibf.count_crystallized(), ibf.count_verified()))
    lp("  Acc_A (end A): %.3f" % acc_A_end_A)
    lp("  Acc_A (end B): %.3f  ->  BT_A = %+.3f" % (acc_A_end_B, BT_A))
    lp("  Acc_B (end B): %.3f" % acc_B_end_B)

    n_inv, n_ctx = 0, 0
    n_cryst_inv, n_cryst_ctx = 0, 0
    n_dissolved_inv, n_dissolved_ctx = 0, 0
    n_verified_inv, n_verified_ctx = 0, 0
    for c in ibf.centers:
        cat = classify_center(c, env)
        is_inv = (cat == 'invariant')
        if is_inv: n_inv += 1
        else:      n_ctx += 1
        if c.is_crystallized():
            if is_inv: n_cryst_inv += 1
            else:      n_cryst_ctx += 1
        if c.is_crystallized() and c.crucible_verified:
            if is_inv: n_verified_inv += 1
            else:      n_verified_ctx += 1
        if c.dissolution_log or (c.was_ever_crystallized and not c.is_crystallized()):
            if is_inv: n_dissolved_inv += 1
            else:      n_dissolved_ctx += 1

    total_dissolved = sum(len(c.dissolution_log) for c in ibf.centers)

    lp("\n  -- Center Classification --")
    lp("  Invariant:     %3d total, %3d cryst, %3d verified, %3d dissolved" %
       (n_inv, n_cryst_inv, n_verified_inv, n_dissolved_inv))
    lp("  Context-spec:  %3d total, %3d cryst, %3d verified, %3d dissolved" %
       (n_ctx, n_cryst_ctx, n_verified_ctx, n_dissolved_ctx))
    lp("  Total dissolution events: %d" % total_dissolved)

    metrics = {
        'acc_A_end_A': acc_A_end_A,
        'acc_A_end_B': acc_A_end_B,
        'acc_B_end_B': acc_B_end_B,
        'BT_A': BT_A,
        'n_centers': len(ibf.centers),
        'n_cryst': ibf.count_crystallized(),
        'n_verified': ibf.count_verified(),
        'n_dissolved': total_dissolved,
        'n_inv': n_inv, 'n_ctx': n_ctx,
        'n_cryst_inv': n_cryst_inv, 'n_cryst_ctx': n_cryst_ctx,
        'n_verified_inv': n_verified_inv, 'n_verified_ctx': n_verified_ctx,
        'n_dissolved_inv': n_dissolved_inv, 'n_dissolved_ctx': n_dissolved_ctx,
    }
    return metrics, log, ibf, env, encoder, base_eval, snapshots
