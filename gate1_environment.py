"""
gate1_environment.py
====================

Phase 1 of the Gate 1 branch: the two-scale toy environment.

A hidden 2D variable u = [u1, u2] is observed only through a fixed nonlinear
20D embedding x20. The learner sees x20 only. The true coordinates u are used
exclusively for:

    1. generating the environment;
    2. the oracle frozen-encoder baseline;
    3. evaluation metrics / plot diagnostics.

The Scale 1 representation learner must never touch u.

Task structure (identical logic to the original 2D toy model, but truth is
computed from the hidden u, not the observation):

    s_j(u, c) = beta * u1 * p_j + alpha * u_c * u2 * r_j
    correct_action = argmax_j s_j(u, c)

with p = r = [+1, -1], u_A = +1, u_B = -1, alpha = beta = 1.0.
"""

import numpy as np
from dataclasses import dataclass


# Action / context constants (mirror the v1 toy model).
P_VEC = np.array([+1.0, -1.0])
R_VEC = np.array([+1.0, -1.0])
U_CTX = {'A': +1.0, 'B': -1.0}
ALPHA = 1.0
BETA = 1.0


@dataclass
class Gate1Config:
    # ---- dimensions ----
    d_hidden: int = 2
    D_obs: int = 20
    k: int = 2                      # number of actions
    n_contexts: int = 2             # number of contexts (A, B)

    # ---- dataset sizes ----
    N_repr_pool: int = 2000         # Scale 1 representation fitting pool
    N_train_pool: int = 1000        # Scale 2 training pool
    N_test: int = 1000              # held-out evaluation
    noise_std: float = 0.03
    manifold: str = "normal"        # "normal" or "uniform"
    generator: str = "1A"           # "1A" (geometry-easy) or "1B" (stress)
    u2_freq: float = 3.0            # Gate 1B: high-frequency aliasing of u2
    u_C: float = None               # Gate 3: third-context coefficient (partial overlap)

    # ---- Scale 1 representation dynamics ----
    E_scale1: int = 30
    eta_repr: float = 0.10
    mu_repr_base: float = 0.03
    mu_repr_cryst: float = 0.001
    n_repr_cryst_min: int = 20
    repr_convergence_threshold: float = 0.05
    creation_thresh_repr: float = 0.35
    activation_thresh_repr: float = 0.15
    capacity_repr: int = 2000
    sigma_x_scale: float = 1.0      # multiplier on auto-calibrated raw bandwidth
    merge_thresh_x_frac: float = 0.5    # fraction of sigma_x
    merge_thresh_signature: float = 0.15

    # ---- emergent graph / embedding ----
    k_nearest: int = 15
    sigma_graph_b_frac: float = 2.0     # behavior bandwidth = frac * median sig-dist
    interp_sigma_frac: float = 0.6      # interpolation bandwidth = frac * sigma_x
    # graph-variant knobs (Gate 1B behavioral-geometry branch)
    graph_alpha: float = 0.5            # additive mode: geometry weight
    graph_beta: float = 0.5             # additive mode: behavior weight
    behavior_first_geom_reg: float = 0.3  # behavior-first: geometry regularizer floor

    # ---- Gate 1C: discrepancy-driven splitting of non-converging particles ----
    enable_splitting: bool = False
    split_threshold: float = 0.10       # split if rolling D-variance exceeds this
    split_sigma_factor: float = 0.7     # child bandwidth = factor * parent
    split_min_log: int = 8              # min logged activations to attempt a split
    min_split_sigma_frac: float = 0.30  # do not split below frac * sigma_x (depth cap)
    activation_log_cap: int = 60        # per-particle activation memory (samples)
    # Gate 1C second lever: behavior-augmented particle creation
    enable_behavioral_creation: bool = False
    behavioral_creation_threshold: float = 0.10

    # ---- Scale 2 ----
    E_scale2: int = 25              # epochs per context


def make_features_1A(U):
    """Gate 1A feature map R^2 -> R^20: a smooth, near-isometric embedding.

    Both hidden coordinates enter through low-frequency / polynomial terms, so
    ambient geometry alone already preserves the manifold (the "geometry-easy"
    regime). This is the original Gate 1 generator.
    """
    u1 = U[:, 0]
    u2 = U[:, 1]
    return np.stack([
        u1, u2, u1**2, u2**2, u1 * u2,
        np.sin(u1), np.sin(u2), np.cos(u1), np.cos(u2),
        np.tanh(u1), np.tanh(u2),
        u1**3, u2**3, u1**2 * u2, u1 * u2**2,
        np.exp(-0.5 * u1**2), np.exp(-0.5 * u2**2),
        np.sin(u1 + u2), np.cos(u1 - u2), u1 - u2,
    ], axis=1)


def make_features_1B(U, f=3.0):
    """Gate 1B stress feature map R^2 -> R^20: geometry-only is insufficient.

    u1 is encoded through smooth, low-frequency terms -> ambient geometry
    recovers u1. u2 is encoded ONLY through high-frequency (aliased) terms
    sin(f*u2), cos(f*u2), ... -> globally, ambient distance does NOT track
    |delta u2| (points a full period apart in u2 are near in x, neighbours can
    have very different true u2), so raw-distance / PCA / spectral embeddings
    cannot order u2. u2 remains *locally* accessible (the high-frequency map is
    locally invertible within a period), so a Scale 1 particle still localizes
    u2 well enough to form a u2-dependent behavioral signature -- the only
    channel through which the global u2 ordering survives.
    """
    u1 = U[:, 0]
    u2 = U[:, 1]
    return np.stack([
        # u1: smooth / geometry-accessible (12 features)
        u1, u1**2, u1**3, np.sin(u1), np.cos(u1), np.tanh(u1),
        np.exp(-0.5 * u1**2), np.sin(2 * u1), np.cos(2 * u1), u1,
        np.sin(3 * u1), np.cos(3 * u1),
        # u2: high-frequency aliased (6 features)
        np.sin(f * u2), np.cos(f * u2), np.sin(2 * f * u2), np.cos(2 * f * u2),
        np.sin(f * u2 + 1.0), np.cos(f * u2 + 1.0),
        # mild aliased coupling (2 features)
        0.2 * np.sin(f * (u1 + u2)), 0.2 * np.cos(f * (u1 - u2)),
    ], axis=1)


def make_features(U, generator="1A", f=3.0):
    """Fixed nonlinear feature map R^2 -> R^20 for the chosen generator."""
    if generator == "1B":
        return make_features_1B(U, f=f)
    return make_features_1A(U)


class TwoScaleToyEnvironment:
    """Hidden 2D manifold observed through a fixed nonlinear 20D embedding."""

    def __init__(self, seed, cfg: Gate1Config = None):
        self.cfg = cfg if cfg is not None else Gate1Config()
        self.seed = seed
        self.k = self.cfg.k
        self.u_ctx = {'A': +1.0, 'B': -1.0}
        if self.cfg.u_C is not None:
            self.u_ctx['C'] = float(self.cfg.u_C)
        # action coefficient vectors: 2 actions keep the original +/-1 contrast;
        # k>2 actions are evenly spaced directions on the unit circle, so the
        # correct action is the angular sector of [u1, u_c*u2] -- a balanced
        # k-way task in which BOTH hidden coordinates matter.
        if self.k == 2:
            self.p_vec = np.array([+1.0, -1.0])
            self.r_vec = np.array([+1.0, -1.0])
        else:
            ang = 2.0 * np.pi * np.arange(self.k) / self.k
            self.p_vec = np.cos(ang)
            self.r_vec = np.sin(ang)
        rng = np.random.RandomState(seed)

        # ---- fixed random orthogonal mixing matrix ----
        A = rng.randn(self.cfg.D_obs, self.cfg.D_obs)
        Q, Rmat = np.linalg.qr(A)
        # make the sign deterministic
        Q = Q * np.sign(np.diag(Rmat))[np.newaxis, :]
        self.R = Q

        # ---- reference standardization statistics (fixed) ----
        ref = self._sample_u(rng, 8000)
        F = make_features(ref, self.cfg.generator, self.cfg.u2_freq)
        self.feat_mean = F.mean(axis=0)
        self.feat_std = F.std(axis=0) + 1e-8
        F_std = (F - self.feat_mean) / self.feat_std
        Xr = F_std @ self.R
        self.obs_mean = Xr.mean(axis=0)
        self.obs_std = Xr.std(axis=0) + 1e-8

        # ---- generate the datasets ----
        self.pool_u2 = self._sample_u(rng, self.cfg.N_repr_pool)
        self.pool_x20 = self.embed(self.pool_u2, rng)

        self.train_u2 = self._sample_u(rng, self.cfg.N_train_pool)
        self.train_x20 = self.embed(self.train_u2, rng)

        self.test_A_u2 = self._sample_u(rng, self.cfg.N_test)
        self.test_A_x20 = self.embed(self.test_A_u2, rng)
        self.test_B_u2 = self._sample_u(rng, self.cfg.N_test)
        self.test_B_x20 = self.embed(self.test_B_u2, rng)

    # ------------------------------------------------------------------
    #  data generation
    # ------------------------------------------------------------------
    def _sample_u(self, rng, n):
        if self.cfg.manifold == "uniform":
            return rng.uniform(-2.5, 2.5, size=(n, self.cfg.d_hidden))
        return rng.randn(n, self.cfg.d_hidden)

    def embed(self, U, rng=None):
        """Map hidden coords U (N,2) to observations x20 (N,20)."""
        F = make_features(np.atleast_2d(U), self.cfg.generator, self.cfg.u2_freq)
        F_std = (F - self.feat_mean) / self.feat_std
        X = F_std @ self.R
        if rng is not None and self.cfg.noise_std > 0:
            X = X + rng.randn(*X.shape) * self.cfg.noise_std
        X = (X - self.obs_mean) / self.obs_std
        return X

    # ------------------------------------------------------------------
    #  task: truth computed from hidden u (NOT from x20)
    # ------------------------------------------------------------------
    def _u_c(self, ctx):
        # context coefficient: A=+1, B=-1, optional Gate 3 context C (partial
        # overlap, set via cfg.u_C). Falls back to the module dict for A/B.
        if hasattr(self, "u_ctx") and ctx in self.u_ctx:
            return self.u_ctx[ctx]
        return U_CTX[ctx]

    def score_clean(self, u, ctx):
        u_c = self._u_c(ctx)
        s = np.zeros(self.k)
        for j in range(self.k):
            s[j] = BETA * u[0] * self.p_vec[j] + ALPHA * u_c * u[1] * self.r_vec[j]
        return s

    def correct_action(self, u, ctx):
        return int(np.argmax(self.score_clean(u, ctx)))

    def correct_actions_batch(self, U, ctx):
        u_c = self._u_c(ctx)
        U = np.atleast_2d(U)
        N = len(U)
        S = np.zeros((N, self.k))
        for j in range(self.k):
            S[:, j] = BETA * U[:, 0] * self.p_vec[j] + ALPHA * u_c * U[:, 1] * self.r_vec[j]
        return np.argmax(S, axis=1)

    # convenience accessors used by the Scale 2 runner
    def get_test_obs(self, ctx):
        return self.test_A_x20 if ctx == 'A' else self.test_B_x20

    def get_test_u(self, ctx):
        return self.test_A_u2 if ctx == 'A' else self.test_B_u2
