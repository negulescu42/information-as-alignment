"""
gate1_encoders.py
=================

Phase 2 of the Gate 1 branch: encoders that turn an observation + action index
into a z-vector for the v1 IBFAgent.

All encoders share the same interface:

    encode(observation, action_idx)            -> 1D z-vector
    encode_batch(observations, action_indices) -> 2D (N, z_dim)

and carry a mutable `action_embedding` matrix (k x k) so the Scale 2 runner can
calibrate action separation per encoder, exactly as the v1 toy model does.

The "observation" fed to an encoder depends on the condition and is chosen
*explicitly by the runner*:

    Oracle2DEncoder      <- u_true   (the hidden 2D coords)
    Raw20DEncoder        <- x20
    PCA2DEncoder         <- x20
    Random2DEncoder      <- x20
    EmergentScaleEncoder <- x20

The emergent encoder never receives u_true.
"""

import numpy as np


class _BaseEncoder:
    """Coords(observation) ++ action_embedding[action]."""

    def __init__(self, action_embedding):
        self.action_embedding = action_embedding

    def _coords(self, obs):
        raise NotImplementedError

    def _coords_batch(self, obs):
        raise NotImplementedError

    def encode(self, observation, action_idx):
        return np.concatenate([self._coords(np.asarray(observation, dtype=float)),
                               self.action_embedding[action_idx]])

    def encode_batch(self, observations, action_indices):
        coords = self._coords_batch(np.asarray(observations, dtype=float))
        return np.concatenate([coords, self.action_embedding[action_indices]], axis=1)


class Oracle2DEncoder(_BaseEncoder):
    """Frozen oracle: the observation passed in IS the true 2D coordinate."""

    def __init__(self, action_embedding):
        super().__init__(action_embedding)

    def _coords(self, obs):
        return obs

    def _coords_batch(self, obs):
        return obs


class Raw20DEncoder(_BaseEncoder):
    """Raw control: pass the 20D observation straight through."""

    def __init__(self, action_embedding):
        super().__init__(action_embedding)

    def _coords(self, obs):
        return obs

    def _coords_batch(self, obs):
        return obs


class PCA2DEncoder(_BaseEncoder):
    """Unsupervised PCA(2) of x20, standardized to unit variance per component."""

    def __init__(self, action_embedding, fit_x20):
        super().__init__(action_embedding)
        X = np.asarray(fit_x20, dtype=float)
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components = Vt[:2]                       # (2, D)
        proj = Xc @ self.components.T
        self.proj_std = proj.std(axis=0) + 1e-8

    def _coords_batch(self, obs):
        return ((obs - self.mean) @ self.components.T) / self.proj_std

    def _coords(self, obs):
        return self._coords_batch(obs[np.newaxis, :])[0]


class Random2DEncoder(_BaseEncoder):
    """Random linear projection 20D -> 2D, standardized to unit variance."""

    def __init__(self, action_embedding, fit_x20, seed):
        super().__init__(action_embedding)
        rng = np.random.RandomState(seed + 7777)
        D = np.asarray(fit_x20).shape[1]
        self.P = rng.randn(D, 2)
        X = np.asarray(fit_x20, dtype=float)
        self.mean = X.mean(axis=0)
        proj = (X - self.mean) @ self.P
        self.proj_std = proj.std(axis=0) + 1e-8

    def _coords_batch(self, obs):
        return ((obs - self.mean) @ self.P) / self.proj_std

    def _coords(self, obs):
        return self._coords_batch(obs[np.newaxis, :])[0]


class PassThroughEncoder(_BaseEncoder):
    """Coords ARE the observation (any dimension) ++ action embedding.

    Used by Gate 1D to feed pre-computed interactive coordinates straight into
    the v1 Scale 2 engine, exactly as Oracle2DEncoder feeds true 2D coords.
    """

    def __init__(self, action_embedding):
        super().__init__(action_embedding)

    def _coords(self, obs):
        return obs

    def _coords_batch(self, obs):
        return obs


def probe_signature(u, env, k, n_probes, rng, contexts=('A', 'B')):
    """
    Interactive behavioral signature at a point, built by taking exploratory
    actions and observing rewards (Gate 1D). Rewards are deterministic given
    (u, context, action): reward = 1 if action is correct else 0. Each probe is
    one (context, action) interaction sampled uniformly; an unobserved slot
    defaults to 0.5 (unknown). The signature dimension is k * len(contexts).

    This is the only channel through which an observation-aliased coordinate can
    enter the representation at test time, because different hidden coordinates
    produce different reward patterns across the same actions.
    """
    n_slots = k * len(contexts)
    seen = np.full(n_slots, 0.5)
    correct = {c: env.correct_action(u, c) for c in contexts}
    for _ in range(n_probes):
        ci = rng.randint(len(contexts))
        a = rng.randint(k)
        ctx = contexts[ci]
        seen[ci * k + a] = 1.0 if a == correct[ctx] else 0.0
    return seen


def build_interactive_coords(static_encoder, X20, U, env, k, n_probes, seed,
                             contexts=('A', 'B')):
    """
    q2 = [ static geometric projection(x20) , interactive probe signature ].

    The probe signature uses U only to query the environment for rewards (i.e.
    to interact); it is never used as a coordinate. Returns (N, 2 + k*|ctx|).
    """
    q_geom = static_encoder.encode_observation_batch(X20)        # (N, 2)
    rng = np.random.RandomState(seed + 4242)
    sig = np.array([probe_signature(U[i], env, k, n_probes, rng, contexts)
                    for i in range(len(U))])
    return np.concatenate([q_geom, sig], axis=1)


def build_interactive_coords_informed(geom_encoder, scout, scout_action_emb,
                                      X20, U, env, k, n_probes, seed,
                                      contexts=('A', 'B')):
    """
    Interactive coords whose probe actions are chosen by AGENCY: a Boltzmann
    policy over the current Scale 2 corrections (the `scout` agent), rather than
    uniform random. This tests whether Scale 2 agency, fed back into Scale 1
    representation formation, produces better probes than random -- the first
    cross-scale agency signal.

    The scout is trained on the static geometric coords (q_geom); its k_eff*R_eff
    Boltzmann allocates exploration to where it is uncertain (near decision
    boundaries, where the aliased coordinate matters) and exploits elsewhere.
    """
    qg = geom_encoder.encode_observation_batch(X20)
    rng = np.random.RandomState(seed + 4242)
    ctx_id = {'A': 0, 'B': 1}
    N = len(U)
    n_slots = k * len(contexts)
    out = np.full((N, n_slots), 0.5)
    for i in range(N):
        correct = {c: env.correct_action(U[i], c) for c in contexts}
        for _ in range(n_probes):
            ci = rng.randint(len(contexts))
            ctx = contexts[ci]
            scout.current_context = ctx_id[ctx]
            zs = [np.concatenate([qg[i], scout_action_emb[a]]) for a in range(k)]
            R = np.array([scout.R_eff(z) for z in zs])
            kk = np.array([scout.k_eff(z) for z in zs])
            logits = kk * R
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            a = int(rng.choice(k, p=p))
            out[i, ci * k + a] = 1.0 if a == correct[ctx] else 0.0
    return np.concatenate([qg, out], axis=1)


class EmergentScaleEncoder(_BaseEncoder):
    """
    The Gate 1 candidate. Interpolates an emergent 2D coordinate q_hat(x) from
    nearby crystallized Scale 1 particles using a raw-space Gaussian kernel:

        q_hat(x) = sum_i K_raw(x, x_i) q_i / sum_i K_raw(x, x_i)

    Falls back to the nearest crystallized particle if the kernel mass is too
    small. The hidden coords u are never used.
    """

    def __init__(self, particles, q_particles, action_embedding, bandwidth=None, eps=1e-8):
        super().__init__(action_embedding)
        self.X = np.array([p.x for p in particles], dtype=float)     # (M, 20)
        if bandwidth is None:
            self.sigma = np.array([p.sigma for p in particles], dtype=float)
        else:
            self.sigma = np.full(len(particles), float(bandwidth))
        self.q = np.asarray(q_particles, dtype=float)                # (M, 2)
        self.eps = eps
        self._Xsq = np.sum(self.X**2, axis=1)                        # (M,)

    def _kernel(self, obs_batch):
        # squared distances (N, M)
        obs_sq = np.sum(obs_batch**2, axis=1, keepdims=True)
        sq = obs_sq + self._Xsq[np.newaxis, :] - 2.0 * (obs_batch @ self.X.T)
        sq = np.maximum(sq, 0.0)
        return np.exp(-sq / (2.0 * self.sigma[np.newaxis, :]**2))

    def encode_observation_batch(self, obs_batch):
        obs_batch = np.atleast_2d(np.asarray(obs_batch, dtype=float))
        K = self._kernel(obs_batch)                                  # (N, M)
        denom = K.sum(axis=1)                                        # (N,)
        q_hat = (K @ self.q)                                         # (N, 2)
        good = denom > 1e-6
        q_hat[good] = q_hat[good] / denom[good, np.newaxis]
        if np.any(~good):
            # fallback: nearest crystallized particle
            obs_sq = np.sum(obs_batch[~good]**2, axis=1, keepdims=True)
            sq = obs_sq + self._Xsq[np.newaxis, :] - 2.0 * (obs_batch[~good] @ self.X.T)
            nn = np.argmin(sq, axis=1)
            q_hat[~good] = self.q[nn]
        return q_hat

    def encode_observation(self, x_obs):
        return self.encode_observation_batch(np.asarray(x_obs, dtype=float)[np.newaxis, :])[0]

    def _coords_batch(self, obs):
        return self.encode_observation_batch(obs)

    def _coords(self, obs):
        return self.encode_observation(obs)
