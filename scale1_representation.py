"""
scale1_representation.py
========================

Phase 3 of the Gate 1 branch: the Scale 1 representation learner.

This is NOT the v1 correction engine. Its job is to crystallize stable
representation particles directly in the 20D observation space, then induce an
emergent 2D configuration space from the crystallized structure via a behavioral
+ geometric affinity graph and a normalized-Laplacian (diffusion) embedding.

The learner never sees the hidden coordinates u. It only receives imposed
results (scalar rewards) from interacting with the environment.

Pipeline:
    fit -> crystallize -> build_particle_graph -> extract_embedding_2d -> make_encoder
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List

from gate1_environment import Gate1Config
from gate1_encoders import EmergentScaleEncoder


@dataclass
class Scale1RepresentationParticle:
    x: np.ndarray                       # raw 20D center
    sigma: float                        # raw-space kernel bandwidth
    mu_eff: float                       # decay / crystallization state
    n_updates: int = 0
    signature: np.ndarray = None        # behavioral/discrepancy signature
    signature_counts: np.ndarray = None
    D_history: List[float] = field(default_factory=list)
    stability: float = 0.0
    was_ever_crystallized: bool = False
    # Gate 1C: recent activating observations + their full reward vector, used to
    # partition a non-converging particle's jurisdiction along the behavioral
    # fault line (discrepancy-driven splitting).
    activation_log: list = field(default_factory=list)
    depth: int = 0                      # number of ancestral splits

    def is_crystallized(self):
        return self.was_ever_crystallized and self.mu_eff < 0.01

    def D_var_rolling(self, window=30):
        rec = self.D_history[-window:]
        return float(np.var(rec)) if len(rec) >= 2 else 0.0


class Scale1RepresentationLearner:
    def __init__(self, cfg: Gate1Config, seed, enable_crystallization=True,
                 graph_mode="multiplicative"):
        self.cfg = cfg
        self.seed = seed
        self.enable_crystallization = enable_crystallization
        self.graph_mode = graph_mode
        self.rng = np.random.RandomState(seed + 31)
        self.k = cfg.k                        # number of actions
        self.n_contexts = getattr(cfg, "n_contexts", 2)
        self.n_slots = self.n_contexts * self.k   # context * action
        self.particles: List[Scale1RepresentationParticle] = []
        self.sigma_x = None
        self.merge_thresh_x = None
        self.n_splits = 0
        self.n_behavioral_creations = 0

    # ------------------------------------------------------------------
    #  calibration of raw-space bandwidth
    # ------------------------------------------------------------------
    def _calibrate_sigma_x(self, X):
        Xc = X - X.mean(axis=0)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        var_exp = np.cumsum(s**2) / np.sum(s**2)
        eff_rank = int(np.searchsorted(var_exp, 0.95)) + 1
        sub = X[self.rng.choice(len(X), min(600, len(X)), replace=False)]
        dists = []
        for i in range(len(sub)):
            for j in range(i + 1, min(i + 20, len(sub))):
                dists.append(np.linalg.norm(sub[i] - sub[j]))
        med = float(np.median(dists))
        sigma = med / np.sqrt(2 * max(eff_rank, 1)) * self.cfg.sigma_x_scale
        self.eff_rank = eff_rank
        return sigma

    # ------------------------------------------------------------------
    #  kernel helpers
    # ------------------------------------------------------------------
    def _kernel_to_particles(self, x_obs):
        if not self.particles:
            return np.array([])
        X = np.array([p.x for p in self.particles])
        sig = np.array([p.sigma for p in self.particles])
        sq = np.sum((X - x_obs[np.newaxis, :])**2, axis=1)
        return np.exp(-sq / (2.0 * sig**2))

    # ------------------------------------------------------------------
    #  single-interaction update (spec interface)
    # ------------------------------------------------------------------
    def update(self, x_obs, context_id, action_id, imposed_result):
        K = self._kernel_to_particles(x_obs)
        max_K = float(np.max(K)) if K.size else 0.0
        if max_K < self.cfg.creation_thresh_repr:
            self._create_particle(x_obs)
            K = self._kernel_to_particles(x_obs)
        self._apply(K, context_id, action_id, imposed_result)

    def _create_particle(self, x_obs):
        if len(self.particles) >= self.cfg.capacity_repr:
            return
        p = Scale1RepresentationParticle(
            x=x_obs.copy(),
            sigma=self.sigma_x,
            mu_eff=self.cfg.mu_repr_base,
            signature=np.zeros(self.n_slots),
            signature_counts=np.zeros(self.n_slots, dtype=int),
        )
        self.particles.append(p)

    def _apply(self, K, context_id, action_id, imposed_result):
        slot = context_id * self.k + action_id
        for i, p in enumerate(self.particles):
            kw = float(K[i])
            if kw < self.cfg.activation_thresh_repr:
                continue
            pred = p.signature[slot]
            D = imposed_result - pred
            p.signature[slot] += self.cfg.eta_repr * kw * D
            p.signature_counts[slot] += 1
            p.D_history.append(D)
            p.n_updates += 1

    # ------------------------------------------------------------------
    #  fit: representation formation over the repr pool
    # ------------------------------------------------------------------
    def fit(self, env, verbose=False):
        X_pool = env.pool_x20
        self.sigma_x = self._calibrate_sigma_x(X_pool)
        self.merge_thresh_x = self.cfg.merge_thresh_x_frac * self.sigma_x
        if verbose:
            print("  [Scale1] sigma_x=%.4f (eff_rank=%d), merge_x=%.4f"
                  % (self.sigma_x, self.eff_rank, self.merge_thresh_x))

        # dense, balanced interaction over both contexts and both actions
        combos = [(c, a) for c in range(self.n_contexts) for a in range(self.k)]
        N = len(X_pool)

        for epoch in range(self.cfg.E_scale1):
            order = self.rng.permutation(N)
            for idx in order:
                x_obs = X_pool[idx]
                u = env.pool_u2[idx]                      # used ONLY to query env reward
                # full reward vector for this observation (one slot per ctx,action)
                reward_vec = np.empty(self.n_slots)
                for (c, a) in combos:
                    correct = env.correct_action(u, 'A' if c == 0 else 'B')
                    reward_vec[c * self.k + a] = 1.0 if a == correct else 0.0

                K = self._kernel_to_particles(x_obs)
                max_K = float(np.max(K)) if K.size else 0.0
                if max_K < self.cfg.creation_thresh_repr:
                    self._create_particle(x_obs)
                    K = self._kernel_to_particles(x_obs)
                elif self.cfg.enable_behavioral_creation:
                    # behaviorally inconsistent incumbent -> spawn a competitor
                    bi = int(np.argmax(K))
                    if self.particles[bi].D_var_rolling() > self.cfg.behavioral_creation_threshold:
                        self._create_particle(x_obs)
                        self.n_behavioral_creations += 1
                        K = self._kernel_to_particles(x_obs)

                for (c, a) in combos:
                    self._apply(K, c, a, reward_vec[c * self.k + a])
                if self.cfg.enable_splitting:
                    self._log_activation(K, x_obs, reward_vec)
            self.end_epoch()
            if verbose and (epoch % 5 == 0 or epoch == self.cfg.E_scale1 - 1):
                nc = sum(p.is_crystallized() for p in self.particles)
                print("  [Scale1] epoch %2d/%d  particles=%d  crystallized=%d  splits=%d"
                      % (epoch + 1, self.cfg.E_scale1, len(self.particles), nc, self.n_splits))
        return self

    def _log_activation(self, K, x_obs, reward_vec):
        cap = self.cfg.activation_log_cap
        for i, p in enumerate(self.particles):
            if float(K[i]) >= self.cfg.activation_thresh_repr:
                p.activation_log.append((x_obs.copy(), reward_vec.copy()))
                if len(p.activation_log) > cap:
                    del p.activation_log[0]

    # ------------------------------------------------------------------
    #  end of epoch: decay, crystallization, merge
    # ------------------------------------------------------------------
    def end_epoch(self):
        for p in self.particles:
            p.signature *= (1.0 - p.mu_eff)

        if self.cfg.enable_splitting:
            self._split_particles()

        if self.enable_crystallization:
            for p in self.particles:
                if p.is_crystallized():
                    continue
                recent = p.D_history[-30:]
                var = float(np.var(recent)) if len(recent) >= 2 else 1.0
                n_support = int(np.sum(p.signature_counts >= 2))
                if (p.n_updates >= self.cfg.n_repr_cryst_min
                        and var < self.cfg.repr_convergence_threshold
                        and n_support >= 2):
                    p.mu_eff = self.cfg.mu_repr_cryst
                    p.was_ever_crystallized = True
                    p.stability = 1.0 / (var + 1e-3)

        # refresh stability for crystallized particles
        for p in self.particles:
            if p.is_crystallized():
                recent = p.D_history[-30:]
                var = float(np.var(recent)) if len(recent) >= 2 else 0.0
                p.stability = 1.0 / (var + 1e-3)

        self._merge()

    # ------------------------------------------------------------------
    #  Gate 1C: discrepancy-driven splitting
    # ------------------------------------------------------------------
    def _split_particles(self):
        """A non-converging particle (high rolling D-variance after enough
        exposure) subdivides its jurisdiction along the behavioral fault line:
        k-means(2) on its logged activating observations -> two child particles
        at the cluster centroids, each at reduced bandwidth, re-seeded from its
        cluster's records and born transient."""
        if len(self.particles) >= self.cfg.capacity_repr:
            return
        new_particles = []
        for p in self.particles:
            if (not p.is_crystallized()
                    and p.n_updates >= self.cfg.n_repr_cryst_min
                    and p.D_var_rolling() > self.cfg.split_threshold
                    and p.sigma > self.sigma_x * self.cfg.min_split_sigma_frac
                    and len(p.activation_log) >= self.cfg.split_min_log):
                children = self._make_children(p)
                if children is not None:
                    new_particles.extend(children)
                    self.n_splits += 1
                    continue
            new_particles.append(p)
        self.particles = new_particles

    def _make_children(self, p):
        from sklearn.cluster import KMeans
        X = np.array([rec[0] for rec in p.activation_log])
        if len(np.unique(X, axis=0)) < 2:
            return None
        try:
            km = KMeans(n_clusters=2, n_init=3, random_state=self.seed).fit(X)
        except Exception:
            return None
        labels = km.labels_
        children = []
        for cl in (0, 1):
            recs = [p.activation_log[t] for t in range(len(labels)) if labels[t] == cl]
            if len(recs) < 3:
                continue
            centroid = km.cluster_centers_[cl]
            child = Scale1RepresentationParticle(
                x=centroid.copy(),
                sigma=p.sigma * self.cfg.split_sigma_factor,
                mu_eff=self.cfg.mu_repr_base,
                signature=np.zeros(self.n_slots),
                signature_counts=np.zeros(self.n_slots, dtype=int),
                depth=p.depth + 1,
            )
            child.activation_log = list(recs[-self.cfg.activation_log_cap:])
            # re-seed the child's signature from its cluster's records
            for (xo, rv) in recs:
                kw = float(np.exp(-np.sum((xo - centroid) ** 2) / (2.0 * child.sigma ** 2)))
                if kw < self.cfg.activation_thresh_repr:
                    continue
                for slot in range(self.n_slots):
                    D = rv[slot] - child.signature[slot]
                    child.signature[slot] += self.cfg.eta_repr * kw * D
                    child.signature_counts[slot] += 1
                    child.D_history.append(D)
                    child.n_updates += 1
            children.append(child)
        if len(children) < 2:
            return None          # only a genuine 2-way split counts
        return children

    def _merge(self):
        M = len(self.particles)
        if M < 2:
            return
        X = np.array([p.x for p in self.particles])
        Xsq = np.sum(X**2, axis=1)
        D2 = Xsq[:, None] + Xsq[None, :] - 2.0 * (X @ X.T)
        np.fill_diagonal(D2, np.inf)
        D2 = np.maximum(D2, 0.0)

        merged = set()
        survivors = []
        for i in range(M):
            if i in merged:
                continue
            base = self.particles[i]
            for j in range(i + 1, M):
                if j in merged:
                    continue
                if np.sqrt(D2[i, j]) >= self.merge_thresh_x:
                    continue
                sig_dist = np.linalg.norm(base.signature - self.particles[j].signature)
                if sig_dist >= self.cfg.merge_thresh_signature:
                    continue
                other = self.particles[j]
                # weighted-average merge into the more-experienced particle
                if other.n_updates > base.n_updates:
                    base, other = other, base
                w1, w2 = base.n_updates + 1, other.n_updates + 1
                base.signature = (base.signature * w1 + other.signature * w2) / (w1 + w2)
                base.signature_counts = base.signature_counts + other.signature_counts
                base.n_updates += other.n_updates
                base.D_history.extend(other.D_history)
                base.was_ever_crystallized = base.was_ever_crystallized or other.was_ever_crystallized
                base.mu_eff = min(base.mu_eff, other.mu_eff)
                base.stability = max(base.stability, other.stability)
                merged.add(j)
            survivors.append(base)

        # capacity pressure: keep crystallized first
        if len(survivors) > self.cfg.capacity_repr:
            cryst = [p for p in survivors if p.is_crystallized()]
            trans = sorted([p for p in survivors if not p.is_crystallized()],
                           key=lambda p: p.n_updates)
            keep = self.cfg.capacity_repr - len(cryst)
            survivors = cryst + (trans[-keep:] if keep > 0 else [])
        self.particles = survivors

    # ------------------------------------------------------------------
    #  extraction of the emergent representation
    # ------------------------------------------------------------------
    def get_crystallized_particles(self):
        return [p for p in self.particles if p.is_crystallized()]

    def get_particles_for_embedding(self, min_support=2):
        """Crystallized particles, or (ablation) transient particles with support."""
        cryst = self.get_crystallized_particles()
        if cryst:
            return cryst
        # no-crystallization ablation: fall back to sufficiently-updated transients
        return [p for p in self.particles
                if int(np.sum(p.signature_counts >= min_support)) >= 2
                and p.n_updates >= self.cfg.n_repr_cryst_min]

    def build_particle_graph(self, particles, shuffle_signatures=False):
        """
        Affinity graph over particles (spec sec 7.1):

            W_ij = exp(-||x_i-x_j||^2 / (2 sigma_gx_ij^2))      (geometry)
                   * exp(-||b_i-b_j||^2 / (2 sigma_gb^2))       (behavior)
                   * sqrt(s_i * s_j)                            (stability)

        Geometry uses a self-tuning *local* bandwidth (distance to the k-th
        neighbour) which is what makes the normalized-Laplacian embedding unfold
        the manifold; the behavior bandwidth is set adaptively (median pairwise
        signature distance) so behavior is a gentle organizing modulation rather
        than a fragmenting gate; stability is normalized to [0.5, 1].
        """
        M = len(particles)
        if M < 3:
            return None
        X = np.array([p.x for p in particles])
        B = np.array([p.signature for p in particles])
        S = np.array([max(p.stability, 1e-3) for p in particles])
        if shuffle_signatures:
            B = B[self.rng.permutation(M)]

        kk = min(self.cfg.k_nearest, M - 1)
        Dx = np.sqrt(np.maximum(
            np.sum(X**2, 1)[:, None] + np.sum(X**2, 1)[None, :] - 2.0 * (X @ X.T), 0.0))
        # self-tuning local geometry bandwidth (Zelnik-Manor & Perona)
        kth = np.maximum(np.sort(Dx, axis=1)[:, kk], 1e-6)
        Wx = np.exp(-Dx**2 / (kth[:, None] * kth[None, :]))

        Db = np.sqrt(np.maximum(
            np.sum(B**2, 1)[:, None] + np.sum(B**2, 1)[None, :] - 2.0 * (B @ B.T), 0.0))
        iu = np.triu_indices(M, 1)
        med_b = float(np.median(Db[iu])) if len(iu[0]) else 1.0
        sigma_gb = max(self.cfg.sigma_graph_b_frac * med_b, 1e-3)
        # behavior self-tuning local bandwidth (for the behavior-driven modes)
        kthb = np.maximum(np.sort(Db, axis=1)[:, kk], 1e-6)
        Wb_local = np.exp(-Db**2 / (kthb[:, None] * kthb[None, :]))
        Wb = np.exp(-Db**2 / (2.0 * sigma_gb**2))

        Sn = 0.5 + 0.5 * (S - S.min()) / (S.max() - S.min() + 1e-9)
        Wstab = np.sqrt(Sn[:, None] * Sn[None, :])

        mode = self.graph_mode
        a = self.cfg.graph_alpha
        b = self.cfg.graph_beta

        def sparsify_by(Dist, Wfull):
            Wsp = np.zeros_like(Wfull)
            for i in range(M):
                nn = np.argsort(Dist[i])[1:kk + 1]
                Wsp[i, nn] = Wfull[i, nn]
            return np.maximum(Wsp, Wsp.T)

        if mode == 'multiplicative':
            # BASELINE (Gate 1A): behavior multiplicatively gates geometric kNN.
            W = sparsify_by(Dx, Wx * Wb * Wstab)
        elif mode == 'additive':
            # mixed-view affinity: geometry OR behavior can carry an edge.
            Wmix = (a * Wx + b * Wb_local) * Wstab
            # sparsify by the combined affinity (largest mixed weight = nearest)
            W = sparsify_by(-Wmix, Wmix)
        elif mode == 'union':
            # neighborhood = kNN_geometry UNION kNN_behavior.
            Wfull = Wx * Wb * Wstab
            Wsp = np.zeros_like(Wfull)
            for i in range(M):
                gi = np.argsort(Dx[i])[1:kk + 1]
                bi = np.argsort(Db[i])[1:kk + 1]
                idx = np.union1d(gi, bi)
                Wsp[i, idx] = Wfull[i, idx]
            W = np.maximum(Wsp, Wsp.T)
        elif mode == 'behavior_first':
            # connectivity primarily behavioral; geometry only a mild regularizer.
            reg = self.cfg.behavior_first_geom_reg
            Wfull = Wb_local * Wstab * (reg + (1.0 - reg) * Wx)
            W = sparsify_by(Db, Wfull)
        else:
            raise ValueError("unknown graph_mode: %s" % mode)

        W[W < 1e-10] = 0.0
        return W

    def extract_embedding_2d(self, particles, shuffle_signatures=False):
        """Return q_particles (M, 2): normalized-Laplacian (diffusion) embedding."""
        from sklearn.manifold import SpectralEmbedding
        M = len(particles)
        if M < 3:
            return np.zeros((M, 2))
        W = self.build_particle_graph(particles, shuffle_signatures=shuffle_signatures)
        try:
            q = SpectralEmbedding(n_components=2, affinity='precomputed',
                                  random_state=self.seed).fit_transform(W)
        except Exception:
            # robustness fallback: PCA of particle positions
            X = np.array([p.x for p in particles])
            Xc = X - X.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            q = Xc @ Vt[:2].T
        # normalize: zero mean, unit variance per dimension
        q = q - q.mean(axis=0)
        std = q.std(axis=0)
        std[std < 1e-8] = 1.0
        return q / std

    def make_encoder(self, action_embedding, particles=None, q_particles=None,
                     shuffle_signatures=False):
        if particles is None:
            particles = self.get_particles_for_embedding()
        if q_particles is None:
            q_particles = self.extract_embedding_2d(
                particles, shuffle_signatures=shuffle_signatures)
        bandwidth = self.sigma_x * self.cfg.interp_sigma_frac
        return EmergentScaleEncoder(particles, q_particles, action_embedding,
                                    bandwidth=bandwidth)
