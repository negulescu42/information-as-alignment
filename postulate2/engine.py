"""
postulate2.engine  --  the Scale-2 correction engine (Postulate 1) + two hooks.

This is the ORIGINAL IBF continual-learning engine (Postulate 1: discrepancy ->
localized correction -> crystallization -> memory / agency / self-correction),
UNCHANGED except for two small, guarded additions used by the Postulate-2 layer.
Both are no-ops unless a caller opts in, so Postulate-1 behavior is identical.

  1. MemoryCenter.frozen  (default False)
       A read-only "reserve" center: it is READ during same-context readout
       (normal context gating still applies) but is NEVER written -- skipped in
       the Crucible, the same-context update, epoch decay/crystallization, and
       merge. Used by context-aware interface promotion (see promotion.py) so a
       compressed interface can preserve prior-context knowledge without letting
       later contexts erode it.

  2. MemoryCenter.interface_group  (default None)
       Optional refinement: when several boundary centers of one promoted
       interface co-activate under cross-context exposure, the interface absorbs
       the reversal pressure as a UNIT (one peak center's worth) instead of N
       copies. Guarded -> exact original behavior when interface_group is None.

Everything else (kernel readout, action selection, discrepancy write path,
crystallization, the Crucible, merge) is the original engine.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from collections import defaultdict


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
    interface_group: int = None      # Gate 3 Path C: promoted-interface membership
    frozen: bool = False             # Gate 3B: read-only reserve (same-context readout
                                     # only; never updated / crystallized / dissolved)

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
            # Gate 3 Path C: a promoted interface absorbs cross-context pressure
            # AS A UNIT. Without this, each of an interface's N boundary centers
            # takes its own full juris_D = D*kw, so the interface's total v-drift
            # is sum over N centers; with fewer centers sharing the load (interior
            # dormant) each drifts faster. We normalise the v-update by the
            # interface's total activation so the interface absorbs the pressure of
            # ONE peak center, not N copies. Detection signal (D_history, used by
            # the reversal test) is left RAW so the Crucible is not weakened.
            group_sum, group_max = {}, {}
            for i, c in enumerate(self.centers):
                g = c.interface_group
                if g is not None and c.is_crystallized() and c.context_id != self.current_context:
                    kw = float(K_all[i])
                    if kw >= C.activation_thresh:
                        group_sum[g] = group_sum.get(g, 0.0) + kw
                        group_max[g] = max(group_max.get(g, 0.0), kw)
            for i, c in enumerate(self.centers):
                if c.frozen:                       # Gate 3B: read-only reserve
                    continue
                if c.is_crystallized() and c.context_id != self.current_context:
                    kw = float(K_all[i])
                    if kw >= C.activation_thresh:
                        g = c.interface_group
                        if g is not None and group_sum.get(g, 0.0) > 1e-9:
                            # interface absorbs the pressure of ONE peak center:
                            # total over the group = D * peak_kw (<= flat D*sum_kw,
                            # and exactly = flat when a single center is activated).
                            eff_kw = group_max[g] * kw / group_sum[g]
                        else:
                            eff_kw = kw                      # standard (flat) path
                        juris_D = D * eff_kw
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
            if c.frozen:                           # Gate 3B: read-only reserve
                continue
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
            if c.frozen:                           # Gate 3B: read-only reserve
                continue
            c.v *= (1.0 - c.mu_eff)
            c.w *= (1.0 - c.mu_eff)

        for c in self.centers:
            if c.frozen:                           # Gate 3B: read-only reserve
                continue
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
        # Gate 3B: frozen read-only reserves never merge; hold them aside and
        # re-append unchanged.
        reserve = [c for c in self.centers if c.frozen]
        if reserve:
            self.centers = [c for c in self.centers if not c.frozen]
        if len(self.centers) < 2:
            self.centers = self.centers + reserve
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
        self.centers = new + reserve

    def count_crystallized(self):
        return sum(1 for c in self.centers if c.is_crystallized())

    def count_verified(self):
        return sum(1 for c in self.centers
                   if c.is_crystallized() and c.crucible_verified)


# ============================================================================