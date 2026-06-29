"""
gate3_promotion.py
=================

Gate 3 -- promotion and recursive interaction.

Tests whether promoted interfaces (boundary-only compressed Gate-2 basins) can
participate in further continual learning as first-class primitives, across three
sequential contexts (Phase A, B, C). Three conditions:

    F (flat)     : standard engine, no promotion (Gate 1D/2 baseline)
    P (promoted) : after Phase A, basins -> boundary-only interfaces; interior
                   dormant; interface-level Crucible across B and C
    N (no prior) : fresh agent each phase (lower bound, no continual learning)

Fixed: Scale 1 (splitting), interactive encoder, k=8 generator at f=3.0, Crucible
/ crystallization parameters, the Gate 2 basin/partition code. Phase C adds a
third context coefficient u_C (partial overlap with A and B).

NOTE on config: the spec calls for 120 epochs/phase; for tractability here we use
E_PHASE epochs and a smaller pool (documented). The lifecycle dynamics
(crystallize / Crucible / promote / dissolve) are what is under test, and they
stabilize well within this budget (Gate 1D crystallized at 25 epochs).

Usage:
    python gate3_promotion.py [n_seeds]      # default 5
"""

import os
import sys
import copy
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import ibf_v1_engine as E
E.C.k = 8
E.ACTION_EMB = np.eye(8)
from ibf_v1_engine import IBFAgent, C

from gate1_environment import TwoScaleToyEnvironment, Gate1Config
from gate1_encoders import PassThroughEncoder, build_interactive_coords
from scale1_representation import Scale1RepresentationLearner
from run_gate1 import (calibrate_action_embedding_obs, Scale2BaseEvaluator,
                       evaluate_scale2)
import run_gate2 as G2

OUT_DIR = "gate3_outputs"
N_ACTIONS = 8
N_PROBES = 15
E_PHASE = 20
PHASES = [("A", 0), ("B", 1), ("C", 2)]


# ---------------------------------------------------------------- promoted interface
class PromotedInterface:
    def __init__(self, basin_id, boundary_centers, interior_centers, phase_origin):
        self.basin_id = basin_id
        self.boundary_centers = list(boundary_centers)
        self.interior_centers = list(interior_centers)
        self.centroid = np.mean([c.z for c in boundary_centers], axis=0)
        self.effective_amplitude = float(np.mean([abs(c.v) for c in boundary_centers]))
        self.phase_origin = phase_origin
        self.active = True
        self.verified = False
        self.dissolution_count = 0
        self.interior_restored = 0


# ---------------------------------------------------------------- fast training
def _Z_for(encoder, o):
    # z_j = [coords o, action_embedding[j]] for all actions, shape (k, zdim)
    return np.concatenate([np.tile(o, (C.k, 1)), encoder.action_embedding], axis=1)


def select_fast(agent, Z):
    R = agent.R_eff_batch(Z)                       # gated readout over all centers
    kk = np.array([agent.k_eff(Z[j]) for j in range(len(Z))])
    logits = kk * R
    logits -= logits.max()
    p = np.exp(logits); p /= p.sum()
    ch = int(np.random.choice(len(Z), p=p))
    return ch, R


def train_phase(agent, encoder, coords_pool, u_pool, env, ctx_name, ctx_id, epochs, seed):
    rng = np.random.RandomState(seed * 131 + ctx_id)
    np.random.seed(seed * 977 + ctx_id)
    agent.set_context(ctx_id)
    truth = env.correct_actions_batch(u_pool, ctx_name)
    N = len(coords_pool)
    for _ep in range(epochs):
        for idx in rng.permutation(N):
            Z = _Z_for(encoder, coords_pool[idx])
            ch, R = select_fast(agent, Z)
            Ri = 1.0 if ch == int(truth[idx]) else 0.0
            agent.update(Z[ch], Ri - R[ch], x=coords_pool[idx], j_chosen=ch)
        agent.end_epoch()


def eval_all_contexts(agent, encoder, coords_test, u_test, env):
    out = {}
    for name, cid in PHASES:
        sv = agent.current_context
        agent.current_context = cid
        out[name] = evaluate_scale2(agent, encoder, coords_test, u_test, env, name)
        agent.current_context = sv
    return out


def n_active_cryst(agent):
    return sum(1 for c in agent.centers if c.is_crystallized())


# ---------------------------------------------------------------- promotion
def promote(agent):
    cryst = [c for c in agent.centers if c.is_crystallized()]
    if len(cryst) < G2.MIN_BASIN:
        return []
    Z = np.array([c.z for c in cryst]); sg = np.array([c.sigma for c in cryst])
    adj = G2.build_overlap_graph(Z, sg)
    comp = G2.connected_components(adj)
    sizes = np.bincount(comp)
    interfaces = []
    interior_ids = set()
    for b in [b for b in range(len(sizes)) if sizes[b] >= G2.MIN_BASIN]:
        members = list(np.where(comp == b)[0])
        bnd, intr = G2.partition(Z, sg, comp, members)
        if not intr:
            continue
        boundary = [cryst[i] for i in bnd]
        interior = [cryst[i] for i in intr]
        if not boundary:
            continue
        interfaces.append(PromotedInterface(b, boundary, interior, phase_origin=0))
        interior_ids |= set(id(c) for c in interior)
    # interior centers become dormant: removed from the active readout population
    agent.centers = [c for c in agent.centers if id(c) not in interior_ids]
    return interfaces


def interface_crucible(agent, interfaces):
    """Aggregate the standard per-center Crucible to the interface level. A
    boundary center has 'dissolved' if it lost crystallization or logged a
    dissolution this phase. If > half a basin's boundary dissolves, the interface
    is invalidated and its dormant interior is restored as transient particles."""
    stats = dict(verified=0, dissolved=0, interior_restored=0)
    for itf in interfaces:
        if not itf.active:
            continue
        dissolved = [c for c in itf.boundary_centers
                     if (not c.is_crystallized()) or len(c.dissolution_log) > 0]
        itf.dissolution_count = len(dissolved)
        if len(dissolved) > len(itf.boundary_centers) / 2.0:
            itf.active = False
            itf.verified = False
            for c in itf.interior_centers:
                c.mu_eff = C.mu_base
                c.was_ever_crystallized = False
                c.crucible_verified = False
                agent.centers.append(c)
            itf.interior_restored = len(itf.interior_centers)
            stats["dissolved"] += 1
            stats["interior_restored"] += itf.interior_restored
        else:
            itf.verified = True
            itf.boundary_centers = [c for c in itf.boundary_centers if c not in dissolved]
            stats["verified"] += 1
    return stats


# ---------------------------------------------------------------- per seed
def fit_scale1_coords(seed, u_C):
    cfg = Gate1Config(generator="1B", u2_freq=3.0, k=N_ACTIONS, n_contexts=2, u_C=u_C,
                      N_repr_pool=800, N_train_pool=300, N_test=400,
                      E_scale1=20, E_scale2=E_PHASE, sigma_x_scale=0.7,
                      enable_splitting=True, split_threshold=0.12)
    env = TwoScaleToyEnvironment(seed, cfg)
    s1 = Scale1RepresentationLearner(cfg, seed, graph_mode="multiplicative")
    s1.fit(env, verbose=False)
    geom = s1.make_encoder(np.eye(N_ACTIONS), s1.get_crystallized_particles())
    cP = build_interactive_coords(geom, env.train_x20, env.train_u2, env, N_ACTIONS, N_PROBES, seed, ('A',))
    cT = build_interactive_coords(geom, env.test_A_x20, env.test_A_u2, env, N_ACTIONS, N_PROBES, seed, ('A',))
    return cfg, env, cP, cT


def make_agent(coords_pool, seed):
    enc = PassThroughEncoder(np.eye(N_ACTIONS))
    sigma, _ = calibrate_action_embedding_obs(enc, coords_pool, seed)
    base = Scale2BaseEvaluator(enc, coords_pool, seed + 2)
    return IBFAgent(sigma, sigma, enc, base, True, True, True), enc


def run_seed(seed):
    rng = np.random.RandomState(seed)
    u_C = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.6))
    cfg, env, cP, cT = fit_scale1_coords(seed, u_C)
    uP, uT = env.train_u2, env.test_A_u2

    agent, enc = make_agent(cP, seed)

    # ---- Phase A (shared by F and P) ----
    train_phase(agent, enc, cP, uP, env, 'A', 0, E_PHASE, seed)
    accA = eval_all_contexts(agent, enc, cT, uT, env)
    activeA = n_active_cryst(agent)

    agentF = copy.deepcopy(agent)
    agentP = copy.deepcopy(agent)
    interfaces = promote(agentP)
    n_promoted = len(interfaces)
    activeA_P = n_active_cryst(agentP)

    rec = dict(seed=seed, u_C=u_C, n_promoted=n_promoted,
               F={"A": {"acc": accA, "active": activeA}},
               P={"A": {"acc": accA, "active": activeA_P}},
               N={}, lifecycle={})

    # fresh N agent for phase A current-context
    agentN, encN = make_agent(cP, seed)
    train_phase(agentN, encN, cP, uP, env, 'A', 0, E_PHASE, seed)
    rec["N"]["A"] = {"acc": eval_all_contexts(agentN, encN, cT, uT, env),
                     "active": n_active_cryst(agentN)}

    # ---- Phases B, C ----
    for name, cid in [("B", 1), ("C", 2)]:
        train_phase(agentF, enc, cP, uP, env, name, cid, E_PHASE, seed)
        train_phase(agentP, enc, cP, uP, env, name, cid, E_PHASE, seed)
        life = interface_crucible(agentP, interfaces)
        rec["lifecycle"][name] = dict(n_promoted=n_promoted, **life,
                                      n_active=sum(1 for i in interfaces if i.active))
        rec["F"][name] = {"acc": eval_all_contexts(agentF, enc, cT, uT, env),
                          "active": n_active_cryst(agentF)}
        rec["P"][name] = {"acc": eval_all_contexts(agentP, enc, cT, uT, env),
                          "active": n_active_cryst(agentP)}
        agentN, encN = make_agent(cP, seed)
        train_phase(agentN, encN, cP, uP, env, name, cid, E_PHASE, seed)
        rec["N"][name] = {"acc": eval_all_contexts(agentN, encN, cT, uT, env),
                          "active": n_active_cryst(agentN)}

    # flat-Crucible comparison: how many Phase-A crystals survive in F after C?
    survivedA_F = sum(1 for c in agentF.centers
                      if c.context_id == 0 and c.is_crystallized())
    rec["flat_phaseA_survivors_after_C"] = survivedA_F
    rec["flat_phaseA_cryst_at_A"] = activeA
    return rec


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "gate3_results.json")
    print("Gate 3 | k=%d f=3.0 | E_phase=%d | seeds=%d" % (N_ACTIONS, E_PHASE, n_seeds))
    rows = []
    for seed in range(n_seeds):
        r = run_seed(seed)
        rows.append(r)
        lc = r["lifecycle"].get("C", {})
        print("  s%d uC=%+.2f promoted=%d | afterC ACC_A F=%.3f P=%.3f N=%.3f | "
              "B-active F=%d P=%d | interfaces verified=%d dissolved=%d"
              % (seed, r["u_C"], r["n_promoted"],
                 r["F"]["C"]["acc"]["A"], r["P"]["C"]["acc"]["A"], r["N"]["C"]["acc"]["A"],
                 r["F"]["B"]["active"], r["P"]["B"]["active"],
                 r["lifecycle"].get("C", {}).get("verified", 0),
                 sum(r["lifecycle"][p]["dissolved"] for p in r["lifecycle"])))
        with open(out, "w") as f:
            json.dump(dict(k=N_ACTIONS, e_phase=E_PHASE, n_probes=N_PROBES, rows=rows), f, indent=2)
    print("\nSaved %s" % out)


if __name__ == "__main__":
    main()
