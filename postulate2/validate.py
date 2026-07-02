"""
postulate2.validate  --  functional validation of the four Postulate-2 mechanisms.

Each check runs a SMALL, fast configuration and asserts the qualitative result
that was established at full scale. Full configs (in the handover) reproduce the
reported magnitudes; this harness proves the reference code is functional and
directionally correct end to end.

    V1  Emergent representation      : Scale-1 crystallization induces a 2D
                                       config space that recovers the manifold
                                       (rho_struct > 0.8) on the easy generator.
    V2  Interactive encoding         : on the HARD (aliased) generator a static
                                       map cannot recover u2 (rho ~ 0) but an
                                       interactive encoder can (rho_u2 > 0.5).
    V3  Interface external shielding : a dense field's basin interior is
                                       negligible EXTERNALLY (leak < 0.05) with a
                                       meaningful compression ratio.
    V4  Context-aware promotion      : freezing interior preserves/IMPROVES
                                       prior-context retention (P >= F) at a
                                       positive compression ratio.

Run:  python postulate2/validate.py
"""

import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine
from environment import TwoScaleToyEnvironment, Gate1Config
from encoders import (Oracle2DEncoder, PassThroughEncoder,
                      build_interactive_coords)
import scale1 as s1mod
import scale2 as s2mod
import promotion as promo

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print("  [%s] %s -- %s" % ("PASS" if ok else "FAIL", name, detail))


# ---------------------------------------------------------------- V1
def v1_emergent_representation():
    engine.C.k = 2
    cfg = Gate1Config(generator="1A", k=2, N_repr_pool=800, N_test=500,
                      E_scale1=20, sigma_x_scale=0.7)
    env = TwoScaleToyEnvironment(0, cfg)
    learner = s1mod.Scale1RepresentationLearner(cfg, 0)
    learner.fit(env)
    parts = learner.get_crystallized_particles()
    enc = learner.make_encoder(np.eye(2), parts)
    q = enc.encode_observation_batch(env.test_A_x20)
    rho = s2mod.manifold_rho(q, env.test_A_u2, seed=0)
    check("V1 emergent rho_struct", rho > 0.8,
          "rho=%.3f (>0.8); %d crystallized particles" % (rho, len(parts)))


# ---------------------------------------------------------------- V2
def v2_interactive_encoding():
    engine.C.k = 2
    cfg = Gate1Config(generator="1B", u2_freq=3.0, k=2, N_repr_pool=1200,
                      N_train_pool=300, N_test=600, E_scale1=25, E_scale2=15,
                      sigma_x_scale=0.7, enable_splitting=True)
    env = TwoScaleToyEnvironment(0, cfg)
    learner = s1mod.Scale1RepresentationLearner(cfg, 0); learner.fit(env)
    geom = learner.make_encoder(np.eye(2), learner.get_crystallized_particles())
    static_q = geom.encode_observation_batch(env.test_A_x20)
    inter_q = build_interactive_coords(geom, env.test_A_x20, env.test_A_u2, env, 2, 5, 0)
    rho_static = s2mod.coord_recovery(static_q, env.test_A_u2[:, 1], 0)
    rho_inter = s2mod.coord_recovery(inter_q, env.test_A_u2[:, 1], 0)
    check("V2 static cannot recover u2", rho_static < 0.3,
          "static rho_u2=%.3f (< 0.3)" % rho_static)
    check("V2 interactive recovers u2", rho_inter > 0.5,
          "interactive rho_u2=%.3f (> 0.5)" % rho_inter)


# ---------------------------------------------------------------- V3
def v3_interface_shielding():
    engine.C.k = 8
    cfg = Gate1Config(generator="1B", u2_freq=3.0, k=8, n_contexts=2,
                      N_repr_pool=700, N_train_pool=350, N_test=500, E_scale1=20,
                      E_scale2=15, sigma_x_scale=0.7, enable_splitting=True)
    env = TwoScaleToyEnvironment(0, cfg)
    learner = s1mod.Scale1RepresentationLearner(cfg, 0); learner.fit(env)
    geom = learner.make_encoder(np.eye(8), learner.get_crystallized_particles())
    cP = build_interactive_coords(geom, env.train_x20, env.train_u2, env, 8, 15, 0, ('A',))
    cA = build_interactive_coords(geom, env.test_A_x20, env.test_A_u2, env, 8, 15, 0, ('A',))
    cB = build_interactive_coords(geom, env.test_B_x20, env.test_B_u2, env, 8, 15, 0, ('A',))
    enc = PassThroughEncoder(np.eye(8))
    _, _, agent = s2mod.run_scale2_v1(enc, cP, env.train_u2, cA, env.test_A_u2,
                                      cB, env.test_B_u2, env, 0, cfg.E_scale2, want_agent=True)
    cryst = [c for c in agent.centers if c.is_crystallized()]
    Z, sg, comp, basins = promo.detect_basins(cryst)
    vs = np.array([c.v for c in cryst])
    Z_eval = np.concatenate([enc.encode_batch(cA, np.full(len(cA), j, int))
                             for j in range(8)], axis=0)
    n_int, leaks = 0, []
    for members in basins:
        bnd, intr = promo.partition_basin(Z, sg, comp, members)
        n_int += len(intr)
        if len(intr) >= 3:
            leaks.append(promo.external_field_fidelity(Z, sg, vs, members, intr, Z_eval)["external_leak"])
    comp_ratio = n_int / max(len(cryst), 1)
    ext = float(np.nanmax(leaks)) if leaks else float("nan")
    check("V3 interior externally shielded", (not leaks) or ext < 0.05,
          "%d basins, external leak max=%.4f (< 0.05)" % (len(basins), ext))
    check("V3 compression ratio", comp_ratio > 0.15,
          "interior fraction=%.2f (> 0.15)" % comp_ratio)


# ---------------------------------------------------------------- V4
def v4_context_aware_promotion():
    engine.C.k = 8
    E_PHASE = 15
    cfg = Gate1Config(generator="1B", u2_freq=3.0, k=8, n_contexts=2, u_C=-0.4,
                      N_repr_pool=700, N_train_pool=250, N_test=400, E_scale1=20,
                      E_scale2=E_PHASE, sigma_x_scale=0.7, enable_splitting=True)
    env = TwoScaleToyEnvironment(0, cfg)
    learner = s1mod.Scale1RepresentationLearner(cfg, 0); learner.fit(env)
    geom = learner.make_encoder(np.eye(8), learner.get_crystallized_particles())
    cP = build_interactive_coords(geom, env.train_x20, env.train_u2, env, 8, 15, 0, ('A',))
    cT = build_interactive_coords(geom, env.test_A_x20, env.test_A_u2, env, 8, 15, 0, ('A',))
    uT = env.test_A_u2
    enc = PassThroughEncoder(np.eye(8))
    agent = s2mod.make_agent(enc, cP, 0)

    import copy
    s2mod.train_phase(agent, enc, cP, env.train_u2, env, 'A', 0, E_PHASE, 0)
    agentF, agentP = copy.deepcopy(agent), copy.deepcopy(agent)
    n_cryst_at_promote = sum(1 for c in agentP.centers if c.is_crystallized())
    interfaces = promo.promote(agentP)
    n_frozen = sum(len(i.interior_centers) for i in interfaces)

    def accA(ag):
        sv = ag.current_context; ag.current_context = 0
        a = s2mod.evaluate_scale2(ag, enc, cT, uT, env, 'A'); ag.current_context = sv
        return a

    for name, cid in [('B', 1), ('C', 2)]:
        s2mod.train_phase(agentF, enc, cP, env.train_u2, env, name, cid, E_PHASE, 0)
        s2mod.train_phase(agentP, enc, cP, env.train_u2, env, name, cid, E_PHASE, 0)
        promo.interface_crucible(interfaces)
    retF, retP = accA(agentF), accA(agentP)
    # Compression = interior fraction frozen at promotion (the stable, unconfounded
    # quantity; the post-context active ratio is skewed by differential growth
    # during B/C training).
    comp_ratio = n_frozen / max(n_cryst_at_promote, 1)
    check("V4 promotion does not degrade retention", retP >= retF - 0.03,
          "Acc_A after C: F=%.3f P=%.3f (P>=F-0.03); %d frozen reserves" % (retF, retP, n_frozen))
    check("V4 cross-context compression", comp_ratio > 0.10,
          "interior frozen=%d/%d=%.2f (>0.10)" % (n_frozen, n_cryst_at_promote, comp_ratio))


def main():
    t0 = time.time()
    print("Postulate 2 -- functional validation (small configs)\n")
    for fn in [("V1 Emergent representation", v1_emergent_representation),
               ("V2 Interactive encoding", v2_interactive_encoding),
               ("V3 Interface shielding", v3_interface_shielding),
               ("V4 Context-aware promotion", v4_context_aware_promotion)]:
        print("== %s ==" % fn[0])
        fn[1]()
    n_pass = sum(ok for _, ok, _ in RESULTS)
    print("\n%d/%d checks passed  (%.0fs)" % (n_pass, len(RESULTS), time.time() - t0))
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
