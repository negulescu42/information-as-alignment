"""
run_gate2.py
============

Gate 2 -- basin detection and interface extraction (post-hoc; NO new training).

Operates on the TRAINED Gate 1D interactive two-scale system at f=3.0:
detect coherent basins in the crystallized Scale 2 correction field, partition
each basin into boundary / interior, then test whether removing interior centers
preserves (a) the correction field and (b) behavioral accuracy. This is the
operational test of the Interface Principle: interior particles contribute
negligibly to external interaction.

Note on boundary/interior. The spec's literal `partition_basin` checks whether a
center's overlap-neighbour is in a different basin -- but basins ARE the
connected components of the overlap graph, so no center has a cross-component
edge and the literal rule labels everything interior. We therefore use the
operationally faithful formulation of the same intent:

    a basin center is BOUNDARY if it has non-negligible kernel overlap
    (> tau_face) with ANY center outside its basin (it faces external
    structure); INTERIOR if it is shielded -- overlap with everything outside
    the basin is below tau_face.

tau_face < tau_basin, so boundary centers are those that *almost* connect to
another structure. Interior centers are deep inside their basin, far from
anything external -- exactly the shielding the Interface Principle predicts.

Pass criteria (all three):
    field fidelity     : relative error < 0.05 for every basin with >= 5 interior
    behavioral fidelity: |ACC_compressed - ACC_full| < 0.02 across seeds
    compression ratio  : >= 20% of centers classified interior

Usage:
    python run_gate2.py [n_seeds]      # default 5
"""

import os
import sys
import json
import copy
import warnings
import numpy as np

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ibf_v1_engine import C
from gate1_encoders import PassThroughEncoder, build_interactive_coords
from run_gate1 import run_scale2_v1, evaluate_scale2
from run_gate1d import fit_scale1, N_PROBES_DEFAULT

OUT_DIR = "gate2_outputs"
FREQ = 3.0
TAU_BASIN = 0.01
TAU_FACE = 1e-4
MIN_BASIN = 3
REL_ERR_THRESHOLD = 0.05
ACC_DELTA_THRESHOLD = 0.02
COMPRESSION_THRESHOLD = 0.20


# ---------------------------------------------------------------- basins
def build_overlap_graph(centers, sigmas, tau=TAU_BASIN):
    n = len(centers)
    adj = np.zeros((n, n), dtype=bool)
    Z = np.asarray(centers)
    for i in range(n):
        for j in range(i + 1, n):
            sigma_ij = np.sqrt(sigmas[i] * sigmas[j])
            d2 = np.sum((Z[i] - Z[j]) ** 2)
            if np.exp(-d2 / (2 * sigma_ij ** 2)) > tau:
                adj[i, j] = adj[j, i] = True
    return adj


def connected_components(adj):
    n = len(adj)
    comp = -np.ones(n, dtype=int)
    cid = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]
        comp[s] = cid
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if comp[v] < 0:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    return comp


def partition(Z, sigmas, comp, basin_ids, tau_face=TAU_FACE):
    """boundary if max kernel overlap with any out-of-basin center > tau_face."""
    boundary, interior = [], []
    for i in basin_ids:
        bi = comp[i]
        faces_outside = False
        for j in range(len(Z)):
            if comp[j] == bi:
                continue
            sigma_ij = np.sqrt(sigmas[i] * sigmas[j])
            d2 = np.sum((Z[i] - Z[j]) ** 2)
            if np.exp(-d2 / (2 * sigma_ij ** 2)) > tau_face:
                faces_outside = True
                break
        (boundary if faces_outside else interior).append(i)
    return boundary, interior


def raw_delta_R(Z_eval, Z_c, sigmas, vs, idx):
    """sum_i v_i K(y, z_i) over centers in idx, for every y in Z_eval."""
    if len(idx) == 0:
        return np.zeros(len(Z_eval))
    Zc = Z_c[idx]
    sg = sigmas[idx]
    vv = vs[idx]
    d2 = (np.sum(Z_eval ** 2, 1)[:, None] + np.sum(Zc ** 2, 1)[None, :]
          - 2 * Z_eval @ Zc.T)
    K = np.exp(-np.maximum(d2, 0) / (2 * sg[None, :] ** 2))
    return K @ vv


# ---------------------------------------------------------------- per seed
def run_seed(seed, n_probes, fig_dir):
    cfg, env, s1 = fit_scale1(FREQ, seed)
    E2 = cfg.E_scale2
    parts = s1.get_crystallized_particles()
    geom = s1.make_encoder(np.eye(C.k), parts)

    cP = build_interactive_coords(geom, env.train_x20, env.train_u2, env, C.k, n_probes, seed)
    cA = build_interactive_coords(geom, env.test_A_x20, env.test_A_u2, env, C.k, n_probes, seed)
    cB = build_interactive_coords(geom, env.test_B_x20, env.test_B_u2, env, C.k, n_probes, seed)
    enc = PassThroughEncoder(np.eye(C.k))
    m, _, agent = run_scale2_v1(enc, cP, env.train_u2, cA, env.test_A_u2,
                                cB, env.test_B_u2, env, seed, E2, want_log=True)

    return analyze_trained_system(agent, enc, cA, cB, env, seed, E2, fig_dir)


def analyze_trained_system(agent, enc, cA, cB, env, seed, E2, fig_dir):
    """Gate 2 post-hoc analysis on ANY trained interactive Scale 2 agent.

    This is the validated, task-agnostic Gate 2 pipeline -- the k=8 retry
    (run_gate2_k8.py) calls it unchanged.
    """
    cryst = [c for c in agent.centers if c.is_crystallized()]
    n_total = len(agent.centers)
    n_cryst = len(cryst)
    if n_cryst < MIN_BASIN:
        return dict(seed=seed, n_total=n_total, n_cryst=n_cryst, n_basins=0,
                    degenerate="too few crystallized centers")

    Z = np.array([c.z for c in cryst])
    sig = np.array([c.sigma for c in cryst])
    vs = np.array([c.v for c in cryst])

    adj = build_overlap_graph(Z, sig)
    comp = connected_components(adj)
    sizes = np.bincount(comp)
    basin_labels = [c for c in range(len(sizes)) if sizes[c] >= MIN_BASIN]
    basin_members = {c: list(np.where(comp == c)[0]) for c in basin_labels}
    n_singleton = int(sum(sizes[c] for c in range(len(sizes)) if sizes[c] < MIN_BASIN))

    # partition every basin
    all_boundary, all_interior = [], []
    per_basin = []
    for b in basin_labels:
        members = basin_members[b]
        bnd, intr = partition(Z, sig, comp, members)
        all_boundary += bnd
        all_interior += intr
        per_basin.append(dict(basin=int(b), size=len(members),
                              boundary=len(bnd), interior=len(intr),
                              members=members, bnd=bnd, intr=intr))
    n_interior = len(all_interior)
    n_boundary = len(all_boundary)
    compression_ratio = n_interior / n_cryst if n_cryst else 0.0

    # ---- Step 3: field fidelity per basin (raw delta_R) ----
    Z_eval = np.concatenate([enc.encode_batch(cA, np.full(len(cA), j, int))
                             for j in range(C.k)], axis=0)
    fidelity_rows = []
    for pb in per_basin:
        members, intr = pb["members"], pb["intr"]
        boundary = pb["bnd"]
        full = raw_delta_R(Z_eval, Z, sig, vs, members)
        bnd_only = raw_delta_R(Z_eval, Z, sig, vs, boundary)
        max_full = float(np.max(np.abs(full)))
        max_err = float(np.max(np.abs(full - bnd_only)))
        rel_err = max_err / (max_full + 1e-10)
        fidelity_rows.append(dict(basin=pb["basin"], size=pb["size"],
                                  boundary=pb["boundary"], interior=pb["interior"],
                                  max_field=max_full, max_error=max_err,
                                  relative_error=rel_err))

    # ---- Step 4: behavioral fidelity (full vs compressed vs oracle) ----
    def acc_gate(ag, encoder, coordsA, coordsB):
        sv = ag.current_context
        ag.current_context = 0
        aA = evaluate_scale2(ag, encoder, coordsA, env.test_A_u2, env, 'A')
        ag.current_context = 1
        aB = evaluate_scale2(ag, encoder, coordsB, env.test_B_u2, env, 'B')
        ag.current_context = sv
        return 0.5 * (aA + aB)

    acc_full = acc_gate(agent, enc, cA, cB)
    # compressed agent: drop the interior crystallized centers (keep all others)
    interior_objs = set(id(cryst[i]) for i in all_interior)
    comp_agent = copy.deepcopy(agent)
    comp_agent.centers = [cp for orig, cp in zip(agent.centers, comp_agent.centers)
                          if id(orig) not in interior_objs]
    acc_comp = acc_gate(comp_agent, enc, cA, cB)

    # oracle
    om, _, oracle_agent = run_scale2_v1(PassThroughEncoder(np.eye(C.k)),
                                        env.train_u2, env.train_u2, env.test_A_u2,
                                        env.test_A_u2, env.test_B_u2, env.test_B_u2,
                                        env, seed, E2, want_log=True)
    acc_oracle = om['ACC_gate']

    # ---- visualization (seed 0) ----
    if seed == 0:
        from sklearn.decomposition import PCA
        os.makedirs(fig_dir, exist_ok=True)
        coords6 = Z[:, :6]
        p2 = PCA(2).fit_transform(coords6)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        col = np.array([comp[i] if sizes[comp[i]] >= MIN_BASIN else -1 for i in range(len(Z))])
        kind = np.array(["singleton" if sizes[comp[i]] < MIN_BASIN else
                         ("interior" if i in all_interior else "boundary")
                         for i in range(len(Z))])
        for k, mk, lbl in [("singleton", "x", "singleton/pair"),
                           ("boundary", "o", "basin boundary"),
                           ("interior", "*", "basin interior")]:
            mask = kind == k
            if mask.any():
                ax.scatter(p2[mask, 0], p2[mask, 1],
                           c=(col[mask] if k != "singleton" else "#bbbbbb"),
                           cmap="tab10", marker=mk, s=(120 if k == "interior" else 60),
                           edgecolors="black", linewidths=0.4, label=lbl)
        ax.set_title("Gate 2 (seed 0) -- crystallized Scale 2 centers\n%d basins, "
                     "%d boundary, %d interior" % (len(basin_labels), n_boundary, n_interior))
        ax.set_xlabel("PCA-1 of q2"); ax.set_ylabel("PCA-2 of q2")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "basin_visualization.png"), dpi=140)
        plt.close(fig)

    return dict(
        seed=seed, n_total=n_total, n_cryst=n_cryst,
        n_basins=len(basin_labels), basin_sizes=[int(sizes[b]) for b in basin_labels],
        n_singleton=n_singleton, n_boundary=n_boundary, n_interior=n_interior,
        compression_ratio=compression_ratio,
        ACC_full=acc_full, ACC_compressed=acc_comp, ACC_oracle=acc_oracle,
        ACC_delta=acc_comp - acc_full,
        per_basin=per_basin_clean(per_basin), fidelity=fidelity_rows)


def per_basin_clean(per_basin):
    return [dict(basin=int(p["basin"]), size=p["size"], boundary=p["boundary"],
                 interior=p["interior"]) for p in per_basin]


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for sub in ["basin_detection", "fidelity", "behavioral"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    fig_dir = os.path.join(OUT_DIR, "basin_detection")
    out = os.path.join(OUT_DIR, "gate2_results.json")
    print("Gate 2 | f=%.1f | tau_basin=%.3g tau_face=%.3g | seeds=%d"
          % (FREQ, TAU_BASIN, TAU_FACE, n_seeds))
    rows = []
    for seed in range(n_seeds):
        r = run_seed(seed, N_PROBES_DEFAULT, fig_dir)
        rows.append(r)
        if r.get("degenerate"):
            print("  s%d: DEGENERATE (%s)" % (seed, r["degenerate"]))
        else:
            print("  s%d: cryst=%d basins=%d sizes=%s interior=%d (%.0f%%) | "
                  "ACC full=%.3f comp=%.3f delta=%+.3f oracle=%.3f"
                  % (seed, r["n_cryst"], r["n_basins"], r["basin_sizes"],
                     r["n_interior"], 100 * r["compression_ratio"],
                     r["ACC_full"], r["ACC_compressed"], r["ACC_delta"], r["ACC_oracle"]))
        with open(out, "w") as f:
            json.dump(dict(freq=FREQ, tau_basin=TAU_BASIN, tau_face=TAU_FACE,
                           rel_err_threshold=REL_ERR_THRESHOLD,
                           acc_delta_threshold=ACC_DELTA_THRESHOLD,
                           compression_threshold=COMPRESSION_THRESHOLD,
                           rows=rows), f, indent=2)
    print("\nSaved %s" % out)


if __name__ == "__main__":
    main()
