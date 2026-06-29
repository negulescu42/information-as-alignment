"""
gate2_k8_diagnostic.py
=====================

Why does Gate 2 (k=8) fail field fidelity even though the field is now dense with
27% interior? Two hypotheses:
  (1) interior is genuinely not shielded -- it contributes to EXTERNAL points too;
  (2) the spec's grid (all Gate 1D test points) samples points INSIDE basins,
      where interior necessarily dominates, so a "global" field test is stricter
      than the Interface Principle's EXTERNAL-shielding claim.

For each basin we measure interior "leakage" = max |delta_R_interior(y)| /
max_y |delta_R_full(y)|, over (a) ALL test points [= spec] and (b) only points
EXTERNAL to the basin (max kernel over basin members < tau_basin). We also
report interior depth = min distance to a same-basin boundary center, in units
of the center bandwidth.

Usage:
    python gate2_k8_diagnostic.py [n_seeds]      # default 2
"""

import os
import sys
import json
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from run_gate2_k8 import fit_k8, N_ACTIONS
import run_gate2 as G2
from gate1_encoders import PassThroughEncoder
from ibf_v1_engine import C

OUT_DIR = "gate2_k8_outputs"
TAU = G2.TAU_BASIN


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    os.makedirs(os.path.join(OUT_DIR, "fidelity"), exist_ok=True)
    lines = []
    rows = []
    for seed in range(n_seeds):
        r = fit_k8(seed, 15)
        agent, enc, cA = r["agent"], r["enc"], r["cA"]
        cryst = [c for c in agent.centers if c.is_crystallized()]
        Z = np.array([c.z for c in cryst]); sg = np.array([c.sigma for c in cryst])
        vs = np.array([c.v for c in cryst])
        adj = G2.build_overlap_graph(Z, sg); comp = G2.connected_components(adj)
        sizes = np.bincount(comp)
        basins = [b for b in range(len(sizes)) if sizes[b] >= G2.MIN_BASIN]
        Z_eval = np.concatenate([enc.encode_batch(cA, np.full(len(cA), j, int))
                                 for j in range(C.k)], axis=0)
        for b in basins:
            members = list(np.where(comp == b)[0])
            bnd, intr = G2.partition(Z, sg, comp, members)
            if len(intr) < 5:
                continue
            full = G2.raw_delta_R(Z_eval, Z, sg, vs, members)
            interior_field = G2.raw_delta_R(Z_eval, Z, sg, vs, intr)
            denom = np.max(np.abs(full)) + 1e-10
            # external points: kernel over basin members below tau
            Zb = Z[members]; sgb = sg[members]
            d2 = (np.sum(Z_eval**2, 1)[:, None] + np.sum(Zb**2, 1)[None, :]
                  - 2 * Z_eval @ Zb.T)
            Kmax = np.max(np.exp(-np.maximum(d2, 0) / (2 * sgb[None, :]**2)), axis=1)
            ext = Kmax < TAU
            global_leak = float(np.max(np.abs(interior_field)) / denom)
            ext_leak = (float(np.max(np.abs(interior_field[ext])) / denom)
                        if ext.any() else float("nan"))
            # interior depth: min dist to a same-basin boundary center / sigma
            depths = []
            for i in intr:
                if not bnd:
                    continue
                dd = [np.linalg.norm(Z[i] - Z[j]) / sg[i] for j in bnd]
                depths.append(min(dd))
            rows.append(dict(seed=seed, basin=int(b), n_interior=len(intr),
                             global_leak=global_leak, external_leak=ext_leak,
                             n_external_pts=int(ext.sum()),
                             median_interior_depth=float(np.median(depths)) if depths else None))
            lines.append("seed %d basin %d: interior=%d | global leak=%.3f | "
                         "external leak=%.3f (%d ext pts) | interior depth(median)=%.2f sigma"
                         % (seed, b, len(intr), global_leak, ext_leak, int(ext.sum()),
                            np.median(depths) if depths else float("nan")))
            print(lines[-1])

    gl = np.median([x["global_leak"] for x in rows]) if rows else float("nan")
    el = np.nanmedian([x["external_leak"] for x in rows]) if rows else float("nan")
    dp = np.nanmedian([x["median_interior_depth"] for x in rows if x["median_interior_depth"]]) if rows else float("nan")
    summary = ("\nMEDIAN over %d basins (>=5 interior): global leak=%.3f, "
               "external leak=%.3f, interior depth=%.2f sigma\n"
               "Interface Principle (interior negligible EXTERNALLY): %s\n"
               % (len(rows), gl, el, dp,
                  "SUPPORTED (external leak < 0.05)" if el < 0.05 else
                  "NOT supported -- interior leaks externally too"))
    print(summary)
    with open(os.path.join(OUT_DIR, "fidelity", "external_vs_global.txt"), "w") as f:
        f.write("\n".join(lines) + "\n" + summary)
    with open(os.path.join(OUT_DIR, "fidelity", "external_vs_global.json"), "w") as f:
        json.dump(dict(rows=rows, median_global_leak=gl, median_external_leak=el,
                       median_interior_depth_sigma=dp), f, indent=2)


if __name__ == "__main__":
    main()
