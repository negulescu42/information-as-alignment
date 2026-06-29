"""
summarize_gate1b.py
===================

Aggregate the Gate 1B behavioral-geometry variant results into:

    gate1b_variants_outputs/gate1b_variants_per_seed.csv
    gate1b_variants_outputs/gate1b_variants_report.md

Reporting format mirrors the Gate 1A report. Thresholds unchanged
(rho_struct > 0.8 and accuracy_gap <= 0.15).

Usage:
    python summarize_gate1b.py [out_dir]
"""

import os
import sys
import json
import numpy as np
import pandas as pd

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "gate1b_variants_outputs"
RESULTS = os.path.join(OUT_DIR, "gate1b_variants_results.json")
RHO_THRESHOLD = 0.8
ACC_GAP_THRESHOLD = 0.15


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(xs)) if xs else float("nan")


def main():
    with open(RESULTS) as f:
        payload = json.load(f)
    rows = payload["rows"]
    modes = payload["modes"]
    seeds = [r["seed"] for r in rows]

    # per-seed long table
    long = []
    for r in rows:
        for m in modes:
            v = r["variants"][m]
            long.append(dict(seed=r["seed"], variant=m, **v))
    df = pd.DataFrame(long)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "gate1b_variants_per_seed.csv"), index=False)

    # geometry-only controls (median over seeds)
    geo = {k: med([r["geometry_controls"][k] for r in rows])
           for k in ["raw20D", "PCA2D", "spectral", "random2D"]}
    geo_best = max(geo.values())
    diag_u2 = med([r["diagnostic"]["sig_vs_u2"] for r in rows])
    diag_u1 = med([r["diagnostic"]["sig_vs_u1"] for r in rows])
    diag_var = med([r["diagnostic"]["per_particle_u2_var"] for r in rows])
    rho_nc = med([r["rho_nocryst"] for r in rows])
    rho_sh = med([r["rho_shuffled"] for r in rows])

    # per-variant aggregate
    agg = {}
    for m in modes:
        sub = df[df.variant == m]
        agg[m] = dict(
            rho_struct=med(sub.rho_struct), rho_u1=med(sub.rho_u1), rho_u2=med(sub.rho_u2),
            ACC_oracle=med(sub.ACC_oracle), ACC_emergent=med(sub.ACC_emergent),
            accuracy_gap=med(sub.accuracy_gap),
            n_cryst=med(sub.n_repr_crystallized),
            n_pass=int(sub.gate1_pass.sum()), n_total=len(sub))

    # any variant passing on the median?
    passing = [m for m in modes
               if agg[m]["rho_struct"] > RHO_THRESHOLD
               and agg[m]["accuracy_gap"] <= ACC_GAP_THRESHOLD]
    gate1b_pass = len(passing) > 0

    L = []
    L.append("# Gate 1B (behavioral-geometry) Report\n")
    L.append("Generator: **1B (stress)** | seeds: **%d** (%s) | thresholds: "
             "rho_struct > %.2f and accuracy_gap <= %.2f\n"
             % (len(seeds), ", ".join(map(str, seeds)), RHO_THRESHOLD, ACC_GAP_THRESHOLD))

    if gate1b_pass:
        concl = ("**1B PASS** via variant(s): %s -- crystallized behavioral structure "
                 "recovers representation when geometry alone fails." % ", ".join(passing))
    else:
        concl = ("**1B FAIL** -- no graph variant clears both thresholds. The current "
                 "Scale 1 dynamics are not sufficient yet for Postulate 2 validation in "
                 "the hard setting.")
    L.append("\n## Conclusion: %s\n" % concl)

    L.append("\n## Geometry-only controls (median rho_struct -- should stay < %.2f)\n"
             % RHO_THRESHOLD)
    L.append("| raw20D | PCA2D | spectral | random2D | best |")
    L.append("|---|---|---|---|---|")
    L.append("| %.3f | %.3f | %.3f | %.3f | **%.3f** |"
             % (geo["raw20D"], geo["PCA2D"], geo["spectral"], geo["random2D"], geo_best))
    L.append("\nGeometry-only best = %.3f %s %.2f -> geometry is %s on Gate 1B.\n"
             % (geo_best, "<=" if geo_best <= RHO_THRESHOLD else ">", RHO_THRESHOLD,
                "INSUFFICIENT" if geo_best <= RHO_THRESHOLD else "sufficient"))

    L.append("\n## Graph variants (median over seeds)\n")
    L.append("| variant | rho_struct | rho_u1 | rho_u2 | ACC_oracle | ACC_emergent | "
             "accuracy_gap | n_cryst | seeds pass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in modes:
        a = agg[m]
        tag = " (BASELINE)" if m == "multiplicative" else ""
        L.append("| %s%s | %.3f | %.3f | %.3f | %.3f | %.3f | %+.3f | %d | %d/%d |"
                 % (m, tag, a["rho_struct"], a["rho_u1"], a["rho_u2"], a["ACC_oracle"],
                    a["ACC_emergent"], a["accuracy_gap"], int(a["n_cryst"]),
                    a["n_pass"], a["n_total"]))

    L.append("\n## Upstream diagnostic: do crystallized signatures encode u2?\n")
    L.append("| signature-dist vs u1 | signature-dist vs u2 | per-particle u2 variance "
             "(global=1.0) |")
    L.append("|---|---|---|")
    L.append("| %.3f | %.3f | %.3f |" % (diag_u1, diag_u2, diag_var))
    L.append("\nno-crystallization control rho=%.3f | shuffled-signature control rho=%.3f\n"
             % (rho_nc, rho_sh))

    L.append("\n## Interpretation\n")
    L.append("On the Gate 1B generator u1 is geometry-accessible while u2 is encoded only "
             "through high-frequency aliased terms. The per-variant table shows rho_u1 "
             "(recovery of u1) and rho_u2 (recovery of u2) separately.\n")
    if not gate1b_pass:
        L.append("\n**Why every variant fails is upstream of the graph.** The crystallized "
                 "signature *distance* tracks u1 (%.2f) but not u2 (%.2f), and each "
                 "particle's activation ball spans most of the u2 range (median u2 "
                 "variance %.2f of the global 1.0). Because the 20D geometry is "
                 "u1-dominated, a particle averages its reward signal over a wide range of "
                 "true u2, so the behavioral signature collapses to a function of u1. With "
                 "no u2 information in the signatures, *no* graph mixing rule "
                 "(multiplicative, additive, union, or behavior-first) can rebuild u2 -- "
                 "the binding constraint is particle localization of the aliased "
                 "coordinate, not the geometry/behavior combination rule.\n"
                 % (diag_u1, diag_u2, diag_var))
        L.append("\n**Conclusion: 1B FAIL.** Current Scale 1 dynamics are not sufficient "
                 "yet for Postulate 2 validation in the hard setting. The next lever is the "
                 "Scale 1 representation dynamics themselves (e.g. a behavior-sensitive "
                 "kernel / metric so particles can localize the aliased coordinate), not "
                 "the graph construction layer tested here.\n")
    else:
        L.append("\nThe passing variant(s) recover u2 (rho_u2 above the geometry-only "
                 "controls) and clear rho_struct > %.2f while keeping the Scale 2 accuracy "
                 "gap <= %.2f. This is the signature of genuine crystallization-driven "
                 "representational buildup: behavioral structure supplied the coordinate "
                 "that ambient geometry had aliased away.\n" % (RHO_THRESHOLD, ACC_GAP_THRESHOLD))

    L.append("\n## Note on the baseline\n")
    L.append("The `multiplicative` row is the unchanged Gate 1A affinity "
             "(W = W_geometry * W_behavior * W_stability), kept as the baseline. Its "
             "failure on Gate 1B is the motivation for this branch and is retained, not "
             "deleted.\n")

    report = "\n".join(L)
    with open(os.path.join(OUT_DIR, "gate1b_variants_report.md"), "w") as f:
        f.write(report)
    print(report)
    print("\nWrote %s/gate1b_variants_report.md and gate1b_variants_per_seed.csv" % OUT_DIR)


if __name__ == "__main__":
    main()
