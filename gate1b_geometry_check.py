"""
gate1b_geometry_check.py
========================

Fast, standalone demonstration that the **Gate 1B** generator makes
geometry-only manifold recovery insufficient -- the property that Gate 1A
lacks. It does NOT run Scale 1 / Scale 2; it only measures how well purely
geometric 2D representations of the 20D observation recover the hidden manifold.

For each seed and each generator it reports the pairwise-distance Spearman rho
(same metric as Gate 1's rho_struct) for:

    raw 20D distances, PCA(2), spectral embedding, random projection(2)

Pass/fail thresholds are unchanged (rho_struct > 0.8). On Gate 1A every
geometry-only baseline clears 0.8 (the manifold is near-isometric in 20D). On
Gate 1B they fall well below 0.8, because u2 is encoded only through
high-frequency aliased terms that ambient geometry cannot order.

Usage:
    python gate1b_geometry_check.py [n_seeds]
"""

import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

from gate1_environment import TwoScaleToyEnvironment, Gate1Config

RHO_THRESHOLD = 0.8


def geom_rhos(env, seed, n_sub=300):
    sub = np.random.RandomState(seed + 5).choice(env.cfg.N_test, n_sub, replace=False)
    X = env.test_A_x20[sub]
    U = env.test_A_u2[sub]
    Du = pdist(U)
    raw = spearmanr(pdist(X), Du).correlation
    pca = spearmanr(pdist(PCA(2).fit_transform(X)), Du).correlation
    spec = spearmanr(pdist(SpectralEmbedding(
        n_components=2, n_neighbors=15, random_state=seed).fit_transform(X)), Du).correlation
    rp = np.random.RandomState(seed + 1).randn(X.shape[1], 2)
    rnd = spearmanr(pdist((X - X.mean(0)) @ rp), Du).correlation
    return dict(raw20D=raw, PCA2D=pca, spectral=spec, random2D=rnd)


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print("Geometry-only manifold recovery (rho_struct), threshold > %.2f" % RHO_THRESHOLD)
    print("(no crystallization, no behavior -- pure geometry of the 20D observation)\n")
    for gen in ["1A", "1B"]:
        print("=== Generator %s ===" % gen)
        print("  %-6s %-8s %-8s %-9s %-9s" % ("seed", "raw20D", "PCA2D", "spectral", "random2D"))
        acc = {k: [] for k in ["raw20D", "PCA2D", "spectral", "random2D"]}
        for s in range(n_seeds):
            cfg = Gate1Config(N_repr_pool=50, N_train_pool=50, N_test=800, generator=gen)
            env = TwoScaleToyEnvironment(s, cfg)
            r = geom_rhos(env, s)
            for k in acc:
                acc[k].append(r[k])
            print("  %-6d %-8.3f %-8.3f %-9.3f %-9.3f"
                  % (s, r["raw20D"], r["PCA2D"], r["spectral"], r["random2D"]))
        meds = {k: float(np.median(v)) for k, v in acc.items()}
        print("  %-6s %-8.3f %-8.3f %-9.3f %-9.3f"
              % ("median", meds["raw20D"], meds["PCA2D"], meds["spectral"], meds["random2D"]))
        best = max(meds.values())
        verdict = ("geometry SUFFICIENT (>%.2f) -> geometry-easy"
                   if best > RHO_THRESHOLD else
                   "geometry INSUFFICIENT (all <=%.2f) -> behavior required")
        print("  => best geometry-only median = %.3f : %s\n"
              % (best, verdict % RHO_THRESHOLD))


if __name__ == "__main__":
    main()
