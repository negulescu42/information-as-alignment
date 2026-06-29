"""
gate1c_oracle_diagnostic.py
===========================

GATING DIAGNOSTIC for Gate 1C -- run BEFORE any method change.

Question: is the Gate 1B task solvable in principle by a *behavioral signature*,
if the signature were built with perfect localization in the true [u1, u2]
space? If yes, the generator is fair and the Scale 1 localization is the
problem. If even a perfectly-localized oracle signature cannot recover u2, the
generator is too hard and u2_freq must be lowered first.

Oracle signature for a query point u:
    for each (context c, action a):
        signature[c,a] = mean over the 10 nearest neighbours of u (in TRUE u-space)
                         of  [ a == correct_action(u_neighbour, c) ]
This is exactly the Scale 1 behavioral signature, but with the activation
neighbourhood taken in hidden-coordinate space instead of observation space --
i.e. perfect localization.

Decision rule (supervisor):
    oracle rho_u2 >= 0.3  -> task solvable, proceed to method changes.
    oracle rho_u2 <  0.3  -> generator too hard, lower u2_freq first.

Usage:
    python gate1c_oracle_diagnostic.py [n_seeds] [u2_freq]
"""

import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from gate1_environment import TwoScaleToyEnvironment, Gate1Config
from run_gate1b_variants import coord_recovery

RHO_THRESHOLD = 0.8
ORACLE_GATE = 0.3


def oracle_signatures(query_u, ref_u, env, knn=10):
    """(N, k*k) perfectly-localized behavioral signatures for query points."""
    k = env.k
    nn = NearestNeighbors(n_neighbors=knn).fit(ref_u)
    _, idx = nn.kneighbors(query_u)               # (N, knn)
    # precompute correct action for every reference point in each context
    correct = {c: env.correct_actions_batch(ref_u, 'A' if c == 0 else 'B')
               for c in range(k)}
    N = len(query_u)
    sig = np.zeros((N, k * k))
    for c in range(k):
        cc = correct[c]                            # (Nref,)
        for a in range(k):
            hit = (cc == a).astype(float)          # reward of action a in context c
            sig[:, c * k + a] = hit[idx].mean(axis=1)
    return sig


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    u2_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    print("Gate 1C oracle-signature diagnostic | generator=1B | u2_freq=%.2f | seeds=%d"
          % (u2_freq, n_seeds))
    print("decision: oracle rho_u2 >= %.2f -> solvable (proceed); < %.2f -> lower u2_freq\n"
          % (ORACLE_GATE, ORACLE_GATE))
    print("  %-5s %-10s %-9s %-9s" % ("seed", "rho_struct", "rho_u1", "rho_u2"))
    rs, r1, r2 = [], [], []
    for s in range(n_seeds):
        cfg = Gate1Config(generator="1B", u2_freq=u2_freq,
                          N_repr_pool=1500, N_test=800)
        env = TwoScaleToyEnvironment(s, cfg)
        sub = np.random.RandomState(s + 5).choice(cfg.N_test, 300, replace=False)
        Uq = env.test_A_u2[sub]
        sig = oracle_signatures(Uq, env.pool_u2, env, knn=10)
        rho_struct = spearmanr(pdist(sig), pdist(Uq)).correlation
        rho_u1 = coord_recovery(sig, Uq[:, 0], s)
        rho_u2 = coord_recovery(sig, Uq[:, 1], s)
        rs.append(rho_struct); r1.append(rho_u1); r2.append(rho_u2)
        print("  %-5d %-10.3f %-9.3f %-9.3f" % (s, rho_struct, rho_u1, rho_u2))
    mrs, mr1, mr2 = np.median(rs), np.median(r1), np.median(r2)
    print("  %-5s %-10.3f %-9.3f %-9.3f" % ("med", mrs, mr1, mr2))
    print()
    if mr2 >= ORACLE_GATE:
        print("=> DIAGNOSTIC PASS: oracle rho_u2 = %.3f >= %.2f." % (mr2, ORACLE_GATE))
        print("   Task is solvable by a well-localized signature. The method (broad")
        print("   Scale 1 particles) is the problem -> proceed to discrepancy-driven")
        print("   splitting on the gate-1c-behavioral-metric branch.")
    else:
        print("=> DIAGNOSTIC FAIL: oracle rho_u2 = %.3f < %.2f." % (mr2, ORACLE_GATE))
        print("   Even perfect localization cannot recover u2 -> generator too hard.")
        print("   Lower u2_freq (re-run this script with a smaller value) before any")
        print("   method change.")


if __name__ == "__main__":
    main()
