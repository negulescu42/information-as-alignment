"""
gate1_plots.py
==============

Diagnostic figures for Gate 1 (spec section 13). Per-seed figures (13.1-13.4)
are produced by `make_seed_figures`; the cross-seed accuracy-gap bar chart
(13.5) is produced by summarize_gate1.py.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


def _procrustes_align(src, tgt):
    """Orthogonally align src (N,2) onto tgt (N,2): scale+rotation+reflection."""
    s = (src - src.mean(0))
    t = (tgt - tgt.mean(0))
    s = s / (np.linalg.norm(s) + 1e-12)
    t_norm = np.linalg.norm(t) + 1e-12
    tt = t / t_norm
    U, S, Vt = np.linalg.svd(tt.T @ s)
    R = U @ Vt
    aligned = s @ R.T
    # rescale to target spread
    aligned = aligned / (np.linalg.norm(aligned) + 1e-12) * t_norm + tgt.mean(0)
    return aligned


def make_seed_figures(seed, env, s1, parts, q_parts, emergent_enc,
                      s2_logs, row, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    n_sub = min(400, env.cfg.N_test)
    sub = np.random.RandomState(seed + 11).choice(env.cfg.N_test, n_sub, replace=False)
    Xt = env.test_A_x20[sub]
    Ut = env.test_A_u2[sub]
    qh = emergent_enc.encode_observation_batch(Xt)
    actA = env.correct_actions_batch(Ut, 'A')

    # ---- 13.1 true manifold vs emergent (Procrustes aligned) ----
    q_al = _procrustes_align(qh, Ut)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    for a in range(env.k):
        m = actA == a
        ax[0].scatter(Ut[m, 0], Ut[m, 1], s=10, alpha=0.7, label='action %d' % a)
        ax[1].scatter(q_al[m, 0], q_al[m, 1], s=10, alpha=0.7, label='action %d' % a)
    ax[0].set_title("True hidden manifold u")
    ax[0].set_xlabel("u1"); ax[0].set_ylabel("u2")
    ax[1].set_title("Emergent q_hat (Procrustes aligned)")
    ax[1].set_xlabel("q1"); ax[1].set_ylabel("q2")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.suptitle("Fig 13.1 (seed %d) -- True vs Emergent manifold, colored by correct action (ctx A)"
                 % seed, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_13_1_manifold_seed%d.png" % seed), dpi=140)
    plt.close(fig)

    # ---- 13.2 pairwise distance correlation ----
    De = pdist(qh)
    Du = pdist(Ut)
    rho = spearmanr(De, Du).correlation
    fig, ax = plt.subplots(figsize=(5.2, 5))
    idx = np.random.RandomState(seed).choice(len(De), min(4000, len(De)), replace=False)
    ax.scatter(Du[idx], De[idx], s=3, alpha=0.25)
    ax.set_xlabel("true pairwise distance ||u_a - u_b||")
    ax.set_ylabel("emergent pairwise distance ||q_a - q_b||")
    ax.set_title("Fig 13.2 (seed %d) -- Distance correlation, Spearman rho=%.3f" % (seed, rho))
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_13_2_distcorr_seed%d.png" % seed), dpi=140)
    plt.close(fig)

    # ---- 13.3 Scale 1 particle graph in emergent coords ----
    fig, ax = plt.subplots(figsize=(5.6, 5))
    sig = np.array([p.signature for p in parts])
    dom = np.argmax(np.abs(sig), axis=1) if len(sig) else np.zeros(len(parts), int)
    nupd = np.array([p.n_updates for p in parts], dtype=float)
    sizes = 20 + 120 * (nupd - nupd.min()) / (nupd.ptp() + 1e-9)
    sc = ax.scatter(q_parts[:, 0], q_parts[:, 1], s=sizes, c=dom, cmap='tab10',
                    alpha=0.85, edgecolors='black', linewidths=0.3)
    ax.set_title("Fig 13.3 (seed %d) -- %d crystallized particles\n(size ~ n_updates, color ~ dominant signature slot)"
                 % (seed, len(parts)), fontsize=9)
    ax.set_xlabel("q1"); ax.set_ylabel("q2")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_13_3_graph_seed%d.png" % seed), dpi=140)
    plt.close(fig)

    # ---- 13.4 Scale 2 learning curves ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {'oracle': 'k', 'emergent': 'C0', 'pca2D': 'C1',
              'raw20D': 'C2', 'random2D': 'C3'}
    for name, log in s2_logs.items():
        if log is None:
            continue
        ep = np.arange(1, len(log['acc_A']) + 1)
        ax[0].plot(ep, log['acc_A'], color=colors.get(name, None), label=name, linewidth=1.3)
        ax[1].plot(ep, log['acc_B'], color=colors.get(name, None), label=name, linewidth=1.3)
    E2 = env.cfg.E_scale2
    for a in ax:
        a.axvline(E2 + 0.5, color='gray', ls='--', alpha=0.5)
        a.set_xlabel("Scale 2 epoch")
        a.grid(alpha=0.3)
    ax[0].set_ylabel("Accuracy"); ax[0].set_title("Acc_A (context A)")
    ax[1].set_title("Acc_B (context B)")
    ax[0].legend(fontsize=7)
    fig.suptitle("Fig 13.4 (seed %d) -- Scale 2 learning curves (A then B)" % seed, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig_13_4_learning_seed%d.png" % seed), dpi=140)
    plt.close(fig)
