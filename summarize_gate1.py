"""
summarize_gate1.py
==================

Phase 5 of the Gate 1 branch: aggregate the per-seed results produced by
run_gate1.py into:

    gate1_outputs/gate1_per_seed.csv
    gate1_outputs/gate1_summary.csv
    gate1_outputs/gate1_report.md
    gate1_outputs/figures/fig_13_5_accuracy_gap.png   (spec 13.5)

and print the PASS / FAIL / INCONCLUSIVE conclusion.

Aggregate pass rule (spec sec 12):
    median rho_struct > 0.8  AND  median accuracy_gap <= 0.15
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "gate1_outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
RESULTS = os.path.join(OUT_DIR, "gate1_results.json")

RHO_THRESHOLD = 0.8
ACC_GAP_THRESHOLD = 0.15

PER_SEED_COLS = [
    'seed', 'rho_struct', 'ACC_oracle', 'ACC_emergent', 'accuracy_gap',
    'Acc_A_end_A', 'Acc_A_after_B', 'Acc_B_after_B', 'BT_A_emergent',
    'n_repr_particles', 'n_repr_crystallized',
    'n_scale2_centers', 'n_scale2_crystallized',
    'rho_nocryst', 'rho_shuffled',
    'ACC_raw20D', 'ACC_pca2D', 'ACC_random2D',
    'rho_pass', 'acc_pass', 'gate1_pass',
]


def load():
    with open(RESULTS) as f:
        return json.load(f)


def accuracy_gap_barchart(df):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(df))
    gaps = df['accuracy_gap'].values
    colors = ['#1b9e77' if g <= ACC_GAP_THRESHOLD else '#d95f02' for g in gaps]
    ax.bar(x, gaps, color=colors)
    ax.axhline(ACC_GAP_THRESHOLD, color='red', ls='--',
               label='threshold = %.2f' % ACC_GAP_THRESHOLD)
    ax.axhline(0.0, color='gray', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df['seed'].astype(int))
    ax.set_xlabel("seed")
    ax.set_ylabel("ACC_oracle - ACC_emergent")
    ax.set_title("Fig 13.5 -- Scale 2 accuracy gap per seed (lower is better)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    p = os.path.join(FIG_DIR, "fig_13_5_accuracy_gap.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def agg(df, col):
    v = df[col].astype(float).values
    return dict(median=float(np.median(v)), mean=float(np.mean(v)),
               std=float(np.std(v)), min=float(np.min(v)), max=float(np.max(v)))


def main():
    payload = load()
    df = pd.DataFrame(payload['rows'])
    for c in PER_SEED_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[PER_SEED_COLS].sort_values('seed').reset_index(drop=True)

    per_seed_path = os.path.join(OUT_DIR, "gate1_per_seed.csv")
    df.to_csv(per_seed_path, index=False)

    rho_stats = agg(df, 'rho_struct')
    gap_stats = agg(df, 'accuracy_gap')
    acc_o = agg(df, 'ACC_oracle')
    acc_e = agg(df, 'ACC_emergent')

    median_rho = rho_stats['median']
    median_gap = gap_stats['median']
    rho_pass = median_rho > RHO_THRESHOLD
    gap_pass = median_gap <= ACC_GAP_THRESHOLD
    both = rho_pass and gap_pass

    if both:
        conclusion = "PASS"
    elif (median_rho > RHO_THRESHOLD - 0.05) or (median_gap < ACC_GAP_THRESHOLD + 0.05):
        conclusion = "INCONCLUSIVE" if not both else "PASS"
    else:
        conclusion = "FAIL"
    if not both and conclusion != "FAIL":
        conclusion = "INCONCLUSIVE"
    if both:
        conclusion = "PASS"

    summary = {
        'n_seeds': len(df),
        'config': payload.get('config'),
        'median_rho_struct': median_rho,
        'mean_rho_struct': rho_stats['mean'],
        'std_rho_struct': rho_stats['std'],
        'median_accuracy_gap': median_gap,
        'mean_accuracy_gap': gap_stats['mean'],
        'std_accuracy_gap': gap_stats['std'],
        'median_ACC_oracle': acc_o['median'],
        'median_ACC_emergent': acc_e['median'],
        'rho_threshold': RHO_THRESHOLD,
        'acc_gap_threshold': ACC_GAP_THRESHOLD,
        'rho_pass': rho_pass,
        'acc_gap_pass': gap_pass,
        'n_seeds_gate1_pass': int(df['gate1_pass'].sum()),
        'conclusion': conclusion,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(OUT_DIR, "gate1_summary.csv"), index=False)

    fig_path = accuracy_gap_barchart(df)

    # -------- markdown report --------
    lines = []
    lines.append("# Gate 1 Report -- Recursive Scale Structure (Postulate 2)\n")
    lines.append("Config: **%s** | seeds: **%d** (%s)\n"
                 % (payload.get('config'), len(df),
                    ", ".join(str(int(s)) for s in df['seed'])))
    lines.append("\n## Conclusion: **%s**\n" % conclusion)
    lines.append("")
    lines.append("| metric | result | threshold | pass |")
    lines.append("|---|---|---|---|")
    lines.append("| median rho_struct | %.3f | > %.2f | %s |"
                 % (median_rho, RHO_THRESHOLD, "YES" if rho_pass else "NO"))
    lines.append("| median accuracy_gap | %.3f | <= %.2f | %s |"
                 % (median_gap, ACC_GAP_THRESHOLD, "YES" if gap_pass else "NO"))
    lines.append("")
    lines.append("- **rho_struct result:** median %.3f (mean %.3f +/- %.3f, "
                 "range [%.3f, %.3f]). Threshold rho_struct > %.2f -> %s.\n"
                 % (median_rho, rho_stats['mean'], rho_stats['std'],
                    rho_stats['min'], rho_stats['max'], RHO_THRESHOLD,
                    "PASS" if rho_pass else "FAIL"))
    lines.append("- **accuracy gap result:** median %.3f (mean %.3f +/- %.3f, "
                 "range [%.3f, %.3f]). Threshold gap <= %.2f -> %s.\n"
                 % (median_gap, gap_stats['mean'], gap_stats['std'],
                    gap_stats['min'], gap_stats['max'], ACC_GAP_THRESHOLD,
                    "PASS" if gap_pass else "FAIL"))
    lines.append("- **both thresholds passed:** %s\n" % ("YES" if both else "NO"))
    lines.append("- Per-seed Gate 1 pass (both conditions): %d / %d.\n"
                 % (int(df['gate1_pass'].sum()), len(df)))

    lines.append("\n## Scale 2 accuracy (ACC_gate) by condition (median over seeds)\n")
    lines.append("| condition | median ACC_gate |")
    lines.append("|---|---|")
    lines.append("| oracle (true 2D) | %.3f |" % acc_o['median'])
    lines.append("| **emergent (Gate 1)** | %.3f |" % acc_e['median'])
    lines.append("| raw 20D control | %.3f |" % agg(df, 'ACC_raw20D')['median'])
    lines.append("| PCA 2D control | %.3f |" % agg(df, 'ACC_pca2D')['median'])
    lines.append("| random 2D control | %.3f |" % agg(df, 'ACC_random2D')['median'])

    lines.append("\n## Manifold recovery diagnostics (median rho over seeds)\n")
    lines.append("| condition | median rho_struct |")
    lines.append("|---|---|")
    lines.append("| **emergent (crystallized)** | %.3f |" % rho_stats['median'])
    lines.append("| no-crystallization ablation | %.3f |" % agg(df, 'rho_nocryst')['median'])
    lines.append("| shuffled-signature control | %.3f |" % agg(df, 'rho_shuffled')['median'])

    lines.append("\n## Per-seed table\n")
    show = ['seed', 'rho_struct', 'ACC_oracle', 'ACC_emergent', 'accuracy_gap',
            'Acc_A_end_A', 'Acc_A_after_B', 'Acc_B_after_B', 'BT_A_emergent',
            'n_repr_particles', 'n_repr_crystallized',
            'n_scale2_centers', 'n_scale2_crystallized', 'gate1_pass']
    lines.append("| " + " | ".join(show) + " |")
    lines.append("|" + "|".join(["---"] * len(show)) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in show:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append("YES" if v else "no")
            elif isinstance(v, (float, np.floating)):
                cells.append("%.3f" % v)
            else:
                cells.append(str(int(v)))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## Figures\n")
    lines.append("- `figures/fig_13_1_manifold_seed0.png` -- true vs emergent manifold")
    lines.append("- `figures/fig_13_2_distcorr_seed0.png` -- pairwise distance correlation")
    lines.append("- `figures/fig_13_3_graph_seed0.png` -- Scale 1 particle graph in emergent coords")
    lines.append("- `figures/fig_13_4_learning_seed0.png` -- Scale 2 learning curves")
    lines.append("- `figures/fig_13_5_accuracy_gap.png` -- accuracy gap bar chart over seeds")

    lines.append("\n## Interpretation\n")
    if both:
        lines.append("rho_struct > 0.8 means Scale 1 crystallization recovered the hidden "
                     "manifold structure. accuracy gap <= 0.15 means the existing v1 "
                     "correction dynamics operate on the emergent representation about as "
                     "well as on the oracle true-2D space. Both thresholds passed: "
                     "**Postulate 2 (Recursive Scale Structure) is validated for this "
                     "computational instantiation at Gate 1.**")
    elif rho_pass and not gap_pass:
        lines.append("rho passed but the accuracy gap exceeds 0.15: the emergent coordinates "
                     "recover geometry but are not operationally usable by the correction "
                     "dynamics. Gate 1 does **not** pass.")
    elif not rho_pass:
        lines.append("rho_struct did not clear 0.8: Scale 1 crystallization did not recover "
                     "the hidden manifold. Gate 1 does **not** pass.")
    lines.append("")

    with open(os.path.join(OUT_DIR, "gate1_report.md"), "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print("\nWrote:")
    print("  ", per_seed_path)
    print("  ", os.path.join(OUT_DIR, "gate1_summary.csv"))
    print("  ", os.path.join(OUT_DIR, "gate1_report.md"))
    print("  ", fig_path)


if __name__ == "__main__":
    main()
