# Gate 1B (behavioral-geometry) Report

Generator: **1B (stress)** | seeds: **5** (0, 1, 2, 3, 4) | thresholds: rho_struct > 0.80 and accuracy_gap <= 0.15


## Conclusion: **1B FAIL** -- no graph variant clears both thresholds. The current Scale 1 dynamics are not sufficient yet for Postulate 2 validation in the hard setting.


## Geometry-only controls (median rho_struct -- should stay < 0.80)

| raw20D | PCA2D | spectral | random2D | best |
|---|---|---|---|---|
| 0.544 | 0.562 | 0.418 | 0.283 | **0.562** |

Geometry-only best = 0.562 <= 0.80 -> geometry is INSUFFICIENT on Gate 1B.


## Graph variants (median over seeds)

| variant | rho_struct | rho_u1 | rho_u2 | ACC_oracle | ACC_emergent | accuracy_gap | n_cryst | seeds pass |
|---|---|---|---|---|---|---|---|---|
| multiplicative (BASELINE) | 0.531 | 0.977 | 0.021 | 0.907 | 0.745 | +0.154 | 138 | 0/5 |
| additive | 0.502 | 0.977 | 0.004 | 0.907 | 0.749 | +0.159 | 138 | 0/5 |
| union | 0.532 | 0.981 | 0.002 | 0.907 | 0.739 | +0.151 | 138 | 0/5 |
| behavior_first | 0.404 | 0.953 | 0.057 | 0.907 | 0.752 | +0.175 | 138 | 0/5 |

## Upstream diagnostic: do crystallized signatures encode u2?

| signature-dist vs u1 | signature-dist vs u2 | per-particle u2 variance (global=1.0) |
|---|---|---|
| 0.865 | 0.003 | 0.794 |

no-crystallization control rho=0.592 | shuffled-signature control rho=0.528


## Interpretation

On the Gate 1B generator u1 is geometry-accessible while u2 is encoded only through high-frequency aliased terms. The per-variant table shows rho_u1 (recovery of u1) and rho_u2 (recovery of u2) separately.


**Why every variant fails is upstream of the graph.** The crystallized signature *distance* tracks u1 (0.86) but not u2 (0.00), and each particle's activation ball spans most of the u2 range (median u2 variance 0.79 of the global 1.0). Because the 20D geometry is u1-dominated, a particle averages its reward signal over a wide range of true u2, so the behavioral signature collapses to a function of u1. With no u2 information in the signatures, *no* graph mixing rule (multiplicative, additive, union, or behavior-first) can rebuild u2 -- the binding constraint is particle localization of the aliased coordinate, not the geometry/behavior combination rule.


**Conclusion: 1B FAIL.** Current Scale 1 dynamics are not sufficient yet for Postulate 2 validation in the hard setting. The next lever is the Scale 1 representation dynamics themselves (e.g. a behavior-sensitive kernel / metric so particles can localize the aliased coordinate), not the graph construction layer tested here.


## Note on the baseline

The `multiplicative` row is the unchanged Gate 1A affinity (W = W_geometry * W_behavior * W_stability), kept as the baseline. Its failure on Gate 1B is the motivation for this branch and is retained, not deleted.
