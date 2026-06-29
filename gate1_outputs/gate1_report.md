# Gate 1A (geometry-easy generator) Report -- Recursive Scale Structure (Postulate 2)

Generator: **1A** | config: **dev** | seeds: **10** (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


## Conclusion: **PASS (Gate 1A only -- not causal validation)**


| metric | result | threshold | pass |
|---|---|---|---|
| median rho_struct | 0.865 | > 0.80 | YES |
| median accuracy_gap | 0.040 | <= 0.15 | YES |

- **rho_struct result:** median 0.865 (mean 0.864 +/- 0.016, range [0.838, 0.891]). Threshold rho_struct > 0.80 -> PASS.

- **accuracy gap result:** median 0.040 (mean 0.031 +/- 0.050, range [-0.076, 0.101]). Threshold gap <= 0.15 -> PASS.

- **both thresholds passed:** YES

- Per-seed Gate 1 pass (both conditions): 10 / 10.


## Scale 2 accuracy (ACC_gate) by condition (median over seeds)

| condition | median ACC_gate |
|---|---|
| oracle (true 2D) | 0.881 |
| **emergent (Gate 1)** | 0.841 |
| raw 20D control | 0.935 |
| PCA 2D control | 0.798 |
| random 2D control | 0.764 |

## Manifold recovery diagnostics (median rho over seeds)

| condition | median rho_struct |
|---|---|
| **emergent (crystallized)** | 0.865 |
| no-crystallization ablation | 0.954 |
| shuffled-signature control | 0.875 |

## Per-seed table

| seed | rho_struct | ACC_oracle | ACC_emergent | accuracy_gap | Acc_A_end_A | Acc_A_after_B | Acc_B_after_B | BT_A_emergent | n_repr_particles | n_repr_crystallized | n_scale2_centers | n_scale2_crystallized | gate1_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.891 | 0.812 | 0.888 | -0.076 | 0.949 | 0.820 | 0.956 | -0.129 | 203 | 180 | 25 | 24 | YES |
| 1 | 0.857 | 0.883 | 0.843 | 0.040 | 0.964 | 0.728 | 0.958 | -0.236 | 176 | 156 | 25 | 21 | YES |
| 2 | 0.875 | 0.861 | 0.847 | 0.014 | 0.951 | 0.727 | 0.966 | -0.224 | 204 | 183 | 25 | 23 | YES |
| 3 | 0.857 | 0.849 | 0.812 | 0.036 | 0.975 | 0.692 | 0.932 | -0.283 | 196 | 178 | 20 | 17 | YES |
| 4 | 0.870 | 0.879 | 0.839 | 0.040 | 0.947 | 0.711 | 0.968 | -0.236 | 176 | 155 | 23 | 22 | YES |
| 5 | 0.871 | 0.873 | 0.833 | 0.040 | 0.972 | 0.694 | 0.973 | -0.278 | 204 | 184 | 22 | 18 | YES |
| 6 | 0.860 | 0.891 | 0.845 | 0.046 | 0.968 | 0.729 | 0.960 | -0.239 | 185 | 165 | 22 | 19 | YES |
| 7 | 0.881 | 0.903 | 0.802 | 0.101 | 0.962 | 0.641 | 0.964 | -0.321 | 213 | 194 | 29 | 26 | YES |
| 8 | 0.840 | 0.890 | 0.790 | 0.100 | 0.962 | 0.605 | 0.975 | -0.357 | 199 | 177 | 18 | 16 | YES |
| 9 | 0.838 | 0.910 | 0.938 | -0.028 | 0.965 | 0.901 | 0.974 | -0.064 | 214 | 194 | 25 | 24 | YES |

## Figures

- `figures/fig_13_1_manifold_seed0.png` -- true vs emergent manifold
- `figures/fig_13_2_distcorr_seed0.png` -- pairwise distance correlation
- `figures/fig_13_3_graph_seed0.png` -- Scale 1 particle graph in emergent coords
- `figures/fig_13_4_learning_seed0.png` -- Scale 2 learning curves
- `figures/fig_13_5_accuracy_gap.png` -- accuracy gap bar chart over seeds

## Interpretation

rho_struct > 0.8 means the emergent representation recovered the hidden manifold structure. accuracy gap <= 0.15 means the existing v1 correction dynamics operate on the emergent representation about as well as on the oracle true-2D space. Both thresholds passed.


## Causal validity of the result

**This generator is geometrically easy.** The no-crystallization ablation (median rho=0.954) and the shuffled-signature control (median rho=0.875) *also* recover the manifold above the 0.80 threshold. That means the manifold geometry is essentially recoverable from the ambient 20D distances alone -- the learned behavioral signature and the crystallization step are **not** the load-bearing cause of manifold recovery here; the near-isometric embedding is.

Therefore, if the thresholds pass, this is a **Gate 1A pass** (the pipeline runs end-to-end and clears the bar on an easy generator) -- it is **not** causal validation of Postulate 2. A geometry-only baseline would pass too. Establishing that lower-scale *crystallization* is what induces the usable configuration space requires a generator where geometry-only recovery is insufficient.

That generator is **Gate 1B** (`--generator 1B`): u1 is encoded smoothly (geometry-accessible) while u2 is encoded only through high-frequency aliased terms, so raw-distance / PCA / spectral embeddings cannot order u2 (geometry-only rho falls well below 0.80). u2 then survives only via the behavioral signature of crystallized particles. See `GATE1B_README.md`.
