# Gate 1B — Stress Generator (geometry-only recovery is insufficient)

## Why Gate 1B exists

Gate 1A passes, but its generator is **geometrically easy**: the 20D embedding
is near-isometric, so the hidden manifold is recoverable from the ambient
distances alone. Concretely, on Gate 1A the controls that *strip out* the
learned behavioral structure still clear the ρ > 0.8 bar:

| geometry-only baseline (no crystallization, no behavior) | median ρ_struct (Gate 1A) |
|---|---|
| raw 20D distances | ~0.92 |
| PCA(2) | ~0.76 |
| spectral embedding | ~0.82 |
| no-crystallization ablation | ~0.97 |
| shuffled-signature control | ~0.90 |

Because a geometry-only baseline passes too, a Gate 1A pass shows the pipeline
**runs end-to-end and clears the bar**, but it does **not** show that
lower-scale *crystallization* is the cause of the usable representation. The
near-isometric embedding is doing the work. That is why a Gate 1A pass is
labelled **"Gate 1A pass — not causal validation."**

## What Gate 1B changes (and what it keeps)

Gate 1B keeps **everything** identical except the observation embedding:

- same hidden manifold `u ~ N(0, I_2)`;
- same task `s_j(u,c) = u1·p_j + u_c·u2·r_j`, truth from `u`;
- same Scale 1 / Scale 2 engines, same encoders, same metrics;
- **same pass/fail thresholds**: `rho_struct > 0.8` and `acc_gap <= 0.15`.

The only change is the feature map (`make_features_1B` in
`gate1_environment.py`):

- **u1** is encoded through smooth, low-frequency terms → ambient geometry
  recovers u1.
- **u2** is encoded **only** through high-frequency *aliased* terms
  `sin(f·u2), cos(f·u2), …` (f = 3). Globally, ambient distance does **not**
  track `|Δu2|`: points a full period apart in u2 sit close in x, and
  geometric neighbours can have very different true u2. So raw-distance / PCA /
  spectral embeddings **cannot order u2**.
- u2 is still **locally** accessible (the high-frequency map is locally
  invertible within one period), so a Scale 1 particle localizes u2 well enough
  to form a u2-dependent **behavioral signature** — the only channel through
  which the global u2 ordering survives.

## Demonstration: geometry alone is insufficient on Gate 1B

```bash
python gate1b_geometry_check.py 5
```

Representative result (pairwise-distance Spearman ρ, same metric as ρ_struct):

| generator | raw20D | PCA2D | spectral | random2D | best geometry-only | verdict |
|---|---|---|---|---|---|---|
| 1A | 0.92 | 0.76 | 0.82 | 0.61 | **0.92** | geometry SUFFICIENT (geometry-easy) |
| 1B | 0.59 | 0.64 | 0.59 | 0.27 | **0.64** | geometry INSUFFICIENT (all < 0.80) |

On Gate 1B, ambient geometry recovers u1 (ρ ≈ 0.80 vs |Δu1|) but is blind to u2
(ρ ≈ 0.09 vs |Δu2|). No geometry-only baseline reaches the 0.8 threshold.

## How to run the full Gate 1B suite

```bash
# full pipeline on the stress generator (writes to gate1b_outputs/)
python run_gate1.py --generator 1B --config dev
python summarize_gate1.py gate1b_outputs
```

Outputs mirror Gate 1A but land in `gate1b_outputs/`.

## Honest status of the *method* on Gate 1B

Gate 1B is the discriminating test; it is **not** pre-tuned to pass. With the
emergent encoder **exactly as implemented for Gate 1A**, the affinity graph is
built by sparsifying on geometric k-NN and only *multiplicatively* gating those
geometric edges by behavioral similarity (`W = Wx · Wb · Wstab`, spec §7.1).
That construction can down-weight geometric edges but cannot *create*
behavior-only edges, so it cannot reconstruct a coordinate (u2) that geometry
has aliased away. In a quick check the unchanged emergent method scores ρ ≈ 0.56
on Gate 1B — i.e. **the current pipeline does not pass Gate 1B.**

This is the intended, informative outcome: Gate 1B isolates exactly the claim
Gate 1A could not test. Passing it requires a representation step that uses the
crystallized **behavioral signature** as a genuine source of connectivity /
coordinates (e.g. an affinity that adds behavior-similarity edges, or a joint
geometry-plus-behavior embedding), rather than as a multiplicative gate on
geometry. Designing and validating that representation step is the real content
of a Gate 1B pass and is deliberately left as the next step — no method change
was made here, per the "do not tune anything" instruction.
