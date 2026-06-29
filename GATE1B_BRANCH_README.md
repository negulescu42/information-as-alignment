# Branch: `gate-1b-behavioral-geometry`

## Goal

Test whether **crystallized behavioral signatures** can supply representational
neighborhood structure when **ambient geometry is insufficient** (the Gate 1B
stress generator).

Gate 1B established that geometry-only recovery of the hidden manifold fails
(u2 is encoded only through high-frequency aliased terms). The Gate 1A affinity

```
W = W_geometry * W_behavior * W_stability
```

multiplicatively gates behavioral similarity by geometric proximity, so it
cannot create behavior-only edges and cannot rebuild a coordinate that geometry
has aliased away. This branch tests behavior-capable graph constructions.

## What is held fixed (unchanged)

- same Gate 1B generator (`--generator 1B`, `make_features_1B`)
- same task and truth (from hidden `u`)
- same v1 Scale 2 engine (`IBFAgent`) and oracle baseline
- same thresholds: `rho_struct > 0.8` and `accuracy_gap <= 0.15`
- same reporting format

**Only the graph construction / representation extraction layer changes.**

## Graph variants (in `scale1_representation.py`, `graph_mode=`)

| mode | construction |
|---|---|
| `multiplicative` | `W = W_geom * W_behavior * W_stab`, kNN by geometry — **Gate 1A BASELINE, kept** |
| `additive` | `W = (alpha*W_geom + beta*W_behavior) * W_stab`, kNN by the combined affinity |
| `union` | neighbors = kNN_geometry ∪ kNN_behavior |
| `behavior_first` | kNN from behavioral signatures; geometry only a regularizer floor |

## Run

```bash
python run_gate1b_variants.py 5        # 5 seeds on the 1B generator
python summarize_gate1b.py             # -> gate1b_variants_outputs/gate1b_variants_report.md
```

Per variant the report gives: `rho_struct, rho_u1, rho_u2, ACC_oracle,
ACC_emergent, accuracy_gap, n_crystallized, seeds passing`, plus geometry-only
controls and an upstream diagnostic (does the crystallized signature distance
encode u2?).

`rho_u1` / `rho_u2` are 5-fold k-NN-regression recoverabilities of each hidden
coordinate from the emergent representation (Spearman of predicted vs true).

## Desired pattern for a true Gate 1B pass

```
geometry-only controls stay below rho 0.8
random stays weak
emergent behavioral graph recovers rho_struct > 0.8
emergent Scale 2 accuracy within 0.15 of oracle
```

## Finding (see the committed report for the exact numbers)

Geometry-only controls do stay below 0.8 (the generator is hard as intended),
but **none of the behavior-capable variants clear the thresholds**, and they all
recover u1 (`rho_u1` high) while failing u2 (`rho_u2 ≈ 0`). The reason is
upstream of the graph: the crystallized **signature distances do not encode u2**
(per-particle activation balls span most of the u2 range, because the 20D
geometry is u1-dominated, so each particle averages its reward over a wide range
of true u2). With no u2 in the signatures, no graph mixing rule can rebuild it.

**Conclusion: 1B FAIL** — current Scale 1 dynamics are not sufficient yet for
Postulate 2 validation in the hard setting. This is an informative negative
result, not a bug: it localizes the binding constraint to **particle
localization of the aliased coordinate** (a Scale 1 representation-dynamics
issue), not to the geometry/behavior graph-combination rule. The multiplicative
baseline is retained for comparison and not deleted.

## Interpretation summary

```
1A pass:  end-to-end pipeline works when geometry is available.
1B pass:  (not achieved) behavioral structure recovers representation when geometry fails.
1B fail:  current Scale 1 dynamics are not sufficient yet for Postulate 2 in the hard setting.
```

No interface nodes, no promotion, no recursion, no RRW/chess.
