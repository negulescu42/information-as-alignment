## Future Industries bridge — weak + strong prior regimes

## Geometry

- sigma_operating: `7.262067926124915`
- kappa_operating: `1.2671588548682307`
- epsilon_global: `0.0005`
- n_eff: `10000`
- d_eff: `32.84418487548828`
- d_shell: `42.10902786254883`

## Regime weighting

- weak repeats: `1`
- strong repeats: `3`
- weak push: `1.5`
- strong push multiplier: `1.8`

## Results

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| weak-prior target | 0.242 | 1.000 | +0.758 |
| strong-prior FI truth | 0.000 | 1.000 | +1.000 |
| strong-prior base selected | 1.000 | 0.000 | -1.000 |
| ordinary control | 1.000 | 1.000 | +0.000 |
| A retention lin | 0.850 | 0.850 | +0.000 |
| C retention lin | 0.987 | 0.987 | +0.000 |

## Criteria

- control_design_clean: `True`
- weak_truth_rises: `True`
- strong_truth_rises: `True`
- weak_target_ok: `True`
- strong_target_ok: `True`
- strong_base_drops: `True`
- strong_base_suppressed: `True`
- control_stable: `True`
- A_retention_stable: `True`
- C_retention_stable: `True`
- field_wrote_centers: `True`

## Interpretation

### bridge_result

The bridge test checks whether one field can support two learning regimes at once: weak-prior supplementation and strong-prior override.

### regime_weighting

V3 uses weak repeats=1, strong repeats=3, weak push=1.500, and strong push multiplier=1.800.

### weak_prior

Weak-prior target accuracy changes from 0.242 to 1.000.

### strong_prior

Strong-prior FI truth changes from 0.000 to 1.000, while strong base/common selection changes from 1.000 to 0.000.

### control_preservation

Ordinary controls remain 1.000 → 1.000; A retention remains 0.850 → 0.850; C retention remains 0.987 → 0.987.

### method_relevance

This unifies factual supplementation, strong-prior correction, and control preservation in a single frozen-model correction field. The failed V2 bridge showed that mixed regimes require regime-aware pressure.

