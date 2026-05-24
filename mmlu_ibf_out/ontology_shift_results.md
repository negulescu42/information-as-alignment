## Future Industries systemic ontology shift

## Geometry

- sigma_operating: `7.262067926124915`
- kappa_operating: `1.2671588548682307`
- epsilon_global: `0.0005`
- n_eff: `10000`
- d_eff: `32.84418487548828`
- d_shell: `42.10902786254883`

## Results

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| FI truth selected | 0.000 | 1.000 | +1.000 |
| common/base selected | 1.000 | 0.000 | -1.000 |
| FI-minus-base margin | -4.156 | 2.726 | +6.881 |
| common-meaning control | 1.000 | 1.000 | +0.000 |
| A retention lin | 0.850 | 0.850 | +0.000 |

## Criteria

- fi_truth_rises: `True`
- base_rate_drops: `True`
- control_stable: `True`
- A_retention_stable: `True`
- margin_moves: `True`
- field_wrote_centers: `True`

## Interpretation

### ontology_shift

The test installs FI-specific meanings for 10 concepts against strong common meanings. FI target accuracy changes from 0.000 to 1.000.

### strong_prior_reversal

The mean FI-minus-base margin moves from -4.156 to +2.726.

### control_preservation

Common-meaning controls remain 1.000 → 1.000; A_Onboarding retention remains 0.850 → 0.850.

### method_relevance

This tests whether IBF can represent a local ontology, not merely a list of facts: inside FI, the same words can point to different operational meanings without globally changing ordinary language.

### durable_alignment_relevance

This is the semantic version of policy alignment: a frozen model keeps its ordinary meaning outside the context, while FI-specific meanings dominate inside the local operational field.

