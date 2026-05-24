## Future Industries strong-prior pilot

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
| base/common selected | 1.000 | 0.000 | -1.000 |
| FI-minus-base margin | -4.156 | 2.728 | +6.884 |
| ordinary control | 1.000 | 1.000 | +0.000 |
| A retention lin | 0.850 | 0.850 | +0.000 |

## Criteria

- fi_truth_rises: `True`
- base_rate_drops: `True`
- control_stable: `True`
- A_retention_stable: `True`
- margin_moves: `True`
- field_wrote_centers: `True`

## Interpretation

### strong_prior

The base/common answer begins with probability 0.957, while the FI truth begins with probability 0.015. The initial mean FI-minus-base margin is -4.156.

### local_correction

After local IBF correction, FI truth accuracy changes from 0.000 to 1.000, while base/common selection changes from 1.000 to 0.000.

### control_preservation

Ordinary-control accuracy changes from 1.000 to 1.000; A_Onboarding retention changes from 0.850 to 0.850.

### method_relevance

This tests whether local correction centers can override a confident base-model prior without globally retraining the model or damaging unrelated operational knowledge.

### durable_alignment_relevance

In practical terms, this is the policy-alignment use case: a frozen model has a strong generic prior, but the organization needs local current policy to win.

