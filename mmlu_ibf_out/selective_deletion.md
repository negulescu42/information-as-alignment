## Geometry

- sigma_operating: `7.262067926124915`
- kappa_operating: `1.2671588548682307`
- epsilon_global: `0.0005`
- n_eff: `10000`
- d_eff: `32.84418487548828`
- d_shell: `42.10902786254883`

## Target

- employee: `Ava Mitchell`
- facts: `5`
- deletion mode: `correct_positive`
- centers deleted: `4`

## Results

| stream | before | after | delta |
| --- | ---: | ---: | ---: |
| target employee | 0.800 | 0.000 | -0.800 |
| everyone else | 0.850 | 0.850 | +0.000 |

## Gate

- target_drop: `0.8000`
- others_abs_drift: `0.0000`
- passed: `True`

## Target facts

| category | answer | before | after | status | before_pred | after_pred |
| --- | --- | ---: | ---: | --- | --- | --- |
| location | Building D, Floor 4 | False | False | still_wrong | Building D, Floor 5 | Building D, Floor 5 |
| manager | Valentina Osei | True | False | erased | Valentina Osei | Tao Watanabe |
| mentor | Valentina Sullivan | True | False | erased | Valentina Sullivan | Tao Flores |
| project | Meridian | True | False | erased | Meridian | Apex |
| team | Accounting | True | False | erased | Accounting | Solutions Engineering |

## Interpretation

### provenance_guided_deletion

Deleting active support centers for Ava Mitchell changed target accuracy from 0.800 to 0.000, a drop of 0.800.

### collateral_effect

Everyone-else accuracy changed from 0.850 to 0.850, with absolute drift 0.000.

### method_relevance

The result tests causal erasure: deleting the centers that actively support a belief, rather than deleting all centers with a matching subject feature.

### durable_alignment_relevance

Right-to-erasure requires provenance-guided removal. The relevant object is not merely an entity tag, but the local support structure that makes a belief win.

### geometry_context

The deletion smoke is evaluated under σ=7.2621, κ=1.2672, ε_global=0.0005.

