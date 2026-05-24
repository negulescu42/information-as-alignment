## Geometry

- sigma_operating: `7.262067926124915`
- kappa_operating: `1.2671588548682307`
- epsilon_global: `0.0005`
- n_eff: `10000`
- d_eff: `32.84418487548828`
- d_shell: `42.10902786254883`

## Sample buckets

- C_decisive: `5`
- D_decisive: `3`
- B_decisive: `3`
- A_mentor_wrong: `5`
- A_manager_wrong: `5`
- A_ambiguous: `5`
- A_decisive: `3`

## Population contributor statistics

| bucket | n | active_mean | dark_mean | top10_frac | |dR| | same_ctx_% | cryst_% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C_decisive | 5 | 1.0 | 1.0 | 1.000 | 2.523 | 100.0 | 100.0 |
| D_decisive | 3 | 1.0 | 0.0 | 1.000 | 1.246 | 100.0 | 100.0 |
| B_decisive | 3 | 1.0 | 0.0 | 1.000 | 0.826 | 100.0 | 100.0 |
| A_mentor_wrong | 5 | 1.0 | 0.0 | 1.000 | 0.030 | 100.0 | 100.0 |
| A_manager_wrong | 5 | 1.0 | 1.0 | 1.000 | 0.057 | 100.0 | 100.0 |
| A_ambiguous | 5 | 1.0 | 1.0 | 1.000 | 0.073 | 100.0 | 100.0 |
| A_decisive | 3 | 1.0 | 0.3 | 1.000 | 0.369 | 100.0 | 100.0 |

## Failure analysis

| bucket | subject | category | correct | pred | dR_correct | dR_pred | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_mentor_wrong | Milo Campbell | mentor | Tao Kowalski | Tao Wells | +0.1009 | -0.0162 | base prior dominates |
| A_mentor_wrong | Amara Kim | mentor | Valentina Griffin | Valentina Foster | +0.2355 | -0.0550 | base prior dominates |
| A_mentor_wrong | Anjali Desai | mentor | Tao Kowalski | Tao Brennan | +0.1341 | -0.0011 | base prior dominates |
| A_mentor_wrong | Liam Saleh | mentor | Valentina Park | Tao Berg | +0.1092 | -0.0311 | base prior dominates |
| A_mentor_wrong | Valentina Bose | mentor | Tao Patel | Tao Moretti | +0.1059 | -0.0471 | base prior dominates |
| A_manager_wrong | Andre Berg | manager | Tao Al-Rashid | Valentina Wong | +0.0133 | +0.0818 | weak support |
| A_manager_wrong | Nia Shah | manager | Valentina Griffin | Tao Flores | +0.0115 | +0.0557 | weak support |
| A_manager_wrong | Amara Rossi | manager | Valentina Patel | Valentina Chen | +0.0121 | -0.0754 | weak support |
| A_manager_wrong | Hassan Dubois | manager | Valentina Campbell | Tao Rao | +0.0186 | -0.0141 | weak support |
| A_manager_wrong | Liam Toure | manager | Valentina Flores | Valentina Dubois | +0.0132 | -0.0560 | weak support |

## Reorganization suppression

| subject | category | correct | original | dR_correct | dR_original | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| Drew Bello | manager | Tao Park | Tao Reed | +2.2207 | -1.5050 | 0.678 |
| Liam Choi | manager | Tao Sato | Valentina Moretti | +2.7324 | -2.3697 | 0.867 |
| Milo Cruz | manager | Tao Foster | Valentina Gupta | +2.7031 | -2.3395 | 0.866 |
| Andre Kumar | project | Odyssey | Frontier | +2.6318 | -1.6782 | 0.638 |
| Amara Kim | project | Summit | Frontier | +2.3276 | -1.8256 | 0.784 |

## Interpretation

### read_only_attribution

This cell attributes final predictions to stored correction centers without modifying the field.

### decisive_cases

Decisive B/C/D cases should show strong localized support from crystallized centers, indicating that lifecycle updates are carried by explicit stored structure rather than hidden model changes.

### residual_errors

The Phase A wrong buckets inspect the main residual weakness identified in Cell 11: retained onboarding mentor/manager associations.

### dark_contributors

Dark contributors estimate the counterfactual contribution of centers that are geometrically nearby but gated off by context. They help audit whether context gating is preventing cross-phase contamination.

### suppression

The reorganization suppression analysis checks whether new C-phase corrections support the updated answer while reducing or outcompeting the original answer.

### geometry_context

The provenance readout is performed under σ=7.2621, κ=1.2672, ε_global=0.0005.

