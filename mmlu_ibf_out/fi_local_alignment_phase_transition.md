## FI local alignment control — single-phase closure

## Run policy

- mode: `strong_smoke`
- rules: `200`
- conflict rules: `40`
- control rules: `80`
- install epochs: `2`
- revision epochs: `2`
- sigma: `7.262067926124915`
- merge threshold: `10.893101889187372`

## Results

| metric | value |
| --- | ---: |
| install target log | 1.000 |
| install target lin | 1.000 |
| revision new selected | 0.975 |
| revision old selected | 0.025 |
| revision base selected | 0.000 |
| non-conflict retention | 1.000 |
| control | 1.000 |
| rollback old selected | 1.000 |
| rollback new selected | 0.000 |
| removal install target selected | 0.000 |
| removal control selected | 1.000 |

## Criteria

- install_success: `True`
- revision_new_wins: `True`
- old_suppressed: `True`
- base_suppressed: `True`
- nonconf_retained: `True`
- control_stable: `True`
- rollback_old_dominates_new: `True`
- removal_drops_install_target: `True`
- removal_control_restored: `True`

## Interpretation

### local_alignment_control

This experiment tests whether one mutable correction field can install local operational constraints, revise a contradicted subset, suppress obsolete rules, preserve unrelated installed rules, preserve ordinary controls, and support rollback/removal behavior.

### single_phase_closure

The revision phase uses one Crucible melt pass, local energy accumulation, and one global closure. This avoids repeated global closure ticks during a local contradiction transition.

### install

Install target log accuracy is 1.000; control log accuracy after install is 1.000.

### revision

Revision new selected is 0.975; old selected is 0.025; base selected is 0.000.

### retention

Non-conflict retention is 1.000, with delta +0.000 from post-install.

