## FI local alignment system — report and validation

## Run

- mode: `strong_smoke`
- rules: `200`
- conflict rules: `40`
- non-conflict rules: `160`
- control rules: `80`

## Results

| metric | value |
| --- | ---: |
| install target log | 1.000 |
| install target linear | 1.000 |
| revision new selected | 0.975 |
| revision old selected | 0.025 |
| revision base selected | 0.000 |
| revision other selected | 0.000 |
| mean new-old margin | +1.731 |
| min new-old margin | -0.263 |
| non-conflict after revision | 1.000 |
| non-conflict delta | +0.000 |
| control after revision | 1.000 |
| rollback old selected | 1.000 |
| rollback new selected | 0.000 |
| removal install selected | 0.000 |
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

### local_alignment_system

The system installed 200 local FI constraints and revised 40 selected constraints through contradiction while preserving unrelated local constraints and ordinary controls.

### mutable_belief_state

The result supports the interpretation of IBF as a mutable local belief-state layer: rules can be installed, revised, rolled back, and removed without retraining the frozen model.

### revision_dynamics

Revision selected the new rule at 0.975, the old rule at 0.025, and the base/common answer at 0.000. Non-conflict retention remained 1.000.

### paper_claim_boundary

If this artifact is from smoke mode, report it as a strong-smoke validation of the mechanism. The full 1K / 200-rule result should be run separately for the final paper-grade scale claim.

