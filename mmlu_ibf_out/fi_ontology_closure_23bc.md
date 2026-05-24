# Cell 22 — Local Ontology Closure

## Objective

Test whether IBF can move beyond local definition override toward bounded local ontology installation.

the prior definition-only experiment (which lived as CELL 23 in an earlier draft and was removed for the paper-grade run) showed that definition-only training installs local definitions perfectly, but does not automatically propagate into downstream inference.

This cell tests two stronger regimes:

- **23B:** train definitions + one-hop implications; test two-hop generalization.
- **23C:** train definitions + one-hop + two-hop implications; test explicit ontology closure.

## Before

- Definition target: 0.000
- One-hop target: 0.000
- Two-hop target: 0.000
- Ordinary definition control: 1.000
- Ordinary one-hop control: 1.000
- Ordinary two-hop control: 1.000

## Results


23B — definition + one-hop closure:
  train groups:                         ['definition_train', 'onehop_train']
  train items:                          32

  definition target:                    1.000
  one-hop target:                       1.000
  two-hop target:                       0.000

  definition base selected:             0.000
  one-hop base selected:                0.000
  two-hop base selected:                1.000

  ordinary definition control:          1.000
  ordinary one-hop control:             1.000
  ordinary two-hop control:             1.000

  definition target-base margin:        +6.137
  one-hop target-base margin:           +6.132
  two-hop target-base margin:           -4.126

  centers added:                        32
  centers after:                        6414
  |v|max after:                          7.520
  time:                                 68.9s

  validation:
    ✓ definition_override
    ✓ onehop_closure
    ✓ ordinary_definition_control_stable
    ✓ ordinary_onehop_control_stable
    ✓ ordinary_twohop_control_stable
    ✓ definition_base_suppressed
    ✓ onehop_base_suppressed
    ✓ field_wrote_centers
    ✗ twohop_generalization_partial
    ✓ A_retention_stable


23C — explicit two-hop closure:
  train groups:                         ['definition_train', 'onehop_train', 'twohop_train']
  train items:                          48

  definition target:                    1.000
  one-hop target:                       1.000
  two-hop target:                       0.875

  definition base selected:             0.000
  one-hop base selected:                0.000
  two-hop base selected:                0.125

  ordinary definition control:          1.000
  ordinary one-hop control:             1.000
  ordinary two-hop control:             1.000

  definition target-base margin:        +6.137
  one-hop target-base margin:           +6.160
  two-hop target-base margin:           +4.851

  centers added:                        46
  centers after:                        6428
  |v|max after:                          7.520
  time:                                 37.4s

  validation:
    ✓ definition_override
    ✓ onehop_closure
    ✓ ordinary_definition_control_stable
    ✓ ordinary_onehop_control_stable
    ✓ ordinary_twohop_control_stable
    ✓ definition_base_suppressed
    ✓ onehop_base_suppressed
    ✓ field_wrote_centers
    ✓ twohop_closure
    ✓ twohop_base_suppressed
    ✓ A_retention_stable

## Interpretation

23B does not fully pass: one-hop closure alone does not fully generalize to all downstream two-hop consequences.

23C passes: local ontology closure works when two-hop implication structure is explicitly represented in the correction field.

This distinguishes three levels:

1. local definition override,
2. one-hop implication closure,
3. downstream two-hop ontology closure.
