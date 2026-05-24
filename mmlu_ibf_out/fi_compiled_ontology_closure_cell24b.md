# Cell 24 — Compiled Ontology Closure

## Objective

Test the practical architecture implied by Cell 24:

1. deterministic policy graph closure,
2. explicit derived consequences,
3. IBF enforcement over the frozen model,
4. revised graph closure after policy change.

IBF is not asked to derive A→C from A→B and B→C. Instead, the closure compiler derives A→C, then IBF enforces it.

## Result


24B_initial_compiled_closure_AB_BC_AC:
  train groups:                         ['initial_train_active']
  train items:                          48

  initial_test_active_AB               target=1.000 base=0.000 margin=+6.138
  initial_test_active_BC               target=0.875 base=0.125 margin=+4.948
  initial_test_active_AC               target=1.000 base=0.000 margin=+6.142
  initial_test_retired_BD              target=1.000 base=1.000 margin=+0.000
  initial_test_retired_AD              target=1.000 base=1.000 margin=+0.000
  ordinary_control_AB                  target=1.000 base=1.000 margin=+0.000
  ordinary_control_BC                  target=1.000 base=1.000 margin=+0.000
  ordinary_control_AC                  target=1.000 base=1.000 margin=+0.000
  ordinary_control_BD                  target=1.000 base=1.000 margin=+0.000
  ordinary_control_AD                  target=1.000 base=1.000 margin=+0.000

  centers added:                        47
  centers after:                        6429
  |v|max after:                          7.520
  time:                                 38.0s

  validation:
    ✓ AB_active
    ✓ BC_active
    ✓ AC_compiled_closure_active
    ✓ BD_retired_inactive
    ✓ AD_retired_inactive
    ✓ ordinary_AB_stable
    ✓ ordinary_BC_stable
    ✓ ordinary_AC_stable
    ✓ field_wrote_centers


24B_revised_compiled_closure_AB_BD_AD:
  train groups:                         ['revised_train_active']
  train items:                          48

  revised_test_active_AB               target=1.000 base=0.000 margin=+6.138
  revised_test_active_BD               target=1.000 base=0.000 margin=+6.142
  revised_test_active_AD               target=1.000 base=0.000 margin=+6.142
  revised_test_retired_BC              target=1.000 base=1.000 margin=+0.000
  revised_test_retired_AC              target=1.000 base=1.000 margin=+0.000
  ordinary_control_AB                  target=1.000 base=1.000 margin=+0.000
  ordinary_control_BC                  target=1.000 base=1.000 margin=+0.000
  ordinary_control_AC                  target=1.000 base=1.000 margin=+0.000
  ordinary_control_BD                  target=1.000 base=1.000 margin=+0.000
  ordinary_control_AD                  target=1.000 base=1.000 margin=+0.000

  centers added:                        48
  centers after:                        6430
  |v|max after:                          7.520
  time:                                 38.2s

  validation:
    ✓ AB_still_active
    ✓ BD_revision_active
    ✓ AD_recompiled_closure_active
    ✓ BC_retired_suppressed
    ✓ AC_old_closure_retired_suppressed
    ✓ ordinary_AB_stable
    ✓ ordinary_BD_stable
    ✓ ordinary_AD_stable
    ✓ field_wrote_centers

## Interpretation

- Initial compiled closure pass: True
- Revised compiled closure pass: True
- Overall pass: True

This tests IBF as an enforcement substrate for compiled local ontology closure, not as an autonomous symbolic reasoner.
