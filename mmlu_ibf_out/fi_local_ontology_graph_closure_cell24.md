# Cell 23 — Local Ontology Graph Closure

## Objective

Test whether IBF supports local ontology graph behavior beyond isolated definitions.

This cell separates:

1. explicit edge installation,
2. transitive closure,
3. explicit closure upper bound,
4. revision of local graph consequences,
5. ordinary control preservation.

## Before

- test_AB: target=0.000, base=1.000
- test_BC: target=0.000, base=1.000
- test_AC_transitive: target=0.000, base=1.000
- test_BD_revision: target=0.000, base=1.000
- test_AD_revised_transitive: target=0.000, base=1.000
- ordinary_A_control: target=1.000, base=1.000
- ordinary_B_control: target=1.000, base=1.000
- ordinary_AC_control: target=1.000, base=1.000

## Results


24A_edge_installation_AB_BC:
  train groups:                         ['train_AB', 'train_BC']
  train items:                          32

  AB edge target:                       1.000
  BC edge target:                       0.875
  AC transitive target:                 0.000
  BD revision target:                   0.000
  AD revised downstream target:         0.000

  AC base selected:                     1.000
  AD base selected:                     1.000

  ordinary A control:                   1.000
  ordinary B control:                   1.000
  ordinary AC control:                  1.000

  AB margin:                            +6.138
  BC margin:                            +4.977
  AC margin:                            -4.156
  BD margin:                            -4.156
  AD margin:                            -4.156

  centers added:                        31
  centers after:                        6413
  |v|max after:                          7.520
  time:                                 69.1s

  validation:
    ✓ AB_edge_installed
    ✗ BC_edge_installed
    ✓ ordinary_A_stable
    ✓ ordinary_B_stable
    ✓ field_wrote_centers


24B_transitive_probe_AB_BC_test_AC:
  train groups:                         ['train_AB', 'train_BC']
  train items:                          32

  AB edge target:                       1.000
  BC edge target:                       0.875
  AC transitive target:                 0.000
  BD revision target:                   0.000
  AD revised downstream target:         0.000

  AC base selected:                     1.000
  AD base selected:                     1.000

  ordinary A control:                   1.000
  ordinary B control:                   1.000
  ordinary AC control:                  1.000

  AB margin:                            +6.138
  BC margin:                            +4.977
  AC margin:                            -4.156
  BD margin:                            -4.156
  AD margin:                            -4.156

  centers added:                        31
  centers after:                        6413
  |v|max after:                          7.520
  time:                                 68.8s

  validation:
    ✓ AB_edge_installed
    ✗ BC_edge_installed
    ✗ AC_transitive_partial
    ✗ AC_base_not_dominant
    ✓ ordinary_A_stable
    ✓ ordinary_B_stable
    ✓ ordinary_AC_stable


24C_explicit_closure_AB_BC_AC:
  train groups:                         ['train_AB', 'train_BC', 'train_AC']
  train items:                          48

  AB edge target:                       1.000
  BC edge target:                       0.875
  AC transitive target:                 1.000
  BD revision target:                   0.000
  AD revised downstream target:         0.000

  AC base selected:                     0.000
  AD base selected:                     1.000

  ordinary A control:                   1.000
  ordinary B control:                   1.000
  ordinary AC control:                  1.000

  AB margin:                            +6.138
  BC margin:                            +4.977
  AC margin:                            +6.142
  BD margin:                            -4.156
  AD margin:                            -4.156

  centers added:                        47
  centers after:                        6429
  |v|max after:                          7.520
  time:                                 74.6s

  validation:
    ✓ AB_edge_installed
    ✗ BC_edge_installed
    ✓ AC_explicit_closure
    ✓ AC_base_suppressed
    ✓ ordinary_A_stable
    ✓ ordinary_B_stable
    ✓ ordinary_AC_stable


24D_revision_BC_to_BD:
  train groups:                         ['train_BD_revision']
  train items:                          16

  AB edge target:                       1.000
  BC edge target:                       0.875
  AC transitive target:                 1.000
  BD revision target:                   1.000
  AD revised downstream target:         0.000

  AC base selected:                     0.000
  AD base selected:                     1.000

  ordinary A control:                   1.000
  ordinary B control:                   1.000
  ordinary AC control:                  1.000

  AB margin:                            +5.520
  BC margin:                            +4.429
  AC margin:                            +5.524
  BD margin:                            +6.142
  AD margin:                            -4.156

  centers added:                        63
  centers after:                        6445
  |v|max after:                          7.520
  time:                                 62.8s

  validation:
    ✓ BD_revision_edge_installed
    ✗ AD_revised_downstream
    ✓ ordinary_A_stable
    ✓ ordinary_B_stable
    ✓ ordinary_AC_stable

## Interpretation

- Transitive closure pass: False
- Explicit closure pass: False
- Revision pass: False

If transitive closure passes, IBF shows evidence of limited emergent relational closure over a local semantic graph.

If explicit closure passes but transitive closure fails, IBF supports bounded local ontology closure when the relevant implication structure is represented directly in the correction field, but not spontaneous transitive composition.

If revision passes, the local ontology graph is not only installable but updateable.
