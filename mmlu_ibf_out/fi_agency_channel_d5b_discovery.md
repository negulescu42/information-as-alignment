# § 24b-D5b — Discovery Training v2 (ungated readout)

- D5 used engine.delta_R (gated by crystallization). Phase 0 supervised install with contrastive pushes doesn't satisfy the convergence_threshold, so new centers stay transient and gated readout returns 0. Boltzmann degenerated to deterministic-on-base.
- D5b uses ungated readout (matches D2/D3/D4 / cell 58 convention).
- Phase 0: 3 epochs × 3 reps supervised AB+BC
- Phase 1: 40 epochs Boltzmann discovery on AC with environmental reward

## Pre vs Post (UNGATED δR readout)

| group | pre | post | Δ |
|---|---:|---:|---:|
| test_AB | 1.000 | 1.000 | +0.000 |
| test_BC | 0.875 | 1.000 | +0.125 |
| test_AC_heldout | 0.000 | 1.000 | +1.000 |

- k_eff at AC: 5.000 → 5.000 (Δ +0.000)
- engine final: value 6436 (6383 cryst), agency cryst 2150

## Trajectory (eval every 5 epochs)

| ep | AC_hit | AC_k̄ | dR_tgt | dR_base | test_AC | test_AB | test_BC | V/cryst |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.312 | 5.00 | +0.05 | -2.86 | 1.000 | 1.000 | 0.875 | 6432/6382 |
| 5 | 1.000 | 5.00 | +6.40 | -3.70 | 1.000 | 1.000 | 0.875 | 6433/6382 |
| 10 | 1.000 | 5.00 | +7.88 | -2.72 | 1.000 | 1.000 | 1.000 | 6436/6383 |
| 15 | 1.000 | 5.00 | +7.88 | -1.99 | 1.000 | 1.000 | 1.000 | 6436/6383 |
| 20 | 1.000 | 5.00 | +7.88 | -1.46 | 1.000 | 1.000 | 1.000 | 6436/6383 |
| 25 | 1.000 | 5.00 | +7.88 | -1.07 | 1.000 | 1.000 | 1.000 | 6436/6383 |
| 30 | 1.000 | 5.00 | +7.88 | -0.79 | 1.000 | 1.000 | 1.000 | 6436/6383 |
| 35 | 1.000 | 5.00 | +7.88 | -0.58 | 1.000 | 1.000 | 1.000 | 6436/6383 |
| 40 | 1.000 | 5.00 | +7.88 | -0.42 | 1.000 | 1.000 | 1.000 | 6436/6383 |

## Verdict

- code: `emergence_at_locality_cost`
- AC emerged (+1.000) but locality drifted (AB Δ=0.000, BC Δ=0.125). Too aggressive.
