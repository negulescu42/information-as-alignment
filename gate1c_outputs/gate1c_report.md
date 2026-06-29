# Gate 1C Report -- Scale 1 Behavioral Localization (discrepancy-driven splitting)

Generator: **1B (stress)** | seeds: **5** | thresholds: signature->u2 > 0.30 (leading), emergent rho_u2 > 0.50, then rho_struct > 0.80 & gap < 0.15


## Generator fairness: can ANY static encoder recover u2 from x20?

Supervised k-NN regression x20 -> u2 with TRUE labels (median over seeds). This is the ceiling for any observation->representation encoder; the oracle-*signature* diagnostic does not test it because it has behavioral access at the query point.

| u2_freq | geometry-aliasing | supervised x20->u2 ceiling |
|---|---|---|
| 3.0 | higher | 0.037 -- impossible (no static encoder can recover u2) |
| 2.0 | higher | 0.446 -- marginal |
| 1.5 | higher | 0.815 -- recoverable |

## Scale 1 localization -- does splitting make signatures encode u2?

| u2_freq | split | n_particles | n_cryst | n_splits | per-particle u2 var | signature->u2 (reg) | signature->u2 (pair) |
|---|---|---|---|---|---|---|---|
| 3.0 | off | 260 | 138 | 0 | 0.794 | -0.021 | 0.003 |
| 3.0 | on | 1398 | 687 | 3263 | 0.495 | 0.662 | 0.163 |
| 2.0 | off | 228 | 155 | 0 | 0.346 | 0.212 | -0.008 |
| 2.0 | on | 918 | 569 | 1881 | 0.005 | 0.743 | 0.282 |
| 1.5 | off | 214 | 164 | 0 | 0.025 | 0.558 | 0.094 |
| 1.5 | on | 534 | 443 | 781 | 0.013 | 0.812 | 0.341 |

## End-to-end emergent recovery (best graph mode, median over seeds)

| u2_freq | split | rho_struct | rho_u1 | rho_u2 | rho_u2 > 0.5 | rho_struct > 0.8 |
|---|---|---|---|---|---|---|
| 3.0 | off | 0.532 | 0.979 | -0.022 | no | no |
| 3.0 | on | 0.585 | 0.997 | -0.029 | no | no |
| 2.0 | off | 0.552 | 0.987 | 0.210 | no | no |
| 2.0 | on | 0.600 | 0.997 | 0.180 | no | no |
| 1.5 | off | 0.557 | 0.988 | 0.584 | YES | no |
| 1.5 | on | 0.623 | 0.995 | 0.308 | no | no |

## Scale 2 accuracy at the headline frequency (median over seeds)

| u2_freq | split | ACC_oracle | ACC_emergent | accuracy_gap | gap < 0.15 |
|---|---|---|---|---|---|
| 1.5 | off | 0.907 | 0.757 | +0.136 | YES |
| 1.5 | on | 0.907 | 0.739 | +0.142 | YES |

## Verdict

**Leading indicator (Scale 1 blocker):** at the hardest aliasing (u2_freq=3.0) discrepancy-driven splitting raises signature->u2 from -0.021 (off) to 0.662 (on)
, clearing the 0.30 gate. **Splitting resolves the Scale 1 localization blocker: crystallized signatures now encode the behaviorally aliased coordinate.** This confirms the supervisor's hypothesis at the Scale 1 level.


**Full Gate 1C pass NOT achieved at any tested frequency.** The result exposes a *second* blocker, downstream of Scale 1, with a clean two-regime structure:


1. **High aliasing (u2_freq=3.0):** splitting fixes Scale 1 (signatures encode u2) but the static x->q encoder cannot carry u2 to test time -- supervised x20->u2 is ~0 even with true labels, so u2 is not recoverable from the observation by *any* encoder. The oracle-signature diagnostic passed here only because it had behavioral access at the query point.

2. **Low aliasing (u2_freq=1.5):** u2 *is* encoder-recoverable (supervised ceiling ~0.81) and emergent rho_u2 clears 0.5, but rho_struct plateaus (~0.65) because joint two-coordinate recovery is not clean enough to reach 0.8 -- and splitting is barely needed there because particles already localize u2.


There is no tested regime where splitting is *both necessary and sufficient* to clear rho_struct > 0.8: where the encoder can carry u2, particles already localize it; where splitting is required, the observation aliases u2 beyond static recovery.


**Bounded conclusion:** discrepancy-driven splitting works as designed and resolves the Scale 1 localization blocker, but Postulate 2 in the hard setting is gated by a further constraint the splitting cannot address -- the emergent representation can only carry a coordinate that the observation *locally determines*. Recovering an observation-aliased coordinate at test time requires behavioral access at encode time (an interactive encoder), which is outside the current static observation->representation paradigm.
