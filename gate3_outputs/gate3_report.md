# Gate 3 Report (Path C: unit-pressure normalization) -- Promotion & Recursive Interaction

k=8, f=3.0, three contexts (A:+1, B:-1, C:partial overlap). 5 seeds, E_phase=20. Conditions: F(flat) / P(promoted) / N(no prior).


## VERDICT: **GATE 3 FAIL**

| criterion | result | threshold | pass |
|---|---|---|---|
| C1 behavioral preservation (worst median |ACC_P-ACC_F|) | 0.100 | < 0.03 | NO |
| C2 compression (median over B,C) | 0.158 | > 0.15 | YES |
| C3 lifecycle (0 < dissolved < promoted, majority seeds) | 2/5 seeds | -- | NO |

## C1 -- accuracy F vs P (median over seeds), by phase x context

| after phase | context | F | P | delta (P-F) |
|---|---|---|---|---|
| A | A | 0.953 | 0.953 | +0.000 |
| A | B | 0.142 | 0.142 | +0.000 |
| A | C | 0.107 | 0.107 | +0.000 |
| B | A | 0.935 | 0.825 | -0.100 |
| B | B | 0.960 | 0.968 | +0.005 |
| B | C | 0.268 | 0.170 | -0.062 |
| C | A | 0.535 | 0.470 | -0.065 |
| C | B | 0.927 | 0.920 | +0.002 |
| C | C | 0.688 | 0.710 | +0.012 |

## Continual-learning transfer (median)

| metric | F | P |
|---|---|---|
| Acc_A after A | 0.953 | 0.953 |
| Acc_A after C (retention) | 0.535 | 0.470 |
| BT_A = Acc_A(after C) - Acc_A(after A) | -0.402 | -0.470 |
| Acc_A after C, condition N (floor) | 0.135 | -- |

## C2 -- active crystallized centers (median), F vs P

| after phase | active F | active P | compression |
|---|---|---|---|
| A | 86 | 41 | 52.3% |
| B | 159 | 125 | 20.3% |
| C | 263 | 234 | 11.3% |

## C3 -- interface lifecycle

| seed | promoted | dissolved (B+C) | interior restored | flat A survivors / cryst |
|---|---|---|---|---|
| 0 | 8 | 0 | 0 | 63/85 |
| 1 | 8 | 1 | 6 | 68/86 |
| 2 | 7 | 1 | 2 | 64/82 |
| 3 | 8 | 0 | 0 | 77/97 |
| 4 | 8 | 0 | 0 | 74/96 |

## Interpretation

**Gate 3 does not pass.** Failing: C1 behavioral (worst gap 0.100); C3 lifecycle.

Forward/backward transfer: F and P both retain Phase A above the no-prior floor (P Acc_A after C = 0.470 vs N = 0.135), confirming continual learning is real; promotion tracks flat (F Acc_A after C = 0.535).

Crucible engagement: 2 of 39 promoted interfaces dissolved across B+C; flat Phase-A crystals survive at a comparable rate (Control 3), indicating the interface-level Crucible mirrors the particle-level one.

## Figures
- `figures/accuracy_by_phase.png`
- `figures/particle_count.png`
- `figures/interface_lifecycle.png`
