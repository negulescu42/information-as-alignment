# § 24b-D6 — α vs β agency gate comparison

## Setup

**α (status quo):** Engine's current agency dynamics. `_update_agency` and `delta_k` gate on `is_crystallized()`.

**β (supervisor Reading C):** Patched `_update_agency` and `delta_k` on `eng_beta`. Both gate on `len(c.D_history) ≥ 20` (history sufficiency for variance estimation) instead of crystallization. `eta_k_transient = 0.05`, `eta_k_cryst = 0.005` (analogous to value channel's `eta` / `eta_cryst`).

Both engines forked from canonical, Phase 0 supervised install of AB+BC identical, Phase 1 Boltzmann discovery on AC.

## Final comparison

| group | α pre | α post | α Δ | β pre | β post | β Δ |
|---|---:|---:|---:|---:|---:|---:|
| test_AB | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| test_BC | 0.875 | 1.000 | +0.125 | 0.875 | 1.000 | +0.125 |
| test_AC | 0.000 | 1.000 | +1.000 | 0.000 | 1.000 | +1.000 |

## k_eff at AC queries (the agency-engagement signal)

- **α**: 5.000 → 5.000  (Δ +0.000)
- **β**: 4.661  → 3.228  (Δ -1.433)

## Agency state — final

| | α | β |
|---|---:|---:|
| n_total | 2176 | 2177 |
| n_crystallized | 2151 | 2152 |
| n_history ≥ 20 | 2176 | 2177 |
| w_mean | 3.123 | 3.102 |
| n_history_max | 447 | 402 |

## Emergence speed (first epoch reaching test_AC ≥ 0.9)

- α: epoch 1
- β: epoch 1

## Verdict

- code: `agency_engages_under_history_gate`
- Under β (history-sufficiency gate), k_eff at AC moved 4.66→3.23 (Δ=-1.43), while under α (crystallization gate) it stayed flat (5.00→5.00). Agency channel engages only when freed from the crystallization gate. β gain on test_AC: +1.000; α gain: +1.000.

## Trajectory (eval every 5 epochs)

| ep | α AC_hit | α k̄ | α test_AC | β AC_hit | β k̄ | β test_AC | β agency n≥hist | β w̄ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.188 | 5.00 | 1.000 | 0.250 | 4.66 | 1.000 | 2154 | 3.288 |
| 5 | 1.000 | 5.00 | 1.000 | 1.000 | 4.31 | 1.000 | 2164 | 3.259 |
| 10 | 1.000 | 5.00 | 1.000 | 1.000 | 3.73 | 1.000 | 2166 | 3.231 |
| 15 | 1.000 | 5.00 | 1.000 | 1.000 | 3.36 | 1.000 | 2169 | 3.204 |
| 20 | 1.000 | 5.00 | 1.000 | 1.000 | 2.94 | 1.000 | 2172 | 3.180 |
| 25 | 1.000 | 5.00 | 1.000 | 1.000 | 2.76 | 1.000 | 2176 | 3.152 |
| 30 | 1.000 | 5.00 | 1.000 | 1.000 | 2.58 | 1.000 | 2177 | 3.133 |
| 35 | 1.000 | 5.00 | 1.000 | 1.000 | 2.93 | 1.000 | 2178 | 3.116 |
| 40 | 1.000 | 5.00 | 1.000 | 1.000 | 3.23 | 1.000 | 2177 | 3.102 |
