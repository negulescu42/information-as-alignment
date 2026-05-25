# § 24b-D7 — De novo A→C emergence (no BC scaffold)

## Setup

D5b/D6 used AC items where the C-answer text was identical to BC's C-answer.
BC install put value centers at z_choice[C], and AC test queries the same
position → kernel scaffold lowered Boltzmann barrier.

**D7 removes the scaffold:** AC items use `C_for_AC` field — lexically
distinct (semantically equivalent) text. BC's trained value centers no
longer fire at AC test items' C-choice position.

**Scaffold-distance diagnostic:** mean BC-C ↔ AC-C distance = 8.09·σ_op

| chain | distance | σ_op multiples | scaffold |
|---|---:|---:|---|
| chain_approval | 57.323 | 7.894 | BROKEN (far) |
| chain_restricted_exposure | 69.030 | 9.506 | BROKEN (far) |
| chain_credential_exposure | 47.629 | 6.559 | BROKEN (far) |
| chain_change_control | 53.676 | 7.391 | BROKEN (far) |
| chain_patient_discharge | 61.640 | 8.488 | BROKEN (far) |
| chain_contract_transfer | 63.431 | 8.735 | BROKEN (far) |
| chain_claim_exclusion | 67.846 | 9.342 | BROKEN (far) |
| chain_data_access | 49.326 | 6.792 | BROKEN (far) |

## Final comparison

| group | α pre | α post | β pre | β post |
|---|---:|---:|---:|---:|
| test_AB | 1.000 | 1.000 | 1.000 | 1.000 |
| test_BC | 0.875 | 1.000 | 0.875 | 1.000 |
| test_AC_de_novo | 0.000 | 1.000 | 0.000 | 1.000 |

## k_eff at AC trajectory

- α: [5.000, 5.000]
- β: [2.483, 4.626]

## β vs α gap on test_AC_de_novo: +0.000

## Verdict

- code: `de_novo_emergence_both`
- De novo emergence works under both α and β (α=1.000, β=1.000). Discovery training is robust enough to handle absent scaffolds even without agency modulation. Discovery as a mechanism is more generic than feared.

## Trajectory (eval every 5 epochs)

| ep | α AC_hit | α k̄ | α test_AC | β AC_hit | β k̄ | β test_AC | β w̄ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.542 | 5.00 | 1.000 | 0.438 | 4.63 | 1.000 | 3.288 |
| 5 | 1.000 | 5.00 | 1.000 | 1.000 | 4.11 | 1.000 | 3.257 |
| 10 | 1.000 | 5.00 | 1.000 | 1.000 | 3.17 | 1.000 | 3.227 |
| 15 | 1.000 | 5.00 | 1.000 | 1.000 | 2.90 | 1.000 | 3.205 |
| 20 | 1.000 | 5.00 | 1.000 | 1.000 | 2.54 | 1.000 | 3.184 |
| 25 | 1.000 | 5.00 | 1.000 | 1.000 | 2.62 | 1.000 | 3.171 |
| 30 | 1.000 | 5.00 | 1.000 | 1.000 | 3.01 | 1.000 | 3.155 |
| 35 | 1.000 | 5.00 | 1.000 | 1.000 | 3.10 | 1.000 | 3.139 |
| 40 | 1.000 | 5.00 | 1.000 | 1.000 | 3.49 | 1.000 | 3.120 |
| 45 | 1.000 | 5.00 | 1.000 | 1.000 | 3.29 | 1.000 | 3.105 |
| 50 | 1.000 | 5.00 | 1.000 | 1.000 | 3.49 | 1.000 | 3.091 |
| 55 | 1.000 | 5.00 | 1.000 | 1.000 | 3.72 | 1.000 | 3.075 |
| 60 | 1.000 | 5.00 | 1.000 | 1.000 | 3.77 | 1.000 | 3.062 |
