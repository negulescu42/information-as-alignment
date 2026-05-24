## Geometry

- sigma_operating: `7.262067926124915`
- kappa_operating: `1.2671588548682307`
- epsilon_global: `0.0005`
- n_eff: `10000`
- d_eff: `32.84418487548828`
- d_shell: `42.10902786254883`

## Overall

- n: `1640`
- acc_lin: `0.9024`
- acc_log: `0.6762`
- margin_lin_mean: `0.6273`
- margin_lin_median: `0.3965`

## Classification by margin

| margin | consistent | ambiguous | wrong |
| --- | --- | --- | --- |
| 0.01 | 1472 | 8 | 160 |
| 0.05 | 1418 | 62 | 160 |
| 0.1 | 1342 | 138 | 160 |

## By phase

| phase | n | acc_lin | acc_log | margin_lin_mean | consistent@0.05 | ambiguous@0.05 | wrong |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_Onboarding | 1000 | 0.8500 | 0.5730 | 0.2863 | 791 | 59 | 150 |
| B_Initiative | 400 | 0.9800 | 0.7750 | 0.5787 | 389 | 3 | 8 |
| C_Reorg | 150 | 0.9867 | 0.9867 | 2.7444 | 148 | 0 | 2 |
| D_Turnover | 90 | 1.0000 | 0.8667 | 1.1041 | 90 | 0 | 0 |

## By category

| category | n | acc_lin | margin_lin_mean | consistent@0.05 | ambiguous@0.05 | wrong |
| --- | --- | --- | --- | --- | --- | --- |
| certification | 200 | 0.9650 | 0.5188 | 191 | 2 | 7 |
| committee | 200 | 0.9950 | 0.6385 | 198 | 1 | 1 |
| location | 200 | 0.9500 | 0.2888 | 185 | 5 | 10 |
| manager | 330 | 0.8333 | 1.0779 | 251 | 24 | 55 |
| mentor | 200 | 0.7900 | 0.2907 | 143 | 15 | 42 |
| project | 280 | 0.9214 | 0.9189 | 252 | 6 | 22 |
| team | 230 | 0.9000 | 0.2976 | 198 | 9 | 23 |

## Sanity check

- passed: `True`
- max_delta: `0.000000`

## Interpretation

### read_only_belief_state

This audit enumerates the final belief state of the frozen-model + IBF field without writing new centers or modifying the engine.

### consistency

At margin 0.05, 1418/1640 items are consistent, 62 ambiguous, and 160 wrong under the primary linear readout.

### margin_stability

The mean linear margin is 0.6273 and the median linear margin is 0.3965. Higher margins indicate that correct beliefs are not merely argmax-correct, but separated from alternatives.

### durable_alignment_relevance

The audit tests whether the correction field has a coherent post-lifecycle belief state. High consistency with low ambiguity supports the interpretation that the field retains and updates knowledge in a stable, inspectable way over the frozen model.

### geometry_context

The audit is performed under operating σ=7.2621, κ=1.2672, ε_global=0.0005, N_eff=10000.

