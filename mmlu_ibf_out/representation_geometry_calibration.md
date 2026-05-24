# Cell 5b — representation + operating geometry V6

This cell derives the IBF correction-field operating bandwidth from active geometric constraints.

Runtime uses only one sigma: `σ_operating`.

## Core law

```text
K(d, σ) = exp(-d² / 2σ²)

σ_operating = max σ such that:
  K(d_pair, σ) <= ε_pair
  N_eff · K(d_shell, σ) <= ε_global
```

## Summary

- d_eff: `32.844185`
- d_pair: `42.109028`
- d_shell: `42.109028`
- pairwise locality bound: `σ <= 11.329004`
- aggregate field bound: `σ <= 7.262068`
- selected operating σ: `7.262068`
- selected operating κ: `1.267159`
- active constraint: `aggregate field`
- ε_global: `0.0005`
- N_eff: `10000`
- operating merge: `10.893102`
- operating agency σ: `4.836129`
- density bound `N_eff*K(d_shell)`: `0.00050000`

## Candidate geometry table

| kind | label | ε | σ | κ | K_shell | N_eff*K | tail_p95 | K_positive | safe | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sigma_multiplier | 0.500x_pair_bound |  | 5.6645 | 0.9884 | 1.00e-12 | 1.00e-08 | 9.0000 | 1.0000 | ✓ | ✗ |
| sigma_multiplier | 0.625x_pair_bound |  | 7.0806 | 1.2355 | 2.09e-08 | 0.0002 | 9.0046 | 1.0000 | ✓ | ✗ |
| sigma_multiplier | 0.700x_pair_bound |  | 7.9303 | 1.3838 | 7.54e-07 | 0.0075 | 9.0316 | 1.0000 | ✗ | ✗ |
| sigma_multiplier | 0.750x_pair_bound |  | 8.4968 | 1.4826 | 4.64e-06 | 0.0464 | 9.0633 | 1.0000 | ✗ | ✗ |
| sigma_multiplier | 0.800x_pair_bound |  | 9.0632 | 1.5814 | 2.05e-05 | 0.2054 | 9.1525 | 1.0000 | ✗ | ✗ |
| sigma_multiplier | 0.875x_pair_bound |  | 9.9129 | 1.7297 | 0.0001 | 1.2068 | 9.2095 | 1.0000 | ✗ | ✗ |
| sigma_multiplier | 1.000x_pair_bound |  | 11.3290 | 1.9768 | 0.0010 | 10.0000 | 10.0956 | 1.0000 | ✗ | ✗ |
| sigma_multiplier | 1.125x_pair_bound |  | 12.7451 | 2.2239 | 0.0043 | 42.6216 | 11.0933 | 1.0000 | ✗ | ✗ |
| sigma_multiplier | 1.250x_pair_bound |  | 14.1613 | 2.4710 | 0.0120 | 120.2264 | 13.8490 | 1.0000 | ✗ | ✗ |
| epsilon_budget | eps_0.0005 | 0.0005 | 7.2621 | 1.2672 | 5.00e-08 | 0.0005 | 9.0069 | 1.0000 | ✗ | ✓ |
| epsilon_budget | eps_0.001 | 0.0010 | 7.4166 | 1.2941 | 1.00e-07 | 0.0010 | 9.0124 | 1.0000 | ✗ | ✗ |
| epsilon_budget | eps_0.0025 | 0.0025 | 7.6368 | 1.3325 | 2.50e-07 | 0.0025 | 9.0196 | 1.0000 | ✗ | ✗ |
| epsilon_budget | eps_0.005 | 0.0050 | 7.8171 | 1.3640 | 5.00e-07 | 0.0050 | 9.0195 | 1.0000 | ✗ | ✗ |
| epsilon_budget | eps_0.01 | 0.0100 | 8.0108 | 1.3978 | 1.00e-06 | 0.0100 | 9.0237 | 1.0000 | ✗ | ✗ |
| epsilon_budget | eps_0.025 | 0.0250 | 8.2905 | 1.4466 | 2.50e-06 | 0.0250 | 9.1110 | 1.0000 | ✗ | ✗ |
| epsilon_budget | eps_0.05 | 0.0500 | 8.5226 | 1.4871 | 5.00e-06 | 0.0500 | 9.1424 | 1.0000 | ✗ | ✗ |
