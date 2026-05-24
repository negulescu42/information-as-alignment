## Geometry

- σ_operating: `7.2621`
- κ_operating: `1.2672`
- ε_global: `0.000500`
- N_eff: `10000`
- d_eff: `32.8442`
- d_shell: `42.1090`
- σ_pairwise_bound: `11.3290`
- σ_field_bound: `7.2621`
- N_eff·K(d_shell): `0.00050000`

## Performance

- avg_log: `0.8003`
- avg_lin: `0.9542`
- Δ avg_log vs base: `0.5839`
- Δ avg_lin vs base: `0.7378`
- A_after_D_lin: `0.8500`
- B_after_D_lin: `0.9800`
- C_after_D_lin: `0.9867`
- D_after_D_lin: `1.0000`

## Phase table

| Phase | Base log | IBF log | Δlog | Base lin | IBF lin | Δlin |
| --- | --- | --- | --- | --- | --- | --- |
| A_Onboarding | 0.2500 | 0.5730 | 0.3230 | 0.2500 | 0.8500 | 0.6000 |
| B_Initiative | 0.2200 | 0.7750 | 0.5550 | 0.2200 | 0.9800 | 0.7600 |
| C_Reorg | 0.2400 | 0.9867 | 0.7467 | 0.2400 | 0.9867 | 0.7467 |
| D_Turnover | 0.1556 | 0.8667 | 0.7111 | 0.1556 | 1.0000 | 0.8444 |

## Engine health

- value centers: `6382`
- crystallized: `6382`
- crystallization rate: `1.0000`
- agency centers: `2139`
- dissolutions: `18786`
- dissolutions / center: `2.9436`
- |v| mean: `0.2964`
- |v| max: `2.8146`
- runtime minutes: `2107.1724`

## Interpretation

### learning

The field installs the Future Industries lifecycle strongly from a weak frozen-model baseline. Average linear accuracy reaches 0.9542 from a base average of 0.2164, a gain of 0.7378.

### retention

After all lifecycle phases, early knowledge remains highly available: A_after_D_lin=0.8500 and B_after_D_lin=0.9800. This indicates that later reorganization and turnover phases did not erase the earlier installed structure.

### adaptation

The reorganization and turnover phases are absorbed cleanly: C_after_D_lin=0.9867 and D_after_D_lin=1.0000. This supports the role of localized discrepancy-driven updates in adapting the correction field without retraining the frozen model.

### field_health

The engine ends with 6382 value centers, 6382 crystallized centers, and a crystallization rate of 1.0000. Dissolutions per value center are 2.9436, with |v|max=2.8146. This suggests a stable, granular correction field rather than amplitude-driven global forcing.

### geometry

The operating geometry uses σ=7.2621 and κ=1.2672 under ε_global=0.000500 and N_eff=10000. Pairwise locality is treated as an upper bound, while aggregate field pressure determines the operating resolution.

### core_takeaway

The canonical run supports the current IBF durable-alignment hypothesis: durable correction is not only a memory problem, but a field-resolution problem. The frozen model remains unchanged; alignment is expressed through a bounded, localized correction field.

