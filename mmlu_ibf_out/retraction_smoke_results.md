## Geometry

- sigma_operating: `7.262067926124915`
- kappa_operating: `1.2671588548682307`
- epsilon_global: `0.0005`
- n_eff: `10000`
- d_eff: `32.84418487548828`
- d_shell: `42.10902786254883`

## Small smoke

| stream | baseline | final | delta |
| --- | ---: | ---: | ---: |
| target_original | 0.800 | 0.200 | -0.600 |
| target_new | 0.067 | 0.800 | +0.733 |
| near_neighbor | 0.847 | 0.847 | +0.000 |

- passed: `True`

## Strong smoke

| stream | baseline | final | delta |
| --- | ---: | ---: | ---: |
| target_original | 0.813 | 0.267 | -0.547 |
| target_new | 0.107 | 0.727 | +0.620 |
| near_neighbor | 0.847 | 0.847 | +0.000 |
| distant | 0.860 | 0.860 | +0.000 |

- passed: `True`
- selectivity_vs_near_neighbor: `620000000.00`
- selectivity_vs_distant: `620000000.00`
- estimated_full_runtime_minutes: `94.83`
- estimated_remaining_after_strong_minutes: `88.51`

## Full run policy

- RUN_FULL_RETRACTION: `False`
- FULL_R_EPOCHS: `30`
- status: `skipped_smoke_only`

## Interpretation

### small_smoke

The 5-target smoke moved target_new from 0.067 to 0.800 and target_original from 0.800 to 0.200, with near-neighbor drift 0.000.

### strong_smoke

The 50-target strong smoke moved target_new from 0.107 to 0.727 and target_original from 0.813 to 0.267, with near-neighbor drift 0.000 and distant drift 0.000.

### selectivity

The target-new rise was 0.620, giving selectivity ratios of 620000000.0x versus near-neighbor drift and 620000000.0x versus distant drift.

### durable_alignment_relevance

The smoke results support targeted belief retraction through contradiction: the correction field can overwrite selected beliefs while preserving near-neighbor and distant controls.

### full_round_policy

This cell performs strong smoke testing only unless RUN_FULL_RETRACTION=True. The full 30-epoch run is reserved for the final mega round after all downstream cells are validated.

