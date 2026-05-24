## FI locality / bleed test — geometric controls V3

## Run

- mode: `strong_smoke`
- target rules: `60`
- near rules: `60`
- distant rules: `60`
- max epochs: `2`
- sigma: `7.262067926124915`
- strong repeats: `3`
- strong push multiplier: `1.8`
- close each epoch: `True`

## Geometric audit

| population | median distance | median σ units | p10 | p90 |
| --- | ---: | ---: | ---: | ---: |
| near | 33.648 | 4.633 | 29.930 | 35.898 |
| distant | 44.123 | 6.076 | 40.876 | 47.929 |

## Results

| metric | before | after | delta |
| --- | ---: | ---: | ---: |
| target FI selected | 0.000 | 0.967 | +0.967 |
| target base selected | 1.000 | 0.033 | -0.967 |
| near control correct | 1.000 | 1.000 | +0.000 |
| distant control correct | 1.000 | 1.000 | +0.000 |
| A retention lin | 0.850 | 0.850 | +0.000 |
| C retention lin | 0.987 | 0.987 | +0.000 |
| near mean abs margin drift | 0.000 | 0.001 | +0.001 |
| distant mean abs margin drift | 0.000 | 0.000 | +0.000 |

## Criteria

- target_success: `True`
- target_base_suppressed: `True`
- near_control_stable: `True`
- distant_control_stable: `True`
- near_margin_bounded: `True`
- distant_margin_bounded: `True`
- A_retention_stable: `True`
- C_retention_stable: `True`

## Interpretation

### locality_claim

The test applies corrections to target rules and measures whether those corrections alter geometrically nearby but unrelated rules or distant controls.

### geometric_near_controls

Near controls were selected by actual 80D proposition-space distance to target FI propositions. The selected near median distance was 4.63σ; the selected distant median distance was 6.08σ.

### bleed_result

Target FI selection changed from 0.000 to 0.967; near-control accuracy changed by +0.000; distant-control accuracy changed by +0.000.

### safety_relevance

This is the locality counterpart of local alignment: the field can correct selected propositions without uncontrolled spread into nearby or distant propositions.

