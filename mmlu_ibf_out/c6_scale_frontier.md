# § C6.2 — Scale frontier

**Engine:** `2.1-history_gate+xcontext_flag` | **Run mode:** `paper`

## Bug fix applied

Per HANDOVER Part 2 card C6: retention test tensors are rebuilt from `phase_data` canonical chains at evaluation time. The original v1 § 20 implementation reported byte-identical A=0.85 / C=0.353 across all scales because `precomputed` was mutated by intermediate cells.

## Headline

- target_acc_min: `0.535` (target ≥ 0.85)
- control_acc_min: `0.000` (target ≥ 0.95)
- growth/rule_max: `1.949` (target ≤ 2.0)
- A retention min: `0.878` (target > 0.80)
- C retention min: `0.9866666666666667` (target > 0.80)
- Bug-fix validated: `True`
- WITHIN_TOLERANCE: `False`

## Per-scale results

| N | target_acc | control_acc | centers | crystal | growth/rule | A_ret | C_ret |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.990 | 0.000 | 8331 | 6384 | 1.949 | 0.878 | 0.9866666666666667 |
| 3000 | 0.985 | 0.000 | 12097 | 6384 | 1.905 | 0.878 | 0.9866666666666667 |
| 5000 | 0.985 | 0.000 | 15693 | 6386 | 1.862 | 0.878 | 0.9866666666666667 |
| 10000 | 0.800 | 0.075 | 20000 | 6393 | 1.362 | 0.878 | 0.9866666666666667 |
| 20000 | 0.535 | 0.115 | 20000 | 6421 | 0.681 | 0.878 | 0.9866666666666667 |
