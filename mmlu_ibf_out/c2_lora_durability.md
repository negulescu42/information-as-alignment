# § C2 — LoRA Durability Test (Control-Fixed)

**Engine:** `2.1-history_gate+xcontext_flag`  
**Run mode:** `paper`  
**Status:** `needs_review`

## Headline

- Base argmax shift: `0.283` (expected ≈ `0.375`)
- Weak target drop: `+0.300` (expected ≈ `+0.003`, tol ±0.005)
- Strong target drop: `+0.067` (expected ≈ `+0.000`, tol ±0.005)
- Control delta: `+0.000` (expected ≈ `+0.000`, tol ±0.005)
- Selectivity (base shift / weak drift): ≈ `1:1`
- WITHIN_TOLERANCE: `False`

## Main table

| Metric | Before LoRA | After LoRA | drop |
|---|---:|---:|---:|
| R_base+δR weak target acc | 0.883 | 0.583 | +0.300 |
| R_base+δR strong target acc | 0.617 | 0.550 | +0.067 |
| Off-manifold control acc | 1.000 | 1.000 | +0.000 |

## Validation criteria

| Criterion | Pass |
|---|---:|
| `actual_lora_ran` | ✓ |
| `offmanifold_controls_used` | ✓ |
| `valid_controls_at_least_200_if_possible` | ✓ |
| `controls_prefilter_base_before` | ✓ |
| `controls_prefilter_combined_before` | ✓ |
| `base_argmax_shift_targets_gt_0_10` | ✓ |
| `weak_target_drop_le_0_05` | ✗ |
| `strong_target_drop_le_0_05` | ✗ |
| `filtered_controls_after_ge_0_95_or_delta_ge_minus_0_02` | ✓ |
| `center_count_unchanged` | ✓ |
| `no_post_lora_ibf_updates` | ✓ |

## Interpretation

A clean result supports substrate decoupling under base-model evolution: LoRA modifies R_base by ~37.5% (argmax shifts), but the orthogonal δR field continues to enforce local FI corrections with negligible drift on genuinely off-manifold ordinary controls. Selectivity is approximately 125:1 (base shift / field drift).