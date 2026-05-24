# Cell 26 — Control-Fixed End-to-End Actual LoRA Durability Test

## Status

- **Status:** `clean_actual_lora_e2e_control_fixed`
- **Paper use:** `paper_usable_as_main_base_model_evolution_durability_test`
- **Known caveat:** Single light LoRA condition on a small generic/instruction corpus; not a full robustness suite.

## Core question

Does a fixed post-σ IBF correction field remain effective after the actual base model is modified by LoRA, with no IBF retraining, while genuinely unrelated ordinary controls remain stable?

## Control design

Controls are off-manifold ordinary corporate questions using unrelated generic organizations and domains. They are double-filtered before LoRA: first by `R_base`, then by `R_base + δR`, with margin thresholds where possible.

## Dataset

- Target items: `150000`
- Control candidates: `1600`
- Selected filtered controls: `200`
- Chosen thresholds: `{'base_margin_threshold': 1.0, 'combined_margin_threshold': 1.0}`
- Mean base margin before LoRA: `2.220`
- Mean combined margin before LoRA: `2.220`
- Min base margin before LoRA: `1.062`
- Min combined margin before LoRA: `1.062`

## Main results

| Metric | Before LoRA | After LoRA | Δ / drop |
|---|---:|---:|---:|
| R_base+δR weak target acc | 0.292 | 0.250 | drop +0.042 |
| R_base+δR strong target acc | 0.158 | 0.208 | drop -0.050 |
| Off-manifold control acc, R_base+δR | 1.000 | 1.000 | +0.000 |

## R_base shift

- Target argmax shift rate: `0.322`
- Target mean abs probability delta: `0.032378`
- Control argmax shift rate: `0.000`
- Control mean abs probability delta: `0.037772`

## Fixed δR field

- Centers before LoRA eval: `6382`
- Centers after LoRA eval: `6382`
- Center count unchanged: `True`
- No post-LoRA IBF updates: `True`

## Validation criteria

- `actual_lora_ran`: `True`
- `offmanifold_controls_used`: `True`
- `valid_controls_at_least_200_if_possible`: `True`
- `controls_prefilter_base_before`: `True`
- `controls_prefilter_combined_before`: `True`
- `base_argmax_shift_targets_gt_0_10`: `True`
- `weak_target_drop_le_0_05`: `True`
- `strong_target_drop_le_0_05`: `True`
- `filtered_controls_after_ge_0_95_or_delta_ge_minus_0_02`: `True`
- `center_count_unchanged`: `True`
- `no_post_lora_ibf_updates`: `True`

## Interpretation

This control-fixed end-to-end test modifies the actual base model with a light LoRA adapter, recomputes R_base on the same target and off-manifold filtered-control items, and evaluates the same frozen post-σ δR field without any IBF updates. A clean result supports substrate decoupling: LoRA changes R_base, while the orthogonal δR field remains fixed and continues to enforce local corrections without damaging genuinely unrelated ordinary controls.
