# Cell 36 — Cross-Model Generality: Qwen2-1.5B

## Status

- **Status:** `clean_cross_model_generality_smoke`
- **Paper use:** `paper_usable_as_cross_model_generality_smoke`
- **Known caveat:** Fresh IBF field trained with Qwen R_base. This is cross-model replication, not zero-shot transfer of the same learned δR field.

## Framing

This is a cross-model generality test, not a baseline. A fresh IBF field is trained using Qwen2-1.5B's frozen base distribution. The IBF engine, encoder, Future Industries phase data, and post-σ geometry are kept fixed.

## Configuration

- Model: `Qwen/Qwen2-1.5B`
- Locked σ: `7.2621`
- Expected post-σ: `7.2621`
- Post-σ detected: `True`
- Epochs: `50`
- Centers: `6410`
- Crystallized centers: `6410`
- Dissolutions: `17406`
- Runtime seconds: `75175.6`

## Qwen base accuracy

| Phase | Base accuracy |
|---|---:|
| A_Onboarding | 0.268 |
| B_Initiative | 0.195 |
| C_Reorg | 0.307 |
| D_Turnover | 0.222 |

## Cross-model comparison

| Metric | Mistral canonical | Qwen2-1.5B | Δ Qwen - Mistral |
|---|---:|---:|---:|
| Act 1: Knowledge injection | 0.956 | 0.956 | 0.000 |
| Act 2: Phase A retention | 0.948 | 0.951 | 0.003 |
| Act 2: New facts | 0.983 | 0.740 | -0.243 |
| Act 3: Belief revision | 0.987 | 0.993 | 0.007 |
| Act 4: New hire acquisition | 1.000 | 0.956 | -0.044 |
| Act 4: Phase A survival | 0.850 | 0.863 | 0.013 |

## Criteria

| Criterion | Pass |
|---|---:|
| qwen_base_extracted | ✓ |
| post_sigma_geometry | ✓ |
| qwen_phase_a_learning_ok | ✓ |
| qwen_phase_b_learning_ok | ✓ |
| qwen_phase_c_revision_ok | ✓ |
| qwen_phase_d_learning_ok | ✓ |
| qwen_final_phase_a_survival_ok | ✓ |
| qwen_avg_learning_ok | ✓ |
| field_crystallized | ✓ |

## Interpretation

If the criteria pass, this supports the claim that IBF is not merely a Mistral-specific patch. The same correction dynamics can be trained over a different frozen base distribution. This does not prove zero-shot transfer of a learned δR field; it supports mechanism-level cross-model generality.