# § C3 — Cross-Model Generality: Qwen2-1.5B

**Engine:** `2.1-history_gate+xcontext_flag`  
**Run mode:** `paper`  
**Status:** `clean_cross_model_generality_smoke`

## Headline

- Metrics within ±0.05: `5` / `6`
- Phase A survival (Qwen, after D_Turnover): `0.861`
- Avg phase learning: `0.911`
- WITHIN_TOLERANCE: `True`

## Configuration

- Model: `Qwen/Qwen2-1.5B`
- Locked σ: `7.2621`
- Epochs: `50`
- Centers: `6409` (crystallised `6409`)
- Dissolutions: `15742`
- Runtime: `75169.3` s

## Qwen base accuracy

| Phase | Base accuracy |
|---|---:|
| A_Onboarding | 0.268 |
| B_Initiative | 0.195 |
| C_Reorg | 0.307 |
| D_Turnover | 0.222 |

## Cross-model comparison

| Metric | Mistral canonical | Qwen2-1.5B | Δ Qwen − Mistral |
|---|---:|---:|---:|
| Act 1: Knowledge injection | 0.957 | 0.957 | 0.000 |
| Act 2: Phase A retention | 0.954 | 0.952 | -0.002 |
| Act 2: New facts | 0.902 | 0.738 | -0.165 |
| Act 3: Belief revision | 0.987 | 0.993 | 0.007 |
| Act 4: New hire acquisition | 1.000 | 0.956 | -0.044 |
| Act 4: Phase A survival | 0.880 | 0.861 | -0.019 |

## Criteria

| Criterion | Pass |
|---|---:|
| `qwen_base_extracted` | ✓ |
| `post_sigma_geometry` | ✓ |
| `qwen_phase_a_learning_ok` | ✓ |
| `qwen_phase_b_learning_ok` | ✓ |
| `qwen_phase_c_revision_ok` | ✓ |
| `qwen_phase_d_learning_ok` | ✓ |
| `qwen_final_phase_a_survival_ok` | ✓ |
| `qwen_avg_learning_ok` | ✓ |
| `field_crystallized` | ✓ |

## Interpretation

Same IBF engine, same encoder, same post-σ geometry, different frozen base model. A clean run supports mechanism-level cross-model generality of the IBF correction field — Discrete Convergence (foundational Claim 5) holds across encoders.