# Paper Tables — IBF over LLMs (v2)

*Generated from artifacts under `mmlu_ibf_out/`.*
*Run mode: `paper` · Engine: `2.1-history_gate+xcontext_flag` · Generated: 2026-06-06 03:58 UTC*

---

## Table C1 — Canonical lifecycle training (Layer 1)
| Metric | Value |
|---|---:|
| avg lin (target 0.954 ± 0.01) | **0.940** |
| WITHIN_TOLERANCE | — |
| Value centers | 6,384 |
| Crystallised | 6,384 |
| |v|_max | 0.942 |
*Source: `mmlu_ibf_out/c1_canonical_lifecycle.json`*

## Table C2 — LoRA durability (Layer 2)
| Metric | Value |
|---|---:|
| Base argmax shift rate | 0.283 |
| Weak target drop | +0.300 |
| Strong target drop | +0.067 |
| Off-manifold control delta | +0.000 |
| Status | needs_review |
| WITHIN_TOLERANCE | no |
*Source: `mmlu_ibf_out/c2_lora_durability.json`*

## Table C3 — Cross-model generality (Layer 2)
| Metric | Value |
|---|---:|
| Qwen Phase A survival (after D) | 0.861 |
| Qwen avg phase learning | 0.911 |
| WITHIN_TOLERANCE | yes |
*Source: `mmlu_ibf_out/c3_qwen_cross_model.json`*

## Table C4 — Lifecycle comparison IBF vs kNN vs RAG (Layer 2)
| Dimension | IBF (native) | kNN (oracle) | RAG (oracle) |
|---|---:|---:|---:|
| direct | 1.000 | 1.000 | 0.967 |
| locality | 1.000 | 0.000 | 0.000 |
| revise | 1.000 | 1.000 | 0.967 |
| remove | 1.000 | 1.000 | 1.000 |
| rollback | 1.000 | 1.000 | 0.967 |
| WITHIN_TOLERANCE: yes | | | |
*Source: `mmlu_ibf_out/c4_lifecycle_comparison.json`*

## Table C5 — Lifecycle operations (Layer 3)
| Subsection | Metric | Value |
|---|---|---:|
| 5.1 Retraction | target_new final | 0.993 |
| 5.1 Retraction | WITHIN_TOLERANCE | yes |
| 5.2 Selective deletion | target drop | 0.600 |
| 5.2 Selective deletion | WITHIN_TOLERANCE | yes |
| 5.3 Forgetting | A retention after D | 0.880 |
| 5.3 Forgetting | Dominant boundary | B_to_C_reorg |
| 5.3 Forgetting | WITHIN_TOLERANCE | yes |
*Source: `mmlu_ibf_out/c5_lifecycle_retraction.json`, `mmlu_ibf_out/c5_lifecycle_selective_deletion.json`, `mmlu_ibf_out/c5_lifecycle_forgetting.json`*

## Table C6 — Locality and scale frontier (Layer 3, with bug fix)
| Subsection | Metric | Value |
|---|---|---:|
| 6.1 Locality | NN drift | +0.000 |
| 6.1 Locality | Distant drift | +0.004 |
| 6.1 Locality | WITHIN_TOLERANCE | yes |
| 6.2 Scale | target_acc min | 0.535 |
| 6.2 Scale | A retention min | 0.878 |
| 6.2 Scale | C retention min | 0.987 |
| 6.2 Scale | center growth/rule max | 1.949 |
| 6.2 Scale | bug-fix validated (distinct retentions) | no |
| 6.2 Scale | WITHIN_TOLERANCE | no |
*Source: `mmlu_ibf_out/c6_locality_bleed.json`, `mmlu_ibf_out/c6_scale_frontier.json`*

## Table C7 — Deductive composition: compiled closure (Layer 4)
| Subsection | Metric | Value |
|---|---|---:|
| 7.1 Explicit closure 23B | pass | — |
| 7.1 Explicit closure 23C | pass | — |
| 7.1 WITHIN_TOLERANCE | | no |
| 7.2 Emergent A→C diagnostic (target_acc) | should be ≤ 0.30 (L1 limit) | — |
| 7.2 WITHIN_TOLERANCE | | yes |
| 7.3 Compiled initial arm passes | | no |
| 7.3 Compiled revised arm passes | | no |
| 7.3 WITHIN_TOLERANCE | | no |
*Source: `mmlu_ibf_out/c7_ontology_closure_23bc.json`, `mmlu_ibf_out/c7_graph_closure_diagnostic.json`, `mmlu_ibf_out/c7_compiled_closure.json`*

## Table C8 — Inductive composition: discovery + adjudication (Layer 4)
| Subsection | Metric | Value |
|---|---|---:|
| 8.1 D5b scaffolded | test_AC final | 0.000 |
| 8.1 D5b scaffolded | WITHIN_TOLERANCE | no |
| 8.2 D7 de novo | test_AC final | 0.000 |
| 8.2 D7 de novo | mean BC↔AC σ-units | 5.17 |
| 8.2 D7 de novo | WITHIN_TOLERANCE | no |
| 8.3 D8 adjudication | verdict | needs_review |
| 8.3 D8 adjudication | resilience multiplier (HIGH/LOW) | 1.00 |
| 8.3 D8 adjudication | WITHIN_TOLERANCE | no |
*Source: `mmlu_ibf_out/c8_discovery_d5b.json`, `mmlu_ibf_out/c8_de_novo_d7.json`, `mmlu_ibf_out/c8_crucible_adjudication_d8.json`*


---
## Regime-dependence taxonomy (foundational § 8.1, extended)

| Regime | Agency develops? | Endpoint contribution |
|---|---|---|
| Chess (§ 7.2) | yes | decisive (+8.3 cp) |
| RRW (§ 7.1) | yes | harmful |
| CIFAR (§ 7.3) | no (no trajectory dependence) | neutral |
| **LLM closure (C8)** | **yes (U-shape k_eff)** | **observable but endpoint-redundant in saturated regime** |
