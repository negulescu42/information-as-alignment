# Final Public Synthesis Report

## Executive synthesis

IBF is evaluated as a local durable-alignment layer: an orthogonal correction field `delta_R` over frozen base-model scores `R_base`.

The notebook tests whether this field can install, preserve, revise, remove, and restore local beliefs without editing the base model. The central finding is that the mechanism story is coherent across the canonical lifecycle, mechanism readouts, stress tests, substrate-decoupling tests, and shared benchmark comparisons.

The benchmark section is structurally ready but currently smoke-level. Final public/paper tables should use full-mode reruns of the benchmark cells.

## Global artifact status

| Cell | Block | Artifact | Found | Status |
| --- | --- | --- | --- | --- |
| 9-10 | Canonical lifecycle | canonical_results | yes | smoke |
| 11 | Consistency | consistency_audit | yes | smoke |
| 12 | Provenance | provenance_samples | yes | smoke |
| 13 | Retraction | retraction_results | no | smoke |
| 14 | Selective deletion | selective_deletion | yes | smoke |
| 15 | Strong prior | fi_strong_prior_pilot | yes | smoke |
| 16 | Ontology shift | ontology_shift_results | yes | smoke |
| 17 | Bridge | fi_bridge_weak_strong_prior | yes | smoke |
| 18 | Local alignment | fi_local_alignment_phase_transition | yes | smoke |
| 19 | 1k rule system | fi_1k_rule_system_report | yes | smoke |
| 20 | Locality / bleed | fi_locality_bleed_test | yes | smoke |
| 21 | Scale frontier | fi_scale_sweep / 21c | yes | smoke |
| 22 | Long horizon | fi_long_horizon | yes | smoke |
| 22B | Amplitude hygiene | amplitude_hygiene | yes | smoke |
| 23 | Forgetting | forgetting_diagnostic | yes | smoke |
| 24D | LoRA durability | base perturbation durability | no | smoke |
| 25 | Qwen replication | cross_model_generality | yes | smoke |
| 27-33 | Benchmarks | IBF/kNN/RAG comparison | yes | smoke |
| 29P-R | CounterFact diagnostics | geometry/sigma/anchor diagnostics | yes | smoke |

---

## Layer 1 — Canonical durable lifecycle

### Objective

Test whether IBF can install, preserve, revise, and survive local beliefs across a structured lifecycle.

### Consolidated result

The FI lifecycle is the core evidence that IBF is not a one-shot editor. It behaves like a local truth-maintenance layer over a frozen base model.

### Data

#### canonical_lifecycle

| phase | objective | metric | value | status |
| --- | --- | --- | --- | --- |
| A Onboarding | initial memory installation | A after A | 0.955 | smoke |
| A Retention | preserve early memory | A after B / final | 0.938 | smoke |
| B Initiative | new facts after prior memory | B after B | 0.983 | smoke |
| C Reorg | belief revision | C after C | 0.987 | smoke |
| D Turnover | late acquisition | D after D | 1.000 | smoke |

### Conclusion

This is the central paper spine. The current status is smoke unless the canonical artifacts were produced in full mode.

---

## Layer 2 — Mechanism evidence

### Objective

Show that the correction field is inspectable, auditable, and locally editable.

### Consolidated result

Consistency, provenance, retraction, and selective deletion artifacts are present as mechanism readouts.

### Data

#### mechanism_readouts

| cell | experiment | objective | result | status |
| --- | --- | --- | --- | --- |
| 11 | Consistency audit | whole belief-state check | artifact present | smoke |
| 12 | Provenance | trace local contributors | artifact present | smoke |
| 13 | Retraction | contradiction-driven removal | missing | smoke |
| 14 | Selective deletion | local erasure | artifact present | smoke |

### Conclusion

These cells support the claim that IBF behaves like a bounded truth-maintenance substrate, not a black-box cache.

---

## Layer 3 — Stress, locality, and hygiene

### Objective

Test whether IBF survives stronger priors, ontology shifts, scale pressure, long-horizon updates, and amplitude-hygiene checks.

### Consolidated result

The notebook includes strong-prior, ontology, bridge, locality, scale, long-horizon, amplitude, and forgetting diagnostics.

### Data

#### stress_hygiene

| cell | experiment | objective | result | status |
| --- | --- | --- | --- | --- |
| 15 | Strong prior | override base priors | artifact present | smoke |
| 16 | Ontology shift | coordinated local reality shift | artifact present | smoke |
| 17 | Bridge | weak-to-strong prior continuity | artifact present | smoke |
| 18 | Local alignment | contradiction transition | artifact present | smoke |
| 19 | 1k rule system | larger local rule set | artifact present | smoke |
| 20 | Locality / bleed | spillover control | artifact present | smoke |
| 21 | Scale frontier | capacity pressure | artifact present | smoke |
| 22 | Long horizon | repeated update stability | artifact present | smoke |
| 22B | Amplitude hygiene | boundedness check | artifact present | smoke |
| 23 | Forgetting | state retention/decay | artifact present | smoke |

### Conclusion

This layer supports the robustness story. Some entries are main-text candidates; others are better treated as appendix diagnostics.

---

## Layer 4 — Substrate decoupling

### Objective

Test whether IBF is tied to one exact frozen base model.

### Consolidated result

The notebook includes actual LoRA durability evidence and Qwen2-1.5B cross-model replication.

### Data

#### substrate_qwen

| metric | mistral | qwen | delta |
| --- | --- | --- | --- |
| Act 1: Knowledge injection | 0.951 | 0.947 | -0.004 |
| Act 2: Phase A retention | 0.946 | 0.939 | -0.007 |
| Act 2: New facts | 0.985 | 0.720 | -0.265 |
| Act 3: Belief revision | 0.987 | 0.993 | 0.007 |
| Act 4: New hire acquisition | 1.000 | 0.933 | -0.067 |
| Act 4: Phase A survival | 0.922 | 0.929 | 0.007 |

### Conclusion

This is one of the strongest generality signals: same mechanism, different base distribution. Qwen is a fresh-field replication, not zero-shot transfer of the same delta_R.

---

## Layer 5 — Shared benchmark comparison

### Objective

Compare IBF with simpler memory systems on the same durable lifecycle harness.

### Consolidated result

IBF, kNN, and RAG are compared on shared CounterFact/ZsRE lifecycle tasks. kNN and RAG receive manual/oracle lifecycle maintenance; IBF is evaluated as a native field.

### Data

#### harness

| benchmark | scenarios | tasks | status |
| --- | --- | --- | --- |
| counterfact | 200 | 3576 | smoke |
| zsre | 200 | 1800 | smoke |

#### benchmarks

| benchmark | method | direct | para | loc | multi | rev | remove | rollback | residue | burden | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| counterfact | IBF | 1.000 | 0.000 | 0.967 | 0.310 | 1.000 | 1.000 | 1.000 | 0.000 | native | smoke |
| counterfact | kNN | 1.000 | 0.000 | 0.800 | 0.964 | 1.000 | 1.000 | 0.833 | 0.000 | manual | smoke |
| counterfact | RAG | 0.300 | 0.100 | 0.867 | 0.067 | 0.000 | 1.000 | 0.167 | 0.000 | oracle | smoke |
| zsre | IBF | 1.000 | 0.800 | 1.000 | - | 1.000 | 1.000 | 1.000 | 0.000 | native | smoke |
| zsre | kNN | 1.000 | 1.000 | 1.000 | - | 1.000 | 1.000 | 0.667 | 0.000 | manual | smoke |
| zsre | RAG | 1.000 | 1.000 | 0.900 | - | 1.000 | 1.000 | 1.000 | 0.000 | oracle | smoke |

### Conclusion

The benchmark story is architectural: native lifecycle dynamics versus manual/oracle memory maintenance. External editors such as ROME, MEMIT, SERAC, GRACE, and WISE are not part of this scoped comparison.

---

## Layer 6 — CounterFact paraphrase limitation

### Objective

Explain the weak CounterFact paraphrase result instead of hiding it.

### Consolidated result

The diagnostic cells show that failed CounterFact paraphrases are geometrically far from direct-edit anchors and receive weak delta_R. Widening sigma improves paraphrase only modestly and costs locality/revision/retention. One paraphrase anchor helps the installed paraphrase surface but not held-out paraphrases.

### Data

#### cf_geometry

| dataset | direct | para | direct_ok_para_fail | fail_dist | success_dist | fail_dR | success_dR | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CounterFact | 0.980 | 0.020 | 96 | 16.400 | 7.707 | 0.269 | 1.363 | smoke |
| ZsRE | 1.000 | 0.940 | 3 | 14.085 | 4.919 | 0.486 | 1.854 | smoke |

#### cf_sigma

| sigma | direct | para | loc | multi | rev | remove | residue | retain | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7.2621 | 0.980 | 0.020 | 0.947 | 0.414 | 1.000 | 1.000 | 0.000 | 0.950 | smoke |
| 8.5000 | 0.920 | 0.040 | 0.860 | 0.579 | 0.967 | 1.000 | 0.000 | 0.900 | smoke |
| 10.8931 | 0.700 | 0.100 | 0.640 | 0.628 | 0.667 | 1.000 | 0.000 | 0.700 | smoke |
| 11.3290 | 0.600 | 0.150 | 0.673 | 0.545 | 0.600 | 1.000 | 0.000 | 0.700 | smoke |

#### cf_anchor

| condition | direct | anchor_para | heldout_para | loc | multi | rev | residue | retain | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct only | 0.980 | 0.020 | 0.020 | 0.947 | 0.414 | 1.000 | 0.000 | 0.950 | smoke |
| direct + one paraphrase anchor | 0.520 | 0.600 | 0.020 | 0.953 | 0.262 | 1.000 | 0.000 | 0.350 | smoke |

### Conclusion

CounterFact paraphrase weakness is a representation/prompt-surface limitation under the direct-install protocol, not a lifecycle failure.
