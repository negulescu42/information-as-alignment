# Reproducibility Appendix

**Engine:** `2.1-history_gate+xcontext_flag` (Reading C agency patch)

**Run mode:** `paper` · **Run ID:** `20260603T164908Z` · **SEED:** `42`

**Branch:** `unknown` · **Commit:** `unknown`

**Hardware:** NVIDIA GeForce RTX 5090 (33.7 GB)

**Notebook:** `(IBF)Companion-LLM-Durable-Alignment-v2.ipynb`

**Base model:** `mistralai/Mistral-7B-v0.1` · **Cross-model:** `Qwen/Qwen2-1.5B` · **Encoder:** `sentence-transformers/all-mpnet-base-v2`


## Per-claim seed offsets

| Claim | Seed offset | Effective seed |
|---|---:|---:|
| C1 | 100 | 142 |
| C2 | 200 | 242 |
| C3 | 300 | 342 |
| C4 | 400 | 442 |
| C5 | 500 | 542 |
| C6 | 600 | 642 |
| C7 | 700 | 742 |
| C8 | 800 | 842 |

## Foundational anchoring (HANDOVER Part 1.5)

| Companion claim | Foundational concept | Falsifier (one line) |
|---|---|---|
| **C1** (Local durable alignment without weight editing) | Foundational Claim 1 (Memory) | Crystallised modifications decay during dormant epochs despite the stability transition, OR persist structurally but yield no measurable behavioural retention. |
| **C2** (Substrate decoupling under base evolution) | Postulate 1 constraint (iii) — D ⊥ motion | 30%+ base perturbation degrades field accuracy by >10%, OR field drift > 5% under any base modification. |
| **C3** (Cross-model mechanism generality) | Foundational Claim 5 (Discrete Convergence) | Mechanism fails on a second base model, OR cross-model behaviour differs qualitatively. |
| **C4** (Distinct from kNN / RAG (architecturally)) | Foundational § 8.2 "Relation to existing frameworks" | Oracle-maintained kNN or RAG matches IBF on every lifecycle dimension with comparable operational burden. |
| **C5** (Truth-maintenance lifecycle) | Foundational Claims 1 + 4 (Memory + Self-Correction) | Retraction doesn't remove; revision creates phantom side effects; rollback doesn't restore. |
| **C6** (Locality preservation under operations) | Postulate 1's localisation kernel K(y, x_S) | NN drift > 0.05 under sustained operations; OR scale-frontier C-retention bug reproduces. |
| **C7** (Compiled semantic structure (deductive composition)) | Foundational § 7.2 chess compiled regularities → ontology graph | Compiled consequences don't survive across queries; revision fails to update derived edges; closure rules interact destructively. |
| **C8** (Discovery-driven extension (inductive composition)) | Foundational Claims 2 + 3 (Agency + Intelligence) + § 8.1 fourth regime | Emergence requires kernel scaffold (D7 falsified); Crucible adjudication fails to respond to resilience knob (D8 falsified). |

## Major artifacts (cN + legacy alias)

| Claim | Artifacts |
|---|---|
| **C1** | `c1_canonical_lifecycle.json`, `canonical_training_results.json (alias)`, `canonical_engine.pkl`, `canonical_metrics.pkl` |
| **C2** | `c2_lora_durability.json`, `actual_lora_e2e_durability_control_fixed_report.json (alias)` |
| **C3** | `c3_qwen_cross_model.json`, `cross_model_generality_qwen2_1_5b.json (alias)`, `cross_model_generality.json (compat)`, `phi3_generality.json (compat)` |
| **C4** | `c4_lifecycle_comparison.json`, `benchmark_ibf_lifecycle.json`, `benchmark_knn_lifecycle.json`, `benchmark_rag_lifecycle.json`, `benchmark_comparison.md` |
| **C5** | `c5_lifecycle_retraction.json`, `c5_lifecycle_selective_deletion.json`, `c5_lifecycle_forgetting.json`, `retraction_full_results.json (alias)`, `selective_deletion.json (alias)`, `forgetting_diagnostic_report.json (alias)` |
| **C6** | `c6_locality_bleed.json`, `c6_scale_frontier.json`, `fi_locality_bleed_test.json (alias)`, `fi_scale_capacity_frontier.json (alias)` |
| **C7** | `c7_ontology_closure_23bc.json`, `c7_graph_closure_diagnostic.json`, `c7_compiled_closure.json`, `fi_ontology_closure_23bc.json (alias)`, `fi_local_ontology_graph_closure_cell24.json (alias)`, `fi_compiled_ontology_closure_cell24b.json (alias)` |
| **C8** | `c8_discovery_d5b.json`, `c8_de_novo_d7.json`, `c8_crucible_adjudication_d8.json`, `fi_agency_channel_d5b_discovery.json (alias)`, `fi_agency_channel_d7_de_novo.json (alias)`, `fi_agency_channel_d8_conflict_adjudication.json (alias)` |
| **S4** | `paper/paper_tables.md`, `paper/abstract_numbers.json`, `paper/claims_status_final.md` |
| **S5** | `paper/reproducibility_manifest.json`, `paper/reproducibility_appendix.md` |

## Deferred from v2 (HANDOVER Part 5)

### Diagnostics
`κ/σ diagnostics (§9 family)`, `amplitude hygiene (§22B)`, `anchor experiments (paraphrase audit subcells)`, `mechanism continuation (§35, §37)`

### Eliminated alternatives
`§24b-D1 (kernel locality diagnostic — informed D5b design)`, `§24b-D2/D3/D4 (static agency wirings, c.z[:64] proxy, z_before storage)`, `§24b-D5 (original buggy version — superseded by D5b)`, `§24b-D6 (α vs β engine-fix validation — moved to engine-patch documentation)`

### Redundant variants
`§15 Strong prior pilot`, `§15b Strong absurdities`, `§16 Bridge experiment`, `§17 Local alignment phase transition`, `§18 Local alignment system report`


## Limitations

| Code | Description |
|---|---|
| **L1** | Compiled closure (C7) is required for transitive A→C — emergent closure does NOT arise automatically; the C8 discovery path complements C7 inductively. |
| **L2** | Paraphrase generalisation depends on representation geometry. ZsRE transfers more easily than CounterFact under direct-only install. |
| **L3** | Qwen replication is fresh-field cross-model generality, not zero-shot transfer of the same learned δR field. |
| **L4** | External editor baselines (ROME / MEMIT / SERAC / GRACE / WISE) are deferred. C4's kNN/RAG comparison instantiates the architectural argument. |

## Estimated runtimes

- *Smoke mode:* ≈ 30 minutes on a single A100.
- *Paper mode:* ≈ 49 hours on a single A100/H100 (pre-Reading-C; post-Reading-C numbers TBD on next pod run).
- *Qwen replication (C3):* ≈ 19.9 hours additional in paper mode.
- *Convergence-stop protocol:* ≈ 2.5× compute savings if validated; currently skipped (paper mode) per session-handover decision.

## Pickup instructions for future sessions

```bash
cd /workspace/information-as-alignment
git pull --rebase origin claude/review-jupyter-notebook-8AU5y
# Open (IBF)Companion-LLM-Durable-Alignment-v2.ipynb in JupyterLab
# Set RUN_MODE = 'paper' (or 'smoke' for fast sanity check)
# Run all cells end-to-end
```

Each Cn cell prints `EXPECTED: X, GOT: Y, WITHIN_TOLERANCE: True/False` at its end. Compare GOT against the headline in the corresponding card (HANDOVER_NOTEBOOK_REBUILD.md Part 2).
