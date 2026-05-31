# Supervisor note — IBF-over-LLM v2 notebook: end-to-end run results

**From:** Radu (LLM substrate work) → Supervisor (foundation paper author)
**Subject:** v2 rebuild ran end-to-end on the pod; 1 of 8 claims clean-passes the gate; results split cleanly between mechanism finding, methodology deltas, and small implementation bugs
**Status:** Decision needed before next pod cycle
**Companions:**
- `HANDOVER_NOTEBOOK_REBUILD.md` — rebuild specification
- `AGENCY_DISCRETIZATION_NOTE.md` — Reading C engine fix rationale (supervisor-approved)
- `C8_CLAIM_PROPOSAL_NOTE.md` — C8 evidence + claim proposal (supervisor-approved)
- `C1_GATE_MISS_DIAGNOSIS.md` — detailed Phase B analysis

---

## TL;DR

The v2 rebuild ran cleanly end-to-end on Mistral-7B (paper mode, 30.8h for C1 alone, ~70h total). All 13 sections execute without error after one in-flight fix. The headline is split:

- **C3 (Qwen cross-model)** is the only claim that passes its gate cleanly (5/6 metrics within ±0.05; Phase A survival 0.861).
- **C1 (canonical lifecycle)** misses by 1.4 points (avg lin = 0.9398 vs 0.954 ± 0.01). The miss is **entirely Phase B underperforming on its own additive training** by 8.75 points. Hypothesised mechanism: Reading C's history-sufficiency gate makes A's crystallised agency centres operationally engaged, and the cross-context broadcast loop applies responsiveness modulation to B's value-centre formation. **The 0.94 number may be the new Reading-C baseline rather than a bug** — needs your read on whether to rebaseline or investigate further.
- **C2, C4, C5, C6, C7, C8** mostly fail their gates, but the failures are **dominated by methodology deltas and implementation bugs the builder agent introduced for self-containment**, not by mechanism issues. Most are cheap to fix (~30 min – 3 hours each).

The architectural skeleton is sound. What needs your guidance is the rebaseline question for C1 and approval of the fix sequence for C2/C4/C5/C6/C7/C8.

---

## Full scorecard

| Claim | Gate result | Headline numbers | Category |
|---|---|---|---|
| **C1** Lifecycle | **FAIL** | avg lin = 0.9398 (target 0.954 ± 0.01) | **Mechanism finding** (Reading C) |
| **C2** LoRA durability | **FAIL** | weak_drop=0.30, strong_drop=0.07, **ctrl_delta=0.00** | Methodology delta |
| **C3** Qwen cross-model | **PASS ✓** | 5/6 metrics within ±0.05; Phase A survival 0.861 (target ≥ 0.85) | — |
| **C4** vs kNN/RAG | **FAIL** | IBF: direct=1.0, loc=1.0, rev=1.0, **rem=0.0**, rb=0.97 | Implementation bug |
| **C5.1** Retraction | **PASS** | target 0.847→0.000; NN/distant drift ~0 | — |
| **C5.2** Selective deletion | **FAIL** | target 1.000→0.600 (incomplete erasure) | Implementation bug |
| **C5.3** Forgetting decomp | runs | artifact written | — |
| **C6.1** Locality / bleed | **FAIL** | populations 200/**0**/**0** — no controls | Implementation bug |
| **C6.2** Scale frontier | partial | target_acc 0.595 at N=20k; **bug-fix worked** (A=0.878, C=0.987, not 0.353) | Mixed |
| **C7.1-3** Compiled closure | **FAIL** | numbers in 0.125–0.500 range everywhere (random-baseline) | Methodology delta |
| **C8.1-3** Discovery | **FAIL** | D5b 0→0 (canonical 0→1 in 4 ep); D7 0→0; D8: HIGH 0.875 vs LOW 0.0 | Methodology delta |

S4 (paper generator) and S5 (reproducibility) both run cleanly and write their artifacts. The 8-claim summary in `paper/claims_status_final.md` reports 1/8 passing.

---

## Category 1 — Mechanism finding (C1 only)

**The single substantive scientific finding is in C1.** Phase A converged at end-of-phase to lin = 0.957 (matching canonical's 0.956). Phase B should have followed the canonical pattern — pure additive learning of two new categories (certification + committee), no overlap with A, no contradictions. Canonical's B reached 0.975 / 0.995 by ep 1 and held flat at 0.98 aggregate through ep 100. **Our run sits at 0.9025 after 85 epochs with stabilised slope.** Same engine code apart from Reading C, same FI dataset (verbatim verified vs v1 cell 11).

Mechanism hypothesis (full text in `C1_GATE_MISS_DIAGNOSIS.md`):

Reading C's history-sufficiency gate (`len(c.D_history) >= n_agency_min`) unlocks A's crystallised agency centres (~1300 by end of A). When B starts:

1. `_update_agency`'s cross-context loop fires on every B-context query — every A-context crystallised agency centre is queried, increments `n_updates`, contributes to `D_history_cross`. Under the original engine, this loop was structurally empty (the crystallization gate on the cross-context filter was never satisfied for these centres given the contrastive install scheme).
2. `delta_k(z)` for B's queries now integrates over A-context agency centres' `w` values, producing a non-zero responsiveness modulation.
3. B's value-centre formation experiences this responsiveness perturbation from semantically unrelated A-context centres. Centres can't crystallise cleanly because their D-trajectories are being modulated by agency centres from a different semantic domain.

This same mechanism explains the **2.5× per-epoch slowdown on Phase B** (more work per query: ~1300 extra kernel evaluations on A-context agency centres) — slowdown and accuracy gap are two symptoms of one Reading-C side-effect.

**Important framing:** this is *not* an indictment of Reading C. The D6 empirical validation (β k_eff U-shape tracking D-variance), D5b's emergent A→C closure, and D7's de novo emergence all depend on Reading C's agency engagement. **The agency channel being operationally engaged is the point.** What we didn't anticipate is that cross-context engagement during multi-phase lifecycle training has a measurable cost on Phase B's value-centre formation.

### The discriminator

We can confirm or refute the mechanism with a single ~15h pod run: add a debug flag in S1 that disables the cross-context agency broadcast loop, and run C1 with the flag set to True only during Phase B's training. If B then hits the canonical 0.97–0.98 in 1–2 epochs, mechanism confirmed.

### The decision

If the mechanism is confirmed, three design options:

1. **Accept 0.94 as the Reading-C baseline.** Cross-context agency engagement is the design intent. The original 0.954 was an artifact of the under-implemented agency channel. Rebaseline `HEADLINE_AVG_LIN` and document the engine-version delta explicitly in the paper.
2. **Per-phase agency reset.** When a new phase context begins, freeze previous-phase agency centres (no further updates, no cross-context firing). Preserves single-phase Reading-C semantics but eliminates cross-phase interference. Re-runs would likely land back near 0.954.
3. **Tighten the cross-context gate** with a semantic-proximity check between the broadcast centre and the query. Adds complexity but preserves both engagement and clean cross-phase training.

My lean is (1), given D5b/D6/D7 validation depends on the engagement being unrestricted. But (2) is a clean fallback if the paper's central existence claim needs to land closer to 0.954 for narrative reasons.

---

## Category 2 — Methodology deltas (C2, C7, C8)

These cells fail their gates because the builder agent's deviations from the v1 spec changed what the cells actually measure.

**C2 — LoRA durability.** Builder built target items inline because v1's §14 (strong-prior pilot) was deferred per handover Part 5. Inline targets have different baseline accuracies (BEFORE LoRA: weak 0.883, strong 0.617) than v1's would have. Result: same LoRA training produces different drift magnitudes. **Critically, controls held perfectly (`ctrl_delta=0.000`) — the architectural decoupling claim is intact.** The gate-failing numbers are the absolute target drops, which are a function of target-item construction, not engine behaviour.

**Fix (~1h dev):** port v1's strong-prior FI target construction directly, with the same 60 weak + 60 strong split and same baseline-margin criteria. Re-run produces apples-to-apples numbers against the canonical 0.003 / 0.000.

**C7 — Compiled closure.** Numbers are in the 0.125–0.500 range across all three subsections — random-baseline-ish. Builder flagged this deviation explicitly:

> *"C7 and C8 use synthetic chain definitions inline rather than the v1 `_graph_chains` / `_make_edge_item` infrastructure built in v1 cell 62 ... I built the same chain definitions inline (8 chains: approval, restricted_exposure, ...) with the C_AC_de_novo field for D7's de novo regime."*

The inline reconstructions don't carry v1's distributional properties — chain-element semantics, embedding distances, the BC↔AC C-text distance structure that D7 specifically depends on. **Tests run cleanly but measure nothing.**

**C8 — Discovery.** Same root cause as C7. D5b's emergent A→C 0→1 trajectory doesn't manifest because the chains aren't carrying the canonical distributional properties. D7 reports "scaffold_was_load_bearing" — but in v1 the diagnostic is "scaffold NOT load-bearing" (mean 8.09σ separation, both engines reach 1.0).

**Fix (~2–3h dev):** port v1 cell 62's `_graph_chains`, `_make_edge_item`, `_with_question`, `_make_strong_prior` infrastructure verbatim into S3 (where the dataset lives) and re-wire C7/C8 to consume them. After this, C7/C8 will either reproduce v1's headline numbers or the failure will be a real mechanism finding (worth reporting either way).

---

## Category 3 — Implementation bugs (C4, C5.2, C6)

**C4 remove operation returns 0.0.** IBF wins decisively on locality (1.0 vs 0.0 for both kNN and RAG oracles — the architectural distinction the card asks for) and matches on direct/revise/rollback. But `remove`=0.0 means the IBF "remove" code path isn't actually dissolving the installed centres in this synthetic-CounterFact configuration. Either the remove code mis-targets the centres, or the test expectation doesn't match what the engine does. **Fix (~30 min dev):** trace the IBF remove path in C4 against the same record IDs that kNN remove succeeds on; should resolve quickly.

**C5.2 selective deletion is partial.** target_acc went 1.000 → 0.600 (should be → ~0.05). Others held perfectly at 0.879. The dissolution criterion is too conservative or the target-set construction misses some active contributors. **Fix (~30 min dev):** instrument the deletion to log which centres are dissolved and which contributors remain; tune criterion.

**C6.1 / C6.2 controls population is empty.** Locality test built populations 200/**0**/**0** — the near-neighbor and distant arrays have zero entries, so all drift measurements are vacuously zero. Same defect cascades into 6.2's control accuracy column (all 0). **Fix (~30 min dev):** populate near and distant arrays from the FI dataset's existing employee pool, same way v1 does. **Note: 6.2's spec-required scale-frontier C-retention bug fix WORKED** — A retention 0.878 and C retention 0.987 across all scales (not the canonical-bug's flat 0.353). The fix is validated even though the cell as a whole doesn't pass.

---

## Decisions requested

1. **C1 rebaseline vs investigate.** Should we run the discriminator (~15h pod, debug-flag disabling cross-context agency during Phase B), or accept the 0.94 baseline now and update `HEADLINE_AVG_LIN` + paper text accordingly? If you want the discriminator first, I'll patch S1 with the debug flag and queue the run.

2. **C2 methodology fix.** OK to port v1's strong-prior FI target construction into C2 to make the headline comparison apples-to-apples? Estimated ~1h dev.

3. **C7/C8 chain infrastructure port.** OK to port v1 cell 62's `_graph_chains` / `_make_edge_item` / `_with_question` / `_make_strong_prior` infrastructure verbatim into S3? This is the largest fix in the queue (~2–3h dev) and changes nothing about the engine; it just gives C7/C8 valid test data.

4. **C4 / C5.2 / C6 implementation bugs.** OK to fix in-place (~30 min each)?

5. **Sequencing.** My proposed order:
   - First: methodology fixes (C2 + C7/C8 chain port + C4/C5.2/C6 bugs) — cheapest path to apples-to-apples re-run
   - Then: end-to-end re-run on pod (~70h)
   - Then: based on which of C1–C8 still miss, decide whether C1's mechanism diagnostic is the bottleneck or whether something else surfaced

   Alternative: run the C1 mechanism diagnostic in parallel with the methodology fixes, get both answers in one pod cycle.

   Either sequence works. The parallel option saves a pod cycle if the diagnostic can run alongside fix-development without blocking. Your preference?

---

## Files of record

All on branch `claude/review-jupyter-notebook-8AU5y`:

- `(IBF)Companion-LLM-Durable-Alignment-v2.ipynb` — the rebuilt notebook with all outputs inline (`73c065f`)
- `mmlu_ibf_out/c{1..8}_*.{json,md}` — per-claim machine-readable + human-readable artifacts
- `mmlu_ibf_out/paper/{paper_tables.md, abstract_numbers.json, claims_status_final.md, reproducibility_manifest.json, reproducibility_appendix.md}` — S4/S5 outputs
- `C1_GATE_MISS_DIAGNOSIS.md` — full Phase B analysis
- This note: `V2_RUN_RESULTS_SUPERVISOR_NOTE.md`

Engine version: `2.0-history_gate` (Reading C applied). Branch HEAD: `73c065f`. Runtime: ~70h on RTX 5090 (paper mode with strong-convergence early-stop active).
