# Handover — IBF-over-LLM Notebook Restructure

**To:** The cell-builder agent
**From:** Radu (LLM substrate work) + Supervisor (foundation paper) + Physicist (agency-channel proposal)
**Status:** Approved 2026-05-XX. Build the new notebook to this spec.

---

## Part 0 — Context

The existing notebook (`(IBF)Companion-LLM-Durable-Alignment.ipynb`) has ~40 sections grown organically during research. All eight claims (C1–C8) are empirically validated; all artifacts live in `mmlu_ibf_out/`. The investigation is complete.

This handover specifies a **clarity-only refactor**: restructure the existing evidence into a logical dependency chain, renumber claims to match layer order, and rebuild a clean ~18-section notebook that maps 1:1 to the claim structure.

**Not your job:** redo experiments, re-derive claims, write paper text. Just rebuild the notebook from the spec below.

---

## Part 1 — The central thesis and four-layer stack

**Central thesis (single sentence):**

> *IBF is a local durable alignment substrate for frozen LLMs whose properties (decoupling, generality, distinction) support a clean operational lifecycle (install / revise / remove with preserved locality) which in turn supports complementary deductive and inductive composition paths — without modifying base-model weights.*

**Four-layer dependency stack:**

```
Layer 4 — COMPOSITION
   C7  Compiled closure (deductive)
   C8  Discovery-driven extension (inductive)
        — includes Crucible-adjudicated conflict resolution (D8 evidence)

Layer 3 — OPERATIONS
   C5  Truth-maintenance lifecycle
   C6  Locality preservation

Layer 2 — PROPERTIES
   C2  Substrate decoupling under base evolution
   C3  Cross-model mechanism generality
   C4  Distinction from kNN/RAG

Layer 1 — EXISTENCE
   C1  Local durable alignment without weight editing
```

Each upper layer is *earned* by the layers below it. Falsifying a lower claim cascades upward.

---

## Part 2 — The eight claim cards

Each card is a complete spec for one notebook section. Build them in order C1 → C2 → ... → C8. Each section's output artifact must be a deterministic function of the engine state and the cell's seed.

---

### C1 — Local durable alignment without weight editing

- **Layer:** 1 (Existence)
- **Presupposes:** nothing (foundational existence claim)
- **Adds:** the substrate exists as a distinct architectural object
- **Enables:** every subsequent claim
- **Falsifier:** alignment installation requires weight modification, period — or canonical lifecycle fails to converge

- **Source cells (current notebook):** § 8 (canonical training at fixed operating geometry)
- **Target cell (new notebook):** § C1 — Canonical lifecycle training on frozen Mistral-7B
- **Headline result to reproduce (within ±0.01 absolute):**
  - Avg lin = 0.954 (vs base 0.216, +0.738 absolute)
  - 4/4 phases converged
  - Engine end-state: 6382 value centers (all crystallized), 2139 agency centers, |v|_max = 2.815, 18786 lifecycle dissolutions

- **Artifacts produced (write to `mmlu_ibf_out/`):**
  - `canonical_training_results.json` (per-phase final metrics)
  - `canonical_training_results.md` (human-readable summary)
  - `canonical_engine.pkl` (~300MB — the engine state)
  - `canonical_metrics.pkl` (training trajectory)

- **Convergence-stop protocol (PRIORITY — test this first; supervisor's note):**
  - Add `EARLY_STOP_STRONG_CONVERGE = True` flag
  - Stop a phase when: `ma_delta < 0.001 AND slope_recent < 0.0001 AND lin ≥ 0.95 AND ep ≥ min_epochs`
  - `min_epochs` defaults to 50 (down from current 100) for phases B/C/D; keep 100 for A as buffer
  - Hard caps unchanged: A:300, B/C/D:200
  - **Validation gate:** if avg lin shifts by more than ±0.01 from the headline (0.954), revert to current parameters and document
  - If validation passes, the same convergence protocol can be applied to all subsequent long-running cells (C5 retraction, C6 scale frontier, C7 compiled-closure arms, C8 discovery)

- **Deferred from this card:** smoke-only canonical, σ sweep variants, anchor experiments, paraphrase audits

---

### C2 — Substrate decouples from base-model evolution

- **Layer:** 2 (Property: decoupling)
- **Presupposes:** C1
- **Adds:** the substrate is structurally orthogonal to base parameters — base evolution doesn't break the alignment layer
- **Enables:** C3 (cross-model is a special case of decoupling), production framing
- **Falsifier:** a 30%+ base perturbation degrades field accuracy by >10%, OR field drift > 5% under any base modification

- **Source cells (current notebook):** § 26 (end-to-end actual LoRA durability, control-fixed)
- **Target cell (new notebook):** § C2 — LoRA durability test (control-fixed)
- **Headline result to reproduce (within ±0.005 absolute):**
  - Base shift: 37.5% (target argmax shift rate)
  - Field drift: +0.003 (weak target accuracy drop)
  - 0.000 drift on strong target and controls
  - Selectivity ≈ 125:1 (base shift / field drift)
  - All 11 validation criteria pass; status: `clean_actual_lora_e2e_control_fixed`

- **Artifacts produced:**
  - `actual_lora_e2e_durability_control_fixed_report.json`
  - `actual_lora_e2e_durability_control_fixed_report.md`

- **Convergence-stop protocol:** none (LoRA training is fixed 24 steps; no convergence loop to optimize)

- **Deferred from this card:** smoke-only LoRA, instruction-tune adversarial, multi-step LoRA sweeps

---

### C3 — Cross-model mechanism generality

- **Layer:** 2 (Property: generality)
- **Presupposes:** C1, C2
- **Adds:** the mechanism transfers across base models in practice — IBF is a substrate, not a Mistral feature
- **Enables:** framing the contribution as architectural, not model-specific
- **Falsifier:** mechanism fails on a second base model, OR cross-model behavior differs qualitatively

- **Source cells (current notebook):** § 36 (cross-model generality: Qwen2-1.5B)
- **Target cell (new notebook):** § C3 — Qwen2-1.5B cross-model replication
- **Headline result to reproduce (within ±0.02 absolute):**
  - 5 of 6 metrics within ±0.05 of Mistral canonical
  - Knowledge injection identical (Mistral 0.956 ↔ Qwen 0.956)
  - Phase A survival: Qwen 0.863 ≥ Mistral 0.850 (slight Qwen edge)
  - Only outlier: New facts (Phase B) — Qwen 0.740 vs Mistral 0.983 (Δ −0.243); known characteristic of Qwen-1.5B's smaller capacity
  - All 9 of 9 criteria pass; status: `clean_cross_model_generality_smoke`

- **Artifacts produced:**
  - `cross_model_generality_qwen2_1_5b.json`
  - `cross_model_generality_qwen2_1_5b.md`

- **Convergence-stop protocol:** apply C1's protocol (Qwen has similar phase-by-phase convergence structure)
  - **Validation gate:** Phase A survival on Qwen must stay ≥ 0.85

- **Deferred from this card:** Phi-3 replication, additional cross-model targets (§ 37 mechanism continuation defers to supplementary)

---

### C4 — Distinct from kNN/RAG (architecturally)

- **Layer:** 2 (Property: distinction)
- **Presupposes:** C1
- **Adds:** the substrate is not reducible to retrieval-based or weight-editing alternatives
- **Enables:** differentiation in the alignment literature
- **Falsifier:** an oracle-maintained kNN or RAG matches IBF on every lifecycle dimension

- **Source cells (current notebook):** § 30 (lifecycle benchmark harness) + § 32 (IBF in lifecycle) + § 33 (kNN baseline) + § 34 (RAG baseline)
- **Target cell (new notebook):** § C4 — Lifecycle comparison: IBF vs kNN vs RAG
  - One consolidated cell that runs all three methods on CounterFact + ZsRE lifecycle
  - Same data, same operations sequence (install / paraphrase / multi-hop / revise / remove / rollback)
  - Comparison emphasizes: native lifecycle (IBF) vs oracle-maintained memory (kNN/RAG)

- **Headline result to reproduce (within reasonable smoke tolerances):**
  - CounterFact: IBF native 1.000 direct / 0.967 loc / 1.000 rev / 1.000 remove / 1.000 rollback
  - kNN (oracle): 1.000 direct / 0.800 loc / 1.000 rev / 1.000 remove / 0.833 rollback
  - RAG (oracle): 0.300 direct / 0.867 loc / 0.000 rev / 1.000 remove / 0.167 rollback
  - Architectural distinction: IBF's `native` burden vs kNN's `manual` vs RAG's `oracle` is the qualitative comparison

- **Artifacts produced:**
  - `benchmark_ibf_lifecycle.json`
  - `benchmark_knn_lifecycle.json`
  - `benchmark_rag_lifecycle.json`
  - `benchmark_comparison.md` (synthesized comparison table)

- **Convergence-stop protocol:** apply C1's protocol to the IBF run only (kNN and RAG are not iterative)

- **Deferred from this card:** MMLU evaluation, full counterfact paraphrase diagnostic suite, σ sensitivity diagnostics

---

### C5 — Truth-maintenance lifecycle

- **Layer:** 3 (Operation: lifecycle)
- **Presupposes:** C1
- **Adds:** install / revise / remove / rollback / retain as discrete operations on the substrate
- **Enables:** C6 (locality is meaningful when ops exist), Layer 4 composition
- **Falsifier:** retraction doesn't remove; revision creates phantom side effects; rollback doesn't restore

- **Source cells (current notebook):** § 12 (retraction via contradiction) + § 14 (selective deletion) + § 25 (forgetting / truth-maintenance)
- **Target cell (new notebook):** § C5 — Lifecycle operations validation
  - Subsection 5.1: Retraction (from § 12)
  - Subsection 5.2: Selective deletion (from § 14)
  - Subsection 5.3: Forgetting decomposition (from § 25)

- **Headline results to reproduce:**
  - **5.1 Retraction (full 30-epoch run, RUN_FULL_RETRACTION=True):**
    - target_orig: 0.947 → 0.020 (Δ = −0.927)
    - target_new: 0.027 → 0.973 (Δ = +0.947)
    - NN drift: 0.000; distant drift: 0.000
    - Selectivity ≈ 950M:1 (vs near-zero drift)
  - **5.2 Selective deletion:**
    - Target deletion accuracy ≥ 0.95
    - Control preservation ≥ 0.95
  - **5.3 Forgetting decomposition:**
    - A retention after D: 0.850 (matches C1 canonical)
    - B→C boundary dominant: −0.081 of −0.106 total A-decay (76%)
    - Reorg selectivity: 8.6× (affected −0.173 vs unaffected −0.020)

- **Artifacts produced:**
  - `retraction_full_results.json` (full 30-epoch result; not just smoke)
  - `selective_deletion.json` + `.md`
  - `forgetting_diagnostic_report.json` + `.md`

- **Convergence-stop protocol:** apply C1's protocol to retraction's full 30-epoch run — saturates well before epoch 30 in practice

- **Deferred from this card:** consistency audit (§ 11), provenance samples (§ 12 v3), interim audits

---

### C6 — Locality preservation under operations

- **Layer:** 3 (Operation: locality)
- **Presupposes:** C1, C5
- **Adds:** operations are spatially localized — installing A doesn't leak into B's region
- **Enables:** safe composition (Layer 4 isn't meaningful if locality fails)
- **Falsifier:** NN drift > 0.05 under sustained operations; or scale-frontier C-retention bug reproduces

- **Source cells (current notebook):** § 19 (locality/bleed test) + § 20 (scale frontier, merged § 20+§ 20b)
- **Target cell (new notebook):** § C6 — Locality and scale frontier
  - Subsection 6.1: Locality/bleed test (from § 19) — operations preserve near/distant neighbor accuracy
  - Subsection 6.2: Scale frontier (from § 20) — 1k–50k rule scales with locality preserved

- **Headline results to reproduce:**
  - **6.1 Locality:** NN drift 0.000 and distant drift 0.000 across all tested operations
  - **6.2 Scale:** target accuracy ≥ 0.93 at every scale up to 20k rules; control accuracy = 1.000 at all scales; center growth/rule ≤ 2.0

- **Artifacts produced:**
  - `fi_locality_bleed_test.json` + `.md`
  - `fi_scale_capacity_frontier.json` + `.md`

- **Critical bug to fix:** § 20's "A retention" and "C retention" columns are byte-identical across all scales (0.85 and 0.353 — see `PAPER_RUN_HANDOVER.md` and § 20 diagnostic notes). This is a measurement artifact, not a real retention drop.
  - **Fix:** rebuild `precomputed["A_Onboarding_test"]` and `precomputed["C_Reorg_test"]` from canonical chains at evaluation time, not from session-state precomputed dicts (which may have been mutated by intermediate cells).
  - **Validation gate:** after fix, both A and C retention should be > 0.80 across all scales, not flat at 0.353.

- **Convergence-stop protocol:** apply C1's protocol to scale-frontier sweep epochs

- **Deferred from this card:** bridge experiment (§ 17), local alignment phase transition (§ 18), long-horizon stability (§ 22 — note this is the old § 22 long-horizon, not the new compiled-closure §22)

---

### C7 — Compiled semantic structure (deductive composition)

- **Layer:** 4 (Composition: deductive)
- **Presupposes:** C1, C5, C6
- **Adds:** structured knowledge (rules + their derived consequences) installable as a unit, revisable as a unit
- **Enables:** closure-style alignment without manual edge enumeration; complementary to C8
- **Falsifier:** compiled consequences don't survive across queries; revision fails to update derived edges; closure rules interact destructively

- **Source cells (current notebook):** § 22 (local ontology closure) + § 23 (ontology closure diagnostic — L1 negative finding) + § 24 (compiled ontology closure with revision)
- **Target cell (new notebook):** § C7 — Deductive composition via compiled closure
  - Subsection 7.1: Local ontology closure (§ 22) — definitions + one-hop + two-hop installable
  - Subsection 7.2: Emergent transitive closure does NOT happen automatically (§ 23) — L1 limitation, motivates § 24
  - Subsection 7.3: Compiled closure with revision (§ 24) — external compiler derives consequences, installs durably, revises cleanly

- **Headline results to reproduce:**
  - **7.1:** 23B (one-hop) pass; 23C (two-hop) pass; both with held-out generalization
  - **7.2:** Emergent A→C closure under value-only readout = 0.000 (`target_acc = 0.0`, `chain_consistency = 0.0`, mean `target − base margin = −4.16`)
  - **7.3:** Compiled closure passes all four regimes (explicit edge, explicit closure, revision); A→C survives compiler-installation; A→D survives compiler-revision

- **Artifacts produced:**
  - `fi_ontology_closure_23bc.json` + `.md`
  - `fi_local_ontology_graph_closure_cell24.json` + `.md`
  - `fi_compiled_ontology_closure_cell24b.json` + `.md`

- **Convergence-stop protocol:** apply C1's protocol to compiled-closure training arms

- **Deferred from this card:** the early ontology graph closure variants that didn't reach final form

---

### C8 — Discovery-driven extension (inductive composition)

- **Layer:** 4 (Composition: inductive)
- **Presupposes:** C1, C5, C6
- **Adds:** the substrate extends itself through interaction + environmental reward; finds edges the compiler missed; conflict with compiled rules resolved via C5's existing Crucible mechanism with operator-tunable timescale resilience
- **Enables:** continual alignment from production feedback; complementary to C7; substantively unifies deductive + inductive
- **Falsifier:** emergence requires kernel scaffold (D7 falsified); emergence requires agency modulation (D6 showed redundant in saturated regimes); Crucible adjudication fails to respond to resilience knob (D8 falsified — operator-tunable timescale confirmed)

- **Source cells (current notebook):** § 24b-D5b (basic emergence) + § 24b-D7 (de novo without scaffold) + § 24b-D8 (Crucible-adjudicated conflict)
- **Target cell (new notebook):** § C8 — Discovery-driven extension and adjudicated unification
  - Subsection 8.1: Discovery training produces emergent A→C closure (from D5b)
  - Subsection 8.2: Emergence is robust without kernel scaffolding (from D7 — de novo at 8.09σ separation)
  - Subsection 8.3: Crucible-adjudicated conflict resolution (from D8 — operator-tunable resilience)

- **Headline results to reproduce:**
  - **8.1:** test_AC: 0.000 → 1.000 in 4 epochs; +0.125 positive backward transfer to BC; AB preserved at 1.000
  - **8.2:** Same emergence achieved with mean BC↔AC C-text distance = 8.09 σ_op (all chains scaffold BROKEN per diagnostic); test_AC: 0.000 → 1.000 at ep 1
  - **8.3:** LOW resilience: compiled rule collapses in ~1 epoch under sustained contradicting evidence; HIGH resilience: compiled rule preserved through ep ~25, catastrophic transition by ep 30. Operator-tunable resistance multiplier ≈ 25×.

- **Artifacts produced:**
  - `fi_agency_channel_d5b_discovery.json` + `.md`
  - `fi_agency_channel_d7_de_novo.json` + `.md`
  - `fi_agency_channel_d8_conflict_adjudication.json` + `.md`

- **Convergence-stop protocol:** discovery saturates within 4-5 epochs typically; reduce default to 20-30 epochs with early-stop at `test_AC ≥ 0.95 AND stable for 5+ epochs`

- **Engine requirement:** Reading C patch must be applied (history-sufficiency gating instead of crystallization gating on agency channel). See Part 7 of this handover.

- **Deferred from this card:**
  - § 24b-D1 (kernel locality diagnostic) — informed the D5b design but not headline evidence
  - § 24b-D2/D3/D4 (eliminated alternatives: static agency wirings, c.z[:64] proxy, phase-2 z_before storage) — all ruled out as approaches; superseded by D5b+
  - § 24b-D5 (the original buggy version) — superseded by D5b
  - § 24b-D6 (α vs β engine-fix validation) — relegated to engine-patch documentation in Part 7; D6's empirical content (k_eff U-shape under Reading C) is part of the engine fix's validation, not C8's evidence

---

## Part 3 — Setup and synthesis sections

These are infrastructure cells; no claim attached. Build them once at the top and bottom of the notebook.

### S1 — Universal IBF engine

- **Source:** current cell 9 (engine definition)
- **Modifications:** apply Reading C patch (see Part 7)
- **Content:** `IBFParams` dataclass, `MemoryCenter` dataclass, `IBFEngine` class with patched `_update_agency` and `delta_k` for history-sufficiency gating

### S2 — Representation geometry calibration

- **Source:** current cells 16-17 (sentence-transformers + representation calibration)
- **Content:** sentence encoder (mpnet-base-v2), PCA + scaler, 80-D augmented proposition space, σ calibration to 7.2621

### S3 — Future Industries dataset

- **Source:** current cell 18 (propositional adapter + FI data definitions)
- **Content:** A/B/C/D phase definitions, train/test splits, subject and answer features

### S4 — Paper deliverable generator

- **Source:** current § 38 v3 (paper-deliverable generator)
- **Content:** reads all `mmlu_ibf_out/*.json` artifacts; produces `paper_tables.md`, `abstract_numbers.json`, `claims_status_final.md`. Verifies all 8 claims have passing evidence.

### S5 — Conclusion and reproducibility appendix

- **Source:** current § 39 (artifact summary) + § 40 (reproducibility)
- **Content:** claims/artifact map updated for new numbering; reproducibility metadata (seeds, env, branch, commit)

---

## Part 4 — Final notebook structure

```
S1.  Universal IBF engine (with Reading C patch)
S2.  Representation geometry calibration
S3.  Future Industries dataset

C1.  Canonical lifecycle training [LAYER 1]

C2.  LoRA durability test [LAYER 2]
C3.  Qwen2-1.5B cross-model replication [LAYER 2]
C4.  Lifecycle comparison: IBF vs kNN vs RAG [LAYER 2]

C5.  Lifecycle operations: retraction + deletion + forgetting [LAYER 3]
C6.  Locality and scale frontier [LAYER 3]

C7.  Deductive composition via compiled closure [LAYER 4]
C8.  Discovery-driven extension and adjudicated unification [LAYER 4]

S4.  Paper deliverable generator
S5.  Conclusion and reproducibility appendix
```

**Total: 13 sections** (3 setup + 8 claims + 2 synthesis).

Each claim section produces its specific artifacts. Each section's `## § Cn — ...` header in markdown should mirror the card title exactly. Each code cell's banner (first 10 lines) should:

1. State the claim number and short statement
2. Reference its dependency on previous claims
3. Note what artifacts it writes
4. Note the convergence-stop protocol active in the cell

---

## Part 5 — Deferred cells (do NOT include in main notebook)

The following are deferred to supplementary materials. They remain in the project for reproducibility but should not appear in the rebuilt main notebook:

**Diagnostics:**
- κ/σ diagnostics (§ 9 family)
- Amplitude hygiene (§ 22B)
- Anchor experiments (paraphrase audit subcells of § 31b/31c)
- Mechanism continuation (§ 35, § 37)

**Eliminated alternatives:**
- § 24b-D1 (kernel locality diagnostic — informed D5b design, superseded)
- § 24b-D2/D3/D4 (static agency wirings, c.z[:64] proxy, z_before storage Phase 2 — all eliminated by D5b+)
- § 24b-D5 (original buggy version — superseded by D5b)
- § 24b-D6 (α vs β engine-fix validation — moved to engine-patch documentation in Part 7)

**Redundant variants:**
- § 15 Strong prior pilot
- § 15b Strong absurdities
- § 16 Bridge experiment
- § 17 Local alignment phase transition
- § 18 Local alignment system report

**Superseded subsections:**
- § 22B Amplitude hygiene
- § 22C Forgetting trajectory (folded into C5)
- § 27 (already deleted in prior commit)
- Multiple benchmark variants subsumed by § 30/§ 32-34

**Cross-model variants beyond Qwen:**
- § 37 mechanism continuation
- Phi-3 generality file

---

## Part 6 — Convergence optimization protocol

**Background (from supervisor's note):**
> *"The 2-3× compute reduction from early stopping is welcome but verify on one claim before applying globally. Run C1's canonical lifecycle with the proposed early-stop criteria and confirm the headline number (avg lin 0.954) is preserved within ±0.01. If it holds, apply the same criteria to all long-running cells. If it doesn't, the convergence detector is too aggressive for some claims and needs per-claim calibration."*

### The early-stop criteria

In every training loop with phase-level convergence detection:

```python
if (ep >= min_epochs
    and ma_delta < 0.001
    and slope_recent < 0.0001
    and lin_acc >= 0.95):
    break  # strong convergence reached
```

Where:
- `ma_delta` = moving-average delta of recent eval scores (window 20)
- `slope_recent` = linear slope of recent eval scores (window 10)
- `lin_acc` = current linear-readout accuracy
- `min_epochs` = 50 for Phases B/C/D; 100 for Phase A (slightly more conservative for first phase)

Hard caps unchanged from current paper run:
- A_Onboarding: 300
- B_Initiative: 200
- C_Reorg: 200
- D_Turnover: 200

### The validation protocol

1. **Test on C1 first.** Run C1's canonical lifecycle with the new early-stop enabled. Save artifact.
2. **Compare to baseline.** The reference number is `avg lin = 0.954` (from the current paper run, recorded in `canonical_training_results.json`).
3. **Validation gate:** If new run's `avg lin` is within `0.954 ± 0.01`, the optimization passes. Apply to all long-running cells (C5 retraction, C6 scale frontier, C7 compiled-closure arms, C8 discovery).
4. **If validation fails** (more than ±0.01 from headline): revert C1 to current parameters, document the divergence, and consider per-claim calibration. Do NOT apply globally.

### Expected savings if validation passes

| Cell | Current runtime | Optimized estimate |
|---|---:|---:|
| C1 canonical lifecycle | ~35h | ~12-15h |
| C5 retraction (full 30 ep) | ~95 min | ~30-40 min |
| C6 scale frontier (20k rules) | ~230 min | ~80-100 min |
| C7 compiled-closure arms | ~10 min × 4 arms | ~5 min × 4 arms |
| C8 discovery (40 ep) | ~20 min × 3 engines | ~10 min × 3 engines |
| C3 Qwen cross-model | ~21h | ~10-12h |

**Total compute reduction: roughly 2.5× across the rebuilt notebook.**

---

## Part 7 — Engine specification (Reading C patch)

The current engine in cell 9 has a structural asymmetry in agency-channel dynamics that under-implements Postulate 1's "parallel equation" framing. The supervisor (foundation paper author) approved the fix (Reading C from `AGENCY_DISCRETIZATION_NOTE.md`).

### The patch (mandatory for the rebuilt notebook)

In `IBFEngine._update_agency`, replace:

```python
if c.is_crystallized():
    dv = c.D_var_rolling()
    tw = np.clip(self.p.w_max*(1-dv/self.p.w_dvar_threshold),-self.p.w_max,self.p.w_max)
    c.w += self.p.eta_k*kw*(tw-c.w); c.w = np.clip(c.w,-self.p.w_max,self.p.w_max)
```

with:

```python
if len(c.D_history) >= self.p.n_agency_min:
    dv = c.D_var_rolling()
    tw = np.clip(self.p.w_max*(1-dv/self.p.w_dvar_threshold),-self.p.w_max,self.p.w_max)
    lr = self.p.eta_k_cryst if c.is_crystallized() else self.p.eta_k
    c.w += lr*kw*(tw-c.w); c.w = np.clip(c.w,-self.p.w_max,self.p.w_max)
```

In `IBFEngine.delta_k`, replace:

```python
if not c.is_crystallized(): continue
```

with:

```python
if len(c.D_history) < self.p.n_agency_min: continue
```

Add to `IBFParams`:
- `n_agency_min: int = 20` (history sufficiency threshold for agency channel)
- `eta_k_cryst: float = 0.005` (crystallized-agency learning rate, parallel to `eta_cryst` for value)

### Pre-merge validation requirement

Before declaring the patched engine canonical:

1. Run C1 with the patched engine. Result must match unpatched within ±0.01 on avg lin.
2. Run C3 (Qwen) with the patched engine. Result must match within ±0.02 on all 6 cross-model metrics.

If both pass, the patched engine is canonical. The current `canonical_engine.pkl` must be **re-saved under the patched engine semantics** with a version stamp (`engine_version: "2.0-history_gate"` in the JSON metadata).

### Why the patch is required

Without Reading C, the agency channel is structurally silenced in the LLM closure regime (the engine's crystallization gate doesn't fire because the contrastive install scheme never satisfies `|mean(D)| < convergence_threshold`). This causes C8's headline mechanism (agency-modulated exploration) to be untestable.

D6 validates the patch on two independent chain geometries: β k_eff at AC traces a U-shape (4.66 → 2.58 → 3.23) tracking D-variance dynamics, while α (status quo) stays flat at k_0 = 5.000. The patch faithfully implements Postulate 1's "parallel equation" framing on small-data substrates.

---

## Part 8 — Coding standards and reproducibility

### Cell structure conventions

Each code cell follows this banner format (first lines):

```python
# ════════════════════════════════════════════════════════════════
# § Cn — [Claim statement]
# Layer: [layer number]
# Presupposes: [previous claims]
# Artifacts: [list of files written]
# Convergence-stop: [yes/no/per-protocol]
# ════════════════════════════════════════════════════════════════
```

### Determinism

- Set `SEED` at the top of the notebook (default `42`).
- Each long-running cell sets its own seed offset from `SEED` (e.g., `SEED + 100` for C1, `SEED + 200` for C2, etc.) — listed in the card if applicable.
- All `np.random.RandomState` and Python `random.Random` instances use these seeds.
- No global `np.random.seed(...)` calls inside cells that share state with other cells.

### Artifact naming

All artifacts go to `mmlu_ibf_out/`. Names follow the pattern:

- `cn_claim_short_name.json` (machine-readable result)
- `cn_claim_short_name.md` (human-readable summary)

Example: `c5_lifecycle_retraction.json`, `c5_lifecycle_retraction.md`.

This is a **break from the current notebook's naming** (which used `fi_*` and verbose descriptive names). Adopt the cN-prefixed naming for new artifacts; the old artifacts remain under their existing names in `mmlu_ibf_out/` for reproducibility.

### Run mode

Add a single mode flag at the top of the notebook:

```python
RUN_MODE = "paper"  # one of: "smoke", "paper", "verify-convergence"
```

- `"smoke"`: minimum epochs, small datasets, smoke-mode hard caps; for development
- `"paper"`: full paper-grade run (target for final submission)
- `"verify-convergence"`: paper-grade BUT with the convergence-stop protocol enabled; for the C1 validation gate

### Logging

Each long-running cell prints:
1. The claim it serves (at start)
2. The convergence trajectory (per evaluation)
3. The final headline result with explicit `EXPECTED: X, GOT: Y, WITHIN_TOLERANCE: True/False`
4. Any optimization metadata (epochs_used vs epochs_capped, time_saved_vs_baseline)

---

## Part 9 — Build sequence

**Recommended order:**

1. **Day 1 (morning):** Build S1 (engine + Reading C patch), S2 (representation), S3 (dataset). Validate S1 with a smoke test that the patched engine produces identical output to the unpatched engine on a small canonical-trained instance (just to confirm the patch is well-formed).

2. **Day 1 (afternoon):** Build C1 with the convergence-stop protocol enabled. Run in `"verify-convergence"` mode. **Validation gate:** confirm avg lin within 0.954 ± 0.01. If yes, the optimization protocol is approved for global application. If no, revert to current parameters and document.

3. **Day 2 (morning):** Build C2 (LoRA), C3 (Qwen), C4 (kNN/RAG comparison). C3 is the second long-running cell — apply the convergence-stop protocol per Part 6 if C1's validation passed.

4. **Day 2 (afternoon):** Build C5 (lifecycle ops), C6 (locality + scale). C6 includes the scale-frontier C-retention bug fix (Part 5 of this handover).

5. **Day 3 (morning):** Build C7 (compiled closure) and C8 (discovery + adjudication). C8 reuses the chain definitions from § 24b cells; consolidate D5b + D7 + D8 evidence cleanly.

6. **Day 3 (afternoon):** Build S4 (paper-deliverable generator updated for new naming) and S5 (reproducibility). Run end-to-end on pod.

7. **Day 4:** End-to-end validation pass. Compare every claim's headline number to the reference (this handover). If all 8 claims pass their validation gates within tolerance, the rebuilt notebook is the new canonical artifact and the paper-run can begin.

**Total estimate from approval to clean rebuilt notebook: ~3-4 days.**

---

## Final notes

- **Don't refactor the engine** beyond Reading C. Keep cell 9 (now S1) as faithful to the foundation paper as possible.
- **Don't add new claims.** If you find something interesting during the rebuild, document it as a deferred experiment for future work. The eight claims are locked.
- **Don't change σ or other geometric parameters.** The operating point is canonical (σ = 7.2621). Keep it.
- **Don't relitigate the deferred cells.** They're in supplementary materials. The main notebook is for the claim chain.
- **If you hit a real bug** (like the C6 scale-frontier C-retention measurement artifact), fix it and document the fix in the cell's banner. Don't extend the bug list silently.

The structure is approved. Build the notebook to this spec.
