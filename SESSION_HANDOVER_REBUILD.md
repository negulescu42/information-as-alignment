# Session handover — pick up from notebook rebuild

**Date:** 2026-05-XX (end of session approaching token ceiling)
**Branch:** `claude/review-jupyter-notebook-8AU5y`
**Last commit at handover:** `8948669` (session-handover note) — but agent commits follow.

---

## What the background builder agent delivered (completed)

Builder agent (general-purpose subagent) finished cleanly. Built S1 + S2 + S3 + C1 of the rebuilt notebook per the handover spec.

**Path to new notebook:** `/home/user/information-as-alignment/(IBF)Companion-LLM-Durable-Alignment-v2.ipynb`

**Commits pushed** on `claude/review-jupyter-notebook-8AU5y`:
- `25065cd` — S1/S2/S3 (engine + representation + dataset)
- `9eb25e3` — C1 with validation gate

**Status:** 11 cells (5 markdown + 6 code), 1683 lines, all code cells `ast.parse`-verified.

### What was built (4 of 13 sections)

| Section | Layer | Source (orig idx) | Lines | Notes |
|---|---|---|---|---|
| Title + run config | — | new | 64+40 | RUN_MODE flag, SEED, HEADLINE_AVG_LIN=0.954, EARLY_STOP_STRONG_CONVERGE |
| S1 — Engine | setup | 9 | 409 | **Reading C patch applied**; ENGINE_VERSION="2.0-history_gate" |
| S2 — Representation | setup | 7, 13, 15 | 110 | mpnet + Mistral loaders; σ calibration deferred to S3 |
| S3 — FI dataset + adapter + FAISS | setup | 11, 16, 18, 20 | 425 | Consolidated; σ ≈ 7.2621 calibrated empirically |
| C1 — Canonical lifecycle | 1 | 23 | 447 | Per-phase early-stop + validation gate + dual-naming artifact |

### Reading C patch — confirmed applied

```python
# IBFParams additions:
n_agency_min: int = 20
eta_k_cryst: float = 0.005

# delta_k (in IBFEngine):
# before: if not c.is_crystallized(): continue
# after:  if len(c.D_history) < self.p.n_agency_min: continue

# _update_agency (in IBFEngine):
# before: if c.is_crystallized():
# after:  if len(c.D_history) >= self.p.n_agency_min:
#             ...
#             lr = self.p.eta_k_cryst if c.is_crystallized() else self.p.eta_k
```

S1's banner documents the Postulate-1-to-engine mapping (9-row table) and the D6 empirical justification.

### C1 validation-gate code (as built)

```python
measured_avg_lin = avg_ai
deviation = measured_avg_lin - HEADLINE_AVG_LIN  # 0.954
within_tolerance = abs(deviation) <= HEADLINE_AVG_LIN_TOL  # 0.01

if RUN_MODE == "verify-convergence":
    CONVERGENCE_GATE_PASSED = bool(within_tolerance)
    if within_tolerance:
        print("    ◆ GATE PASSED — early-stop protocol approved for C2-C8")
    else:
        print("    ✗ GATE FAILED — revert to RUN_MODE='paper' for C2-C8")
```

Gate result is written to `c1_canonical_lifecycle.json` so C2-C8 cells can read it programmatically.

### Three deviations from spec flagged by agent

**(1) S3 consolidation — ACCEPTED.**
Agent merged S2/S3 because σ calibration depends on dataset existence. Architecturally necessary, not a content change. Mechanisms remain 1:1 with original cells.

**(2) FI dataset uses representative templates instead of verbatim §11 content — NEEDS FIX.**
Agent says: *"FI dataset content is faithful to the original but generated fresh in the cell rather than copied verbatim. The original cell 11 is ~3500 lines of templates; I kept the deterministic generation logic and the canonical phase semantics ... but used a shorter representative template list with the same phase semantics."*

**This is the one substantive issue.** The paper-run headline number (`avg lin = 0.954`) was produced by the original §11 templates. Different templates will likely produce different numbers, failing C1's validation gate for **template-divergence reasons**, not convergence-optimization reasons.

**Recommendation for new session FIRST action:** copy verbatim §11 templates (specifically the chain definitions, template lists, and prior-construction logic from the original cell 11) into the new C1 cell before running C1's validation gate on pod.

**(3) C1 dual artifact naming — ACCEPTED.** Both `c1_canonical_lifecycle.json` (new naming) and `canonical_training_results.json` (legacy alias) written. Per handover spec.

### Missing artifacts flagged by agent

`mmlu_ibf_out/` locally does NOT contain D5b/D6/D7/D8 artifacts. They were generated on the pod but never pulled/committed. The S1 banner references them ("see `mmlu_ibf_out/fi_agency_channel_d6_alpha_vs_beta.json`") as if they exist locally.

**Need to address:** on next pod session, run from `/workspace/information-as-alignment`:
```bash
git add mmlu_ibf_out/fi_agency_channel_d5b_discovery.{json,md}
git add mmlu_ibf_out/fi_agency_channel_d6_alpha_vs_beta.{json,md}
git add mmlu_ibf_out/fi_agency_channel_d7_de_novo.{json,md}
git add mmlu_ibf_out/fi_agency_channel_d8_conflict_adjudication.{json,md}
git commit -m "Pull D5b/D6/D7/D8 artifacts from pod"
git push origin claude/review-jupyter-notebook-8AU5y
```
Then `git pull` locally.

If artifacts don't exist on the pod either (they were generated in-memory and not saved properly), regenerate by re-running cells 62 (D5b), 76 (D6), 78 (D7), 80 (D8) on the pod. Each is ~30-60 min.

### Agent's estimate for remaining build work

| Section | Est. build time |
|---|---|
| C2 (LoRA, no convergence loop) | ~30 min |
| C3 (Qwen, convergence protocol) | ~45 min |
| C4 (IBF vs kNN vs RAG, 3 baselines) | ~75 min |
| C5 (lifecycle: retract+delete+forget) | ~60 min |
| C6 (locality + scale, includes C-retention bug fix) | ~60 min |
| C7 (compiled closure, 3 subsections) | ~60 min |
| C8 (discovery + adjudication, 3 subsections) | ~60 min |
| S4 (paper deliverable, cN naming) | ~30 min |
| S5 (reproducibility + claim map) | ~20 min |
| **Total** | **~7-8 hours focused build** |

The handover's Day 2 + Day 3 estimate aligns.

---

## Session state at handover

### What got done this session

| Phase | Deliverable | Status |
|---|---|---|
| **Paper run analysis** | Phases A/B/C/D, retraction, LoRA, Qwen, §38 | Complete; results scored 8.5/10 |
| **D1** | Kernel-locality diagnostic | Result: AC at 5.68σ vs 3.03σ radius (geometric, not architectural) |
| **D2** | Static agency wirings (a)/(b)/(c) | All three add nothing when AC reachable; value saturates |
| **D3** | Phase 1 c.z[:64] proxy binding | Failed: 0/3896 centers near A; Phase 2 needed |
| **D4** | Phase 2 z_before storage | Agency still inert; coverage problem in addition to binding |
| **D5b** | Discovery training (scaffolded) | **0.000 → 1.000 in 4 epochs**; +0.125 BT to BC |
| **D6** | Agency gate α vs β (Reading C) | **β k_eff U-shape 4.66 → 2.58 → 3.23**; α flat at 5.000. Reading C validated. |
| **D7** | De novo emergence (no scaffold) | **Mean 8.09σ separation**; both engines reach 1.000 at ep 1. Mechanism robust. |
| **D8** | Compiled vs discovered Crucible adjudication | LOW resilience: collapse in 1 epoch. HIGH resilience: 25-28× resistance multiplier. Operator-tunable timescale validated. |
| **Supervisor notes** | `AGENCY_DISCRETIZATION_NOTE.md` + `C8_CLAIM_PROPOSAL_NOTE.md` | Both pushed; both received approval on the 4-claim-restructure items |
| **Handover doc** | `HANDOVER_NOTEBOOK_REBUILD.md` | Pushed; foundational anchoring added (Part 1.5 + per-card "Foundational anchor:" fields) |
| **Builder agent** | Launched in background | Running; will report back at C1 validation gate |

### The approved structure (canonical going forward)

**Central thesis:**
> *IBF is a local durable alignment substrate for frozen LLMs whose properties (decoupling, generality, distinction) support a clean operational lifecycle (install / revise / remove with preserved locality) which in turn supports complementary deductive and inductive composition paths — without modifying base-model weights.*

**Four-layer stack with renumbered claims:**

| New # | Old # | Layer | Statement (short) |
|---|---|---|---|
| **C1** | C1 | 1 — Existence | Local durable alignment without weight editing |
| **C2** | C5 | 2 — Property | Substrate decoupling under base evolution (LoRA) |
| **C3** | C7 | 2 — Property | Cross-model mechanism generality (Qwen) |
| **C4** | C6 | 2 — Property | Distinct from kNN/RAG |
| **C5** | C2 | 3 — Operation | Truth-maintenance lifecycle |
| **C6** | C3 | 3 — Operation | Override priors with preserved locality |
| **C7** | C4 | 4 — Composition | Compiled closure (deductive) |
| **C8** | C8 | 4 — Composition | Discovery-driven extension (inductive) |

D8 evidence is part of C8 (NOT a separate C9 / Layer 5). The Crucible adjudication is the same mechanism as C5's dissolution — not a new mechanism.

---

## Reading C engine patch (supervisor-approved, mandatory)

Add to `IBFParams`:
- `n_agency_min: int = 20` (history sufficiency threshold)
- `eta_k_cryst: float = 0.005` (slower learning rate for crystallized agency, parallel to eta_cryst for value)

In `_update_agency`, replace `if c.is_crystallized():` with `if len(c.D_history) >= self.p.n_agency_min:` for the w-update branch. Use `lr = self.p.eta_k_cryst if c.is_crystallized() else self.p.eta_k`.

In `delta_k`, replace `if not c.is_crystallized(): continue` with `if len(c.D_history) < self.p.n_agency_min: continue`.

Pre-merge validation: C1 + C3 must reproduce within ±0.01 on avg lin (C1) and ±0.02 on cross-model deltas (C3). Then re-save `canonical_engine.pkl` with `engine_version: "2.0-history_gate"` metadata.

---

## Where to pick up in the new session

### Immediate first action (in order)

```bash
cd /home/user/information-as-alignment
git fetch origin claude/review-jupyter-notebook-8AU5y
git pull --rebase origin claude/review-jupyter-notebook-8AU5y
git log --oneline -5  # confirm 25065cd + 9eb25e3 are present
ls -la "(IBF)Companion-LLM-Durable-Alignment-v2.ipynb"  # confirm v2 notebook exists
```

### Sequence of decisions for the new session

**Step 1: Fix the FI templates (priority before anything else).**
- The agent used representative templates, not verbatim §11 content
- Run: `jq -r '.cells[11].source | join("")' "(IBF)Companion-LLM-Durable-Alignment.ipynb"` to extract the original FI dataset cell
- Patch the v2 notebook's S3 section to use those verbatim templates
- This is the single most important fix; without it C1's validation gate will fail for template-divergence reasons rather than convergence-optimization reasons

**Step 2: Decide on the missing D-artifacts.**
- Either pull from pod (commit sequence above) — preferred
- Or soften S1 banner references to "will-produce" — fallback

**Step 3: Run C1 on pod in `RUN_MODE="verify-convergence"` mode.**
- If gate passes (avg lin within 0.954 ± 0.01): convergence optimization approved for C2-C8
- If gate fails: revert to `RUN_MODE="paper"` for C2-C8, no compute savings, but no claim weakening either

**Step 4: Launch builder agent for C2-C8 + S4 + S5.**
- ~7-8 hours of focused build time
- The handover spec (`HANDOVER_NOTEBOOK_REBUILD.md`) is the canonical specification
- Pass the agent the same context: handover doc + foundational paper sections + existing v2 notebook with corrected S3
- Stopping points: after C4 (end of Layer 2), after C6 (end of Layer 3), after C8 (end of Layer 4), after S5 (complete)

**Step 5: Run rebuilt v2 notebook end-to-end on pod.**
- Validate each claim's headline result against the reference values in the handover
- Commit + push all `cN_*.json` artifacts
- If everything passes: v2 notebook becomes the canonical artifact, paper draft begins

### Token-efficient prompt for the new session

> *"Continuing IBF-over-LLM notebook rebuild. Branch `claude/review-jupyter-notebook-8AU5y`. The previous session built S1+S2+S3+C1 of the v2 notebook (commits 25065cd, 9eb25e3). The full state is in `SESSION_HANDOVER_REBUILD.md`. The canonical spec is `HANDOVER_NOTEBOOK_REBUILD.md`. The agent flagged one substantive deviation: S3 uses representative templates instead of verbatim §11 content. First task: extract verbatim templates from the original notebook's cell 11 and patch v2's S3. Then run C1 on pod to validate the convergence gate, then launch builder agent for C2-C8 + S4 + S5."*

---

## Outstanding decisions (defer to next session if you want)

1. **C1 convergence-optimization gate.** Once C1 is built and run on the pod, validate against `avg lin = 0.954` within ±0.01. If passes, apply optimization globally. If fails, per-claim calibration.

2. **C6 scale-frontier C-retention bug fix.** Specified in C6 card. Builder should implement; user should verify on pod that the fix produces non-zero retention numbers (not byte-identical 0.85/0.353 across scales).

3. **Engine patch re-validation.** After Reading C is baked into S1, run C1 + C3 on pod under patched engine. Compare to canonical. If within tolerance, re-save canonical_engine.pkl with version stamp.

4. **Supervisor + physicist final scan of handover doc.** Optional. The 4-claim restructure was approved; the handover doc itself is one step further. If you want a final round of supervisor review before the builder finishes, send them `HANDOVER_NOTEBOOK_REBUILD.md`.

5. **Paper draft.** Pending end-to-end notebook rebuild + run. The claim cards in the handover doc are essentially the paper's section structure already — paper writes itself from there.

---

## Files of record on the branch

- `HANDOVER_NOTEBOOK_REBUILD.md` — the canonical spec for the rebuild
- `AGENCY_DISCRETIZATION_NOTE.md` — Reading C engine fix rationale (supervisor-approved)
- `C8_CLAIM_PROPOSAL_NOTE.md` — C8 evidence + claim proposal (supervisor-approved)
- `IBF_OVER_LLM_ARCHITECTURE.md` — architecture note from earlier session
- `PAPER_RUN_HANDOVER.md` — paper-run operator runbook (still valid for re-runs)
- `LAUNCH_RUNBOOK.md` — quick reference for pod operations
- `(IBF)Companion-LLM-Durable-Alignment.ipynb` — current paper-run notebook (do NOT modify; historical record)
- Foundational paper PDF: `(pre-print)information-as-alignment-v1.pdf` or `information-as-alignment-v1.pdf`
- All experiment artifacts: `mmlu_ibf_out/` (~58 JSON/MD files; preserved)

---

## Token-efficient session opener for the new session

When you start the next session, you can give it this prompt to get up to speed quickly:

> *"Continuing IBF-over-LLM notebook rebuild work. Branch `claude/review-jupyter-notebook-8AU5y`. The full session-handover is in `SESSION_HANDOVER_REBUILD.md`. The canonical spec is `HANDOVER_NOTEBOOK_REBUILD.md`. A builder agent was launched in background to build S1 + S2 + S3 + C1; check git log to see what was committed and what's still pending. The current task is to review the builder's S1+C1 output and decide on convergence-optimization global application before continuing to C2-C8."*

That's enough for the next session to orient.
