# Session handover — pick up from notebook rebuild

**Date:** 2026-05-XX (end of session approaching token ceiling)
**Branch:** `claude/review-jupyter-notebook-8AU5y`
**Last commit at handover:** `d262849` (handover doc updated with foundational anchoring)

---

## What's running in the background

Builder agent launched (`general-purpose` subagent, ID `a07b70af413e2fe74`, background mode). It's building S1 + S2 + S3 + C1 of the new notebook structure per the handover spec, stopping at the C1 convergence-validation-gate checkpoint to report back.

**Where to find its work when the new session starts:**
- New notebook file: `/home/user/information-as-alignment/(IBF)Companion-LLM-Durable-Alignment-v2.ipynb` (or wherever it placed it — check git log)
- Commits on the branch: `git log --oneline` will show its commits since `d262849`
- Plan file: `/tmp/build_plan.md` (if it followed instructions)
- Background output (DO NOT read directly — too large for context): `/tmp/claude-0/.../tasks/a07b70af413e2fe74.output`

**Notification:** the agent's completion notification will fire in the OLD (current) session, not the new one. The new session won't see it. So in the new session, **check git log + the notebook file directly** to see what was built.

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

### Immediate first action

```bash
cd /home/user/information-as-alignment
git fetch origin claude/review-jupyter-notebook-8AU5y
git log --oneline -10  # see what the background agent committed
ls -lat *.ipynb         # check if v2 notebook was created
ls -lat *.md            # check if build_plan.md or new docs appeared
```

### If the agent finished and committed S1+S2+S3+C1

1. Read what the agent built. Verify it parses (`python3 -c "import ast; ast.parse(...)"`).
2. Review the agent's report (in commits + maybe a new markdown file).
3. Apply C1's convergence-validation-gate logic: if it's well-formed, approve continuation to C2-C8.
4. Relaunch builder agent (or extend the existing one if its context is still alive) with the next phase: "Build C2 through C8 + S4 + S5. Apply the convergence-stop protocol globally per Part 6 of the handover. Stop and report when complete."

### If the agent is still running / didn't finish

1. Check `/tmp/build_plan.md` for the planning output.
2. Check git for partial commits.
3. Consider waiting (the agent's output file persists; you can launch a new agent to "continue" with `Agent` tool's SendMessage if the ID is still valid: `a07b70af413e2fe74`).

### If the agent failed or got stuck

1. Read the error in git log / commit messages.
2. Relaunch with corrections.
3. The handover doc (`HANDOVER_NOTEBOOK_REBUILD.md`) is the canonical spec — work from it.

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
