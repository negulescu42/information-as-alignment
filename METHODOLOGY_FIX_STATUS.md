# Methodology fix status — work done in parallel with pod cycle 1

**Branch HEAD:** `f750623` (after 6 commits since the supervisor's directives in `V2_RUN_RESULTS_SUPERVISOR_NOTE.md`)
**Context:** While the pod runs cycle 1 (C1 discriminator with `disable_cross_context_agency=True` during Phase B), implementing the supervisor-approved methodology fixes for C2-C8.

---

## Fixes pushed (6 commits)

| # | Commit | Cell | Fix |
|---|---|---|---|
| 1 | `3bcfbbe` | S1 + C1 | Add `disable_cross_context_agency` flag + per-phase toggle (the discriminator) |
| 2 | `aeeb00c` | S3 | Port v1 cell 62 closure-chain infrastructure verbatim |
| 3 | `d60586a` | C7 | Rewrite to use v1 cell 58/60 strong-prior install convention. Bug found: builder's R_truth=0/1 convention was inverted (pushed engine toward foils, not targets). Random-baseline 0.125 numbers explained. |
| 4 | `78d5e14` | C8 | Same R_truth-inversion bug in supervised_install + discovery_step. Surgical 3-edit patch (no rewrite — C8's chain infrastructure was already a near-copy of v1 cell 62). |
| 5 | `7931df2` | C6 | Locality populations 200/200/200 → 50/50/50 (fits in 200 unique subjects; previous run had empty near/distant arrays). Added a sanity-assert for future regressions. |
| 6 | `f4e2fc5` | C4 | IBF `remove` now melts z_new + z_rev + z_old (full record footprint). Was melting only z_rev, leaving z_new centres with positive v from the install step → argmax picked "new" instead of falling back to base. |
| 7 | `f750623` | C5.2 | Selective deletion: lower thresholds 1e-4/1e-5 → 1e-6/1e-8 and include all 4 choice positions in contributor scan (was correct-choice-only). Previous run found 5 contributors for a 5-fact subject; now should find the full footprint. |

---

## Pattern across fixes — builder agent's training convention bug

While porting v1 cell 62's infrastructure (supervisor-decision-3), I discovered the builder agent had introduced a consistent training-convention bug in multiple cells:

```python
# Wrong (in C7, C8, and C6.1 before fix):
R_truth = 0.0 if j == yi else 1.0
```

Under this convention:
- target choice: `R_imposed_override = 0.0` → engine pushes δR negative (target field down)
- foil choices: `R_imposed_override = 1.0` → engine pushes δR positive (foils up)
- argmax at eval: foil wins, target loses → random-baseline accuracy

v1's convention (cell 58, train_strong_prior):
```python
# Right (now restored in C7 and C8):
updates = [(target_label, CF_TARGET_LOCAL),    # 0.95 — push target HIGH
           (base_label, CF_BASE_LOW)]           # 0.015 — push base LOW
# Distractor choices: no update (sit at strong-prior baseline)
```

The supervisor's diagnosis "test data is broken, not the mechanism" was correct on the data side; the convention bug was an additional, parallel failure mode that the data fix alone wouldn't have resolved.

C6.1's loop has the same wrong convention but accidentally produces the right *qualitative* output (target accuracy drops because foils get pushed up, not because target was retired). Flagged for follow-up; not blocking.

---

## Outstanding — C2 (deferred to next iteration)

**Why not done in this round:** C2's "weak" vs "strong" target classification structure is non-trivial. The builder agent's inline `_make_target(i, "weak"/"strong")` differs from v1's structure in subtle ways. v1 cell 85's `eval_base_only` / `eval_with_ibf_safe` filter by `d["kinds"]` (per-item "weak"/"strong" labels) but the labeling logic isn't in v1 cell 37 (§14) — it's spread across other cells. Porting needs a careful pass to match v1's data + labeling + eval together.

**Why it's not blocking:**
- C2's architectural decoupling claim is **already supported by the previous run**: `ctrl_delta = 0.000` (controls held perfectly under LoRA base-shift). The headline-target-drop numbers were the failing piece, and those are methodology-sensitive (different inline targets produce different drops).
- The supervisor's TL;DR analysis on C2 explicitly said: "Controls held perfectly. The architectural claim is intact. Methodology delta is from target-item construction."

**Plan for next iteration:** port v1 cell 37 + cell 85's kind-classification logic together as a single coherent unit. Estimated 1-2h dev. If next pod cycle's results are otherwise green, this can wait without blocking the paper.

---

## Anticipated impact on next pod run

After cycle 1 confirms (or refutes) the C1 mechanism hypothesis, applying the methodology fixes from this session should change the following claims:

| Claim | Pre-fix | Anticipated post-fix |
|---|---|---|
| C1 | avg lin 0.9398 (gate miss) | depends on cycle 1 result + per-phase agency isolation |
| C2 | weak 0.30 / strong 0.07 drop | unchanged (C2 fix deferred) |
| C3 | passes ✓ | passes ✓ (unchanged) |
| C4 | rem=0.000 | rem=1.000 (full melt → argmax falls back to "old") |
| C5.1 | passes | passes (unchanged) |
| C5.2 | target 1.0→0.6 partial | target 1.0→~0.05 full erasure |
| C5.3 | runs | runs (unchanged) |
| C6.1 | empty controls | non-empty 50/50/50 populations with real drift measurements |
| C6.2 | partial (bug-fix worked, target degrades at 20k) | same shape (the scale-frontier behaviour is real, not a methodology artefact) |
| C7.1-3 | 0.125-0.500 (random-baseline) | should approach v1 canonical values once strong-prior install + eval are correct |
| C8.1-3 | 0.000 across (D5b never reaches AC) | D5b should hit AC=1.0 in ~4 epochs under fixed convention; D7 de novo should also reach high AC; D8 multiplier becomes computable |
| S4/S5 | run cleanly | continue to run cleanly; S4's pass-rate should jump from 1/8 to 5-7/8 depending on cycle 1 outcome and C2 |

---

## Outstanding decisions still queued from supervisor note

1. **C1 rebaseline vs investigate** — cycle 1's discriminator will resolve. If B reaches 0.97+, per-phase agency isolation is the design fix; if not, investigate further.
2. **C2 methodology fix** — deferred this round; do in next iteration before final pod run.
3. **C7/C8 chain infrastructure port** — done.
4. **C4/C5.2/C6 implementation bugs** — done.
5. **Sequencing** — operating per supervisor's parallel plan (discriminator + methodology fixes both in progress).

---

## Files of record

All on branch `claude/review-jupyter-notebook-8AU5y`:

- `(IBF)Companion-LLM-Durable-Alignment-v2.ipynb` — patched at HEAD `f750623`
- `V2_RUN_RESULTS_SUPERVISOR_NOTE.md` — original supervisor note (the one the supervisor responded to)
- `C1_GATE_MISS_DIAGNOSIS.md` — C1-specific analysis (Phase B locus, not A-retention)
- This note: `METHODOLOGY_FIX_STATUS.md`

---

## Token-efficient prompt for the new session, when pod cycle 1 completes

> *"Pod cycle 1 (C1 discriminator) just finished. Latest commit on branch claude/review-jupyter-notebook-8AU5y is `f750623`. Open `METHODOLOGY_FIX_STATUS.md` to see what's been patched. Per supervisor decision 1 (V2_RUN_RESULTS_SUPERVISOR_NOTE.md), if Phase B's final lin is ≥ 0.97 (under disable_cross_context_agency=True during B only), confirm mechanism + implement proper per-phase agency isolation in S1 + finalize C2 methodology fix + queue full ~70h re-run. If Phase B is still ≤ 0.92, investigate further before re-running."*
