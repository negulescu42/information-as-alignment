# C1 paper-mode gate failure — A-retention drop after C/D training

**Status:** Open diagnostic. Gate missed by ~1.4 points beyond tolerance.
**Run:** C1 paper-mode (commit `9456053`, post-mode-collapse, post-Reading-C, post-strong-convergence-relaxation).
**Headline:** `avg lin = 0.9398` vs target `0.954 ± 0.01` → `WITHIN_TOLERANCE = False`.

---

## Per-phase trajectory

End-of-phase values (mid-lifecycle, before subsequent phases interfere):

| Phase | End-of-phase lin | Notes |
|---|---:|---|
| A_Onboarding | **0.957** | Stops at strong-conv around ep ~110 (min_epochs floor) |
| B_Initiative | 0.9025 | |
| C_Reorg | 0.9867 | |
| D_Turnover | 1.000 | |

Final values (post-D, what the gate evaluates):

| Phase | Final lin | Δ from end-of-phase |
|---|---:|---:|
| A_Onboarding | **0.880** | **−0.077** |
| B_Initiative | 0.8925 | −0.010 |
| C_Reorg | 0.9867 | 0.000 (just trained) |
| D_Turnover | 1.000 | 0.000 (just trained) |
| **avg** | **0.9398** | gate miss = −0.014 |

**The entire 1.4-point miss is A-retention degradation during C+D training.** A loses 7.7 points; B/C/D are stable.

---

## Three candidate reads

### Reading 1 — Reading-C-engine side-effect (rebaselining hypothesis)

The history-gated agency channel (Reading C, S1) now actively modulates throughout C/D training. Agency centre count growth across phases:

```
end A → 1773 agency centres
end B → 1781
end C → 2016
end D → 2153
```

That's new interference pressure on A's value centres that the original *crystallization-gated* engine didn't apply. The original canonical's `avg lin = 0.954` was measured under the OLD agency-channel discretization, where agency was structurally silent in the LLM-closure regime. Under Reading C the agency channel is *operationally engaged* (the whole point of the patch — see `AGENCY_DISCRETIZATION_NOTE.md`), and that engagement appears to actively reshape value-centre dynamics during later phases.

**If true:** `0.9398` IS the new Reading-C baseline. The 0.954 anchor was an artifact of the under-implemented agency channel. The HANDOVER's headline tolerance was pinned to an engine that no longer exists. Rebaseline.

**Implication for the paper:** C1 reproduces durable alignment to within an absolute lin of ~0.94 (vs base ~0.22), still a +0.72 absolute improvement. The Reading-C engine is now the canonical engine and 0.94 is the canonical number.

### Reading 2 — Convergence-stop on A under-entrenches

Phase A stopped at the `MIN_EPOCHS_PER_PHASE['A_Onboarding'] = 100` floor with `slope = +0.0000109` (i.e., still drifting slowly up). The original canonical (no early-stop, ran to hard cap 300) had ~200 additional epochs of A-training to deepen A's centres' crystallization mass, |v| values, and D-history sufficiency. Deeper-entrenched A centres might survive C/D's interference where shallower ones don't.

**Easy test:** rerun with `MIN_EPOCHS_PER_PHASE['A_Onboarding'] = 200`. Phase A runs ~6h instead of ~3h. If A retention holds after C/D under min_ep=200, then the convergence-stop floor was too aggressive for A specifically.

**If true:** bump A's floor to 200; B/C/D stay at 50. No engine change. Gate likely lands within tolerance.

### Reading 3 — Phase B is doing heavier work than canonical (separate concern)

Phase B ran 85 epochs at 700–800s/epoch — **2.5× the smoke-run pace**. Smoke gave ~510 s/ep; paper-mode B is ~750 s/ep. Same item count (4000), same engine code. The slowdown implies extra per-item work: more agency updates firing, more dissolutions, more cross-context broadcast loop iterations, or more new-centre creations during B.

This isn't a gate-blocker — B's final lin is stable (0.9025 → 0.8925, only −1 point loss). But the perf delta is a real engine-perf signal worth investigating separately. Candidates: (a) Reading C's history-sufficiency gate causes more centres to cross the agency-update threshold during B (new behavior); (b) FAISS still isn't wired into the engine hot path (known); (c) something else.

**Doesn't gate the paper. Defer to Phase 2 engine-perf work.**

---

## Recommended next action

**Run Reading 2's diagnostic first** — it's the cheap deterministic discriminator:

1. Edit C1's mode dispatch to set `MIN_EPOCHS_PER_PHASE['A_Onboarding'] = 200`.
2. Re-run C1 in paper mode.
3. **If avg lin lands within 0.954 ± 0.01:** Reading 2 wins. Bump the floor permanently for A; keep the headline target at 0.954.
4. **If A still drops 7+ points after C/D:** Reading 1 wins. The new canonical is ~0.94. Rebaseline the HEADLINE_AVG_LIN constant + the gate tolerance, update C1's banner, document the engine-version delta from the paper-run's 0.954.

Wall-clock for the discriminator: A ~6h + B ~9h + C ~4h + D ~2h ≈ **~21h** (vs full re-run ~30h). Deterministic answer either way.

Phase B perf is a **separate** Phase 2 work item; it doesn't gate the paper and the diagnostic above doesn't need it fixed to resolve.

---

## Files / references

- Run artifact: `mmlu_ibf_out/c1_canonical_lifecycle.json` (current paper-mode result; see `validation_gate.measured_avg_lin = 0.9398`)
- Engine version: `2.0-history_gate` (Reading C patch applied; `(IBF)Companion-LLM-Durable-Alignment-v2.ipynb` cell 4)
- Convergence criterion: `check_strong_convergence` in C1, `ma_delta < 0.001 AND |slope| < 0.0001` over last-10-evals window (commit `d3c8a1c`)
- Mode-collapse commit: `9456053` (paper now uses early-stop by default)
- Reading C rationale: `AGENCY_DISCRETIZATION_NOTE.md`
