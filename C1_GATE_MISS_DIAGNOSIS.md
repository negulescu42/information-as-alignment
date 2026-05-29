# C1 paper-mode gate miss — locus is Phase B underperformance, not A-retention

**Status:** Open diagnostic, revised after canonical reference data became available.
**Run:** C1 paper-mode (commit `9456053`, post-mode-collapse, post-Reading-C, post-strong-convergence-relaxation).
**Headline:** `avg lin = 0.9398` vs target `0.954 ± 0.01` → `WITHIN_TOLERANCE = False`, gate miss `−0.0142`.

> **Supersedes** the earlier `C1_A_RETENTION_DIAGNOSIS.md`, which mis-attributed the miss to A-retention. Once the original canonical's per-phase finals (from the prior dev-run's Cell 25 truth-maintenance diagnostic) were available, the comparison flipped: A is *retaining better* under Reading C than the original engine did. The miss is **B's own training underperformance**.

---

## Per-phase comparison: Reading-C run vs original canonical

Reconstructed from the prior dev-run's `forgetting_diagnostic_report` (Cell 25):

| Phase | Current Reading-C run | Original canonical (pre-Reading-C) | Δ (current − canonical) |
|---|---:|---:|---:|
| A_Onboarding | 0.880 | 0.850 | **+0.030 (current better)** |
| B_Initiative | 0.8925 | 0.980 | **−0.0875 (current worse)** |
| C_Reorg | 0.9867 | 0.9867 | 0.000 |
| D_Turnover | 1.000 | 1.000 | 0.000 |
| **avg** | **0.9398** | **0.954** | **−0.0144** |

Contribution decomposition of the −0.0144 gate miss (per-phase Δ ÷ 4):

| Phase | Contribution to miss |
|---|---:|
| A | +0.0075 (reducing the miss — Reading C is helping A retention) |
| **B** | **−0.0219 (creating the miss)** |
| C | 0.000 |
| D | 0.000 |
| Net | −0.0144 ✓ |

**Phase B is the entire gate-miss source.** A is doing better than the canonical reference, B is doing 8.75 points worse than it.

---

## Within-phase training dynamics — original canonical's B was nearly instantaneous

From the prior dev-run's Cell 25, ANALYSIS 6 (within-phase deltas, ep 1 → 100):

```
B_Initiative (epoch 1 → 100):
  certification : 0.975 → 0.970 (Δ=-0.005, range=[0.970,0.975])
  committee     : 0.995 → 0.995 (Δ=+0.000, range=[0.995,0.995])
```

The original engine fit B's data essentially within the first epoch — by ep 1, certification was already 0.975 and committee 0.995. The remaining 99 epochs barely changed those values.

Our Reading-C run can't get past ~0.90 on B even with 85 epochs and a stabilized slope (`|slope| < 0.0001` triggered the early-stop). Same engine code apart from Reading C, same dataset, same prompts.

**Reading C is impairing B's value-centre formation in real time, not just degrading its retention afterwards.**

---

## Revised candidate reads

### Reading 1 (revised) — Reading C interferes with Phase B's training

When B starts, Phase A's crystallised agency centres are already in place (~1300+ of them). With the original engine they were structurally silent — `_update_agency`'s w-update was gated on `is_crystallized()`, which the contrastive install scheme never satisfied for the cross-context broadcast loop.

Reading C's history-sufficiency gate (`len(c.D_history) >= n_agency_min`) makes those A-context crystallised agency centres operationally engaged. During B's training:

1. Cross-context loop fires: for each B-context query, every crystallised A-context agency centre is queried, increments `n_updates`, and contributes its `D_history_cross`.
2. `delta_k(z)` for B's queries integrates over A-context agency centres' `w` values (via the patched history-sufficiency gate, which is now satisfied for A-context centres).
3. The non-zero `δk(z)` from A-context agency centres applies a responsiveness multiplier to B's value-centre formation that wasn't present in the original engine.

That's the mechanism for both the 2.5× slowdown (more work per query) AND the 8-point accuracy gap (responsiveness modulation prevents B's centres from settling at high accuracy).

**If true:** the headline target of 0.954 was a pre-Reading-C number. Reading C is operating as designed — modulating responsiveness across contexts — but the design has a side-effect on multi-phase lifecycle training that wasn't anticipated. The 0.94 result is the new canonical for the Reading-C engine.

### Reading 2 (invalidated) — A under-entrenched

The original diagnosis suggested A stopped too early (ep 110 vs canonical 300) and lost retention to C/D. The canonical reference data invalidates this: A's retention drop in our run (−0.077) is *less* than the canonical's (−0.106). A is fine. Bumping `MIN_EPOCHS_PER_PHASE['A_Onboarding']` would not help.

### Reading 3 (folded into Reading 1) — Phase B slowness

The 2.5× per-epoch slowdown in B is the same mechanism as Reading 1: cross-context agency engagement adds per-query work. The slowdown and the underperformance are two symptoms of one Reading-C side-effect.

---

## Recommended next action

**Mechanism-isolating diagnostic:** disable the cross-context agency loop during B's training and confirm B reaches the canonical 0.98.

Specifically, gate `_update_agency`'s `for c in [c for c in self.agency_centers if c.is_crystallized() and c.context_id != ctx]:` cross-context broadcast on a debug flag, and run C1 with the flag set to False just during B's phase. If B then reaches ~0.98 within a few epochs (canonical behaviour), Reading 1 is confirmed and the Reading-C side-effect is localised to cross-context agency engagement during multi-phase training.

Estimated cost: re-run C1 with the patch ≈ ~15-18h (same as a clean run, since A unaffected and B should now converge fast).

**If Reading 1 is confirmed**, the design choices are:

1. **Accept the Reading-C baseline of 0.94.** The cross-context agency engagement IS the point of the patch. The original 0.954 was an artifact of the under-implemented agency channel. Rebaseline `HEADLINE_AVG_LIN = 0.94`, tighten or loosen tolerance, document the engine-version delta.
2. **Phase-scoped agency engagement.** Disable cross-context broadcast during training, enable only during evaluation. Probably defeats the point of the patch (agency needs to update during training to learn anything). Likely the wrong call.
3. **Per-phase agency reset.** Crystallised agency centres from phase N get "frozen" (no further updates) when phase N+1 begins. Preserves the Reading-C semantics within a phase but stops cross-phase interference. Worth a follow-up if (1) isn't palatable.

My read: option (1) is the right call. The whole point of Reading C is to make the agency channel operationally engaged. If that engagement reduces multi-phase lifecycle headline by 1.4 points but produces the empirically validated D6 U-shape and D5b/D7 emergence results, that's a feature, not a bug. The paper should reflect the engine version distinction explicitly.

---

## Files / references

- **Comparison source**: prior dev-run's `mmlu_ibf_out/forgetting_diagnostic_report.{json,md}` — Cell 25 truth-maintenance breakdown of the original canonical. Reproduces avg lin = 0.954 from per-phase finals.
- **Current run**: `mmlu_ibf_out/c1_canonical_lifecycle.json` (commit `9456053` engine version `2.0-history_gate`).
- **Reading C rationale**: `AGENCY_DISCRETIZATION_NOTE.md`, D6 validation in `mmlu_ibf_out/fi_agency_channel_d6_alpha_vs_beta.json`.
- **Original diagnosis (superseded)**: `C1_A_RETENTION_DIAGNOSIS.md` — misidentified A-retention as the culprit before canonical reference data was available.
- **Convergence criterion**: `check_strong_convergence` in C1, `ma_delta < 0.001 AND |slope| < 0.0001` over last-10-evals window (commit `d3c8a1c`).
- **Mode-collapse commit**: `9456053` (paper now uses early-stop by default).
