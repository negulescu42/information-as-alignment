# C1 paper-mode gate miss — Phase B's additive learning is being blocked

**Status:** Open diagnostic.
**Run:** C1 paper-mode (commit `9456053`, post-mode-collapse, post-Reading-C, post-strong-convergence-relaxation).
**Headline:** `avg lin = 0.9398` vs target `0.954 ± 0.01` → `WITHIN_TOLERANCE = False`, gate miss `−0.0144`.

> **Supersedes** `C1_A_RETENTION_DIAGNOSIS.md`. This rev also corrects framing errors in the previous version of this file:
> 1. Phase A's drop after D is not "retention degradation" — it's the *expected* truth-maintenance signature of D contradicting ~15% of A's test items.
> 2. Phase B is pure additive learning (new categories, no contradictions); fitting should be near-instantaneous. Our run failing to fit it is a *training-dynamics* failure, not "convergence at a lower equilibrium."

---

## What each phase is structurally doing

Understanding what the engine *should* be doing in each phase is necessary to read the per-phase numbers correctly.

- **A_Onboarding** — first-write: 1000 employees × 5 categories. No prior state. The engine builds value centres from scratch. Final A retention is the headline existence claim — durable alignment without weight editing.
- **B_Initiative** — pure additive: 200 employees × 2 NEW categories (certification, committee). **No overlap, no contradiction with A.** The engine should add new value centres for the new categories and fit them essentially immediately — the centres are alone in their region of representation space.
- **C_Reorg** — revision: 150 counterfactual manager/project changes. Contradicts A's manager/project facts for those 150 employees. The engine is expected to update — A's revised categories should drop on the affected items (truth-maintenance), unaffected items hold.
- **D_Turnover** — partial contradiction + additive: 30 employees leave (their ~150 facts retired) + 30 new hires (90 new facts). Phase A's test set has ~150 items referring to the departed; those become stale.

So **Phase A's test set after D**:
- ~150 of 1000 items reference contradicted state (departed employees) — expected to drop
- ~100 of 1000 items hit the C-revised manager/project facts — expected to drop on stale ones
- Remaining ~750 items should hold

Natural-truth-maintenance drop estimate for A's aggregate post-D: roughly 0.10-0.15. A drop of zero would mean the engine isn't retiring contradicted information — a different failure mode (no truth-maintenance). A drop deep into the 0.20+ range would suggest broader interference.

**Phase B's behaviour post-A** (i.e., before any B-specific training):
- B's certification and committee categories are entirely new; no value centres exist there yet.
- After A trained, the engine has 0 value centres in B's representational region.
- One epoch of B training should create the centres and fit them — they're alone, no competing signal.
- Expected B aggregate after 1-2 epochs: 0.97+.

---

## Per-phase finals: Reading-C run vs original canonical

Reconstructed from the prior dev-run's `forgetting_diagnostic_report` (Cell 25):

| Phase | Current run | Original canonical | Δ | Reading |
|---|---:|---:|---:|---|
| A_Onboarding | 0.880 | 0.850 | +0.030 | Both in natural-TM range; small delta is contradiction-handling variance, not "better retention" |
| **B_Initiative** | **0.8925** | **0.980** | **−0.0875** | **Both should be near-1.0 (pure additive); canonical is, ours isn't** |
| C_Reorg | 0.9867 | 0.9867 | 0.000 | Identical |
| D_Turnover | 1.000 | 1.000 | 0.000 | Identical (just trained) |
| **avg** | **0.9398** | **0.954** | **−0.0144** | |

Contribution decomposition of the −0.0144 gate miss (per-phase Δ ÷ 4):

| Phase | Contribution | Reading |
|---|---:|---|
| A | +0.0075 | Natural-TM variance; should not be characterised as "Reading C helping A" |
| **B** | **−0.0219** | **The actual miss source — B failing to fit new info during training** |
| C | 0.000 | |
| D | 0.000 | |
| Net | −0.0144 ✓ | |

**The structural anomaly is B's training failure.** A's behaviour is normal truth-maintenance for both runs. C and D are unchanged.

---

## Within-phase training dynamics — original canonical's B fits in 1 epoch

From the prior dev-run's Cell 25, ANALYSIS 6 (within-phase deltas, ep 1 → 100):

```
B_Initiative (epoch 1 → 100):
  certification : 0.975 → 0.970 (Δ=-0.005, range=[0.970,0.975])
  committee     : 0.995 → 0.995 (Δ=+0.000, range=[0.995,0.995])
```

By ep 1 the original engine had already fit certification at 0.975 and committee at 0.995. Subsequent 99 epochs barely changed the values. **That's the correct signature of pure-additive learning: the centres get built once, they fit, the trajectory is flat.**

Our run for B in smoke (the early evidence):
```
Phase B Ep 1: lin = 0.8925
Phase B Ep 2: lin = 0.8950
```

And in paper mode, 85 epochs to 0.9025. Same pattern: B starts near 0.89 and barely moves. **B isn't learning to fit the new info.** Our centres for certification and committee are getting built (val centre count grows from 3850 to 5424 across A→B), but they aren't crystallising properly to produce the correct readout.

---

## Mechanism hypothesis — cross-context agency interference is blocking B's value-centre formation

Reading C's history-sufficiency gate makes Phase A's crystallised agency centres operationally engaged from the moment they have 20+ history items. By the time B starts, A has ~1300 crystallised agency centres ready to fire.

The interference path during a B training query `z`:

1. `_update_agency` runs the cross-context loop: `for c in [c for c in self.agency_centers if c.is_crystallized() and c.context_id != ctx]`. The A-context agency centres now satisfy this filter (Reading C unlocked them). Each fires `kernel_batch(z, [c])` and appends to `D_history_cross`.
2. When B's value centres do their `delta_R(z)` and `compute_D_and_update(z)`, the agency-derived `δk(z)` is now non-zero — A-context agency centres' `w` values contribute via the patched `delta_k`. The responsiveness modulation gets applied to B's value-centre updates.
3. B's value centres can't crystallise cleanly because their D-trajectory is being perturbed by responsiveness modulation from A-context centres that have no semantic relationship to B's data.

This same mechanism would produce:
- **The 8.75-point accuracy gap** — B's value centres get D-history that includes responsiveness perturbations, so the kernel readout `delta_R` doesn't converge to a clean +1 / −1 on B's test items.
- **The 2.5× per-epoch slowdown** — every B query now triggers the full A-context agency loop (~1300 kernel evaluations against A centres), where in the original engine those were structurally silent.

Both symptoms come out of the same Reading-C-side-effect.

---

## What's still consistent with the existing evidence

This is *not* an indictment of Reading C. The D6 empirical validation (`mmlu_ibf_out/fi_agency_channel_d6_alpha_vs_beta.json`) shows the β k_eff U-shape under Reading C tracking D-variance dynamics — exactly the "parallel equation" behaviour Postulate 1 specifies. D5b's emergent A→C closure and D7's de novo emergence both depend on Reading C's agency engagement. The agency channel being operationally engaged is the *point*.

The unanticipated side-effect is specifically the **cross-phase interference during multi-phase training**: A's agency centres operate on B's training contexts, perturbing B's learning. Within a single phase (single training context) Reading C produces the desired behaviour.

---

## Recommended diagnostic — confirm the mechanism

Add a debug flag to S1's engine that disables the cross-context agency broadcast loop. Specifically:

```python
# In IBFParams, add:
disable_cross_context_agency: bool = False

# In _update_agency, gate the cross-context loop:
if not self.p.disable_cross_context_agency:
    for c in [c for c in self.agency_centers if c.is_crystallized() and c.context_id != ctx]:
        ...
```

Run C1 with the flag set to True only during Phase B's training, restored to False for A/C/D. If B reaches the canonical 0.97-0.98 within 1-2 epochs and the post-D B aggregate lands at ~0.98, the cross-context-agency mechanism is confirmed and the design choice is clean.

Estimated cost: ~15-18h (A unaffected, B converges fast, C/D unchanged).

---

## Design implications if the mechanism is confirmed

1. **Accept 0.94 as the Reading-C baseline.** Cross-context agency engagement is the point of the patch. The original 0.954 was an artifact of the unlocked agency channel. Rebaseline `HEADLINE_AVG_LIN` and document the engine-version distinction in the paper.
2. **Phase-scoped agency engagement.** Disable cross-context broadcast during training, enable during evaluation only. Probably defeats the agency channel's whole purpose.
3. **Per-phase agency reset.** When a new phase context begins, freeze previous-phase agency centres (no further updates, no cross-context firing). Preserves single-phase Reading-C semantics without cross-phase interference. Worth a follow-up if (1) isn't acceptable.
4. **Tighten the cross-context gate.** Require not just `is_crystallized() and context_id != ctx` but also a semantic-proximity check between the broadcast centre and the query. Adds complexity but might preserve both Reading C's agency engagement and B's clean training.

My read: option (1) remains the right call given D5b/D6/D7 validation, but option (3) is the natural fallback if the paper needs to land closer to 0.954.

---

## Files / references

- **Comparison source**: prior dev-run's `mmlu_ibf_out/forgetting_diagnostic_report.{json,md}` — Cell 25 truth-maintenance breakdown of the original canonical. Reproduces avg lin = 0.954 from per-phase finals.
- **Current run**: `mmlu_ibf_out/c1_canonical_lifecycle.json` (commit `9456053` engine version `2.0-history_gate`).
- **Reading C rationale**: `AGENCY_DISCRETIZATION_NOTE.md`, D6 validation in `mmlu_ibf_out/fi_agency_channel_d6_alpha_vs_beta.json`.
- **Earlier (mis-framed) diagnoses (superseded)**: `C1_A_RETENTION_DIAGNOSIS.md` and the previous rev of this file (in git history at commit `bfc75ae`).
- **Convergence criterion**: `check_strong_convergence` in C1, `ma_delta < 0.001 AND |slope| < 0.0001` over last-10-evals window (commit `d3c8a1c`).
- **Mode-collapse commit**: `9456053` (paper now uses early-stop by default).
