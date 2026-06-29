# Supervisor Note — Gate 3: Promotion & Recursive Interaction → **FAIL** (the anticipated recursion boundary)

Branch `gate-3-promotion`, all committed and pushed. Three contexts (A:+1, B:−1,
C: partial overlap), conditions F/P/N, 5 seeds.

## Result

| criterion | result | threshold | pass |
|---|---|---|---|
| C1 behavioral preservation | worst median \|ACC_P−ACC_F\| = **0.110** | < 0.03 | **NO** |
| C2 compression | 16.2% (median over B,C) | > 0.15 | YES |
| C3 lifecycle (0 < dissolved < promoted) | 2/5 seeds | majority | **NO** |

**VERDICT: GATE 3 FAIL.**

## The mechanism — and it's clean

The F-vs-P accuracy gap is **entirely retention**, not current learning:

| after C | context A (retention) | context B | context C (current) |
|---|---|---|---|
| F | 0.535 | 0.927 | 0.688 |
| P | 0.425 | 0.920 | 0.705 |
| Δ (P−F) | **−0.110** | −0.003 | +0.010 |

Promotion preserves the ability to *learn the new context* but worsens
*forgetting of the promoted one* (BT_A: −0.402 → −0.512).

**Why:** dormant interior is excluded from readout, so prior-context retention
rests on the boundary alone, which gets diluted by B/C training. **Gate 2's
static external-shielding (interior negligible for immediate readout) does not
imply the interior is dispensable for dynamic reuse across interfering
contexts.** That distinction — static compression vs dynamic reuse — is the
finding.

This is exactly the "Gate 3 failing" scope statement from the spec: *Compression
is possible (Gate 2) but compressed structures cannot participate as active
primitives in further learning. Postulate 2's recursive claim is not validated.
This bounds the theory to single-scale dynamics with static compression.*

Two specific diagnostics from the spec confirmed:

- **C1 fail via BT_A degradation** — predicted: "if BT_A degrades, the promoted
  interface doesn't carry enough information to survive dormant epochs." It does
  (−0.11).
- **C3 near-inert** — interfaces mostly *verify* (3/5 seeds dissolve 0), so the
  interface-level Crucible barely engages. The reversal threshold inherited from
  the particle Crucible looks too conservative once aggregated to whole
  interfaces.

C2 (compression) decays as new contexts add centers: 52% after A → 20% after B →
12% after C. Real but shrinking.

## What I did and one caveat

- Implemented promotion, dormant interior, interface-level Crucible (aggregating
  the **unmodified** per-center Crucible) with interior restoration on
  dissolution, all three conditions, reusing the validated
  engine/encoder/Gate-2 code unchanged.
- **Config deviation (flagged):** E_phase=20, not the spec's 120, for
  tractability. The *direction* (P < F on retention) is consistent across all 5
  seeds, so the qualitative result is robust; magnitudes could shift with more
  training.

## Decisions / next levers (yours)

1. **Accept the bound and write it up?** The program now has a clean three-part
   story: representation-through-interaction (1D ✓), static interface
   compression with external shielding (2 ✓), recursive reuse (3 ✗). A complete,
   publishable arc with a sharp negative.
2. **Or attack the promotion design** (the likely culprit, not the principle):
   the failure is caused by making interior *dormant-and-unread*. A promotion
   operator that folds a **compressed summary of interior contribution** into the
   interface readout (instead of discarding it) might preserve retention at lower
   compression. That changes the promotion operator, which the spec fixed — so it
   needs approval as a Gate 3b.
3. **C3 fix** is independent: an interface-level reversal threshold (less
   conservative than the per-particle one) would let the Crucible actually
   engage. Also a spec change.

## Full ladder

| gate | result |
|---|---|
| 1A | PASS (geometry-easy, not causal) |
| 1B | geometry-only insufficient (confirmed) |
| 1C | PASS at Scale 1; encoder wall exposed |
| 1D | PASS — interactive encoding |
| 2 | PASS — Interface Principle (external shielding) |
| **3** | **FAIL — dynamic reuse lossy (retention −11pp)** |

Eight branches pushed, every gate documented with diagnostics.

Open question: (a) draft a Gate 3b proposal (interior-summary promotion +
interface-level Crucible threshold), or (b) stop and leave Gate 3 as the
documented recursion bound pending review?
