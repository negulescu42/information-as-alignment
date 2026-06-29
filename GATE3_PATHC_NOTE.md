# Supervisor Note — Gate 3 Path C: one change, decisive answer → **structural (Path A)**

Branch `gate-3-promotion`. Implemented the one change you specified: the promoted
interface absorbs cross-context pressure **as a unit**. Re-ran Gate 3, 5 seeds.

## The change (and a correction worth flagging)
Engine `update()` Crucible block now normalizes the cross-context v-update within
each promoted interface. My first cut used `eff_kw = kw / sum_kw` (interface total
= D), which **boosts a lone boundary center** (kw≈0.5 → 1.0) and made P *worse*.
Corrected to your stated intent — **one peak center's** worth of pressure:

```
eff_kw = peak_kw * kw / sum_kw      # interface total = D * peak_kw
```

This never exceeds the flat path, equals flat when a single boundary center is
activated, and only attenuates when several co-activate. Detection signal
(`D_history`, the reversal test) left **raw** so the Crucible is not weakened.
Guarded by `interface_group` → exact no-op for the flat condition and Gates 1/2
(regression still PASS).

## Result: the gap narrowed but did NOT close

| C1 metric (median P − F) | no Path C | **Path C** | target |
|---|---|---|---|
| ACC_A after C (late retention) | −0.110 | **−0.065** | |
| ACC_A after B (first transition) | −0.103 | −0.100 | |
| **worst gap (C1)** | 0.110 | **0.100** | < 0.03 |
| BT_A (condition P) | −0.512 | −0.470 | |
| ACC_A after C: F / P | 0.535 / 0.425 | 0.535 / **0.470** | |

C2 compression 15.8% (PASS); C3 lifecycle 2/5 seeds (FAIL, unchanged).

**Verdict: GATE 3 FAIL persists.** Per your decision rule ("if the gap closes
below 0.03 the concept is sound; if it persists the buffering is structural and
Path A is correct"), **the answer is Path A.**

## What the partial improvement tells us
Your diagnosis was **half right**, and the data now separates the two components:

1. **Late-phase drift (after C): real and bufferable.** Path C halved this gap
   (−0.110 → −0.065) and improved BT_A (−0.512 → −0.470). So pressure
   concentration *was* a genuine contributor to the slow drift across Phase C —
   exactly your mechanism.
2. **First-transition damage (after B): structural, not bufferable.** The B/A gap
   is ~0.100 with and without Path C. Pressure normalization during Phase B does
   not touch it. This is the part that fails C1, and it is **not** v-drift —
   normalizing the cross-context v-update leaves it unchanged.

The residual is the readout deficit made dynamic: with interior dormant and
unread, the A-correction at the first context switch rests on the boundary alone,
and the single B-phase is enough to expose that deficit (~10pp), independent of
how the cross-context pressure is distributed. Buffering protects what remains;
it cannot replace what was removed from the readout.

## Conclusion
**Path A.** The Interface Principle gives faithful **static** compression (Gate 2,
external shielding 0.3%), but the promoted boundary-only interface is **not a
sufficient operational primitive for dynamic reuse**: it loses ~10pp of
prior-context retention at the first context switch, and that loss is structural
(survives the unit-pressure fix). Postulate 2's recursive-reuse claim is **not
validated** in this instantiation; the theory is bounded to single-scale dynamics
plus static compression.

The one clean lever that *might* recover it is no longer "buffer the pressure" but
"don't remove the interior from readout" — i.e. fold a compressed **summary of
interior contribution** into the interface's readout (not just its boundary
centers). That changes the promotion *operator* (a Gate 3b), not the Crucible, and
is the natural next proposal if you want to keep pushing. Otherwise the bound
stands and is publishable as written.

## Final ladder
| gate | result |
|---|---|
| 1A / 1B / 1C / 1D | PASS / confirmed / PASS(Scale 1) / PASS |
| 2 | PASS — Interface Principle (external shielding) |
| **3** | **FAIL (structural)** — static compression OK; dynamic reuse loses ~10pp retention, survives the unit-pressure fix |
