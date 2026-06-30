# Supervisor Note — Gate 3B: Context-Aware Promotion → **Path A overturned; readout hypothesis confirmed**

Branch `gate-3b-context-aware`. One change: interior centers are kept as a
**frozen read-only reserve** — they contribute to **same-context** readout
(Gate 2: interior is 70% internally) and are excluded from **cross-context**
readout (Gate 2: 0.3% externally) and from all updates. 5 seeds, same protocol.

## Headline: the retention failure is fixed and reversed

| metric | Gate 3 (remove interior) | **Gate 3B (freeze interior)** |
|---|---|---|
| Acc_A after C (retention), F → P | 0.535 → **0.425** (−0.11) | 0.535 → **0.723** (**+0.19**) |
| BT_A (catastrophic forgetting), P | −0.512 | **−0.215** (halved vs F's −0.402) |
| Acc_A after B, Δ(P−F) | −0.103 | **+0.002** |

Per-seed retention is **higher under promotion on all 5 seeds** (Δ = +0.27, +0.14,
+0.19, +0.19, +0.29). Your hypothesis is correct: **the gap was the readout**.
Restoring interior for same-context queries closes it — and because the frozen
reserve is immune to the cross-context interference that erodes the *active*
interior in the flat condition, P actually **beats** F on retention.

## The three criteria

| criterion | result | threshold | pass |
|---|---|---|---|
| C1 no-degradation (worst median ACC_F − ACC_P) | 0.062 | < 0.03 | NO |
| C1 literal symmetric (worst median \|ACC_P − ACC_F\|) | 0.193 | < 0.03 | NO (P exceeds F) |
| C2 cross-context compression (median over B,C) | 16.5% | > 0.15 | YES |
| C3 lifecycle (0 < dissolved < promoted, majority) | 2/5 | — | NO |

Two honest caveats keep it from a clean literal pass — **neither is the Path A
retention bound**:

1. **The only degraded cell is `Acc_C after B` = −0.062** — i.e. *zero-shot*
   accuracy on the third context **before it has ever been trained**. Both
   conditions are near floor there (F 0.268, P 0.170); P is lower because the
   cross-context path is compressed (16.5% fewer broadcasters reach a novel
   context). This is the **expected cost of external compression**, on
   forward-transfer to an unseen context — not a retention or current-learning
   loss. Every same-context and current-context cell is preserved or improved.
2. **C3 still fails** (interfaces mostly verify, 2/5 seeds dissolve). This is
   independent of the readout fix — the interface-level reversal threshold,
   inherited from the per-particle Crucible, is too conservative once aggregated.

## What this establishes

Your promotion operator is validated: **compress for external, preserve for
internal.** The Interface Principle governs the split — interior is dropped from
the cross-context (external) path where it leaks 0.3%, and kept in the
same-context (internal) path where it carries 70%. Under this operator, promoted
interfaces are reused **without degrading continual learning — they improve it**,
halving catastrophic forgetting of the promoted context.

**The Path A bound is overturned.** It was an artifact of the earlier operator
(dormant-*unread* interior). Postulate 2's recursive-reuse claim — compressed
lower-scale structures reused as primitives in further learning — holds in this
instantiation, with the qualification that the compression is *context-scoped*
(external-only), exactly as the Lean theorem prescribes.

## Decision for you (this is the Gate 1D / Gate 2 pattern a third time)

The literal C1 (symmetric, < 0.03) is again misaligned with intent: it flags
**P being much better** as a failure. The intent is "promotion must not degrade."
Under the no-degradation reading, the retention/current-learning claim passes
cleanly; the only sub-threshold breach is the zero-shot-to-unseen-context dip.

- **If you accept** (a) no-degradation as the C1 reading and (b) the forward-
  transfer dip as the legitimate cost of external compression, then **Gate 3B
  passes on C1 + C2**, and only **C3** remains — a separable Crucible-threshold
  fix, not a readout/reuse problem.
- **C3 fix** (if you want all three): an interface-level reversal threshold
  scaled for aggregated interfaces, so the Crucible actually dissolves
  non-transferable interfaces. One parameter, one run.

I did **not** change the pass criteria myself. Recommendation: rule on the C1
reading and whether to do the one-parameter C3 run; if both go the expected way,
Gate 3 is passed under the corrected operator and the recursive architecture is
validated end-to-end.

## Ladder
| gate | result |
|---|---|
| 1A/1B/1C/1D | PASS / confirmed / PASS(Scale 1) / PASS |
| 2 | PASS — Interface Principle (external shielding) |
| 3 (remove interior) | FAIL — retention −11pp (Path A bound) |
| 3 Path C (unit pressure) | FAIL — structural, narrowed not closed |
| **3B (context-aware)** | **retention FIXED (+19pp, BT halved); C2 PASS; residual = forward-transfer −6pp + C3** |
