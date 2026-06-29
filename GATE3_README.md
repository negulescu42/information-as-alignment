# Branch: `gate-3-promotion`

## Goal
Test whether **promoted interfaces** (boundary-only compressed Gate-2 basins) can
participate in further continual learning as first-class primitives — the
recursive-reuse claim of Postulate 2. Three sequential contexts (Phase A:+1,
B:−1, C: partial overlap), three conditions:

- **F (flat)** — standard engine, no promotion (Gate 1D/2 baseline)
- **P (promoted)** — basins → boundary-only interfaces after Phase A; interior
  dormant; interface-level Crucible across B and C
- **N (no prior)** — fresh agent each phase (lower-bound floor)

Fixed: Scale 1, interactive encoder, k=8 generator at f=3.0, Crucible /
crystallization params, Gate 2 basin code. (Config reduced from the spec's 120
epochs to E_phase=20 for tractability; the result direction is consistent across
all 5 seeds.)

## Run
```bash
python gate3_promotion.py 5
python summarize_gate3.py
```

## Result (5 seeds) — **GATE 3 FAIL**

| criterion | result | threshold | pass |
|---|---|---|---|
| C1 behavioral preservation (worst median \|ACC_P−ACC_F\|) | **0.110** | < 0.03 | NO |
| C2 compression (median over B,C) | 16.2% | > 0.15 | YES |
| C3 lifecycle (0 < dissolved < promoted, majority seeds) | 2/5 | — | NO |

## The mechanism (this is the finding)
The F-vs-P gap is **entirely in retention of prior contexts**:

| after phase | context | F | P | Δ(P−F) |
|---|---|---|---|---|
| C | **A** (retention) | 0.535 | 0.425 | **−0.110** |
| C | B | 0.927 | 0.920 | −0.003 |
| C | C (current) | 0.688 | 0.705 | +0.010 |

Current-context accuracy (B-after-B, C-after-C) is fully preserved; only
*retention* of the promoted context degrades. Promotion worsens catastrophic
forgetting: **BT_A = −0.402 (F) → −0.512 (P)**.

**Why:** the dormant interior is excluded from readout, so prior-context
retention rests on the **boundary only**, which is then diluted/overwritten by
B/C training. Gate 2 proved interior is negligible for *static, immediate*
external readout — but that does **not** imply it is dispensable for *dynamic*
reuse across interfering contexts. The interior carried redundancy that, in the
flat condition, keeps Phase A alive through later phases.

C2 compression decays as new contexts add centers (52% after A → 20% after B →
12% after C; median over B,C = 16.2%, just above the bar). C3: promoted
interfaces mostly **verify** rather than dissolve (3/5 seeds dissolve 0), so the
interface-level Crucible barely engages and interior is rarely restored — the
promoted level is near-inert under this reversal threshold.

## What this establishes (per the spec's "Gate 3 failing")
The modification dynamics work **within** a scale and support **static**
compression (Gate 2), but the compressed boundary-only structures **cannot fully
participate as active primitives in further learning** — reuse across interfering
contexts is lossy. **Postulate 2's recursive-reuse claim is not validated in this
computational instantiation.** The theory is bounded to single-scale dynamics
with static compression. This is a publishable scope statement.

## Honest caveats / next levers (not pursued — would need supervisor approval)
- **Promotion design is the likely culprit, not the principle.** Making interior
  *dormant-and-unread* is what costs retention. A variant that keeps a compressed
  *summary* of interior contribution in the interface readout (not just boundary
  centers) might preserve retention at lower compression — but that changes the
  promotion operator, which the spec fixed.
- **Reduced epochs (20 vs 120).** More training could change magnitudes; the
  *direction* (P < F on retention, all 5 seeds) is robust.
- **C3 near-inert** suggests the interface-level reversal threshold (inherited
  from the particle Crucible) is too conservative for aggregated interfaces.

## Gate ladder (final)
1A PASS · 1B confirmed · 1C PASS(Scale 1) · 1D PASS · 2 PASS · **3 FAIL** (static
compression validated; dynamic recursive reuse lossy).
