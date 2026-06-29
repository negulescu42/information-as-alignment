# IBF Recursive Scale Structure — Gate Ladder

Status of the Gate 1 → Gate 2 program testing **Postulate 2 (Recursive Scale
Structure)** and the **Interface Principle** in the two-scale toy model. Each gate
is a separate branch with its own README, runner, summarizer, and committed
results.

| gate | branch | claim tested | result |
|---|---|---|---|
| **1A** | `claude/gate-1-recursive-scale-j80grv` | lower-scale crystallization → usable emergent 2D space | **PASS** (10/10) — but geometry-easy, *not* causal validation |
| **1B** | `gate-1b-behavioral-geometry` | geometry-only recovery insufficient (stress generator) | confirmed; multiplicative graph cannot rebuild aliased coord |
| **1C** | `gate-1c-behavioral-metric` | discrepancy-driven splitting localizes the aliased coordinate | **PASS at Scale 1** (signature→u2 0.66–0.81); exposed the encoder wall |
| **1D** | `gate-1d-interactive-encoder` | interactive encoding externalizes the aliased coordinate | **PASS** (accuracy_gap < 0.15, 10/10 seeds) |
| **2 (k=2)** | `gate-2-interface-extraction` | basins partition; interior removable (Interface Principle) | FAIL — field too sparse (all-surface) |
| **2 (k=8)** | `gate-2-retry-k8` | same, on a *dense* field (8-action task) | **PASS** — external shielding 0.3%, compression 27% |
| **3** | `gate-3-promotion` | promoted interfaces reused as primitives in continual learning | **FAIL** — static compression OK, dynamic reuse lossy (retention −11pp) |

## The two load-bearing findings

1. **Postulate 2 requires *interactive* representation formation (Gate 1D).**
   A static observation→representation map cannot externalize a coordinate the
   observation aliases (supervised x→u2 ceiling 0.04 at f=3.0). An encoder that
   takes a few reward probes recovers it (ρ_u2 0.77) and yields a configuration
   space on which Scale 2 matches/beats oracle coordinates. Inference cost ≈ 3
   probes.

2. **The Interface Principle holds for dense fields (Gate 2, k=8).** With 8
   actions the correction field is dense (~180 crystallized centers, 8 basins,
   27% genuinely deep interior at 1.71σ). Interior centers contribute **0.3% to
   the field external to their basin** — externally negligible, as the Lean
   theorem predicts. Removing all 27% interior costs ~2.2pp accuracy.

## Two recurring methodological lessons

- **Literal metric vs the principle's operational claim.** Twice a literal metric
  diverged from the principle and the operational metric was the correct test:
  Gate 1D (geometric ρ_struct failed; operational accuracy_gap passed) and
  Gate 2 (global field fidelity failed; external shielding passed). **Spec
  correction carried forward: field fidelity is measured at points *external* to
  each basin, never inside them.**
- **Argmax robustness.** Scale 2 action selection tolerates large correction-field
  perturbations (≈48–100% field change → ≈2pp accuracy change), because the
  decision margins are wide. Compression is behaviourally cheap even where it is
  not field-faithful.

## Scope bounds established (honest negatives)

- Static encoders cannot externalize observation-aliased coordinates (Gate 1C).
- A 2-action toy produces an all-surface correction field with no shielded
  interior — interface extraction needs a dense field (Gate 2 k=2 → k=8).

## Gate 3 outcome — the recursion boundary

Gate 3 ran (supervisor-approved spec) and **failed**: promoted boundary-only
interfaces preserve current-context accuracy and compress the population (16%),
but **degrade retention of the promoted context by ~11pp** (BT_A −0.40 → −0.51).
The interior is externally negligible for *static* readout (Gate 2) yet carries
redundancy needed to survive *interference* during later phases. Compression is
real; **recursive reuse of compressed interfaces is lossy**.

**Net program result.** Postulate 2 is validated for **representation through
interaction** (Gate 1D) and **static interface compression with external
shielding** (Gate 2). Its **recursive-reuse** extension (Gate 3) is **not**
validated in this instantiation — the theory is bounded to single-scale dynamics
plus static compression. Every result, positive and negative, is committed with
its diagnostics.
