# Branch: `gate-1c-behavioral-metric`

## Goal
Break the Gate 1B blocker: Scale 1 particles were too broad in observation
space, so each particle's behavioral signature averaged over contradictory
rewards and collapsed to a function of u1 only. Test whether **discrepancy-driven
splitting** lets particles localize — and signatures encode — the behaviorally
aliased coordinate u2.

## Gating diagnostic (run first — `gate1c_oracle_diagnostic.py`)
Oracle signature = mean reward over a query point's 10 nearest neighbours in
**true u-space** (perfect localization). Result: median oracle ρ_u2 = **0.784 ≥
0.30** → the task is solvable by a well-localized signature; the broad particles
are the blocker → proceed.

> Caveat discovered later (see fairness check below): the oracle-signature
> diagnostic has *behavioral access at the query point*, so it does not test
> whether a static **x→q** encoder can recover u2. That is a separate, stricter
> condition.

## Method (in `scale1_representation.py`, `enable_splitting=True`)
At each lifecycle step, a non-converging particle (rolling D-variance above
`split_threshold` after `n_repr_cryst_min` exposure) subdivides its jurisdiction:
k-means(2) on its logged activating observations → two child particles at the
cluster centroids, each at 0.7× bandwidth, re-seeded from its cluster's records,
born transient. Second lever available: `enable_behavioral_creation`.

## Run
```bash
python gate1c_oracle_diagnostic.py 5        # gating diagnostic
python run_gate1c.py 5                       # splitting ablation x frequency
python summarize_gate1c.py                   # -> gate1c_outputs/gate1c_report.md
```

## Finding (5 seeds)

**Leading indicator — splitting works at the Scale 1 level (all frequencies):**

| u2_freq | signature→u2 (off → on) | per-particle u2 var (off → on) |
|---|---|---|
| 3.0 | −0.02 → **0.66** | 0.79 → 0.50 |
| 2.0 | 0.21 → **0.74** | 0.35 → 0.005 |
| 1.5 | 0.56 → **0.81** | 0.025 → 0.013 |

Crystallized signatures now encode the aliased coordinate (clears the 0.30 gate).
**Supervisor's hypothesis confirmed at the Scale 1 level.**

**But the full Gate 1C pass (ρ_u2 > 0.5 AND ρ_struct > 0.8) is not achieved at any
frequency** — a second, downstream blocker appears, with a clean two-regime
structure revealed by the generator-fairness check (supervised x20→u2 ceiling):

| u2_freq | supervised x20→u2 (any encoder) | emergent ρ_u2 (split on) | emergent ρ_struct |
|---|---|---|---|
| 3.0 | **0.04 (impossible)** | −0.03 | 0.59 |
| 2.0 | 0.45 (marginal) | 0.18 | 0.60 |
| 1.5 | 0.81 (recoverable) | 0.31 | 0.62 |

- **High aliasing (3.0):** splitting fixes Scale 1, but a static x→q encoder
  cannot carry u2 to test time — even supervised regression with true labels gets
  ρ≈0. u2 is not recoverable from the observation by *any* encoder.
- **Low aliasing (1.5):** u2 is encoder-recoverable and emergent ρ_u2 can clear
  0.5 (split *off*: 0.58), but ρ_struct plateaus ~0.62 and splitting is barely
  needed — it slightly *hurts* via graph fragmentation.

There is no tested regime where splitting is **both necessary and sufficient** to
clear ρ_struct > 0.8.

## Bounded conclusion
Discrepancy-driven splitting works as designed and resolves the Scale 1
localization blocker. But Postulate 2 in the hard setting is gated by a further
constraint splitting cannot address: **the emergent representation can only carry
a coordinate the observation locally determines.** Recovering an
observation-aliased coordinate at test time requires *behavioral access at encode
time* (an interactive encoder), which is outside the current static
observation→representation paradigm. This bounds the postulate and is a
publishable negative result.

## Status of the criteria
```
Leading indicator (signature->u2 > 0.3):   PASS (splitting works at Scale 1)
Full pass (rho_u2 > 0.5 & rho_struct > 0.8): NOT achieved (encoder blocker)
```
No interface nodes, promotion, recursion, RRW/chess.
