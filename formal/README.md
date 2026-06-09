# formal/ — IBF theory results used by the MSCN experiments

Two machine-checked results (Lean 4 / Mathlib, arXiv:2604.07108) underpin the
chess representation-learning work. The full Lean sources live in the IBF
formalization; this file records the statements and **where the code uses them**.

## 1. Operating Resolution — the optimal kernel bandwidth

For a finite kernel-supported correction field, the optimal Gaussian bandwidth is

```
    σ*  =  d_shell / √( 2 · log( N_eff / ε ) )
```

* `d_shell` — characteristic shell distance (nearest interfering, "non-local" centre),
* `N_eff`  — effective interference count = participation ratio `(Σwᵢ)² / Σwᵢ²`,
* `ε`      — locality tolerance (allowed aggregate bleed from non-local centres).

Proven properties (`OperatingResolution.lean`):
- **Locality (exact):** at `σ*` the aggregate bleed equals `ε` exactly
  (`bleedBound_at_operatingBandwidth`).
- **Optimality:** `σ*` is the *largest* bandwidth with bleed ≤ ε — it is the LUB of
  the feasible set (`operatingBandwidth_isSupremum`); any wider σ violates locality.
- **Monotonicity:** `σ*` decreases as `N_eff` grows (denser fields ⇒ tighter kernel);
  the non-local participation ratio is non-decreasing in σ (`nEffFinset_mono_sigma`).
- **Universality:** `N_eff = 1` recovers the pairwise isolation bandwidth.
- **Dynamical bridge:** the discretised modification PDE *produces* such a field, so
  these bounds apply to it (`modification_crystallization`), with σ*-coupled
  discrete→continuum convergence at rate `O(log N_eff / N)`.

**Used in:** `mscn/chess_kernel.py` — `operating_bandwidth(d_shell, n_eff, eps)` sets
the kernel bandwidth. `d_shell` is the median local k-NN radius; `N_eff` is the
participation ratio of the kernel weights measured at reference scale `σ_ref =
d_shell` (the "Path A" prescription that breaks the σ↔N_eff circularity); `ε`
defaults to 0.01.

## 2. Causal States — the state-tracking gap (theory-team response)

Formalizes the gap analysis in `../mscn/NOTE-state-tracking-gap.md`. For a
deterministic sequential process (states = positions, actions = moves, observable =
move legality/value):

- **Predictive sufficiency:** a history representation `φ` is sufficient iff
  `φ(h₁)=φ(h₂) ⇒ h₁, h₂` reach states with identical observable profiles.
- **Route B (causal-state optimality):** when observable profiles are injective, a
  representation is sufficient **iff it refines the state partition**
  (`causal_state_optimal`) — the state map is the *unique coarsest* sufficient
  representation. "Learning the causal-state partition = learning a sufficient
  statistic." An IBF coarse-graining (Postulate II) is valid iff it is sufficient
  (`ibf_coarsegraining_requires_sufficiency`).
- **Route A (recurrent sufficiency):** a recurrent update `z_t = G(z_{t-1}, a_t)` is
  sufficient **iff it simulates the process** — i.e. there is a decoding `σ` with
  `σ(init)=s₀` and `σ(G(z,a)) = T(σ(z), a)` (an algebraic homomorphism from the move
  monoid to the state space) (`recurrent_sufficient_of_simulation`).
- **Window insufficiency (counterexample):** the toggle process shows a fixed
  last-token window is *not* sufficient — two histories with the same last move
  reach states with different legality (`window1_not_sufficient`).

**Engineering implication (acted on):** the kernel chess model's context vector is a
recency-weighted **sum of move embeddings** — a lossy map of exactly the
window/sum kind the toggle counterexample rules out. So the pure kernel model is
*provably* not predictively sufficient, which matches the measured result (it does
not beat the exact n-gram on legality overall; see below). Closing the gap requires
a **simulation-faithful recurrent state** (Route A), not a richer kernel over a
lossy context. This is the next build.
