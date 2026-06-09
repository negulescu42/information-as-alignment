# MSCN → AGI: A Theory-Grounded Roadmap

> Provided by the IBF theory team. This document identifies nine architectural
> upgrades to move the MSCN toward general-intelligence *capabilities*, each
> grounded in an existing IBF theorem or a new one in `formal/AGIFoundations.lean`.
> The Python system implementation of each upgrade lives in `mscn/agi_*.py`, with
> a runnable validation that measures a concrete advantage (or an honest null).
> See `mscn/ARCHITECTURE.md` §4.x for the implementation log and measured results.

## Executive Summary

The MSCN instantiation validates the IBF core: gradient flow converges (Thm 1–2),
memory crystallizes (Thm 3), agency differentiates via Boltzmann (Thm 7),
cooperation emerges from relational coherence (Thm 4–5), and the Lawvere
obstruction limits self-knowledge (§7). The chess study stress-tests the
representation substrate and identifies the precise bottleneck: *the quality of the
internal state representation determines all downstream capability*. The
simulation-faithful state (Route A) closes the legality gap from 33% → 50%, and the
position-value coherence layer opens the strategy frontier.

This document identifies **nine architectural upgrades**, each grounded in existing
IBF theorems and new formalizations (`formal/AGIFoundations.lean`), that collectively
move the MSCN toward general intelligence. The upgrades are ordered by expected
impact and feasibility.

---

## §1. Simulation-Faithful Internal Models (the decisive lever)

The chess results deliver one unambiguous lesson: **bounded move-history suffixes
cannot determine the board state** (the causal-state theorem), so no amount of kernel
smoothing or n-gram depth can close the legality gap. The simulation-faithful state
(`chess_simstate.py`) achieves 0.998 legal-set purity vs ~0.01 for suffix-based
states. This is the single highest-impact finding.

### Upgrade 1: Learnable Simulation Homomorphisms
Replace the hand-coded occupancy transfer with a *learned* simulation map
`G: Z × A → Z` such that `σ(G(z,a)) = T(σ(z),a)`. The learning signal is the
discrepancy between predicted and observed next-state features. This is IBF
coarse-graining (Postulate II) made dynamic. New theorem: `simulation_is_sufficient`
— a recurrent state is sufficient iff it forms a homomorphism with the dynamics.

### Upgrade 2: Hierarchical Simulation (Multi-Scale Internal Models)
Stack simulation layers: raw state → abstract state → strategic state. Each layer is
a coarse-graining of the one below (RG flow, `RenormalizationGroup.lean`). Coherence
is non-increasing under coarsening; iterated coarsening kills fine-scale noise.

---

## §2. Adaptive Responsiveness and Exploration (the agency frontier)

### Upgrade 3: Coherence-Gradient-Driven Exploration
Modulate `k` adaptively from the gradient variance: confident (low variance) →
exploit (raise `k`); uncertain (high variance/flat) → explore (lower `k`). Formal
measure: entropy production `σ(x) = k·‖∇R_eff‖²`.

### Upgrade 4: Directed Exploration via Information Gain
When exploring, choose actions maximizing expected information gain (KL / discrepancy
magnitude), not just immediate coherence. `directed_beats_random_exploration`: the
best exploratory action achieves at least the average information gain.

---

## §3. Compositional Coherence Transfer (the generalization frontier)

### Upgrade 5: Cross-Domain Coherence Morphisms
A coherence morphism `φ: F₁ → F₂` (continuous, coherence-non-decreasing) maps learned
coherence from one domain to another. `transfer_preserves_superlevel` /
`transfer_preserves_viability`: morphisms map basins to basins, so skills transfer as
viable regions.

### Upgrade 6: Compositional Coherence Algebras
Compose coherence landscapes via the network coherence formula
`R_total = Σ R_i + Σ J_ij R_pair`. New skills reuse existing sub-coherences with
learned couplings, not new monolithic δR maps.

---

## §4. Temporal Abstraction and Planning (the reasoning frontier)

### Upgrade 7: Coherence-Based Planning via Imagined Trajectories
Use the learned simulation (Upgrade 1) to imagine trajectories and evaluate them via
the coherence landscape; select the action whose imagined trajectory reaches the
highest coherence. `planning_horizon_value_bound`: depth-`d` value ≤ `d`·max per-step
gain; bounded non-negative gains ⇒ planning value converges.

### Upgrade 8: Temporal Coherence Hierarchies (Options / Macro-Actions)
Learn macro-actions as coherence basins: entering a basin guarantees convergence
(Thm 2). `temporal_abstraction_safety` / `macro_action_composition`: safe
sub-trajectories compose to safe macro-trajectories. Plan over basins.

---

## §5. Self-Monitoring and Meta-Cognition (the consciousness frontier)

### Upgrade 9: Active Self-Improvement via Reflexive Coherence
Close the reflexive loop: the monitor's output drives meta-level modification,
allocating driving signal `α` to under-performing domains subject to the
consciousness–competence budget `Σ α_i ≤ Γ`. `self_improvement_equilibrium_positive`
/ `self_improvement_exceeds_half_gap`: when `gain_rate > μ` the equilibrium
modification exceeds `(θ−c)/2`, reliably crossing the threshold.
`optimal_linear_allocation'`: concentrate the budget on the highest-rate domain.

---

## §6. Priority Ordering (implementation schedule)

| Priority | Upgrade | Expected Impact | Difficulty | Depends On |
|---|---|---|---|---|
| 1 | Learnable Simulation (§1.1) | Critical — unlocks all others | Medium | — |
| 2 | Coherence-Gradient Exploration (§2.3) | High — fixes deceptive functions | Low | — |
| 3 | Planning via Imagined Trajectories (§4.7) | High — enables reasoning | Medium | 1 |
| 4 | Hierarchical Simulation (§1.2) | High — enables abstraction | Medium | 1 |
| 5 | Directed Exploration (§2.4) | Medium — sample efficiency | Medium | 2 |
| 6 | Cross-Domain Transfer (§3.5) | High — generalization | High | 1, 4 |
| 7 | Temporal Abstraction (§4.8) | High — strategic behavior | High | 1, 3 |
| 8 | Compositional Coherence (§3.6) | Medium — knowledge reuse | High | 6 |
| 9 | Active Self-Improvement (§5.9) | Transformative | Very High | all |

---

## §7. Quantitative Predictions and Testable Milestones

**Chess** — legal@1 0.50 → 0.95 (learned simulation); acc@1 0.148 → 0.30 (planning).
**Optimization** — close the CMA-ES rank gap via exploration + planning; raise
Schwefel-10D success via directed exploration. **Multi-agent** — scale agents, add
task diversity via transfer, superlinear coherence growth via self-improvement.

The honest measured outcomes (which of these are achieved, partially achieved, or
null) are logged per-upgrade in `mscn/ARCHITECTURE.md` §4.

---

## §8. Honest Assessment: What's Hard and What's Missing

1. **Credit assignment over long horizons** — the local modification ODE needs a
   coherence-valued TD analog (`tdError'`, `td_is_modification_step` is the seed).
   *The most important open problem.*
2. **Scalable representation learning** — Gaussian-kernel memory is costly at scale;
   a compressed basis is needed (`compression_fidelity_tradeoff'`). The Lawvere
   obstruction guarantees any compressed self-model is lossy — lose the *right*
   information.
3. **Language and symbolic reasoning** — continuous coherence vs discrete
   compositional structure; genuinely hard.
4. **Safe exploration** — basin invariance (Thm 2) guarantees safety *within* a
   basin; exploring *between* basins leaves the safe region.

### What the IBF provides that pure-neural approaches don't
Guaranteed monotonicity (Thm 1); interpretable, causally-attributable memory (Thm 3);
provable agency (Thm 7); formal self-knowledge bounds (Lawvere); thermodynamic
consistency. These constrain the design space: every upgrade must preserve them.

---

## §9. Connection to the Formal Library

| Upgrade | Existing Theorems | New (AGIFoundations.lean) |
|---|---|---|
| 1. Simulation | CoarseGraining | `simulation_is_sufficient`, `bounded_history_insufficient` |
| 2. Hierarchical | RenormalizationGroup | `hierarchical_rescaling_bound'`, `deep_coarsening_kills_fine_structure` |
| 3. Exploration | StochasticDynamics | `high_variance_favors_exploration`, `exploration_cost_nonneg` |
| 4. Info-Gain | InformationTheory | `directed_beats_random_exploration` |
| 5. Transfer | CategoryTheory | `transfer_preserves_superlevel`, `transfer_preserves_viability` |
| 6. Composition | NetworkCoherence | `compositional_coherence_lower_bound'`, `optimal_linear_allocation'` |
| 7. Planning | Thm 1 (convergence) | `planning_horizon_value_bound`, `planning_converges_if_gains_bounded` |
| 8. Temporal | Thm 2 (basin inv.) | `temporal_abstraction_safety`, `macro_action_composition` |
| 9. Self-Improve | Thm 9 (consciousness) | `self_improvement_exceeds_half_gap` |
| 11. Credit | — | `td_equilibrium'`, `td_is_modification_step` |
| 12. Scalable rep. | — | `compression_fidelity_tradeoff'`, `more_basis_less_error` |

All Lean statements are in `formal/AGIFoundations.lean` (verified by the theory team
via `lake build`; not lake-built in this PyPI-only sandbox — see the file header).
