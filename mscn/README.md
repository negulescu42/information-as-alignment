# MSCN — a Multi-Scale Coherence Network toy model

A small, self-contained, **runnable** implementation of the three-layer IBF
apparatus described in the *Most Complex IBF-Based ML/AI Apparatus* design note.
It is a *toy model*: the scales are deliberately small (tens of agents, a few
hundred rounds) so that every mechanism is visible and every run finishes in
seconds. Knowledge is stored as **coherence modifications `δR` on configuration
space** — a kernel-based associative memory — never as neural weights.

The point of the toy model is not to beat production ML; it is to show that the
qualitative behaviours the framework *proves* (basin expansion, selective
retention, agency, emergent cooperation, bounded self-knowledge, a consciousness
phase transition, multi-scale coarse-graining) actually appear in a faithful,
executable reimplementation.

## Quick start

```bash
python -m mscn.demo          # full guided tour (all pilots + the integrated MSCN)
python -m mscn.demo pilot1   # one stage: pilot1 | pilot2 | pilot3 | pilot4 | mscn
python -m mscn.demo --figures   # also write plots to mscn_outputs/  (needs matplotlib)
python -m mscn.tests         # 24 behavioural checks of the proven guarantees
```

Only `numpy` is required (`matplotlib` only for the optional figures).

## Architecture → code map

```
Layer 3  meta-learning / self-knowledge / phase control
         hierarchy.py   coarse-graining, RG flow, hierarchical optimiser
         selfmodel.py   Lawvere obstruction, self-knowledge ceiling, tradeoff
         phase.py       phase transition, dissipative lifetime, Zombie-Twin
Layer 2  network cooperation
         network.py     coupling graphs, network coherence, mean-field
         games.py       iterated Prisoner's Dilemma, IBF game agent, tournament
Layer 1  individual learning
         learner.py     DiscreteIBFLearner + IBFLearner (kernel memory)
         landscapes.py  coherence landscapes (sphere/Rastrigin/Ackley/…)
Integration
         mscn.py        couples all three layers into one running network
         demo.py        guided tour;  tests.py  guarantee checks
```

## The four pilots

**Pilot 1 — individual coherence-gradient learning.** The learner senses
`R_eff = R̂ + δR`, picks moves with a Boltzmann policy whose responsiveness `k`
adapts, and updates `δR` by the discrepancy signal. Demonstrated guarantees:
agency advantage (P(best) ↑ in `k`, → greedy), selective retention
(`δR → α/μ` when reinforced, → 0 otherwise), basin expansion (viable set only
grows; `δR ≥ 0`), Euler → ODE convergence. As an optimiser it beats random
search in nearly every regime and is **best of all four methods on the
higher-dimensional multimodal landscapes** (Rastrigin-5d, Ackley-5d) where
memory and basin expansion pay off. (Schwefel-2d, a deceptive landscape with the
optimum at the domain edge, is the honest weak spot.)

**Pilot 2 — emergent cooperation.** IBF agents play the iterated Prisoner's
Dilemma. They sense only the immediate own payoff (which favours defection), but
their modification map accumulates the *relational* coherence of mutual
cooperation. Result: IBF-memory agents reach **100% mutual cooperation** with
reciprocators and each other, **defect against pure defectors** (not exploited),
while memoryless agents stay in the defection trap. The causal test (EC-4): with
the relational coupling removed (`J = 0`) cooperation collapses (4%); with
`J = 4` it is full (100%) — so it is the coherence coupling, not the optimistic
initialisation, that sustains cooperation. No cooperation reward is added to the
payoffs.

**Pilot 3 — coarse-graining & hierarchy.** Renormalisation flow suppresses
small-scale (irrelevant) noise and the relevant scale emerges; the peak is
non-increasing under coarsening. A hierarchical optimiser that coarse-grains a
block-structured landscape into independent sub-problems **beats a flat learner
8/8** at equal evaluation budget.

**Pilot 4 — bounded self-knowledge & the phase transition.** The Lawvere
diagonal shows no self-model is surjective (perfect self-knowledge is
impossible) at every size tested; introspection depth is capped at
`⌊Γ/base_cost⌋`; the consciousness–competence tradeoff is explicit. The
self-monitoring equilibrium `E* = c + α/μ` has an exact sub-/critical/super-
critical structure around the threshold, consciousness is a dissipative
structure with finite undriven lifetime, and in the **Zombie-Twin** experiment a
self-correcting agent outlasts its non-monitoring twin in 100% of adversarial
episodes.

**Full MSCN.** 27 coupled learners on a scale-free graph: best coherence rises,
coupled agents reach consensus (cooperation), network coherence exceeds the sum
of individual coherences, the population recovers from periodic adversarial
perturbations via phase-controlled self-correction, and the final population
coarse-grains into a 27 → 9 → 3 → 1 macro-hierarchy.

## Guarantee → check map

Every row is verified empirically in `tests.py` (the theorems themselves are
machine-checked in Lean; these keep the *implementation* honest).

| Property (theorem)                        | Check |
|-------------------------------------------|-------|
| Boltzmann monotone in k / greedy (Thm 7)  | `test_boltzmann_monotone_in_k` |
| Basin expansion (Thm 8a)                  | `test_basin_expansion_superset`, `test_discrete_basin_nonshrinking` |
| Selective retention (Thm 8b)              | `test_selective_retention` |
| Forgetting → 0 (Thm 10a)                  | `test_forgetting_decays_to_zero` |
| Crystallization μ=0 ⇒ constant (Thm 3a)   | `test_crystallization_constant` |
| Euler → ODE (Thm 11)                      | `test_euler_converges_to_ode` |
| Coupling only helps (NetworkCoherence)    | `test_coupling_only_helps` |
| Mean-field σ/√N (MeanFieldTheory)         | `test_mean_field_suppression` |
| Cooperation needs memory + coupling (EC-4)| `test_cooperation_requires_memory`, `test_cooperation_needs_coupling` |
| Differential persistence (Thm 6 cor.)     | `test_differential_persistence` |
| RG noise suppression (Renormalization)    | `test_rg_flow_suppresses_noise` |
| Hierarchy beats flat                      | `test_hierarchy_beats_flat` |
| Lawvere: no surjective self-model         | `test_lawvere_no_surjective_self_model` |
| Self-knowledge ceiling                    | `test_self_knowledge_ceiling` |
| Consciousness–competence tradeoff         | `test_competence_tradeoff_monotone` |
| Phase transition / critical point         | `test_phase_transition` |
| Dissipative lifetime                      | `test_dissipative_lifetime` |
| Zombie-Twin survival advantage            | `test_zombie_twin_survival` |

## Honest scope

This is a toy model. Agent counts are in the tens, runs are seconds, and the
optimiser is a competitive *local* search rather than a state-of-the-art global
optimiser. What it demonstrates is **mechanism fidelity**: the proven IBF
behaviours reproduce when you implement the dynamics directly, with no neural
networks and no backpropagation. Scaling the same mechanisms (efficient kernel
data structures, sparse coupling, larger hierarchies) is the engineering path
sketched in the design note.
