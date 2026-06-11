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
Agents (ARCHITECTURE §9–§11)
         ibf_asi.py     the nine-stage coupled agent (+ regimes/gauntlet/benchmark)
         ibf_engine.py  the preprint's classic evaluator-corrector lifecycle
         ibf_unified.py the three-ODE blend (engine as memory organ + local k)
         ibf_ultra.py   self-detected contexts + equipment policy (§11)
Showcase (ARCHITECTURE §12)
         terrarium.py            render stack (headless PNG/GIF)
         terrarium_episodes.py   the 7 scored episodes + report generator
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

## Benchmarks

```bash
python -m mscn.benchmark            # full report (20 seeds): optimiser, strong,
                                    # convergence, cooperation, hierarchy, scaling
python -m mscn.benchmark --quick    # 8 seeds
python -m mscn.benchmark strong     # vs CMA-ES / DE / dual-annealing (needs cma, scipy)
python -m mscn.benchmark optimiser --figures
```

Headline results (eval-budget matched, 20 seeds, objective = best `f`, lower is
better):

* **Optimiser, mean rank over all function × dim × seed (1 = best of four):**
  **IBF 1.89** · SA 2.11 · hill 2.39 · random 3.61. The IBF learner has the best
  overall rank, **beats random search almost everywhere**, and **wins on every
  10-D function** against hill climbing and simulated annealing (where memory +
  basin expansion matter most and random search collapses). It is weakest in
  2-D (hill/SA edge it out) and on the deceptive Schwefel (high variance).
* **Cost:** ~140 ms/run vs SA's ~38 ms — about 4× slower per evaluation (richer
  per-step computation: proposals, kernel memory, modification update).
* **Vs state-of-the-art (`python -m mscn.benchmark strong`):** against CMA-ES
  (BIPOP restarts), differential evolution and dual annealing, IBF is **not**
  competitive on raw continuous optimisation — mean rank CMA-ES 1.61 <
  dual-annealing 2.27 < DE 2.86 < **IBF 3.92** < SA 4.34. IBF beats the classical
  heuristic (SA) and occasionally DE in 10-D, but the decades-tuned global
  optimisers dominate. This is the honest finding and the intended framing: the
  IBF advantage is interpretability, formal guarantees, emergent cooperation and
  representation — not beating CMA-ES at single-objective black-box search.
* **Convergence:** IBF's best-so-far is consistently below random search and
  competitive with SA (it overtakes SA late on Ackley-5D).
* **Cooperation:** IBF-memory tops the round-robin tournament
  (score 2.46 ± 0.02), above TitForTat and Pavlov.
* **Hierarchy:** coarse-grained block optimisation beats the flat learner with a
  win-rate of 1.00 on the small-block configurations.
* **Scaling:** the integrated network runs 9 → 125 agents for 60 rounds in
  0.3 s → 5.2 s; cost grows with the O(N²) pairwise coupling (sparse graphs are
  the path to larger N).

## Emergent chess representation (experiment)

`chess_world.py` is a representation-learning / generalisation probe: can the IBF
coherence mechanism learn the *structure* of chess — rules and the board — from
sequences of **opaque move tokens**, with no board, pieces, rules or strategy
built in? (Same spirit as Othello-GPT / Chess-GPT, but with non-neural coherence
memory instead of a transformer.)

```bash
python -m mscn.chess_world --figures     # needs: pip install chess scipy
```

Strict no-priors: the model sees each move only as an atomic id; it is never told
tokens contain squares, that squares form a grid, or that pieces have movement
rules. `python-chess` is used only to generate legal games (data) and as an
evaluation oracle. (This environment's network policy blocks lichess, so games
are generated locally; emergent *strategy* — Stage 3 — wants real human games and
is left for a network that can reach lichess.)

* **Stage 1 — emergent rules:** trained only on token transitions, the model's
  top move is legal far above chance (opening 72%, overall 38% legal@1; 79%
  legal@5; legal-probability-mass ≈ 14× a random token's 1.5%) — it learned to
  prefer legal moves *with no rules supplied*. Legality is strongest in openings
  and decays in novel late positions: an associative n-gram memory does not fully
  track board state (the gap a transformer's full-history attention closes). On
  near-random generated data, next-move *accuracy* is low by construction —
  that's the Stage-3 strategy question, which needs real games.
* **Stage 2 — emergent board geometry:** embedding the 64 squares from move
  co-occurrence in the learned representation recovers the 8×8 grid (Procrustes
  disparity ≈ 0.06 vs the true board; ≈ 72% of true king-adjacencies are nearest
  neighbours) — the board's 2-D geometry emerges from move tokens alone, no
  spatial prior. See the generated `chess_board_emergence.png`.
* **Stage 3 — emergent strategy** (`chess_strategy.py`): the real-lichess data
  path (`stream_lichess`, `load_pgn`: moves + Elo + `%eval`) is built and ready,
  but this sandbox's network policy blocks the lichess CDN, so the methodology is
  validated on a labelled **skill-stratified proxy** (a tiny pure-Python minimax
  engine: random / greedy / depth-2). Findings: learned **coherence rises
  monotonically with player strength** (0.01 → 0.10 → 0.14) and the more-coherent
  player tends to win (outcome correlation ≈ +0.3), so coherence is a good *judge*
  of strength. The model's own top move is **far above random** (≈ +370 cp) but
  **well below strong play** (engine/actual): partial, capped generative skill —
  a strong discriminator, a limited generator. That ceiling is the state-tracking
  gap analysed in `NOTE-state-tracking-gap.md`. **Run all three stages on a real
  PGN** (lichess Elite, etc.) with one command:
  `python -m mscn.chess_strategy --pgn path/to/games.pgn` — Stage 3 then keys on
  real player Elo and results. `load_pgn` handles `.pgn` and `.pgn.zst`.
* **Kernel model** (`chess_kernel.py`): the *faithful* IBF mechanism — Gaussian-
  kernel memory over learned move embeddings, with bandwidth set by the **Operating
  Resolution** law `σ* = d_shell/√(2·log(N_eff/ε))` (see `../formal/README.md`).
  Honest finding: on real Elite games it **does not beat the exact n-gram** on
  legality at any novelty level (seen 0.37 vs 0.46; novel 0.11 vs 0.17). This
  confirms the causal-states theorem: the recency-weighted-sum context is a lossy,
  non-sufficient map, so a kernel over it cannot recover board state. The fix is a
  simulation-faithful recurrent state (Route A in `NOTE-state-tracking-gap.md`),
  not a richer kernel — the bottleneck is the context representation.
* **Route A — recurrent model** (`chess_recurrent.py`): a variable-order recurrent
  state `z_t = G(z_{t-1}, a_t)` (deepest supported move-suffix; IBF adaptive-
  resolution coarse-graining). It **beats the n-gram** on legality, most where
  longer history matters (opening legal@1 0.84→0.91, acc@1 0.38→0.45; all 0.32→0.34),
  so **VOM > n-gram > kernel**. But the predictive-sufficiency probe shows even the
  depth-6 state has low legal-set purity (~0.29): no bounded move-history statistic
  determines the legal set. A longer suffix helps but isn't sufficient — only a
  simulation-faithful state (`σ∘G = T∘σ`) closes it (see `ARCHITECTURE.md`).
* **Full IBF unit** (`chess_ibf.py`): the closed Layer-1 acting loop online over the
  game stream (Boltzmann-`k` selection, discrepancy modification ODE with
  selective-retention decay, adaptive `k`/agency — `k` grows 1→8). Retest: in the
  **crystallization limit** (`μ=0`, strong drive) it *recovers* the batch predictor
  (legal@1 0.30 ≈ n-gram), so the batch model is its `μ→0` special case (Thm 3a);
  with its characteristic **forgetting** (`μ>0`) legality falls monotonically (the
  selective-retention vs exhaustive-memory tradeoff). The full unit's dynamics are
  built for *acting* in persistent/non-stationary environments, not stationary
  exhaustive prediction — see `ARCHITECTURE.md` §3.4.
* **Route A, full — simulation-faithful state** (`chess_simstate.py`): reconstructs
  the **board** as the recurrent state by replaying move tokens as occupancy
  transfers (`σ∘G = T∘σ`), prior-free — no grid, no piece types, no rules; even the
  **32-piece start position is discovered from data**. This **closes the gap**: the
  reconstructed board has **legal-set purity 0.998** (it determines the legal set;
  move-suffix states ≈0.01), and using it as a legality mask lifts legality to
  **legal@1 0.50 / legal@5 0.83** overall (from 0.33/0.60), with endgame doubling
  0.18→0.38. The headline Route-A result — see `ARCHITECTURE.md` §3.5.
* **Board-conditioned IBF agent** (`chess_board_ibf.py`): feeds the sufficient board
  state into the IBF coherence — a board-conditioned `δR(piece→destination)` with
  full IBF-unit dynamics (online, decay, adaptive `k`), mixed with context and
  occupancy-masked. Honest finding: this **barely improves ranking** (acc@1 +0.001,
  top-move quality −599→−592 cp vs human −14) — the board gives *legality*, not which
  legal move is *strong*. Strong ranking needs position **evaluation** (engine-level);
  the real strategy lever is a coherence *landscape over board states*, not a
  move-affinity memory. See `ARCHITECTURE.md` §3.6.
* **Position-value coherence layer** (`chess_value_ibf.py`): learns a coherence over
  board *states* (position value) from game **outcomes**, then picks the move whose
  resulting board has highest coherence (IBF gradient-flow over positions). **Piece
  values emerge from outcomes** (queen most valuable, prior-free). Value-guided
  selection among context-plausible moves lifts **move quality −606 → −149 cp**
  (sound material play), though **acc@1 drops 0.14→0.05** — value-greedy ≠ human
  *strategy*; human-level needs positional coherence beyond material (the
  engine-level frontier). See `ARCHITECTURE.md` §3.7.
* **Positional coherence layer** (`chess_positional_ibf.py`): adds an emergent
  **piece-square table** (learned from outcomes) and selects by *context-prior +
  value-gain* (not value-greedy). **Positional structure emerges** — pawn value
  rises with advancement (rank 2→7: 0.002→0.022), prior-free. The mixture improves
  objective move quality (−528→−367 cp) **without hurting** human-matching (acc@1
  ≈0.146). Honest limit: acc@1 doesn't rise — human-level *strategy* is the
  engine-level frontier (richer positional coherence / search needed). `ARCHITECTURE.md` §3.8.
* **Playing strength — head-to-head** (`chess_arena.py`): with python-chess as
  *referee* (legality + result only), a value+SEE agent (emergent values + static
  exchange evaluation + positional coherence) **beats the context-only model 40–0**.
  Real, measurable strength — it plays *sound* chess while context-only blunders out
  of book. (acc@1 still ≈0.146: stronger *play* ≠ predicting human *moves*.)
  `ARCHITECTURE.md` §3.9.
* **Deeper search + the acc@1 lever** (`chess_search_ibf.py`): negamax (context =
  move generator, emergent value = leaf eval). **Strength**: depth-2 beats value+SEE
  12–12–0 and context 24–0. **acc@1**: search does *not* crack it (flat-to-negative —
  stronger play ≠ human moves), but **data does** — acc@1 scales 0.090→0.120 over
  500→4000 games, unsaturated. The move-predictor is data-limited; more games (and
  ultimately a stronger learner) raise acc@1, not search. `ARCHITECTURE.md` §3.10.
* **Per-query local σ\*** (`chess_kernel.py`, `local_sigma=True`): per a resolution-
  principle roadmap, σ should be local (`σ*(y) = d_shell(y)/√(2·log(N_eff(y)/ε))`).
  Validated: σ*(y) tracks density (opening 0.057 → endgame 0.111 vs global 0.097),
  de-percolates dense regions (overlap-degree 309→35), and lifts opening legal@1
  0.849→0.907. A real, theory-demanded refinement — though the kernel's main limit is
  still its lossy context (the sufficient state fixes that). `ARCHITECTURE.md` §3.11.
* **Interface-Principle pruning** (`correction_field.py`): a kernel field's tail is
  controlled by the *boundary* subset, so `δR(y)` sums only centres within `c·σ`
  (k-d-tree ball), skipping the deep interior. **O(M) → O(M_boundary)**: speedup
  3×/14×/**34×** at M = 10k/100k/500k centres, error ~2–3% (under the tail bound) —
  the lever for scaling kernel δR memories to 10⁵–10⁶ centres. Run
  `python -m mscn.correction_field`. `ARCHITECTURE.md` §3.12.
* **Scalable MSCN coupling** (`mscn.py` + `network.scale_free_adjacency`): the same
  Interface-Principle idea on the integrated network — each agent couples only to its
  sparse (boundary) neighbours, so coupling is **O(N·k_eff), not O(N²)**. Per-agent
  cost is constant (~0.46 ms/round) from 64 → **2048 agents** (N×4 → time×4); the old
  O(N²) scan choked at ~125. All 24 guarantee checks still pass. `ARCHITECTURE.md` §3.13.

## MSCN → AGI upgrades (`agi_*.py`)

The theory team's `MSCN_AGI_ROADMAP.md` (+ `formal/AGIFoundations.lean`) lists nine
architectural upgrades toward general-intelligence *capabilities*. All nine are
implemented as runnable mechanisms, each with a **measured advantage or an honest null**
(`ARCHITECTURE.md` §8; one-shot `python -m mscn.agi_all`):

* **U1 learnable simulation** (`agi_simulation.py`) — learn `G` with `σ∘G=T∘σ` from
  **atomic** tokens (ε-machine reconstruction): sufficient + faithful, **7–9×**
  compression vs windows; honest beyond-horizon frontier (this *is* §2.2-general).
* **U2 hierarchical simulation** (`agi_hierarchical_sim.py`) — coarse-graining kills fine
  noise (corr 0.58→0.92), **2.9×** cheaper planning.
* **U3 gradient-driven exploration** (`agi_exploration.py`) — fixes deceptive Schwefel
  (+19%); honest funnel tradeoff.
* **U4 directed exploration** (`agi_directed.py`) — info-gain sampling, better coverage,
  8–14% lower regret.
* **U5 cross-domain morphisms** (`agi_transfer.py`) — basin preservation **100%**,
  warm-start **+75–93%** head-start.
* **U6 compositional coherence** (`agi_compositional.py`) — skill-library reuse, −85% /
  **5.4×** amortized.
* **U7 coherence-based planning** (`agi_planning.py`) — plan over the learned sim, goal
  **0%→100%**, diminishing returns.
* **U8 temporal abstraction** (`agi_temporal.py`) — macro-actions, **3.6×** fewer
  decisions, basin-invariance safety.
* **U9 active self-improvement** (`agi_selfimprove.py`) — cost-aware reflexive
  α-allocation, **+6.5** more domains cleared.

No AGI is claimed; the deliverable is functional mechanisms with faithful measurements
(negative/mixed findings included).

## Honest scope

This is a toy model. Agent counts are in the tens, runs are seconds, and the
optimiser is a competitive *local* search rather than a state-of-the-art global
optimiser. What it demonstrates is **mechanism fidelity**: the proven IBF
behaviours reproduce when you implement the dynamics directly, with no neural
networks and no backpropagation. Scaling the same mechanisms (efficient kernel
data structures, sparse coupling, larger hierarchies) is the engineering path
sketched in the design note.
