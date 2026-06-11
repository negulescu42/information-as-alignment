# MSCN — IBF instantiation: architecture & results note (for theory review)

**Scope.** This note documents a runnable, non-neural instantiation of the IBF
framework: (A) the **Multi-Scale Coherence Network (MSCN)** toy model of the
three-layer apparatus, and (B) an **emergent-chess** representation-learning study
that stress-tests the mechanism and connects to two formal results
(`../formal/README.md`: Operating Resolution; Causal States). Everything is in
`mscn/`; reproduce with `python -m mscn.demo | tests | benchmark` and
`python -m mscn.chess_world | chess_strategy [--pgn …]`.

Design invariant throughout: **knowledge is coherence modification `δR` over a
configuration space**, selection is Boltzmann in a responsiveness `k`, and dynamics
follow the modification ODE `δR' = α − μ·δR`. No weight matrices, no backprop.

---

## 1. The IBF unit (Layer 1)

`mscn/learner.py`. State = (configuration `x`, modification map `δR`,
responsiveness `k`). Loop: sense `R_eff = R̂ + δR` → Boltzmann-select →
act → modify (`δR(next) += α·max(discrepancy,0) − μ·δR`) → adapt `k`.

Two realizations: a discrete 1-D learner (mirrors `IBFLearner.lean`
`floatLearnerStepV2`) and a continuous `R^d` learner whose `δR` is a sum of
Gaussian kernels (the non-neural associative memory).

**Tested / proven-property checks (`mscn/tests.py`, 24/24 pass):** Boltzmann
monotone in `k` + greedy limit (Thm 7); basin expansion `viable(R_eff) ⊇
viable(baseline)`, `δR ≥ 0` (Thm 8a); selective retention `δR→α/μ` vs decay→0
(Thm 8b/10a); crystallization `μ=0 ⇒` constant (Thm 3a); Euler→ODE convergence
(Thm 11).

**As an optimiser (`mscn/benchmark.py`, eval-budget matched, 20 seeds):** mean rank
over 5 functions × {2,5,10}-D — **IBF 1.89** < SA 2.11 < hill 2.39 < random 3.61;
IBF is best of the four on every 10-D function (memory + basin expansion), weakest
in 2-D and on deceptive Schwefel. Against SOTA (`benchmark strong`): CMA-ES 1.61 <
dual-annealing 2.27 < DE 2.86 < **IBF 3.92** < SA 4.34 — IBF is not competitive with
decades-tuned global optimisers, as expected; its value is interpretability,
guarantees, and the multi-agent/representational regimes below.

---

## 2. The apparatus (Layers 2–3 + integration)

**Layer 2 — network & cooperation** (`network.py`, `games.py`).
Coupling graphs with symmetric non-negative `J`; network coherence
`Σ R_eff(x_i) + Σ J_ij R_pair(x_i,x_j)`. Verified: coupling only helps (network ≥
sum of individuals), isolated nodes independent, mean-field spread ~`σ/√N`.
Emergent cooperation in the iterated Prisoner's Dilemma: an IBF agent senses only
immediate payoff (favours defection) but accumulates *relational* coherence; result
— IBF-memory agents reach **100% mutual cooperation** with reciprocators/each other
and **defect against pure defectors**, while memoryless agents stay trapped. Causal
test (EC-4): coupling `J=0 → 4%` cooperation vs `J=4 → 100%` — the relational
coherence, not the optimistic init, breaks the trap. IBF-memory tops a 7-strategy
round-robin (2.46 ± 0.02).

**Layer 3 — meta-learning / self-knowledge / phase** (`hierarchy.py`,
`selfmodel.py`, `phase.py`). RG flow suppresses small-scale noise (S/N peaks at the
characteristic scale; peak non-increasing under coarsening); a coarse-grained
hierarchical optimiser beats a flat learner (win-rate ~1.0 on block-structured
problems). Lawvere obstruction: no self-model is surjective at any size; capacity
ceiling `⌊Γ/base⌋`; explicit consciousness–competence tradeoff. Phase transition
`E* = c + α/μ` with exact sub/critical/super structure; finite dissipative lifetime;
**Zombie-Twin** — a self-monitoring agent outlasts its non-monitoring twin in 100%
of adversarial episodes.

**Integrated MSCN** (`mscn.py`). 27 coupled learners on a scale-free graph: best
coherence rises, coupled agents reach consensus (cooperation), network coherence ≥
sum of individuals, the population recovers from periodic adversarial perturbations
via phase-controlled self-correction, and coarse-grains 27→9→3→1. Scales 9→125
agents (0.3→5 s/60 rounds; O(N²) coupling).

---

## 3. The chess instantiation (representation learning, strict no-priors)

Goal: can the IBF mechanism learn the *structure* of chess — rules, board, strategy
— from sequences of **opaque move tokens** (no board, pieces, rules, strategy)?
`python-chess` is used only to generate/parse games and as a legality oracle; it
never feeds rules to the model. Data: locally-generated legal games (for
rules/geometry) and **5,000 real Lichess Elite 2024-01 games**
(`data/elite_2024-01_5k.pgn`, Elo 2314–2950, avg 89 plies) for the headline run.

### 3.0 Is the chess model an instance of the IBF unit?

**Partly.** It instantiates the IBF unit's *coherence-memory + effective-coherence
selection* substrate, in **predictive (learn-from-observation) mode** — not the full
acting agent loop of §1. The mapping:

| IBF unit (§1) | chess instantiation |
|---|---|
| configuration `x` | game position / move-context |
| baseline coherence `R̂` | immediate prior over moves |
| modification `δR` | learned coherence over `(context, move)` — Gaussian-kernel memory in `chess_kernel.py` |
| effective coherence `R_eff = R̂ + δR` | the model's scored move distribution |
| Boltzmann selection `P(a) ∝ exp(k·R_eff)` | the predictive softmax over moves |
| MODIFY `δR' = α·discr − μ·δR` | accumulation of `δR` from observed games |

The **kernel model is the literal Layer-1 memory** (Gaussian-kernel `δR` +
Operating-Resolution bandwidth); the n-gram/VOM are the same coherence-memory
principle in discrete / variable-order form.

**What differs from the full unit (honest gaps):** (i) `δR` is fit by *batch
accumulation over a corpus*, not online discrepancy-driven self-modification while
*acting* in an environment; (ii) responsiveness `k` is a fixed softmax temperature,
not adaptively grown (the agency dynamics); (iii) decay `μ ≈ 0` (crystallized counts)
for n-gram/VOM; (iv) there is no continuous coherence **gradient flow / basin**
dynamics — context is discrete, so no `∇R_eff`. So the chess study exercises the IBF
*representation/selection* substrate on a sequence-prediction task; it does not run
the closed sense→select→**act**→modify→adapt loop. (The "acting agent" form of the
unit is exercised instead by Layer 1 in §1–2: the optimiser and the IPD agents.)
**Route A** then adds a recurrent history coarse-graining `G` (Postulate II /
recursive scale) on top of the IBF readout — it *extends* the unit, it is not part of
the basic unit.

Three model families share one interface (`predict`, `inv_vocab`, `move_salience`):

| model | context / state | mechanism | file |
|---|---|---|---|
| n-gram coherence | last ≤2 tokens (exact) | counts + interpolated backoff | `chess_world.py` |
| kernel (faithful IBF) | recency-weighted **sum** of learned move embeddings | Gaussian-kernel kNN memory, bandwidth from Operating Resolution | `chess_kernel.py` |
| recurrent VOM (**Route A**) | deepest supported move-suffix (adaptive) | variable-order recurrent state `z_t=G(z_{t-1},a_t)` | `chess_recurrent.py` |

### 3.1 Emergent rules & strategy (n-gram, real Elite)
- **Rules:** opening legal@1 **85%**, legal@5 **99%**; **38.6% exact next-move match
  in the opening** (it learned opening theory); midgame 52%; endgame 18% (long novel
  positions). vs unigram 4%, random 1.6%.
- **Board geometry (Stage 2):** embedding the 64 squares from move co-occurrence
  recovers the 8×8 grid — Procrustes disparity **0.10**, **71%** king-adjacency.
- **Strategy (Stage 3):** coherence by Elo tercile 0.042/0.043/**0.070**;
  corr(coherence,Elo) **+0.14** (narrow Elite band); model-top move beats random
  (−486 vs −700 cp) but trails human (−54); corr(coherence-diff, outcome) **+0.003**
  — a clean null: among uniformly strong players coherence does not pick the winner
  (vs +0.31 on a skill-stratified proxy, where players *differ* in strength).

### 3.2 Faithful kernel + Operating Resolution bandwidth
`chess_kernel.py` uses the optimal bandwidth `σ* = d_shell/√(2·log(N_eff/ε))`
(`OperatingResolution.lean`): `d_shell` = median local k-NN radius, `N_eff` =
participation ratio at reference scale `σ_ref=d_shell` (Path A), `ε=0.01` (computed
σ*≈0.10, N_eff≈63). **Honest finding:** the kernel model does **not** beat the exact
n-gram on legality at any novelty level (matched split, overall legal@1 0.242 vs
0.318; by novelty: seen-context 0.37 vs 0.46, **novel-context 0.11 vs 0.17**). The
recency-weighted-sum context is a lossy
window/sum map, provably not predictively sufficient (`CausalStates.lean`, toggle
counterexample); kernel smoothing over it conflates positions that differ in
legality. **The bottleneck is the context representation, not the memory mechanism.**

### 3.3 Route A — recurrent sufficient-statistic (this build)
`chess_recurrent.py`: recurrent state = deepest move-suffix with support; `G` walks
deeper or backs off (interpolated variable-order) — an IBF adaptive-resolution
coarse-graining toward the causal-state partition. **Result (matched 2000-train /
200-test split, all three models), legal@1:**

| phase | n-gram | kernel | VOM (Route A) |
|---|---|---|---|
| opening | 0.849 | 0.832 | **0.909** |
| midgame | 0.499 | 0.384 | **0.520** |
| endgame | 0.178 | 0.104 | **0.181** |
| all     | 0.318 | 0.242 | **0.331** |

(next-move acc@1, all: n-gram 0.111, kernel 0.095, **VOM 0.125**; opening acc@1
rises 0.38→0.45.) So **VOM > n-gram > kernel**: a deeper/adaptive recurrent statistic improves
legality and next-move accuracy, most where longer history is informative
(openings). **But it is not sufficient:** the predictive-sufficiency probe (does the
recurrent state determine the true legal-move set?) gives low legal-set purity for
*every* bounded statistic — n-gram order-2 ≈ 0.50 (coverage 0.29), depth-6 state
≈ 0.27 (coverage 0.78); none approach 1.0. This empirically confirms the theorem:
a bounded move-history suffix helps but cannot determine board state. Closing the gap
needs a **simulation-faithful** recurrent state (`σ(G(z,a)) = T(σ(z),a)`), not a
longer suffix.

### 3.4 Full IBF unit (closed acting loop) + retest

`chess_ibf.py` (`IBFChessAgent`) runs the **complete Layer-1 loop** online over the
game stream — `R_eff = R̂ + Σ_ℓ w_ℓ δR(c_ℓ,·)`, Boltzmann-`k` selection,
discrepancy-driven modification `δR' = α·max(T_ext−R_eff,0) − μ·δR` with
selective-retention decay, and adaptive `k` (agency). The unit properties manifest:
**`k` grows 1.0→8.0 (agency rate ≈ 0.9)** and δR shows selective retention.

**Retest (matched split, legal@1 / legal@5, all):**

| model | legal@1 | legal@5 |
|---|---|---|
| batch n-gram | 0.306 | 0.586 |
| batch VOM (Route A) | 0.331 | 0.597 |
| **full IBF unit**, crystallization + strong drive (`μ=0, T_ext=50`) | **0.299** | **0.579** |
| **full IBF unit**, acting dynamics (`μ=0.04, T_ext=4`) | 0.096 | 0.232 |

μ-sweep (legal@1): `μ=0` 0.205 → `0.005` 0.138 → `0.02` 0.110 → `0.06` 0.086 —
legality falls **monotonically as forgetting rises**.

**Reading.** (1) The full unit **subsumes the batch predictor**: in the
crystallization limit (`μ=0`, Thm 3a) with strong drive (so `δR` tracks frequency)
and a broad readout it *recovers* batch legality (0.299 ≈ 0.306). So the n-gram
substrate is the `μ→0` special case of the full unit. (2) The unit's *characteristic*
dynamics — **forgetting (`μ>0`) + coherence-saturation + agency** — **trade off
against stationary exhaustive prediction**: they prune memory to the recently/strongly
reinforced moves, but legality rewards remembering *every* legal continuation. These
dynamics are matched to **acting in persistent / non-stationary environments** (where
the MSCN optimiser and IPD agents benefit), not to memorising a fixed corpus. So
"make the chess model a full IBF unit" is faithful and instructive, but for the
*prediction* task the right operating point is the crystallization limit; the
acting-unit dynamics are a liability here by design.

### 3.5 Route A, full — a simulation-faithful state (closes the gap)

`chess_simstate.py` reconstructs the **board** as the recurrent state, prior-free.
A move token in from-to form is an **occupancy transfer**; replaying transfers gives
`z` = {location → piece-lineage tag}, i.e. `G` = occupancy transfer and `σ` =
board decode (`σ(G(z,a)) = T(σ(z),a)`, `recurrent_sufficient_of_simulation`). No
grid, no piece types, no rules are supplied — the only structure beyond atomic
tokens is that a move has a source and a destination location (the "from-to pairs"
encoding). The **start position is discovered from data** (a location used as a
source before ever being a destination is initially occupied → **32 start
locations**, exactly chess's starting pieces).

**Sufficiency (the headline):** grouping test positions by `z`, the reconstructed
board has **legal-set purity 0.998** — it *determines* the legal-move set — versus
~0.01 for move-history-suffix states. The board is predictively sufficient; Route A
**closes the state-tracking gap** the theorem identified.

**Legality (sim-state masks impossible moves on the VOM predictor):**

| phase | VOM legal@1 / @5 | → sim-state legal@1 / @5 |
|---|---|---|
| opening | 0.909 / 0.992 | **0.945 / 0.999** |
| midgame | 0.520 / 0.888 | **0.693 / 0.972** |
| endgame | 0.181 / 0.451 | **0.377 / 0.768** |
| all     | 0.331 / 0.597 | **0.503 / 0.833** |

Overall legal@1 **0.33→0.50**, legal@5 **0.60→0.83**; endgame **doubles** — exactly
the regime where the bounded-history models failed. (Honest caveats: castling moves
only the king token, en-passant/promotion are not special-cased, so purity is 0.998
not exactly 1; and the predictor still *ranks* legal candidates by move-frequency —
masking removes impossible moves but full board-conditioned ranking is the
engine-level problem, so next-move acc@1 rises only modestly 0.125→0.148.)

**Reading.** This is the empirical realization of Route A: a recurrent state that
*simulates* the process is sufficient (purity ≈1) and substantially closes the
legality gap. It is also an IBF recursive-scale / coarse-graining result — the board
is the coarse-grained sufficient statistic of the unbounded move history, recovered
without rules. The remaining gap (purity 0.998→1, ranking quality) is castling/
en-passant/promotion bookkeeping plus board-conditioned strategy, not a structural
ceiling.

### 3.6 Board-conditioned IBF agent — and the ranking frontier

`chess_board_ibf.py` feeds the sufficient board state into the IBF coherence: it
learns a board-conditioned modification `δR(piece-lineage → destination)` with full
IBF-unit dynamics (online discrepancy reinforcement, selective-retention decay,
adaptive `k` — `k` grows 2→8), and predicts by mixing (`λ`) this board signal with
the move-context coherence, occupancy-masked.

**Retest (matched split):**

| λ | legal@1 | legal@5 | acc@1 | top-move quality (cp) |
|---|---|---|---|---|
| 0.0 (sim-masked context) | 0.500 | 0.833 | 0.145 | −599 |
| 0.5 (board-conditioned mix) | 0.505 | 0.836 | 0.146 | −592 |
| 1.0 (board feature only) | 0.291 | 0.753 | 0.012 | — |

(human actual move quality: **−14 cp**.)

**Honest finding.** Board-conditioning via the piece→destination feature gives only a
**marginal** lift in ranking (acc@1 +0.001, move-quality −599→−592 cp), and the
agent's top move is far worse than the human's (−592 vs −14 cp); the feature *alone*
(`λ=1`) is a weak predictor. So the sufficient board state's value is overwhelmingly
**legality (sufficiency)**, *not* selecting which legal move is strong. Strong move
*ranking* requires position **evaluation** — material/activity/threats/king-safety —
which a destination-affinity table cannot represent. That is genuinely the
engine-level frontier and the **real Stage-3 strategy lever**: learn a coherence
*landscape over board states* (position value, e.g. from outcomes/strong play) and
select the move whose resulting board has highest coherence — the IBF
gradient-flow/basin formulation applied to the board, rather than a move-affinity
memory. The board-conditioned agent confirms the diagnosis; closing it is the next
research step.

### 3.7 Position-value coherence layer (strategy)

`chess_value_ibf.py` learns a coherence *over board states* — position value — from
game **outcomes**, and selects the move whose resulting board has highest coherence
(the IBF gradient-flow/basin formulation applied to the position). Prior-free: a
piece's **owner side** is inferred from move parity, and each piece's **value** is
learned by regressing the final board's material onto the result (Widrow-Hoff = IBF
discrepancy modification of the value coherence).

**Emergent piece values (from outcomes only — no values/types/rules given),
relative to pawn = 1:** Q **2.7** · B 1.5 · R 1.3 · P 1.0 · N 0.9 · K 0.3. The
**queen emerges as most valuable**. Honest imperfections: the rook is undervalued and
the scale compressed; and regressing over *all* positions (not final ones) mis-ranks
the queen *lowest* — it weights pieces by capture frequency (pawns) rather than
importance — so the final-position signal is essential.

**Move selection.** Pure value-greedy *fails* — the occupancy state has no movement
legality, so it chases impossible captures (`d1d8`, `b2g7`) at −1542 cp. Ranking by
value **among the context model's plausible moves** (which respect learned movement),
with a 1-ply material lookahead, gives sound play:

| selector | move quality (cp) | acc@1 |
|---|---|---|
| context only (sim-masked) | −606 | 0.141 |
| **value-guided** | **−149** | 0.049 |

(human actual move ≈ +61 cp.) Move quality jumps **−606 → −149 cp** — the value layer
grabs free material and avoids hanging pieces. **But acc@1 *drops* 0.14 → 0.05**:
value-greedy ≠ human *strategy* (strong players play positionally, not just for
material), and the quality oracle is itself material-based (favouring a material
agent). So the position-value layer delivers **sound material play and emergent piece
values**, confirming the coherence-landscape-over-boards approach — while
human-level play needs **positional** coherence (activity, king safety, threats)
beyond material: the engine-level frontier.

### 3.8 Positional coherence layer (strategy frontier)

`chess_positional_ibf.py` enriches the value landscape with an emergent
**piece-square table** `pst[lineage, location]` (how much a piece on a square
correlates with winning, learned from outcomes), and selects with the IBF-agency
mixture `score(m) = log P_context(m) + β·(R(z'_m) − R(z))` — the human move-prior
nudged by the value gradient (1-ply lookahead), *not* pure value-greedy.

**Emergent positional structure (prior-free):** the learned pawn piece-square value
**rises with advancement** — white-pawn PST by rank: 2:0.002, 4:−0.001, 6:0.013,
7:0.022, 8:0.020 — i.e. advanced pawns (toward promotion) are valued higher, learned
from outcomes alone. Positional knowledge emerges, as piece values did.

**Selection (β sweep):**

| β | acc@1 | move quality (cp) |
|---|---|---|
| 0.0 (context only) | 0.146 | −528 |
| 1.0 | 0.146 | −412 |
| 2.0 | 0.146 | −367 |

The mixture **improves objective move quality (−528 → −367 cp) while preserving
human-matching (acc@1 ≈ 0.146)** — the correct combination, vs pure value-greedy
which crashed acc@1 to 0.05. **Honest limit:** acc@1 does **not** rise — an
outcome-regressed material+PST eval with 1-ply lookahead plays *sounder* but not more
*human*; matching 2400+ strategy is the genuine engine-level frontier, and the
quality oracle is itself material-based (some circularity). So positional coherence
**emerges and helps objective play**, but human-level strategy needs richer positional
coherence (mobility, king safety, threats, deeper search) or a stronger learner — the
open frontier the layer cleanly localizes.

### 3.9 Playing strength — head-to-head (referee-judged)

acc@1 (matching strong-human moves) plateaus and move-quality is confounded by the
material oracle, so strength is measured **honestly** by playing agents against each
other with python-chess as **referee** (legality + result only — never given to the
models): each agent proposes its ranked moves, the referee plays the highest-ranked
*legal* one, and the actual game outcome decides. The value agent adds **static
exchange evaluation** (SEE) on the emergent piece values (resolve capture sequences
via the occupancy state's attackers) and selects `log P_context(m) + β·SEE-value(m)`.

**Result (40 games):**

| matchup | wins–draws–losses | score |
|---|---|---|
| value+SEE **vs** context-only | **40 – 0 – 0** | **1.00** |

The value+SEE agent is **decisively the stronger player** — it grabs material soundly
and avoids hanging pieces, while the context-only (move-frequency) model blunders once
out of book. So **playing strength is pushed up clearly** by the emergent-value +
tactical-search coherence.

**Honest caveats.** The referee decides by checkmate or end-material, which aligns
with the value agent's material objective; still, the agent achieves material
superiority in *actual play* (sound chess) — a real strength gap. And **acc@1 stays
≈0.146**: stronger *play* is not the same as predicting 2400+ human *move choice*,
which remains the engine/large-learner frontier. Summary of the strength push:
sufficiency/legality (Route A) → emergent values → SEE tactics gives a non-neural,
prior-free agent that plays *sound* chess and crushes the context-only baseline;
human-level *prediction* is the part that needs a fundamentally stronger learner.

### 3.10 Deeper search (more strength) + the acc@1 lever (data, not search)

`chess_search_ibf.py` runs negamax/alpha-beta where the **context model is the move
generator** (plausible occ-valid moves per node), the emergent **position value**
scores leaves, and occupancy transfer applies moves — no hand-coded movement rules.

**More strength (arena, referee-judged):** depth-2 search vs the 1-ply value+SEE
agent = **12–12–0 (0.75)**; vs context-only = **24–0–0**. Deeper search is the
stronger player — Phase-1 goal met.

**Cracking acc@1 — two findings:**
1. **Search does *not* do it.** Mixing the search value into selection
   (`log P_ctx + β·minimax`) leaves acc@1 flat-to-negative: β=0 → 0.149, β=1 → 0.146,
   β=3 → 0.143, β=6 → 0.124 (it *diverges* from human moves as value-weight grows).
   Stronger *play* ≠ predicting strong-human *move choice* — confirmed again.
2. **Data does.** The move-predictor is data-limited; acc@1 scales monotonically and
   is **not saturated**:

   | train games | 500 | 1000 | 2000 | 4000 |
   |---|---|---|---|---|
   | acc@1 | 0.090 | 0.100 | 0.112 | **0.120** |
   | opening | 0.348 | 0.360 | 0.389 | 0.395 |
   | rest | 0.057 | 0.067 | 0.076 | 0.085 |

   acc@1 rises ~33% over an 8× data increase, in both opening and midgame, with no
   plateau. So the route to higher acc@1 is **more games** (we used a 5k slice of the
   235 MB file) and ultimately a **stronger learner** (a neural predictor à la Maia
   reaches ~0.5 with millions of games) — *not* search/value on this substrate.

**Reading.** Strength and human-prediction are different objectives: search/value
push *strength* (depth-2 beats every prior agent), while acc@1 is governed by the
*move-predictor's* data and learner capacity. Both levers are now identified and
quantified; acc@1's is data + learner, and is demonstrably unsaturated at 5k games.

### 3.11 Resolution-roadmap validation: per-query local σ* (and percolation)

A colleague's resolution-principle roadmap notes (correctly) that the locality
theorem is stated *per query point y* (`Keystone.lean`), so the kernel should use a
**local** `σ*(y) = d_shell(y)/√(2·log(N_eff(y)/ε))`, not a single global σ. Chess is
the heterogeneous case the theory warns about (dense openings vs sparse endgames).
Implemented in `chess_kernel.py` (`local_sigma=True`, `_local_operating_bandwidth`)
plus a percolation diagnostic (`overlap_degree`, §2.4).

**Validated, exactly as predicted:** σ*(y) tracks local density — opening **0.057**
(dense), midgame 0.076, endgame **0.111** (sparse) — vs the global 0.097. The
overlap-degree (centres within 3σ — the percolation proxy) at the *global* σ is
**309** (heavily overlapped); local σ* cuts the dense-opening degree to **35**, i.e.
it de-percolates exactly where the global bandwidth bled. Legal@1 improves, most in
the dense opening:

| phase | kernel-global | kernel-**local** |
|---|---|---|
| opening | 0.849 | **0.907** |
| midgame | 0.398 | 0.400 |
| endgame | 0.114 | 0.116 |
| all | 0.257 | **0.266** |

**Reading.** The §1.1 correction is right and helps — per-query σ* is a real,
theory-demanded refinement, and the percolation diagnostic (§2.4) flags the dense
over-overlap it fixes. But it does **not** close the kernel→n-gram gap on its own:
the kernel's dominant limitation is its *lossy context* (the recency-weighted sum —
not predictively sufficient, `CausalStates.lean`), which bandwidth cannot fix. So the
roadmap correctly identifies a valid architectural fix (bandwidth heterogeneity),
while the larger lever for this model was the *representation* — the sufficient board
state (§3.5), which is itself another roadmap item (§2.2, causal-state discovery).

### 3.12 Resolution-roadmap: Interface-Principle pruning (O(M) → O(M_boundary))

Roadmap §1.2: the Interface Principle (`InterfacePrinciple.lean`) proves a kernel
field's tail is controlled by the **boundary** subset — a centre at distance > r
contributes ≤ `V_max·exp(-r²/2σ²)`. So `δR(y)` can be summed over only the centres
within `r = c·σ` of the query (a k-d-tree ball), skipping the deep interior, with
total error ≤ `V_max·M·exp(-c²/2)`. Implemented in `correction_field.py`
(`CorrectionField.eval_pruned`).

**Benchmark (dim 8, fixed local density, growing volume):**

| M | M_boundary | full ms/q | pruned ms/q | speedup | max error |
|---|---|---|---|---|---|
| 10 000 | 81 | 0.37 | 0.14 | 3× | 2.5e-2 |
| 100 000 | 115 | 4.07 | 0.30 | 14× | 2.0e-2 |
| 500 000 | 137 | 26.5 | 0.77 | **34×** | 3.2e-2 |

Full evaluation is O(M); pruned stays ~O(M_boundary) (≈ constant), so the **speedup
grows with M** (34× at 500k, and unbounded as M→∞) while the error stays ~2–3% —
far under the conservative tail bound. This is what lets a kernel `δR` memory scale
to 10⁵–10⁶ centres; it is the drop-in evaluation for large static fields (an
`IBFLearner` at scale, the MSCN coupling). **Caveat (`no_shielding_equal_weights`):**
pruning relies on genuine Gaussian distance-decay — it does **not** apply to the
count-based, distance-free n-gram/VOM models (no metric to prune on).

### 3.13 Resolution-roadmap: scalable MSCN coupling (O(N²) → O(N·k_eff))

Roadmap §3.5. The integrated MSCN coupling was O(N²) (the implementation scanned all
pairs). But the relational coupling `J·R_pair` decays with state distance and each
agent only couples to its sparse (boundary) neighbours — the Interface Principle
applied to coupling. Refactored to **sparse adjacency** (`network.scale_free_adjacency`,
O(N·m), no dense N×N matrix); coupling and all network metrics now run over the edge
set only.

**Scaling (Rastrigin-2D, 30 rounds):**

| agents | time (s) | ms/agent/round | best coh | consensus |
|---|---|---|---|---|
| 64 | 0.90 | 0.467 | 3.03 | 1.76 |
| 256 | 3.48 | 0.453 | 6.59 | 1.62 |
| 1024 | 14.0 | 0.457 | 14.5 | 1.53 |
| 2048 | 28.0 | 0.456 | 18.1 | 1.52 |

The per-agent-per-round cost is **constant** (~0.46 ms) — total cost is **O(N)**:
N×4 → time×4. MSCN now reaches **2048 agents in 28 s**; the old O(N²) scan choked at
125 (≈5 s/60 rounds, → ~10+ min extrapolated to 2048). Behaviour is preserved (all 24
guarantee checks pass; best coherence rises with N, coarse-graining now goes
2048→683→…→1). For *dense* (all-to-all) coupling intent, a per-round k-d-tree ball
over agent states gives the same O(N·k_eff) (operating bandwidth for coupling). The
O(N²) ceiling flagged in the earlier scaling note is removed; the network scales to
1000+ agents.

### 3.14 Resolution-roadmap: per-center adaptive μ (honest, mixed)

Roadmap §1.3: per-center decay `μ_i` — crystallise repeatedly-reinforced centres
(μ→0), keep one-off centres plastic (μ>0). Implemented in `IBFChessAgent`
(`adaptive_mu=True`, `μ_i = μ/(1+λ·count_i)`) and stress-tested on non-stationary
streams (`nonstationary.py`).

- **Stationary (chess): a clear win.** Adaptive μ recovers most of the
  crystallisation advantage over a fixed μ: legal@1 **0.096 (fixed μ=0.04) → 0.147
  (adaptive) → 0.205 (μ=0)** — frequent legal moves crystallise while rare ones still
  fade.
- **Non-stationary: the count scheme is the *wrong* signal.** On a mixed-timescale
  stream it *crystallises now-stale patterns* and fails drift (mean 0.50, ≈ μ=0),
  while a tuned fixed μ=0.4 gets 0.765. **Error-gating** (`μ_i = μ·recent_error_i`,
  forget centres that became wrong) is the correct signal (0.743) but only *matches*
  a fixed μ tuned to the change rate — it is not a clear win.

**Reading.** §1.3's premise (μ>0 for non-stationarity) holds — crystallisation fails
drift — but the proposed *count-based* `μ_i` is counterproductive there (it locks in
stale knowledge); the right adaptive signal is recent prediction *error*, and even
then adaptive μ ties rather than beats a well-tuned fixed μ on single/dual-timescale
tasks. Constructive feedback for the resolution team: key `μ_i` on error, not count;
adaptive μ's real edge needs strongly heterogeneous timescales (no single optimal μ).

### 3.15 Resolution-roadmap: hierarchical σ* — RG flow as a σ*-sequence (§2.1/§3.1)

Roadmap §2.1/§3.1: read the renormalisation-group flow (`hierarchy.py`) as a
**sequence of operating bandwidths** `σ*_0, σ*_1, …` — one per coarse-graining level —
so coarsening carries its own Operating Resolution at each scale rather than an
arbitrary block size. Implemented in `hierarchical_sigma.py` (reuses the kernel's
`operating_bandwidth` and the Path-A `(d_shell, N_eff)` read-off).

Operating Resolution depends only on the **local** geometry, and one RG step rescales
the lattice spacing by `factor` while leaving the local neighbour pattern
self-similar. So on a 1024-point multi-scale field (signal λ=64 + white noise,
factor 2):

| level | #centres | spacing | d_shell | N_eff | σ* | ratio | S/N |
|---|---|---|---|---|---|---|---|
| 0 | 1024 | 1 | 4 | 7.77 | 1.10 | — | 0.61 |
| 3 | 128 | 8 | 32 | 7.73 | 8.78 | 2.00 | 3.86 |
| 5 | 32 | 32 | 128 | 7.58 | 35.2 | 2.00 | **3.88** |
| 6–8 | 16→4 | — | — | ↓ | — | — | finite-size |

**Validated (three predictions, scaling regime size ≥ 4k):** (1) σ* is **monotone
increasing** under coarsening (Operating-Resolution monotonicity: lower N_eff / larger
d_shell ⇒ larger σ*); (2) the per-step ratio is **exactly the RG factor** (2.00 across
every scaling level) because the local `N_eff` is scale-invariant (7.7 throughout) —
the **RG semigroup acts on the operating resolution as a pure rescaling**; (3) the S/N
peaks at the level whose scale **brackets the signal correlation length** (spacing 32,
σ*≈35, λ=64) — the σ*-sequence is the RG flow's natural ruler and yields a **principled
stopping rule** (finer σ* over-resolves noise = irrelevant operators; coarser washes the
signal out). At the coarsest levels (centres < 4k) σ* saturates — the honest finite-size
signal that the flow has reached the system size; reported, not asserted. This is the
multi-scale-resolution foundation under the hierarchical-simulation upgrade (§4.2).

### 3.16 End-to-end MSCN chess player — honest benchmark vs Maia/LLM (`chess_mscn_player.py`)

Composes the **full apparatus** into one prior-free agent — U1 board sim-state (legality)
+ emergent material value + U2 hierarchical positional eval (PST) + U7 negamax planning
over the learned simulation — and benchmarks it on real Lichess Elite (2250-game train /
220-game held-out test) on the **two** senses of "human-level". No LLM is run (PyPI-only
sandbox, no model API/weights); comparators are **published**.

**Playing strength** (self-play referee, randomised openings so games are distinct):

| MSCN end-to-end vs | random | context-only | value+SEE |
|---|---|---|---|
| score | **0.93** | **0.85** | **0.78** |

**Human-move matching (acc@1)** and legality:

| phase | legal@1 | legal@5 | acc@1 |
|---|---|---|---|
| opening | 91.6% | 99.9% | **39.6%** |
| midgame | 69.7% | 97.3% | 24.6% |
| endgame | 37.8% | 76.7% | 7.5% |
| all | **50.3%** | 83.4% | **14.6%** |

(search-played move acc@1 = 15.3% — **flat** vs the 14.6% move-prior: stronger play does
*not* improve human-matching.) Published comparators: **Maia ~0.50 acc@1** (neural,
millions of games); **gpt-3.5-turbo-instruct ~1750 Elo / 99.8% legal**; **Karvonen
chess-GPT (50M) ~1500 Elo**; random/unigram ~0.02 acc@1.

**Honest verdict.** The end-to-end apparatus is the **strongest prior-free agent** — it
beats random (0.93), context-only (0.85) and value+SEE (0.78), i.e. depth-2 planning over
the hierarchical emergent value dominates the 1-ply and context baselines — but it is
**club-level at best, not human-level (2400 Elo)**, and **human-move acc@1 plateaus at
~0.15, far below Maia's ~0.50**, with search/value unable to lift it. The gap is **data +
a neural-capacity value/policy learner** (5k games vs Maia's millions; the non-neural
substrate caps out), *not* the architecture — composing every upgrade end-to-end confirms
§3.10 cleanly. (Methodological note: deterministic agents collapse self-play to two
repeated lines; the strength numbers use randomised openings to be statistically real.)

### 3.17 IBF vs an LLM (Haiku) — positional head-to-head (`chess_vs_llm.py`)

The user asked for an IBF-vs-LLM tournament. A full-game tournament needs a model call
per move; this sandbox's only channel to Haiku is the Agent tool (no subprocess API
key), impractical per-move — so we run the tractable form: on **30 real Elite positions**
(10 per phase) compare the move chosen by the IBF player, by **Haiku** (batched Agent
call, given the FENs), and by the human, on **legality / human-match / cp quality**.

| mover | legal@1 | acc@1 | move quality (cp) |
|---|---|---|---|
| human (Elite 2400+) | 100% | 100% | −101 |
| **IBF end-to-end player** | **100%** | **23%** | −101 |
| **Haiku (LLM)** | **37%** | 10% | −67* |

**Result.** On identical real positions the **grounded IBF player is the more reliable
mover**: **100% legal** (its learned board model masks illegal moves) and 23% human-match,
vs **Haiku's 37% legal / 10% match** — Haiku emits **null moves** (`d4d4`), impossible
piece moves, and non-UCI tokens because it has **no reliable board model from FEN**.
(*Haiku's −67 cp is **survivorship-biased**: averaged over only its 37% legal moves,
mostly easy opening positions; its illegal moves are real-game forfeits.) So the LLM has
genuine book/tactical knowledge but is **ungrounded**, while the prior-free IBF player is
**grounded but shallow** — and both are far below human. Honest caveat: a *chess-tuned*
or board-API-equipped LLM (gpt-3.5-turbo-instruct is ~99.8% legal) would erase this
legality gap; small general Haiku from raw FEN is the weak case. The interesting, real
finding is that a non-neural state machine **wins the grounding axis outright** against a
general LLM.

### 3.18 IBF strength tournament — does stronger search help a weak eval? (`chess_tournament.py`)

Pushing strength: added **quiescence search** (extend captures at leaves, the horizon-
effect fix), **MVV capture-ordering**, and a **depth knob** to the search agent
(`StrongSearchIBFAgent`) — no new priors. Round-robin (referee-judged, randomised
openings, 14 games/pair).

| rank | agent | tournament score (max 4) |
|---|---|---|
| 1 | **search-d3+q** | 2.68 |
| 2 | search-d2+q | 2.68 |
| 3 | search-d2 | 2.50 |
| 4 | value+SEE | 1.25 |
| 5 | context-only | 0.89 |

**Result.** Stronger search **genuinely helps**, even on the shallow material+PST eval:
**quiescence d2+q vs d2 = 0.64**, **depth d3+q vs d2+q = 0.61** — both real gains, with a
clean transitive ladder (search ≫ value+SEE ≫ context). (A 4-game pilot had falsely
shown "quiescence hurts" — pure small-sample noise; 14 games/pair resolves it.) So the
classical levers (quiescence, ordering, depth) **do** lift strength on the prior-free
substrate — "we can do better" holds on the strength axis. It is still club-level: the
ceiling is the **eval** (material+PST, no king-safety/mobility/strategy), which is the
next lever (see the discrepancy-vs-oracle training, §3.19).

### 3.20 D2c — discovering the from-to factorization from ATOMIC tokens (`chess_factor_discovery.py`)

Route A was *given* the from-to decomposition; §8.10/§8.11 sharpened why (structure
buys estimability). The deepest remaining representation question: is the structure
itself statistically recoverable from atomic opaque tokens? Protocol: thresholds
frozen on a 1000-game dev audit; discovery on 3500 games; all evaluation (ground
truth, probes) on 800 held-out games.

**Signals** (audited first): same-player +2 *continuation* pairs (to(x)=from(u),
precision 0.76 at the frozen gate); co-neighbourhood edges (tokens sharing ≥3
continuation successors are same-TO, 0.977; sharing ≥3 predecessors are same-FROM,
0.942); and two **exact impossibility relations** (0 violations in 153k events)
used as hard cannot-link constraints. **Method lessons, each measured**: plain
union-find cascades (at 0.977 edge precision, ~30 wrong edges chain true squares
together — pair precision collapsed to 0.02); the avalanche-proof rule is
support-based agglomeration (≥2 independent edges between clusters). Grouping
**purity is not a valid headline metric** here — at this sample size most states
are singletons, and a *shuffled* factorization scores 0.97; the honest probe is
Matthews correlation of occupancy *changes* (a garbage state scores ≈0).

**Results (held-out):**

| metric | discovered | given from-to (bound) | shuffled control |
|---|---|---|---|
| from+to both correct (occurrence-weighted, best bijection) | **0.503** | 1.0 | ~0.0002 |
| to-square co-clustering precision / recall (top-400) | 0.871 / 0.703 | — | — |
| stream coverage (tokens factored: 757) | 0.737 | 1.0 | — |
| occupancy change-MCC (the functional probe) | **0.107** | 0.946 | −0.03 |
| prior-free self-inconsistency (transfers from empty source) | **0.081** | 0.017 | 0.369 |

**Reading (a partially-positive result with a sharp structural finding).** Half of
all token occurrences get their full (from, to) factorization exactly right,
discovered from co-occurrence statistics alone — the structure *is* substantially
recoverable. But the functional payoff is small (change-MCC 0.11 vs 0.95): **the
occupancy replay is an error amplifier** — a position's reconstruction is a long
product of per-move correctness, so board-state sufficiency demands *near-perfect*
per-token accuracy. Route A's given structure was structurally necessary, not a
convenience: §8.10's "structure buys it cheaply" is now quantified at the
functional level. The named next mechanism is visible in the table: the
**prior-free self-inconsistency diagnostic** separates discovered from garbage by
4.6× *with no oracle* — an EM refinement (re-assign tokens to minimise
transfers-from-empty) has a gradient to descend; closing the loop between
factorization and replay-consistency is the open follow-up.

Run: `python -m mscn.chess_factor_discovery` (~6 min; all asserts green).

### 3.21 The EM replay-consistency loop — a null with a mechanism (`chess_factor_em.py`)

§3.20's named follow-up, pre-registered as a question: *does descending the
prior-free self-consistency objective improve the ground-truth factorization?*
Built properly: a comparable objective J = (misses + skips)/occurrences (a skipped
transfer corrupts the state exactly like a miss — without this, coverage growth is
unaccountable and J is incomparable across rounds, measured), three guarded
M-moves (mutual-best location merges, bounded re-assignments, coverage
assignments), every batch trial-and-reverted against J, exact (A)-vetoes
throughout.

**Result: a clean null with a sharp attribution.** J descends 30% (0.320 → 0.225,
fixed point in 4 rounds) and coverage rises 0.74 → 0.84, but exact accuracy
*dilutes* (0.503 → 0.447 — under J, a half-right assignment profitably beats a
skip) and held-out change-MCC stays flat (0.115 → 0.110). Every proposed location
merge was rejected by the guard. The attribution diagnostic (oracle-assisted,
diagnosis only): **injecting TRUE assignments for 60 wrongly-assigned tokens makes
J slightly *worse* (+0.004)** — the objective, not the search, is the binding
limit.

**Reading.** Replay-consistency separates factorization *classes* (true 0.017 /
discovered 0.081 / shuffled 0.369) but is **truth-blind within the neighbourhood
of a partial solution**: below a correctness threshold, the corrupted majority
defines local consistency, so truth looks deviant against the noise it sits in —
the objective locks in the noise. This is a general caution for self-supervised
consistency objectives in error-amplifying replay systems. The named next
mechanism is therefore not a better optimiser but a better *state to score
against*: a trust-region bootstrap that replays only the high-confidence core
(pair-precision 0.87 tokens), scores candidates against that cleaner partial
state, and extends the core gradually.

Run: `python -m mscn.chess_factor_em` (~6 min; asserts green — J descends, ground
truth not degraded; the null is the registered answer).

---

## 4. Theory connections (what instantiates what)

- **Operating Resolution** (`formal/`) → kernel bandwidth `σ*` in `chess_kernel.py`;
  also the principled bandwidth story for all Gaussian-kernel `δR` memories.
- **Causal States** (`formal/`, theory-team response to `NOTE-state-tracking-gap.md`)
  → defines the Route-A target. Route B: sufficient ⇔ refines the state partition
  (causal-state = unique coarsest sufficient rep; IBF coarse-graining valid iff
  sufficient). Route A: recurrent state sufficient iff it simulates the process
  (homomorphism). Window-insufficiency is a theorem (toggle process). The kernel
  and VOM results are the empirical face of these statements.
- MSCN guarantees map to Thms 1–11 + Lawvere + reflexive-coherence results; see the
  table in `mscn/README.md`.

---

## 5. Honest findings & limitations

1. **IBF optimiser** is competitive with classical heuristics and best in high-D
   multimodal landscapes, but below CMA-ES/DE/dual-annealing on raw black-box search.
2. **Kernel chess model < n-gram** — lossy context, not the memory mechanism.
3. **Route-A VOM > n-gram** but **not predictively sufficient** (bounded-history
   ceiling): the deep recurrent state still doesn't determine legality.
4. **Strategy null among equals** (+0.003 outcome corr on Elite): coherence measures
   strength *relative to the training distribution*, not who outplays whom at parity.
5. **Endgame collapse** (legal@1 ≈ 0.18): long, novel late positions — the
   state-tracking gap in its purest form.

---

## 6. Open problems / next steps

1. **Simulation-faithful recurrent state (the real Route A):** learn `G` with a
   decoding `σ` approximately satisfying `σ∘G = T∘σ`, non-neurally and prior-free —
   e.g. ε-machine/PSR state-merging (bounded automaton that merges transpositions),
   or a factored per-square state discovered from the emergent geometry of §3.1.
   Target metric: legal-set purity → 1 and a linear board-state probe (Othello-GPT
   style) → high.
2. **Prove Route B on a tractable world** (4×4 or K+R endgame): IBF coarse-graining
   → ε-machine optimality, with legality-vs-depth and purity curves as the anchor.
3. **Scale data:** only the 5k slice is reachable here (network policy blocks the
   235 MB file and all URLs); the kernel<n-gram<VOM ordering is scale-invariant by
   the theorem, but more data helps n-gram/VOM legality and Route-A training. A
   wide-Elo (non-Elite) corpus should also turn the weak coherence↔Elo signal
   strongly monotone (the skill-stratified proxy already shows it).

---

## 7. Reproduce

```bash
python -m mscn.tests                       # 24 guarantee checks
python -m mscn.benchmark [--quick]         # optimiser + strong + cooperation + hierarchy + scaling
python -m mscn.demo                        # MSCN guided tour
python -m mscn.chess_world --figures       # emergent rules + board geometry (generated games)
python -m mscn.chess_strategy --pgn data/elite_2024-01_5k.pgn   # Stages 1-3 on real games
```
Deps: numpy (core); matplotlib (figures); scipy, cma (benchmark `strong`, kernel);
python-chess (chess). See `mscn/README.md` for the full map and `formal/README.md`
for the two theory results.

---

## 8. AGI-roadmap upgrades (implementation log)

Implements the theory team's `MSCN_AGI_ROADMAP.md` (9 architectural upgrades),
each grounded in `formal/AGIFoundations.lean`. Norm: every upgrade is a runnable
mechanism with a **measured advantage on a concrete task, or an honest null** —
no AGI is claimed; functional mechanisms with faithful measurements are.

### 8.1 Upgrade 1 — Learnable Simulation Homomorphism (`agi_simulation.py`)

The decisive lever (and roadmap §2.2-general): **learn** the simulation `G` with a
decode `σ` satisfying `σ(G(z,a)) = T(σ(z),a)` from **atomic opaque tokens** — no
from-to decomposition, no rules (`chess_simstate.py` was *given* from-to). Prior-free,
observation-only, this is exactly ε-machine / causal-state reconstruction (Route B):
a CSSR-style learner seeds a partition by the (frequent-token) legal set, **Moore-
minimises** by successor-block refinement (the determinisation / homomorphism step),
and infers **recurrently** `z_{t+1}=G(z_t,a_t)`. Grounding: `simulation_is_sufficient`
(faithful ⇒ sufficient), `bounded_history_insufficient` (the window ceiling).

**Validated on worlds whose causal states are enumerable + known:**

| world | true states | learned | purity | faithful σ∘G=T∘σ | window match | compression |
|---|---|---|---|---|---|---|
| even (canonical ε-machine) | 2 | 7 | 0.951 | 0.946 | window-6 (0.959) | 64 ctx → **9×** |
| toggle (`CausalStates.lean`) | 2 | 13 | 0.990 | 0.980 | window-5 (0.985) | 93 ctx → **7×** |

**The real, measured advantage is ε-machine *compression*** (`causal_state_optimal`,
`more_basis_less_error`): the learned recurrent state is **predictively sufficient
and faithful** while matching a fixed window's legality with **7–9× fewer states**
than the window needs contexts — a compact sufficient state, recovered from atomic
tokens.

**Honest frontier (reported, not asserted).** A dependency that persists *beyond the
learner's estimable suffix horizon* — the persistent-flag latch (purity 0.974) and
the mod-N counter (0.901) — is **not** recoverable from bounded suffixes; neither is
a fixed window. Carrying unbounded latent state needs a latent-variable model
(HMM/RNN), not suffix clustering. This is the genuine §2.2-general frontier, and
quantifies *why* Route A used the from-to token structure: it sidesteps the
statistical-estimation barrier by reading state off the token structure directly.
So Upgrade 1's mechanism is functional and its compression advantage is real; the
"learn the homomorphism with *no* structure for an arbitrarily long-range process"
case remains the open frontier, now measured.

Run: `python -m mscn.agi_simulation`.

### 8.3 Upgrade 3 — Coherence-Gradient-Driven Exploration (`agi_exploration.py`)

The base `IBFLearner` grows responsiveness `k` *monotonically* on improvement (Thm 8c
agency, capped at `k_max`) — correct for unimodal climbs but a liability on the
**deceptive Schwefel** function, the IBF optimiser's named weakness (§1): a greedy `k`
commits to a basin. Upgrade 3 makes `k` a two-sided control loop on the local
coherence gradient (entropy-production view `σ=k·‖∇R_eff‖²`): exploit on a confident
improving gradient; when the gradient stays flat (the high-uncertainty "maybe-not-
global" signal) **explore** via a memory-guided restart, the persistent `δR` memory and
`best_*` keeping the best basin. Grounding: `high_variance_favors_exploration`,
`exploration_cost_nonneg`.

**Measured (mechanism isolated: both learners local-only, so the only exploration is
the gradient-driven `k`+restart; mean best `f`, matched budget):**

| function | structure | monotone-k base | grad-adaptive | improvement |
|---|---|---|---|---|
| schwefel-5d | deceptive | 585.98 | 407.28 | **+30.5%** |
| schwefel-10d | deceptive | 1525.59 | 1409.67 | **+7.6%** |
| rastrigin/ackley | funnel | — | — | *worse* |

**Honest result.** Upgrade 3 **fixes the named weakness**: on deceptive Schwefel
(optimum isolated from the other minima) the stall-restart escapes the wrong region a
monotone-k climber commits to — **+19% mean (local-only), +3.4% on the full learner**.
The faithful flip side: on **funnel**-structured multimodal landscapes (Rastrigin,
Ackley) the restart is a *loss* — it abandons the funnel descent. So the advantage is a
real explore/exploit tradeoff **keyed to landscape structure** (helps deceptive-isolated
optima, not funnels). Second honest caveat: on the full architecture the existing global
jumps already supply most exploration, shrinking the marginal benefit. The mechanism is
functional and improves exactly the case the roadmap targeted; it is not a free lunch.

Run: `python -m mscn.agi_exploration`.

### 8.5 Upgrade 5 — Cross-Domain Coherence Morphisms (`agi_transfer.py`)

A coherence morphism `φ: F1→F2` (continuous, coherence-non-decreasing) carries learned
coherence to a related domain. Formal (`AGIFoundations.lean` §2):
`transfer_preserves_superlevel`/`transfer_preserves_viability` — φ maps basins to
basins, so a learned skill transfers as a *viable region*. Realization: the learned
`δR` is a sum of Gaussian centres `{(z_c,v_c)}`; for a structure-preserving φ (a
coordinate permutation+shift relating two landscapes of one family, so
`coh₂(φ(z))=coh₁(z)`) transfer maps every centre `z_c→φ(z_c)` and warm-starts at
`φ(best_x_source)`.

**Measured (target = φ(source); mean best f, lower better):**

| | basin preservation | budget 200 | 500 | 1500 |
|---|---|---|---|---|
| rastrigin-5d | **100%** | scratch 32.7 → **full 8.1** (+75%) | +69% | +56% |
| ackley-5d | **100%** | scratch 14.3 → **full 0.99** (+93%) | +92% | +87% |

**Honest result.** The basin/viability-preservation theorem holds **exactly (100%)** —
every source super-level point maps into the target's. Operationally, the
morphism-mapped **viable warm-start** (the theorem's guarantee) gives a **large
head-start (75–93%) at small budgets**, narrowing as from-scratch catches up — transfer
learning, the IBF way. Honest scope: **passive `δR`-memory transfer alone** is only a
few-% lift in moderate dim (≈ scratch for rastrigin-5d) — a passive memory must be
re-sampled to act, and in ≥5-D the learner rarely re-samples a transferred centre early
(the same dimensionality barrier as exploration). The decisive, theorem-backed transfer
is the mapped viable warm-start.

Run: `python -m mscn.agi_transfer`.

### 8.9 Upgrade 9 — Active Self-Improvement via Reflexive Coherence (`agi_selfimprove.py`)

Closes the reflexive loop: the monitor drives the scarce driving signal `α` to weak
domains under the consciousness-competence budget `Σα_i ≤ Γ`. Each domain improves by
the modification ODE `R_i' = α_i·k_i − μ(R_i − c_i)` (equilibrium `c_i + α_i k_i/μ`);
domain `i` clears `θ_i` iff its *sustained* allocation exceeds `cost_i = μ(θ_i−c_i)/k_i`.
Grounding: `optimal_linear_allocation'`, `self_improvement_exceeds_half_gap`.

**Measured (14 domains, scarce budget Γ = 50% of the total clearing cost, #above-threshold):**

| policy | seed0–3 above-threshold | vs uniform |
|---|---|---|
| uniform (`Γ/n` each) | ~2/14 | — |
| reflexive, naive (`α ∝ gap·k`) | ~3–4/14 | +1 to +2 |
| **reflexive, cost-aware** (fund cheapest-to-maintain first) | **9–10/14** | **+6.5 avg** |

**Honest result.** Closing the reflexive loop with a **cost-aware** allocation — drive
`α` to the cheapest-to-maintain *responsive* domains (highest `k_i/(θ_i−c_i)`, i.e.
`optimal_linear_allocation'` "concentrate on the highest rate") — clears **+6.5 more
domains** than uniform at the same budget. Important modeling point (found, not
assumed): domains need *sustained* `α` to **stay** above threshold (decay pulls them
back), so this is water-filling on maintenance cost. Honest caveat: the **naive**
reflexive rule `α ∝ gap·k` (the roadmap's first form) helps only marginally — it wastes
budget on expensive high-gap domains; the effective reflexive signal is **marginal
cost-effectiveness**, not raw gap. So the mechanism is functional and the advantage is
real, with a precise correction to the allocation rule.

Run: `python -m mscn.agi_selfimprove`.

### 8.6 Upgrade 6 — Compositional Coherence Algebras (`agi_compositional.py`)

Skills compose via the network-coherence law `R_total = Σ_i R_i + Σ_ij J_ij R_pair`
(`compositional_coherence_lower_bound'`: non-negative coupling only helps). A composite
landscape (B block sub-landscapes + bounded coupling) need not be learned
monolithically: **reuse a library of learned sub-skills** (block optima) and refine
only the residual coupling.

**Measured (4 blocks × 2-D = 8-D, bounded coupling; mean over 6 tasks):**

| | best f at 400 new evals | amortized cost (20 tasks) |
|---|---|---|
| monolithic from scratch | 53.8 | 56 000 evals |
| **compositional (reuse + refine)** | **7.9 (−85%)** | **10 400 evals (5.4× fewer)** |

**Result.** Composing reusable sub-coherences and learning only the residual coupling
reaches **85% lower objective** at the same *new*-eval budget and amortizes to **5.4×
fewer evaluations** across tasks sharing the library; the coupling-only-helps bound holds
(100% in the bounded-coupling regime). The mechanism is functional — knowledge reuse via
the composition law. (It shares the warm-start lever with Upgrade 5; the distinct content
here is the *library amortization* across many composite tasks.)

Run: `python -m mscn.agi_compositional`.

### 8.7 Upgrade 7 — Coherence-Based Planning via Imagined Trajectories (`agi_planning.py`)

With a learned simulation `G` (Upgrade 1) the agent *imagines* action sequences and
evaluates them on the coherence landscape instead of acting reactively. Grounding:
Thm 1 (gradient-flow convergence makes a rollout meaningful), `planning_horizon_value_bound`,
`planning_diminishing_returns`. Setup: a 1-D corridor with a deceptive trap bump near
the start and a higher goal peak past a low-coherence valley. The transition model is
**learned** from random exploration (tabular `G_hat`, faithful 1.0, 100% coverage — the
Upgrade-1 mechanism); planning is BFS through it to the highest-coherence reachable
state.

**Measured (goal-reaching success vs planning depth):**

| depth | 1 (reactive) | 6 | 12 | 16 | 24 | 28 |
|---|---|---|---|---|---|---|
| success | **0%** | 0% | **100%** | 100% | 100% | 100% |

**Result.** A reactive (depth-1) agent climbs onto the trap bump and oscillates —
**0%**. Planning over the learned simulation sees across the valley and crosses to the
goal — **100%** once the horizon spans it, with **clear diminishing returns** (gain
depth 1→16 = +100% ≫ gain depth 24→28 = +0%), exactly `planning_diminishing_returns` /
`planning_horizon_value_bound`. The mechanism is functional: imagined trajectories
convert a trapped reactive agent into one that solves the task.

Run: `python -m mscn.agi_planning`.

### 8.8 Upgrade 8 — Temporal Abstraction (macro-actions as basins) (`agi_temporal.py`)

A macro-action is a coherence basin: following the gradient within it converges to a
peak, coherence non-decreasing (Thm 2 basin invariance), so the macro is *safe* —
`temporal_abstraction_safety`; composing safe in-basin macros stays safe
(`macro_action_composition`). Setup: a multi-basin corridor (a chain of bumps of
increasing height, last = goal).

**Measured (mean over 8 starts):**

| | decisions to goal | reached | ascent macros safe |
|---|---|---|---|
| primitive (per-cell lookahead) | 32.5 | 100% | — |
| **macro (per-basin)** | **9.0 (3.6× fewer)** | 100% | **100% monotone** |

**Result.** Planning over macro-actions (ascend the current basin; transit to the next
toward the goal) reaches the goal in **3.6× fewer high-level decisions** than primitive
per-cell planning, and **every in-basin ascent macro is coherence-non-decreasing**
(basin invariance) — so the composed plan's ascent segments are all safe. The mechanism
is functional: temporal abstraction with the basin-invariance safety guarantee.

Run: `python -m mscn.agi_temporal`.

### 8.2 Upgrade 2 — Hierarchical Simulation (multi-scale internal models) (`agi_hierarchical_sim.py`)

Stack a coarse-grained state layer over the fine one (Postulate II / recursive scale).
By the RG results (`deep_coarsening_kills_fine_structure`, `fine_to_coarse_coherence`)
the block coarse-graining **suppresses fine-scale noise** while **preserving the
relevant large-scale structure** (the per-scale bandwidth is the σ*-sequence of §3.15).

**Measured (24×24 fine, 6×6 coarse, block 4):**

| | result |
|---|---|
| coarse argmax super-cell == goal super-cell | **True** (relevant structure kept) |
| corr with clean signal: fine → coarse | **0.58 → 0.92** (fine noise suppressed) |
| planning cost: flat fine vs hierarchical | **461 → 160 states (2.9× fewer)** |

**Result.** The coarse simulation layer keeps the goal's super-cell as its maximum and
correlates with the clean signal far better than the noisy fine field (0.58→0.92 —
coarse-graining kills the fine noise, the RG result). Planning coarse-first to get a
super-cell corridor, then refining within it, expands **2.9× fewer states** than flat
fine planning. The mechanism is functional: multi-scale internal models give noise-robust
abstraction and cheaper hierarchical planning. (Together with §3.15 this completes the
roadmap §2.1/§3.1 multi-scale-resolution story.)

Run: `python -m mscn.agi_hierarchical_sim`.

### 8.4 Upgrade 4 — Directed Exploration via Information Gain (`agi_directed.py`)

When exploring, sample the highest-**information-gain** candidate, not at random. In
IBF terms the model is the `δR` memory; the info gain of `x` is the size of the update
it would induce — large where `x` is far from every stored centre (novel). Grounding:
`directed_beats_random_exploration` (the best info-gain candidate ≥ the average).
Demonstration: active search for the global max of an 8-mode 2-D coherence function.

**Measured (mean over 30 landscapes; simple regret = gmax − best found, lower better):**

| samples | random | directed | regret cut |
|---|---|---|---|
| 15 | 0.297 | 0.394 | −32% |
| 30 | 0.229 | 0.211 | +8% |
| 60 | 0.179 | **0.155** | **+14%** |

info gain: directed **0.575** vs random 0.362; coverage gap: **0.054** vs 0.068.

**Honest result.** Directed exploration gains more information and covers the space
better (always), and once coverage matters (30–60 samples) it finds the global mode
**more reliably** (8–14% lower regret) where random can miss it. Faithful caveats: at a
**tiny** budget (15) pure space-filling spreads thin and has not concentrated near any
peak, so it does not win there; and to *refine* the best mode it must hand off to
**exploitation** (Upgrade 3) — directed exploration is the *coverage* lever, and the
intended use is the explore/exploit blend. The mechanism is functional for its purpose.

Run: `python -m mscn.agi_directed`.

### 8.10 Research finding — the U1 barrier is bounded-suffix, not fundamental (`agi_state_merging.py`)

A follow-up to U1 (§8.1) that **sharpens the §2.2-general claim**. U1's
`LearnedSimulation` keys states on the last ≤ L tokens, so it stalls on long-range
worlds (counter, flag). Is that the *estimation* barrier, or just the bounded-suffix
design? Tested with **unbounded recurrent state-merging** (RPNI/EDSM): build the
prefix-tree acceptor over *full* histories, greedily merge state-classes with compatible
futures, folding transitions deterministically — merging deep nodes into shallow ones
**creates cycles**, so the machine generalizes beyond observed depth (unbounded memory).

| world | true | suffix-CSSR (U1) | RPNI merge |
|---|---|---|---|
| even | 2 | 7 / 0.94 | 43 / 0.95 |
| toggle | 2 | 13 / 0.98 | 27 / 0.92 |
| flag (long-range, stochastic) | 2 | 80 / **0.98** | 39 / **0.84** |
| counter (long-range, deterministic) | 4 | 7 / **0.90** | 61 / **0.99** |

mod-4 counter, data sweep: **50 seqs → 4 states, purity 1.000**; 150→25/0.98; 900→92/0.98.

**Finding (genuinely sharper than "windows are insufficient").**
1. **The bounded-suffix ceiling is a design artifact, not a hard wall.** RPNI **crosses
   the counter** suffix-CSSR could not (0.90→0.99) and recovers the **exact 4-state
   machine (purity 1.000) from modest data** — it learns unbounded memory by merging
   prefixes into cycles. So the U1 barrier *moves*.
2. **The true residual is noise-robust *estimation* of the merge, not history length.**
   RPNI's exact-legal-set merge is brittle: on the *stochastic* flag world it over-splits
   and **loses** to suffix-CSSR (0.84 vs 0.98), and on the counter **more** data over-splits
   it (1.00@50 → 0.98@900). A statistical merge test (ALERGIA) or domain structure fixes this.
3. **Structure buys noise-robustness/data-efficiency, not expressivity in principle.**
   Chess's from-to token decomposition is a *deterministic, noise-free* factorization —
   exactly what RPNI lacks. Unbounded merging crosses the barrier *in principle but
   brittly*; structure crosses it *cheaply*. The next step on §2.2-general is therefore an
   ALERGIA-style statistical merge or a learned factored (register) decomposition — not a
   longer history.

Run: `python -m mscn.agi_state_merging`.

### 8.11 D2 — statistical state-merging crosses the estimation barrier (`agi_alergia.py`)

§8.10 left a precise target: a **noise-robust** merge test. Implemented: same red-blue
frame and unbounded recurrent fold as RPNI (cycles = unbounded memory), with the exact
legal-set equality replaced by a **statistical compatibility decision**. A per-node
Hoeffding test (classic ALERGIA) is *not* enough — measured: on the flag world the
mode-distinguishing evidence lives in rare-token subtrees (~2% of histories), so every
single node pair is statistically marginal and the modes collapse (purity 0.77). The
working fix is the MDI move: a **G-test pooled across the entire speculative fold**
(2·Σ O·ln(O/E) summed over every folded node pair, accepted iff below the chi-square
critical value at the pooled degrees of freedom; pairs with < 4 observations on either
side carry no evidence — the G-test is anti-conservative there), plus EDSM ordering
(most-evidenced blue first, merge into the best-fitting red by G/df, not the first
compatible one).

**Result (legal-set purity, mean over 3 data seeds, 400 train sequences):**

| world | true states | suffix-CSSR | RPNI (exact) | **statistical** |
|---|---|---|---|---|
| even | 2 | 7 / 0.94 | 43 / 0.95 | **4 / 1.00** |
| toggle | 2 | 13 / 0.98 | 27 / 0.90 | **4 / 1.00** |
| flag (stochastic, long-range) | 2 | 89 / 0.98 | 39 / 0.79 | **2 / 1.00** |
| counter (deterministic, long-range) | 4 | 7 / 0.90 | 71 / 0.98 | **5 / 1.00** |

Data sweeps (the two named §8.10 failures): counter **1.000 at every size 50→900**
(4–5 states; RPNI decays 1.00→0.98); flag **1.000 at every size** (2–3 states; RPNI
wobbles 0.72→0.77). More data now *helps*; the learned machines are exact or
near-minimal everywhere (RPNI's were 27–71 states).

**Reading.** Both §8.10 residuals — noise-robustness and data monotonicity — are
removed by one principled change, and the §8.10 crossing (unbounded memory via merge
cycles) is kept. On worlds with enumerable causal states the estimation barrier is now
**fully crossed**: prior-free, observation-only, exact recovery. What remains for
chess-from-atomic-tokens is **combinatorial state growth** (the flat ε-machine of chess
does not fit any merge table), which needs a *factored* state — a factor-discovery
study — not a better merge test.

---

## 9. IBF-ASI — the coupled single-agent mechanism (`ibf_asi.py`, spec §8)

The IBF-ASI specification (audited clause-by-clause in `IBF_ASI_GAP.md` — its cited
Lean library is **not** in this repo; the ✅ rows map to the mscn behavioural
validations) asks for one agent running the full 9-stage cognitive cycle over the
whole state tuple `(x, R̂, {δR̂ₛ, wₛ}, k, monitor, self-model, Γ)` with runtime
invariants and telemetry. Every stage existed here as a separate validated module;
**the coupled mechanism did not**. Built: `IBFASI` — SENSE (multiscale kernel
memory) → PLAN (H-step lookahead) → SELECT (Boltzmann-k) → ACT (monotone ascent
segment) → LEARN (raw-improvement discrepancy writes, fine scale) → TRANSFER
(crystallised, error-surviving centres consolidate to coarser scales / replicate to a
partner under a reciprocity ledger) → DISSOLVE (error-gated extra decay, scale-aware)
→ REFLECT (θ-monitor + boost; self-model with a measured **conflation floor** > 0 —
the Lawvere gap reported, never claimed zero) → ADAPT (two-sided k, wₛ, capacity
projection to Γ). Invariants asserted **every tick**: I1 ascent monotonicity, I2
δR ≥ 0 (basin expansion over baseline), I3 bounded below-θ transients, I4 Σ|δR| ≤ Γ.

**Measured (CI-graded — every comparison a paired per-seed difference with a 95%
t-interval, eval-matched, 16–24 seeds; see §9.1 for why earlier 8-seed numbers
were revised):**

| claim | CI-graded outcome |
|---|---|
| memory load-bearing | **decisive + significant**: full−no-memory **+1.18 [+0.88, +1.47]** coherence, **+0.21 (sig)** reflexive viability — and CI-significant in **all six regimes** of the matrix (+0.69…+0.91, every cell) |
| error-gated dissolution | **null at proper power**: −0.05 [−0.21, +0.12] at 16 seeds (an 8-seed +0.37 did not replicate); the matrix nulls it in its own drift regime too. Consistent with §3.14: error-gating only ties a tuned fixed μ — base decay already does the work |
| planning / extra scales / reflect | nulls (CIs straddle 0), incl. planning in the moat regime: a stochastic greedy rollout on a noisy sensed field is **not** the U7 planner (BFS over a learned *discrete* simulation) — lookahead pays only with a structured internal model |
| 6.1 (global-basin agency) | **directional, not significant**: +0.13 [−0.06, +0.31] at 24 seeds; and the matrix shows two-sided-k **significantly harmful** under shocks (−0.30*) and in the moat (−0.15*) — restarting abandons position exactly when position is the asset |
| 6.3 (honest budget reserve) | **null both ways**: +0.03 [−0.36, +0.42] coherence, −0.006 viability; the floor telemetry (measured conflation floor > 0, never claimed zero) is what stands |
| 6.4 (aligned interaction) | **CI-significant in the scarce-information regime** (3-D, narrow optimum): cooperation−solo joint **+0.54 [+0.01, +1.08]**; defection neither pays nor costs (+0.03 ns). In 2-D, where solo discovery is cheap, sharing is worthless — null kept |

Five design corrections were forced by measurement (each logged in
`IBF_ASI_GAP.md` §3): reinforce on **raw** sensed improvement (reinforcing R_eff
self-manufactures traps — the system-level re-confirmation of the learner.py rule);
coarse scales fill **only by consolidation**; the world's scale-stability ordering
must be physical (coarse slow, fine fast — and adversarial shocks damage the fast
modes, which is what consolidation is *for*); the agent's "best known" is the
de-noised per-centre quality EWMA, not the raw high-water mark; reciprocity is a
ledger, not a time-window, and transfer across independently-drifting worlds without
a morphism is misinformation (U5 at system level).

Run: `python -m mscn.ibf_asi [--quick]` (numpy+scipy; all asserts green).

### 9.1 Testing the architecture: CI-grading, invariant fuzzing, the regime matrix

Three testing layers were added on top of the per-module validations, and they
materially **changed the conclusions** — which is the point.

**(a) Paired-CI statistics (`stats.py`).** Every comparative claim in `ibf_asi` is
now a paired per-seed difference with a 95% t-interval, asserted on CI bounds, not
point means (institutionalising the §3.18 small-sample lesson). This immediately
killed three earlier 8-seed headlines: "dissolution load-bearing (+0.37)" (null at
16 seeds), "6.4 supported in 2-D" (null at 16 — solo discovery is cheap there), and
"6.3 costs ~3%" (null both ways). It also *established* one: cooperation in the
scarce-information regime is significant (+0.54 [+0.01, +1.08], 24 seeds). Three
mechanism bugs surfaced while chasing significance: memory-guided warm jumps were
silently **undoing stall-restarts** one tick later (fixed: restarts open a
memory-free exploration phase); a deceptive gap smaller than the fine-structure
amplitude is **physically undiscriminable** (both arms tie exactly); and a
quality-record is required because the raw high-water mark is noise-dominated.

**(b) Property-based invariant fuzzing (`ibf_asi_fuzz.py`).** The agent asserts its
spec invariants (I1 ascent monotonicity, I2 δR ≥ 0, I4 capacity ≤ Γ) on every tick,
which makes it fuzzable: 150+40 random configurations across wide ranges (dim 1–4,
0 decoys, zero noise, zero memory, tiny Γ, coupled + parasitic pairs) with
finiteness checks. It caught one real crash (`shock_point` with no decoys — fixed);
zero violations since. I3 (bounded transients) is deliberately reported, not
asserted: configurations whose equilibrium cannot reach θ are legitimately
**sub-critical** (the phase.py phase structure).

**(c) The regimes × mechanisms matrix (`ibf_asi_regimes.py`).** Six regimes
(clean / noisy / drift / shocked / deceptive / moat), six ablations, every cell a
paired CI. The map: **memory is the one universally significant stage** (all six
cells, +0.69…+0.91*); dissolution is redundant with base decay at these timescales;
this plan operator buys nothing even in the moat; two-sided-k restarts are harmful
under shocks and in the moat. A stage's value is a property of
(mechanism × regime) — the honest spec for the integrated agent is the table
itself.

Run: `python -m mscn.ibf_asi_fuzz [--n 150]` · `python -m mscn.ibf_asi_regimes
[--seeds 8]`.

### 9.2 The planning-boundary result (a pre-registered failure, mapped)

The matrix indicted the rollout planner, so a **real model-based planner** was
built into the agent (`model_planner=True`): a learned discrete cell map fed by
already-paid senses (exact eval parity), frontier-directed optimism (U4) scored on
the **R_eff scale** (a raw-scale plan is vetoed by the agent's own δR at
remembered peaks — measured), BFS through low/unknown cells (U7), U3
ascent-arbitration (don't interrupt a climb — without it the sweep drags the agent
off a freshly-found peak before it summits), and empirically-calibrated satiation
(optimism = first-visit mean + 2σ once 30% of the map is seen — a fixed bonus
sweeps forever). A locomotion-constrained regime (`moat-local`: no jump
candidates, no restart teleports, spawn outside the ring, decoys cleared off the
ring after a measured bridge-leak) was built as the planner's pre-registered
proving ground: **the (model-plan, moat-local) cell must be CI-significantly
positive.**

**Outcome: NOT MET — and the failure maps a boundary.** Measured along the way
(isolated arms, shallow→deep moats, 8–10 seeds each):

- **shallow moat** (−2.5, ~2 steps wide): reactive Boltzmann *diffusion* crosses
  anyway (3/10 end inside; downhill picks at `e^(−kΔ)` accumulate over ~250
  ticks) — planning unnecessary;
- **deep moat** (−4): nothing observable signals what is behind the barrier, so
  the crossing is a needle-in-a-haystack *exploration* problem — the undirected
  frontier sweep finds the hidden basin 1/10 isolated — planning insufficient;
- between them, no niche: the integrated and isolated paired differences are ~0
  at every depth tried; the final matrix row is −0.17…+0.27, all ns, never
  significantly harmful.

**Reading.** U7's 0→100% lived in a 1-D corridor where "beyond the trap" is the
*only* unexplored direction. Planning binds in **discrete, low-branching state
spaces** — exactly where the chess search gains live (§3.18: quiescence +0.64,
depth +0.61) — and does not bind in continuous fields, where Boltzmann
exploration channels dominate and barriers either yield to diffusion or hide
their prize. This sharpens the §8.3 honest caveat ("global jumps already supply
most exploration") into a structural statement about the PLAN stage's validity
domain.

**The positive side, pre-registered and MET (the `corridor` regime).** U7's
corridor was then built *inside* the integrated agent (1-D: start mound,
deceptive trap, wide low valley, higher goal; locomotion-constrained for all
arms). Two further mechanisms had to be measured into existence:

1. **δR-selection structurally vetoes unrealized frontiers.** Instrumented: the
   planning agent crossed the whole valley and was yanked home from **one cell
   short of the goal** — the coarse warm-jump candidate carries the trap's
   δR-inflated value (R_eff ≈ 6 vs the unconsolidated goal slope ≈ 2). The same
   memory that wins V1 (homing, shock recovery) is the anti-exploration force.
2. **U8 option-commitment is the cure**: an embarked plan is a macro-action —
   homing candidates are suspended until arrival/expiry, and the U3 gate is
   applied symmetrically (neither the planner nor warm jumps interrupt an
   ascent).

**Result: +1.41 [+1.37, +1.45] — goal reached 10/10 vs trap-locked 10/10**, the
tightest significant interval in the codebase; in the final 8×7 matrix the
(model-plan, corridor) cell is **+1.40\***, the only starred positive planning
cell, while the old rollout operator is significantly *harmful* there (−0.08\*).
Both sides of the boundary are now measured. Isolation footnote (honest): a
memory-less Boltzmann walker crosses this corridor by 1-D diffusion 10/10 — the
corridor's binding difficulty *inside the full agent* is the agent's own memory;
planning + commitment is what defeats it.

### 9.3 Full-capacity benchmark (`ibf_asi_benchmark.py`)

The integrated agent vs external baselines across all eight regimes —
eval-budget matched on a **common world clock** (time advances per 16 senses for
every agent type; locomotion regimes bound everyone's step scale; shocks damage
every memory agent's fast modes) — plus capacity scaling. 8 seeds, paired CIs.

| agent | clean | noisy | drift | shocked | deceptive | moat | moat-local | corridor | **AGG** |
|---|---|---|---|---|---|---|---|---|---|
| **full IBF-ASI** | 3.48 | 3.09 | 3.00 | 3.00 | 3.89 | 2.36 | 2.10 | **2.84** | 2.97 |
| layer-1 IBFLearner | **3.84** | **3.49** | **3.69** | **3.63** | **4.41** | **2.87** | 1.75 | 1.35 | **3.13** |
| CMA-ES (restarts) | 2.46 | 3.08 | 1.44 | 2.42 | 2.78 | 1.53 | 1.53 | n/a | 2.18 |
| reactive | 2.63 | 2.45 | 2.45 | 2.31 | 3.55 | 1.65 | 1.80 | 1.35 | 2.27 |
| random | 0.74 | 0.51 | 0.65 | 0.68 | 0.97 | 0.31 | 0.54 | 0.83 | 0.65 |

**Honest verdicts.** (1) The full agent decisively beats random (+2.32 [+1.95,
+2.69] sig), beats reactive in 6/8 regimes (sig), and beats CMA-ES on the
drifting/structured worlds (CMA collapses to 1.44 under drift — its static-world
assumptions, noted). (2) **The Layer-1 IBFLearner sig-beats the full agent in 4
open-landscape regimes and wins the aggregate (3.13 vs 2.97)** — the 9-stage
apparatus does *not* dominate its own ancestor at free-roaming optimisation
(layer-1's annealed wide steps + jumps are better tuned for it); its edge is
**structural competence**: the corridor (+1.49\* where layer-1 is trap-locked at
1.35) and moat-local (+0.35 ns). This is the §1 finding (IBF < CMA-ES on raw
black-box) reproduced one level up, with the same shape: generality costs
raw-landscape speed and buys regime robustness. (3) **Capacity scaling** (drift
regime): performance saturates at tiny memory budgets (Γ=8; flat to Γ=240 —
capacity-bounded, the v_cap/projection design), extra scales mildly cost
(S=1: 3.06 → S=4: 2.72, consistent with the multiscale nulls), experience keeps
paying (2.70 → 3.07 from 2k → 18k evals, unsaturated), and **cost/tick is flat**
(~4.6 ms, 40–70 centres — decay + capacity projection self-bound the memory, so
the O(M) kernel concern only bites at the 10⁵-centre scale where
`correction_field.py` pruning is the named remedy).

Run: `python -m mscn.ibf_asi_benchmark [--seeds 8]` (numpy+scipy+cma; ~20 min).

### 9.4 The closed online loop on K+R-vs-K (`krk_world.py`, `krk_closed_loop.py`)

The original handover's D1+D4: **one agent, one stream** — representation
(movement legality), value, and policy all learned *while acting*,
discrepancy-driven, with strength rising and gap-to-oracle falling measured
**online** against an exact anchor. KRK is the tractable world where this is
fully measurable.

**The oracle/referee** (`krk_world.py`, teacher-side like `chess_oracle.py`): a
vectorised KRK rules engine + retrograde DTM tablebase over 524,288 states (built
+ solved in ~5 s). Correctness chain: **exact** cross-validation against
python-chess on sampled positions (position legality, complete move sets,
mate/stalemate flags incl. targeted terminal samples), Bellman consistency on the
solved table, and the known KRK bound reproduced exactly (**max DTM = 16
moves**). Cached at `data/krk_dtm.npz`.

**The agent** (strict no-priors): sees three piece locations (its two pieces as
opaque persistent tags) and terminal signals only — no movement rules, no notion
of check/mate. Learns online, in one stream: (i) **movement legality** from
referee rejections (the §3.9 arena protocol: full ranked proposals, the referee
plays the highest legal one, everything above it is revealed illegal); (ii)
**value over afterstates** from terminal outcomes only, by `V += α·(G − V)` —
literally the IBF modification step (`tdError'`, `td_is_modification_step`,
AGIFoundations §11), with terminal-exact updates and truncation bootstrapping;
(iii) **policy** = Boltzmann-k over (legality coherence + k·value), realised as
Plackett-Luce ranking, k adapted on improvement (episode-level Thm 8c agency).

**Measured online (3 seeds × 150k episodes; paired CIs; random-black opponent):**

| episode window | mate rate | legal@1 | TD-err | win-preserving | DTM-optimal | k |
|---|---|---|---|---|---|---|
| 2k | 0.008 | 0.684 | 0.006 | 0.853 | 0.104 | 2.0 |
| 16k | 0.526 | 0.916 | 0.212 | 0.944 | 0.210 | 5.0 |
| 58k | 0.708 | 0.906 | 0.144 | 0.995 | 0.254 | 11.7 |
| 150k | **0.732** | **0.952** | 0.136 | **0.999** | 0.261 | 12.0 |

- **Strength rises online**: mate rate 0.008 → **0.732** (+0.72 [+0.70, +0.75]
  sig; random baseline 0.007; legality-only ablation 0.006 — the value loop *is*
  the strength, +0.73 sig).
- **The gap to the exact oracle falls online**: win-preserving moves 0.853 →
  **0.999** — by the end the agent virtually never throws away a tablebase-won
  position.
- **TD discrepancy shows the predicted shape**: rises to 0.212 as reward signal
  arrives into an empty table, then falls to 0.136 as the value coherence
  converges — the MODIFY driver doing exactly what Postulate IV says.
- Movement rules **emerge from rejections alone** (legal@1 0.68 → 0.95; the rook
  lines and king steps exist nowhere but in the learned acceptance tables).

**Honest ceilings (the finding, not a footnote).** DTM-optimality plateaus at
**~0.26** and mates take ~40 plies (optimal ≈ 16–32): the agent becomes
*safe* (never loses a won game) long before it becomes *fast* — the residual
~27% of episodes are 80-ply cap timeouts while still in won positions, not
losses. Tabular value over 262k afterstates is visited too sparsely to refine
technique beyond "preserve and shuffle toward mate". This is the roadmap's #2
open problem (**scalable representation**) made quantitative on an exact
yardstick: closing the optimality gap needs a generalising value substrate
(learned state geometry / kernel over emergent coordinates), not more episodes.

Run: `python -m mscn.krk_world` (build + validate the oracle) ·
`python -m mscn.krk_closed_loop [--episodes 150000 --seeds 3]` (~45 min full).

### 9.5 Generalising value on KRK — the technique ceiling broken (`krk_value_general.py`)

§9.4's named residual: tabular value gets *safe* (win-preservation 0.999) but not
*fast* (DTM-optimality ~0.25) — the scalable-representation problem on an exact
yardstick. Attacked prior-free, pre-registered:

**Emergent geometry from the agent's own movement model.** The learned legality
tables double as the geometry source: the king-tag's acceptance graph (which tag
is king-like is itself decided by smaller out-degree) is the adjacency graph of
the 64 opaque squares; its Laplacian eigenmaps recover the board at **Procrustes
0.013** — an order of magnitude tighter than the offline Stage-2 result (0.10),
produced *inside the acting loop*.

**Two generalising substrates** on those emergent coordinates, both trained by the
same TD-as-MODIFY updates, both count-shrinkage hybrids over the exact table
(a one-visit Monte-Carlo entry must not override the prior): *tile-relative*
(linear tile-code over the three relative displacements + one pairwise-conjunction
plane — one generic relational bias) and *kernel-absolute* (the Layer-1 Gaussian
memory over the 6-D absolute embedded state, σ **self-calibrated by the
operating-bandwidth principle** — measured d_shell, nonlocal N_eff, ε — no
hand-tuning, GEMM-vectorised).

**Pre-registered criterion — MET on every axis** (50k episodes × 3 seeds, paired
CIs, mean of last 3 windows):

| vs tabular | tile-relative | kernel-absolute |
|---|---|---|
| DTM-optimality | **+0.082 [+0.046, +0.118] sig** | +0.019 ns |
| plies-to-mate | **−7.6 [−9.7, −5.5] sig** | −2.4 ns |
| mate rate | **+0.218 [+0.169, +0.266] sig** | +0.074 ns |
| endpoint | **0.923 mate / 32.5 plies / 0.338 optimal** | 0.773 / 38.4 / 0.271 |

**Reading.** One generic relational inductive bias over self-derived coordinates
breaks the tabular technique ceiling decisively — mate rate 0.70 → 0.92 (most of
the 80-ply timeouts eliminated), mates 8 plies faster, optimality up a third. The
kernel-absolute arm is directionally positive everywhere but ns: the
**dimensionality wall reproduces at the value-learning level** even with
principled bandwidth calibration (which did move it from harmful-when-hand-tuned
to competitive) — relational *structure*, not smoothing, is what buys
generalisation. §8.10's "structure buys data-efficiency" theme, now measured in
value space; and a third in-the-loop validation of the Operating-Bandwidth
calibration as the correct no-tuning default for kernel memories.

Run: `python -m mscn.krk_value_general [--episodes 50000 --seeds 3]` (~45 min).

### 9.6 The gauntlet — compound, continual, adversarial, and perfect defence

Four benchmark dimensions the single-stressor matrix cannot see
(`ibf_asi_gauntlet.py`, `krk_gauntlet.py`; 8 seeds, paired CIs, asserts green).

**G1 — compound-stress ladder** (stressors stack L0 clean → L5 noise+drift+
shocks+deception+moat all at once): the full agent degrades gracefully and beats
no-memory significantly even at L5 (+0.60 sig); CMA-ES survives better than its
drift-collapse in isolation suggested (restarts), never leads; **layer-1's
robustness composes** — it stays on top at every level and sig-beats full at L5
(−0.44). **G2 — continual A→B→A** (the world changes character and returns):
memory buys the **recovered asymptote** (full A2≈A1 at 3.33/3.27; no-memory stuck
at 2.11, CI-sig) but **not relearning speed** — time-to-threshold savings are
*negative* (the B-residue slows the re-climb): a cleanly measured
continual-learning cost. Dissolve-interference is directional (+0.52) but ns.
**G3 — adversarial, memory-targeted shocks** (the world reads the agent's
q-record and aims at it): the 2-seed "+0.65 sig" deflated to +0.22 ns at 8 seeds
(the CI discipline's fourth catch); the robustness ordering — full's targeted
damage ≈ half its ablations' (0.22 vs 0.37–0.38) — is directional, ns.

**G4 — KRK vs the tablebase-optimal defender** (frozen 30k-episode policies,
maximum distribution shift): the most sobering number of the campaign. Mate rate
collapses **0.64 → 0.03** for tabular *and* tile — per-move win-preservation of
0.97–0.98 cannot finish against a defender that forces the full DTM path inside
the 80-ply cap, and the tile agent's technique advantage **vanishes** (−0.001
ns; its preservation is even slightly worse, −0.009 sig). The §9.4/9.5
"safe-but-slow" ceiling re-reads as: *wins only against weak defence*. The
optimality gap (0.26–0.34) is not a cosmetic metric — under adversarial play it
is the whole game. Closing it (deeper value refinement, search at move time) is
the named frontier for the discrete substrate.

### 9.7 The generality tax, located and removed (`lean` mode)

The benchmark and the ladder both showed layer-1 beating the full agent on open
landscapes. Hypothesis #1 (step-width schedule) was **wrong** — annealed
candidate steps alone changed nothing (measured). The real cause: **eval
overhead per decision** — the H=2 plan rollout burns 9 senses/tick for
measured-nothing on open ground (its value lives in the corridor's model-planner
candidate, which is rollout-independent), so eval-matching gives layer-1 ~2.5×
more decisions. The **lean mode** (`horizon=1, anneal_steps=True`, registered as
a benchmark contestant) doesn't just close the gap — at full benchmark power it
**reverses it**:

| agent | clean | noisy | drift | shocked | decept. | moat | moat-loc | corridor | **AGG** |
|---|---|---|---|---|---|---|---|---|---|
| **full ASI (lean)** | 3.64 | 3.23 | 3.67 | 3.13 | 4.27 | 2.40 | **2.73** | **2.92** | **3.25** |
| layer-1 IBFLearner | 3.84 | 3.49 | 3.69 | 3.63 | 4.41 | 2.87 | 1.75 | 1.35 | 3.13 |
| full IBF-ASI (canonical) | 3.48 | 3.09 | 3.00 | 3.00 | 3.89 | 2.36 | 2.10 | 2.84 | 2.97 |

Lean wins the aggregate over its ancestor (3.25 vs 3.13) — within-ns on every
open regime, decisively ahead in both locomotion regimes layer-1 cannot solve —
and sig-beats the canonical config in 4/8 regimes. Generality is not
intrinsically taxed; *unused machinery* is, and once the per-regime cost is
measured, the integrated agent is strictly the better machine.

---

## 10. The paper engine — the preprint's continual-learning instantiation (`ibf_engine.py`)

The "Information as Structural Alignment" preprint (repo root) validates one
concrete engine across RRW / chess-with-Stockfish / Split-CIFAR-100 with
replay-superior retention. Its lifecycle carries machinery the mscn/IBF-ASI
memory lacked: **context-gated reading** (cross-context particles silent unless
crystallized *and verified*), a **crystallization state machine**
(convergence-triggered, μ_cryst ≪ μ_base), the **two-pass write** (cross-context
validity *testing* separated from same-context *learning*), the **Crucible**
(contradiction-triggered de-crystallization, phase-local verification), and an
intensive-readout responsiveness channel. Implemented faithfully (vectorised),
with a faithful mini Rotating-Rules World (phase B = the *exact reversal* of
phase A's contextual component; contexts given — task-incremental, as in the
paper), and **decomposed by ablation** (6 seeds):

| arm | acc(A\|A) | acc(A\|B) | acc(B\|B) | forget(A) after C |
|---|---|---|---|---|
| full lifecycle | 0.898 | 0.884 | 0.834 | +0.217 |
| **no-context-gating** (≈ the old mscn memory) | 0.898 | **0.469** | 0.908 | **+0.402** |
| **no-crucible (gating only)** | 0.898 | 0.892 | 0.902 | **+0.012** |
| no-verification | 0.898 | 0.889 | 0.902 | +0.022 |

**Findings.** (1) **Context gating alone is the retention mechanism**: ungated
memory loses 0.40 of phase-A accuracy under B's exact contradiction — the
gauntlet's G2 poisoning pathology reproduced and CI-significantly attributed in
its home domain (+0.19 [+0.09, +0.28]) — while gating-only forgetting is
**+0.012, near zero, even with phases sharing one input region**. (2) The
crucible *costs* retention here (+0.217) and verified broadcast pollutes rather
than transfers (B|B 0.834 vs 0.902): when every phase revisits the same latent
region, home-context truth is cross-tested constantly and dissolution erodes it.
This is the paper's own regime-dependence theme applied to its own lifecycle:
crucible/verification want **spatially separated contexts** (the CIFAR regime,
where the paper's near-zero headline lives); gating works everywhere. (3) The
architecture lesson for the ASI agent: adopt context-gated reading for continual
regimes; treat crucible/verification as regime-conditional equipment.

Run: `python -m mscn.ibf_engine` (~2 min; asserts green).

### 10.1 IBF classic vs IBF-ASI — head to head, each in the other's arena (`ibf_classic_vs_asi.py`)

The direct comparison the engine ablations only proxied. **Arena 1** — the
classic's home (mini-RRW, A → B=exact reversal → C): the ASI's memory substrate
is ported as an evaluator-corrector under its *actual* laws (multiscale,
non-negative corrections, error-gated dissolution, capacity projection, no
contexts) and faces the classic engine. **Arena 2** — the ASI's home (the G2
continual switching world): the classic's proven mechanism (context gating) is
transplanted into the ASI (`gate_contexts`, phases signalled at boundaries,
task-incremental as in the preprint) and faces the canonical agent.

| Arena 1 (classic's home) | acc(A\|A) | acc(A\|B) | acc(B\|B) | forget(A) |
|---|---|---|---|---|
| classic (gating only) | **0.910** | **0.901** | **0.901** | **0.014** |
| classic (full lifecycle) | 0.910 | 0.906 | 0.703 | 0.014 |
| ASI memory substrate | 0.773 | 0.419 | 0.722 | 0.336 |

| Arena 2 (ASI's home) | A1 tail | B tail | A2 tail | relearning savings |
|---|---|---|---|---|
| ASI canonical | 3.27 | 4.31 | 3.33 | **−12** |
| **ASI + classic gating** | 3.29 | 4.31 | 3.42 | **+22** |
| ASI no-memory | 2.42 | 3.05 | 2.11 | 0 |

**Verdicts.** (1) **Each architecture dominates its home.** In the classic's
arena the ASI substrate is doubly handicapped: its non-negative corrections
cannot suppress wrong actions (lower asymptote 0.77 vs 0.91) and its lack of
contexts makes B's reversal destroy A (0.419; forgetting +0.32 [+0.13, +0.51]
sig vs the gated classic). (2) **The transplant works**: classic gating inside
the ASI flips the G2 relearning savings from −12 to +22 ticks (**+34.6 [+3.3,
+65.9] sig**) — the measured continual pathology repaired — at zero cost to
phase-B adaptation (+0.005 ns). (3) The architectures are **complementary, not
rivals**: the classic is the continual/contextual evaluator (signed corrections
+ gating); the ASI is the embodied navigator (exploration, warm jumps, planning,
options — none of which the classic carries); and the bridge between them is one
mechanism that ports cleanly. `gate_contexts` is now a first-class ASI feature.

Run: `python -m mscn.ibf_classic_vs_asi` (~8 min; asserts green).

### 10.2 IBF Unified — the theory-prescribed blend (`ibf_unified.py`)

The theory writes the agent as **three coupled ODEs** (paper §3 / spec §2.2):
motion along `k·∇R_eff`; modification by **signed** discrepancy (the ASI's
non-negative memory is the Thm-8a special case — §10.1 measured its cost: it
cannot suppress); and **responsiveness itself modified**, locally. `UnifiedASI`
is that decomposition made literal: the **ASI shell** keeps motion, exploration,
warm jumps, the planner with U8 options, and the reflect loop; the **classic
engine becomes the memory organ** (signed particles, crystallization Thm 3,
crucible Thm 10, context gating, capacity control); and the engine's
**responsiveness channel** — which *neither* architecture had wired — modulates
selection spatially: `k_eff(x) = clip(k_base + δk(x))`, intensive readout, trust
where local discrepancy variance is low. The I2 invariant generalizes with the
law: non-negativity (basin expansion) for canonical agents, **bounded
modification** for signed memory.

**Pre-registered validation (8 seeds, paired CIs; asserts green):**

| criterion | result |
|---|---|
| continual repair kept (G2) | **yes, positive-class**: savings +4 (canonical −12) — but *below* the pure gating transplant's +22 (−18 ns); B-tail 3.70 vs 4.31 |
| deceptive gain (signed suppression) | **directional only**: +0.09 [−0.27, +0.45] ns |
| open-regime sanity | **directionally better everywhere**: clean +0.15 [−0.00, +0.30], drift +0.23 ns — nothing worse anywhere |

**Honest reading: generalization without regression.** The unified agent carries
the full three-ODE law — signed corrections live from the first particles
(learned avoidance operating), local trust modulation, the complete lifecycle —
at **no measured cost in any regime**, with directional gains on every open
regime. But at 8 seeds it produces no CI-significant win over the best
*specialized* configuration in any single arena: the targeted gating transplant
remains (ns) better on the continual metrics, and the signed-suppression
hypothesis on deception is unconfirmed. The blend's value is **law-level
unification**, not a benchmark headline: each component's significant win was
already established in its own arena (§10.1's +34.6 gating repair, Arena 1's
signed-correction necessity), and the unified form makes them one agent — the
general machine of which everything measured this session is a special case.

Run: `python -m mscn.ibf_unified` (~13 min; asserts green).

---

## 11. IBF ULTRA — self-detected contexts + regime-aware equipment (`ibf_ultra.py`)

The preprint names task-incremental context signalling as its simplification;
ULTRA removes it. `UltraASI` (extends `UnifiedASI`: engine memory organ, local
k, lean economics) detects its own regime boundaries and re-binds to
recognised old regimes, with the U-2 equipment policy from the §9–§10 claim
map (planner+options auto-engaged on learned low-branching maps; crucible/
verification only on separated context clouds; restart teleports never).

### 11.1 U-1 self-detection — the mechanism, and the pre-registered exam

**Mechanism (final form, after seven measured design iterations on held-out
calibration seeds 100–105 — each iteration forced by an instrumented failure,
logged in the module docstring):**

- **SPLIT**: per-tick prediction discrepancy of the always-learning cell
  world-model at known cells, z-scored against the current context's own
  running statistics; boundary = k-of-m window (4 of 6 above z=3) where a
  surging tick must ALSO be large in raw scale (pe ≥ 1.6× baseline — the
  guard that separates layout switches from drift's high-z-tiny-pe
  coincidences). **The epistemic rule that made it work: anomalous ticks
  teach nothing permanent** — baseline, world-model cells, and landmarks all
  freeze during an anomaly run (entry z≥2.5 ∧ pe≥1.45×, 10-tick timeout):
  without it the baseline absorbed the anomaly (mean crept 0.59→0.83 across
  an undetected boundary) and the cell EWMAs erased the surge in ~3 ticks.
- **RE-BIND**: recognition by **active probing** — passive scoring of recent
  samples against archived models is structurally biased toward the current,
  always-adapting model (measured: the contrast closes in ~20 ticks), so the
  agent redirects one uniform-jump candidate (exact eval parity) to an
  archived context's **landmark** (champion point + de-noised value; up to
  three, ≥2 apart; champions quarantined 6 calm ticks — the detection
  latency itself otherwise poisons the old context's landmark with new-world
  values, measured). A probe is informative only against the prober's own
  **site reference** at that exact point (coarse cell averages mistake
  within-cell spread for model disagreement — measured false re-bind);
  confirmation needs hits at two distinct landmarks (a correlated ripple
  swing fakes one landmark for tens of ticks — measured), or a stricter
  single-landmark rule; agreement-merge folds false splits back; a fresh
  split runs 40 ticks of fast due-diligence probing.

**Pre-registered exam** (criteria frozen before the first acceptance run;
acceptance seeds 0–11 G2 / 0–7 stationary never used in calibration; THREE
rounds run — rounds 1–2 failed, diagnosed, mechanism fixed, re-run; round 3 =
the frozen-design record):

| criterion | round 3 verdict |
|---|---|
| P1 savings repair ≥ 70% of the given-bell transplant's | **NOT MET** (−235%: ULTRA −58.6 mean savings vs transplant +28.9, canonical +4.0; ULTRA−gated −83.5 [−147.3, −19.7] sig) |
| P2 recognition quality (1 split + 1 correct re-bind, ≥10/12) | **NOT MET** (8/12 correct re-binds; split latency 4 ticks (11/12), re-bind latency 18 (8/12); zero FALSE binds) |
| P3 zero false context events on stationary worlds (24 runs) | **MET** (0 events: clean 0, noisy 0, drift 0) |
| P4 G2 asymptote not significantly below canonical Unified | **MET** (A2 −0.105 [−0.587, +0.378] ns; ULTRA A2 3.38 ≈ gated 3.42) |

**Honest decomposition of P1.** (a) Substrate: even GIVEN the bell, the
engine-memory unified agent earns +11.2 savings vs the Scale-memory
transplant's +28.9 (the §10.2 finding reproduced — the savings metric never
favoured the unified substrate). (b) Detection: self − given = −65.8
[−134.3, +2.6] ns, driven by the 4/12 lives whose re-bind never fires (each
catastrophic on the time-to-threshold metric: the agent re-learns phase A in
a fresh context) plus the ~18-tick recognition latency tax on success lives.
The G2 gauntlet's own conclusion (§9.6) was that memory buys the recovered
**asymptote**, not relearning speed — and the asymptote is exactly what
self-detection keeps without any bell (P4 MET). The savings economics of
self-detection remain open; the named residual is **recognition recall**
(landmark informativeness is seed-dependent: an old context whose landmarks
happen to be value-ambiguous against the new world cannot be re-recognised
by point probes; richer fingerprints — more landmarks, distributional
probes — are the next move).

**What stands at CI grade:** self-detected SPLITS are reliable (11/12 within
~4 ticks of the boundary), and the detector is **false-positive-free** across
every stationary regime tried (24/24 runs, plus zero false binds in 36 G2
lives) — the safety half of the claim, fully held. Self-detected RE-BINDING
works in 2/3 of lives with zero false binds. `--smoke` (gate-sized) asserts
the calibration signature, zero FP, and the corridor level.

### 11.2 U-2 equipment matrix (ULTRA vs lean, 8 regimes × 8 seeds)

No significantly harmful cell (pre-registered: held). Directional costs
everywhere (aggregate 3.04 vs lean 3.25; worst drift −0.48 [−1.00, +0.05] and
moat-local −0.49 [−1.10, +0.13], both ns) — the unified organ set is not free
on open landscapes at 8-seed power; attribution (engine memory vs no-restart
policy) not yet isolated. Corridor: ULTRA 2.91 ≈ lean 2.92 (−0.007 ns) — the
auto-engaged planner equipment keeps the structural win, with every
engagement logged. The corridor ATTRIBUTION pre-registration (ULTRA vs
ULTRA-no-planner, lo > 0.5) came back **NOT MET** (+0.50 [−0.05, +1.06] ns)
for a mechanism reason worth the price: **the signed memory organ partially
substitutes for planning in the corridor** — negative writes at the trap
push the agent off it without lookahead, something the non-negative ancestor
structurally could not do. Crucible policy: stays OFF in G2 (context clouds
overlap — consistent with §10's shared-region finding); the separated-cloud
ON branch is exercised by construction in tests only.

Run: `python -m mscn.ibf_ultra` (exam, ~35 min) · `--matrix` (~40 min) ·
`--smoke` (~4 min, in the gate).

## 12. THE TERRARIUM — the visual showcase benchmark (`terrarium.py`, `terrarium_episodes.py`)

One visual language (terrain heatmap, signed memory particles with
crystallization rings, trust halo = local k_eff, option flags, fog, event
banners), rendered headlessly to PNG/GIF with hard size budgets; every
episode doubles as a scoring run without `--render` (the gate runs them
quick). Front-stage numbers are EXACTLY the backstage paired-CI means; each
episode pre-registers its §9–§10 ancestor and reports MET/NOT MET.

| episode | ancestor | backstage outcome (8+ seeds unless noted) |
|---|---|---|
| E1 Two Twins (seasons+fog) | §9.6 G2 asymptote | **MET**: A2 keeper−amnesiac +1.221 [+0.583, +1.860] sig (3.33 vs 2.11 — the ancestor numbers to the digit); honest note kept: keeper's own savings −12 [−76, +51] |
| E2 Earthquake | §9 V1 / shocked cell | **MET**: tail +1.042 [+0.741, +1.344] sig; post-quake recovery 3 vs 8 ticks, −4.7 [−8.6, −0.9] sig (presentation metric, CI'd) |
| E3 Mirage Field | §10.2 null + §10.1 sig | the tripwire episode: its first full run FLAGGED ITSELF (net coherence sig-negative "contradicts ancestor"); investigation: the pair is matched on ULTRA's no-restart policy, NOT the ancestor's restart-enabled protocol — a NEW fact, recorded: signed organ costs −0.41 [−0.68, −0.14] on the deceptive tail vs nonneg under matched no-restart policy (attribution open); dwell registration **NOT MET** (+0.03 ns); evaluator arena re-demonstrated: nonneg extra forgetting +0.32 [+0.13, +0.51] sig |
| E4 Canyon | §9.2 corridor +1.40* | **MET**: +1.432 [+1.345, +1.520] sig; goal 10/10 vs trap-locked 0/10 |
| E5 Trade | §9 V5 / 6.4 (24 seeds) | re-run at ancestor power; assert at suite level (mean > 0.2); 2-D null carried in the panel |
| E6 Chess Garden | §9.4 (30k re-demo) | fresh 30k×3 curve (mate ≥0.4, legal@1 ≥0.85, preserve ≥0.9 registered); 150k asymptote, value-ablation, DTM ceiling and G4 collapse CITED not re-run; learned-legality halo rendered on the board |
| E7 Grand Tour | §11 | ULTRA through seasons/quake/mirage with NO bell; acts scored vs ancestors |

Report generator: `terrarium_report/README.md` with embedded GIFs, novice
scoreboard (= the measured means), and the "what doesn't help, and why"
panel (planner in open 2-D, dissolution, restarts, 2-D trade, crucible under
shared regions, E3's navigator null).

Run: `python -m mscn.terrarium` (render self-check) ·
`python -m mscn.terrarium_episodes --episode all [--render] [--quick]`.
