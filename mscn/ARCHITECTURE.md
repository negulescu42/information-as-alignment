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
