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
