# HANDOVER — MSCN / IBF work (for a fresh conversation)

This file lets a new session resume the work with full context. Read it top to
bottom, then skim `mscn/ARCHITECTURE.md` (the detailed log) before doing anything.

---

## 0. TL;DR — what this is

We built **MSCN** (`mscn/`): a runnable, **non-neural, prior-free** instantiation of
the IBF / Operating-Resolution theory. Two bodies of work:

1. **The MSCN toy model** — the 3-layer apparatus (individual coherence-gradient
   learner → coupled network + emergent cooperation → meta-learning / self-knowledge
   / phase control), with **24 behavioural guarantee checks (all pass)** and
   benchmarks vs classical + SOTA optimisers.
2. **The emergent-chess study** — can the IBF mechanism learn chess structure from
   **opaque move tokens** (no board/pieces/rules)? It progresses rules → board
   geometry → a **sufficient state (Route A)** → value/strategy → search, each step
   tied to a theorem.
3. **Resolution-roadmap items** (from a colleague's note) implemented & validated:
   local σ*, Interface-Principle pruning, percolation diagnostic, O(N²)→O(N) coupling,
   per-center adaptive μ, **hierarchical σ* (RG flow as a σ*-sequence, §2.1/3.1)**.
4. **MSCN→AGI roadmap** (`MSCN_AGI_ROADMAP.md` + `formal/AGIFoundations.lean`, theory
   team): **all 9 architectural upgrades implemented** as runnable `mscn/agi_*.py`
   mechanisms, each with a **measured advantage or an honest null** (see ARCHITECTURE §8;
   one-shot check `python -m mscn.agi_all`). Upgrade 1 (learnable simulation from atomic
   tokens) *is* the §2.2-general frontier — done, with its honest limit quantified.

Everything is committed to branch **`claude/happy-carson-3zhihq`** (push there only).
The repo is the public companion to the "Information as Alignment" paper; MSCN is an
exploratory addition under `mscn/`.

---

## 1. How to resume (procedure)

1. `cd /home/user/information-as-alignment` (or wherever the repo is cloned fresh).
2. `git checkout claude/happy-carson-3zhihq && git pull origin claude/happy-carson-3zhihq`.
3. Install deps (PyPI is reachable; see §2): `pip install numpy scipy matplotlib cma chess zstandard`
   (numpy is the only hard core dep; the rest enable specific modules).
4. Sanity check: `python -m mscn.tests` (expect **24 passed**).
5. Read `mscn/ARCHITECTURE.md` (§1–§3.14) — it is the authoritative, honest log of
   what was built, what was tested, and every result (with caveats). Then `mscn/README.md`.
6. Pick up from §6 (open work) below.

**Working norms (important):**
- Develop on `claude/happy-carson-3zhihq`; commit with clear messages; push after each
  meaningful unit (`git push -u origin claude/happy-carson-3zhihq`, retry w/ backoff).
- A stop-hook nags if there are uncommitted/untracked files — keep the tree clean.
- **Honesty is the contract here.** Every result in ARCHITECTURE.md is reported
  faithfully, *including negative/mixed findings* (kernel < n-gram; search doesn't
  crack acc@1; count-based μ fails drift). Do not spin. State caveats (e.g. the
  move-quality oracle is material-based; arena outcome is material/mate).
- Long runs: use background bash (`run_in_background: true`); you'll be notified.
  Don't poll with `sleep`. Many evals here take 1–4 min.
- Don't open a PR unless asked.

---

## 2. Environment & data (read before trying to fetch anything)

- **Network policy is PyPI-only.** `pip install` works; *every other host returns
  HTTP 403* — lichess, the nikonoel Elite DB, even a RunPod proxy URL the user
  pasted. So you **cannot download data/games/datasets** from the web. The only
  inbound channel is **git** (the remote works).
- **Data on hand:** `data/elite_2024-01_5k.pgn` — 5,000 real Lichess Elite games
  (committed). The user has the full 235 MB file on a RunPod pod but it's not
  reachable here. To get more data, the **user commits a bigger slice** to the branch
  (recipe in `data/README.md`); then `git pull` and rerun.
- The chess models use `python-chess` only as a data parser / legality oracle —
  **never** as a rule input to the model (strict no-priors).
- `chess` installs only after `pip install -U setuptools` (Debian setuptools bug).

---

## 3. Repo / module map (`mscn/`)

**Core MSCN toy model**
- `landscapes.py` — coherence landscapes (sphere/rastrigin/ackley/schwefel/rosenbrock).
- `learner.py` — Layer 1: `IBFLearner` (continuous, Gaussian-kernel δR) + `DiscreteIBFLearner`; theorem helpers (Boltzmann, Euler, decay).
- `baselines.py` / `strong_baselines.py` — random/hill/SA ; CMA-ES/DE/dual-annealing (opt).
- `network.py` — coupling graphs incl. `scale_free_adjacency` (sparse, O(N·m)); network coherence; mean-field.
- `games.py` — iterated PD, IBF game agent, tournament (emergent cooperation).
- `hierarchy.py` — coarse-graining / RG flow / hierarchical optimiser.
- `selfmodel.py` — Lawvere obstruction, capacity ceiling, consciousness–competence tradeoff.
- `phase.py` — consciousness phase transition, dissipative lifetime, Zombie-Twin.
- `mscn.py` — the integrated network (now O(N·k_eff) coupling, scales to 2048 agents).
- `demo.py` (`python -m mscn.demo`), `tests.py` (24 checks), `benchmark.py`.

**Emergent-chess instantiation** (built in this order; see ARCHITECTURE §3)
- `chess_world.py` — n-gram coherence model + Stage 1 (rules) / Stage 2 (board geometry) evals (`evaluate_rules`, `recover_board_geometry`).
- `chess_strategy.py` — Stage 3 (strategy), **real-PGN path** (`load_pgn`, `stream_lichess`), skill-stratified generator, `run_pgn` (Stages 1–3 on a PGN), `_move_quality` oracle.
- `chess_kernel.py` — kernel (faithful-IBF) model + Operating-Resolution σ* (global and **per-query local**) + percolation diagnostic.
- `chess_recurrent.py` — Route A (v1): variable-order recurrent state (VOM).
- `chess_simstate.py` — **Route A (full): the sufficient board state** (occupancy-transfer replay; purity 0.998).
- `chess_ibf.py` — the **full IBF unit** (closed acting loop) + per-center adaptive μ.
- `chess_value_ibf.py` / `chess_positional_ibf.py` — value & positional coherence (emergent piece values, PST).
- `chess_search_ibf.py` — negamax search (context move-gen + value leaves).
- `chess_arena.py` — head-to-head referee + SEE; strength comparison.
- `chess_mscn_player.py` — **the end-to-end MSCN player** (board sim-state + value + positional + planning composed) + honest benchmark vs published Maia/LLM/Karvonen (ARCHITECTURE §3.16).
- `correction_field.py` — Interface-Principle pruned field eval (O(M)→O(M_boundary)).
- `nonstationary.py` — adaptive-μ evaluation on drift streams.

**MSCN→AGI upgrades** (`agi_*.py`; each has a runnable `main()` that prints + asserts a
measured result; `python -m mscn.agi_all` runs all nine):
- `agi_simulation.py` — U1 learnable simulation homomorphism from **atomic** tokens
  (CSSR/ε-machine, recurrent), compression win + honest beyond-horizon frontier (§2.2-general).
- `agi_hierarchical_sim.py` — U2 multi-scale simulation (coarse-graining suppresses noise, cheaper planning).
- `agi_exploration.py` — U3 coherence-gradient-driven exploration (fixes deceptive Schwefel; honest funnel tradeoff).
- `agi_directed.py` — U4 directed exploration via information gain (coverage lever).
- `agi_transfer.py` — U5 cross-domain coherence morphisms (basin preservation 100%, warm-start head-start).
- `agi_compositional.py` — U6 compositional coherence algebras (skill-library reuse, 5.4× amortized).
- `agi_planning.py` — U7 coherence-based planning over the learned simulation (0%→100%, diminishing returns).
- `agi_temporal.py` — U8 temporal abstraction / macro-actions (3.6× fewer decisions, basin-invariance safety).
- `agi_selfimprove.py` — U9 active self-improvement (cost-aware reflexive α-allocation).
- `hierarchical_sigma.py` — roadmap §2.1/3.1 (RG flow as a σ*-sequence).

**Docs**: `mscn/ARCHITECTURE.md` (the log; §8 = AGI upgrades), `MSCN_AGI_ROADMAP.md` (the
theory-team roadmap), `formal/AGIFoundations.lean` (the new Lean grounding, as provided —
not lake-built here), `mscn/NOTE-state-tracking-gap.md` (theory note + team response),
`formal/README.md` (Operating Resolution + Causal States Lean results), `data/README.md`.

---

## 4. Key results (so you don't re-derive them)

**MSCN toy model**
- 24/24 guarantee checks pass (basin expansion, selective retention, Boltzmann, Euler→ODE, coupling-helps, cooperation, Lawvere, phase transition, Zombie-Twin, …).
- Optimiser: IBF best **mean rank 1.89** of {IBF, random, hill, SA}; best in high-D multimodal. Vs SOTA: CMA-ES 1.61 < dual-anneal 2.27 < DE 2.86 < **IBF 3.92** < SA 4.34 (IBF not competitive with tuned global optimisers — expected).
- Cooperation: IBF-memory tops the tournament; needs memory **and** relational coupling (EC-4).
- Coupling now **O(N·k_eff)** → scales to **2048 agents** (constant ms/agent/round).

**Chess (5k real Elite games), the arc**
- Rules emerge (opening legal@1 85%, 38.6% exact next-move); **board geometry emerges** (Procrustes 0.10).
- **Route A sufficient state**: reconstructed board has **legal-set purity 0.998** (vs ~0.01 for move-suffix states) → closes the state-tracking gap; legality 0.33→**0.50** (endgame doubles).
- Full IBF unit: subsumes the n-gram in the crystallisation limit (μ→0); its forgetting dynamics suit *acting*, not stationary prediction.
- Value/positional: **piece values emerge from outcomes** (queen highest); pawn value rises with advancement. Sound material play but ≠ human strategy.
- Strength: value+SEE **beats context-only 40–0**; depth-2 search beats value+SEE 12–12–0.
- **acc@1 (matching 2400+ human moves) plateaus ~0.15** — search/value don't help (stronger play ≠ human moves); it's **data-limited** (scales 0.090→0.120 over 500→4000 games, unsaturated). The lever is **more data / a stronger learner**.

**Resolution roadmap (colleague's note)**: §1.1 local σ* ✓ (de-percolates dense regions, opening +0.058), §1.2 pruning ✓ (34× @500k), §2.4 percolation ✓, §2.2 sufficient statistic ✓ (Route A), §3.4 value ✓, §3.5 coupling ✓ (O(N)), §1.3 adaptive μ ✓ (helps stationary; count-based is wrong for drift — use error-gating).

---

## 5. Honest open limitations (don't claim these are solved)
- **acc@1 ceiling** on the non-neural substrate: needs more data and/or a stronger learner (neural à la Maia). Search/value/positional features do not crack it.
- **Strategy** = engine-level position evaluation; only material+PST so far.
- **§2.2-general** (learn the simulation homomorphism *without* domain from-to structure) — the deep open problem the colleague and we both flag.
- Move-quality oracle is material-based (some circularity); arena outcome is material/mate.

---

## 6. Open work / suggested next steps (pick up here)
Prioritised; all are theory-grounded and reuse existing modules.

**Done since the last handover (see ARCHITECTURE §3.15, §8):**
- §2.1/§3.1 **hierarchical σ*** (RG flow as a σ*-sequence) — `hierarchical_sigma.py`.
- §2.2-general / Route B — **learnable simulation homomorphism from atomic tokens** —
  `agi_simulation.py` (Upgrade 1). Recovers causal states, sufficient + faithful, an
  ε-machine *compression* win; the honest limit (dependencies beyond the estimable suffix
  horizon need a latent-state model) is quantified — that residue is the remaining frontier.
- **All 9 MSCN→AGI upgrades** — `agi_*.py`, each a measured advantage or honest null
  (ARCHITECTURE §8; `python -m mscn.agi_all`).

**Still open:**
1. **Wide-Elo / scale-up acc@1** (highest-leverage chess lever): user commits a larger
   PGN slice (50k+); rerun `run_pgn` and the acc@1-vs-data curve. *Blocked only on data via git.*
2. **§2.2-general, the hard residue** — *sharpened* (ARCHITECTURE §8.10,
   `agi_state_merging.py`): unbounded recurrent state-merging (RPNI/EDSM) **crosses** the
   long-range counter U1's bounded-suffix learner could not (recovers the exact 4-state
   machine, purity 1.0), so the ceiling was a *design artifact*. The real residual is
   **noise-robust estimation of the merge** (RPNI is brittle on the stochastic flag world;
   more data over-splits). Next step: an **ALERGIA-style statistical merge** or a **learned
   factored (register) decomposition** — *not* longer history; then apply to chess atomic
   tokens and compare to the from-to Route A (0.998 purity).
3. **Upgrade chess instantiations**: wire the AGI mechanisms into the chess agent
   (planning over the learned sim for move choice; hierarchical sim = piece→material→
   strategic layers) and re-measure acc@1 / strength.
4. **Remaining minor roadmap items**: §2.3 N_eff module-isolation metric; §3.2
   vector-valued δR; §3.3 Lawvere metacognition; §3.5 dense-coupling k-d-tree variant.

When in doubt about scope/data/direction, ask the user one crisp question (they
steer actively) — but act on sensible defaults where the path is clear.

---

## 7. One-command references
```bash
python -m mscn.tests                      # 24 guarantee checks
python -m mscn.demo                       # MSCN guided tour
python -m mscn.benchmark [--quick]        # optimiser/strong/cooperation/hierarchy/scaling
python -m mscn.chess_world --figures      # emergent rules + board geometry (generated games)
python -m mscn.chess_strategy --pgn data/elite_2024-01_5k.pgn   # Stages 1-3 on real games
python -m mscn.chess_mscn_player --pgn data/elite_2024-01_5k.pgn  # end-to-end player + Maia/LLM benchmark
python -m mscn.chess_tournament --pgn data/elite_2024-01_5k.pgn   # IBF strength tournament (quiescence/depth)
python -m mscn.chess_vs_llm --emit /tmp/pos.json                 # IBF-vs-Haiku positional benchmark (2-phase)
python -m mscn.correction_field           # Interface-Principle pruning scaling
python -m mscn.nonstationary              # adaptive-μ on drift streams
python -m mscn.hierarchical_sigma         # RG flow as a σ*-sequence (roadmap 2.1/3.1)
python -m mscn.agi_all                    # all 9 MSCN→AGI upgrade validations (§8)
```
