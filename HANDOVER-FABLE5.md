# HANDOVER — for a frontier successor (Fable 5): *mega-deepen* the IBF/MSCN architecture

You are a frontier model inheriting a mature, **honest** research codebase. The brief is
not "continue the to-do list" — it is **deepen the architecture**: unify the substrate,
close the loops, push the representation and the theory until the thing is one coherent
machine instead of a drawer of validated modules. Read this top to bottom, then go deep.

The non-negotiable inherited standard: **every claim is measured, every negative is
reported, every caveat is stated.** The most valuable results in this repo are the ones
that proved an *earlier* claim wrong (the U1 barrier moved — §8.10; the arena artifact —
§3.18). Mega-deepening means deeper **truth**, not louder claims. Hold that bar.

---

## 0. Orient fast (don't re-derive — it's all logged)

```bash
git checkout claude/happy-carson-3zhihq && git pull origin claude/happy-carson-3zhihq
pip install -U setuptools && pip install numpy scipy matplotlib cma chess zstandard
python -m mscn.tests        # 24 behavioural guarantee checks (all pass)
python -m mscn.agi_all      # 9 AGI-upgrade validations + hierarchical sigma* (10/10 pass)
```
- `mscn/ARCHITECTURE.md` is the **authoritative, honest log** — §1–2 (toy model), §3.1–3.19
  (chess arc), §8 (AGI upgrades + the §8.10 state-merging finding). Read it.
- `HANDOVER.md` (state + module map), `MSCN_AGI_ROADMAP.md` (the 9 upgrades + measured
  outcomes), `formal/README.md` + `formal/AGIFoundations.lean` (the Lean grounding).

## 1. The whole architecture in one invariant

**Knowledge is coherence modification `δR` over a configuration space; selection is
Boltzmann in a responsiveness `k`; dynamics follow the modification ODE
`δR' = α·discrepancy − μ·δR`.** No weights, no backprop. *Everything* below is this one
unit, instantiated at different scales:

- **Layer-1 acting unit** (`learner.py`) — the closed loop sense→select→act→**modify**→adapt;
  exercised as an optimiser (mean-rank 1.89 of classical heuristics; below tuned CMA-ES)
  and as IPD agents (emergent cooperation from *relational* coherence).
- **Layers 2–3** — coupling/cooperation (`network.py`,`games.py`), RG coarse-graining
  (`hierarchy.py`), Lawvere self-model + phase transition (`selfmodel.py`,`phase.py`);
  integrated in `mscn.py` (now O(N·k_eff), scales to 2048 agents).
- **Chess arc** (the substrate stress-test, strict no-priors, opaque tokens):
  rules→geometry→**Route A sufficient board state** (purity 0.998 *with* from-to) →
  emergent value/PST→search→**end-to-end player** (§3.16) → **discrepancy-vs-oracle
  distillation** (§3.19).
- **9 AGI upgrades** (`agi_*.py`) — each a runnable mechanism with a measured advantage
  *or an honest null* (compression, planning 0→100%, transfer basins 100%, etc.).

## 2. The honest meta-picture (internalise before deepening)

Across *every* experiment, two walls recur. Name them so you don't waste cycles:

1. **Dimensionality.** Passive δR-memory transfer, info-gain exploration, kernel context
   — all great in low-D / structured regimes, weak in high-D (the curse re-sampling
   barrier). Wins concentrate where structure is local.
2. **Estimation / horizon.** Bounded-suffix learning can't acquire long-range structure
   (U1). **§8.10 sharpened this:** the wall is *not* history length — unbounded
   recurrent state-merging (RPNI) recovers the exact mod-N counter from 50 sequences;
   the real residual is **noise-robust estimation of the merge**. Domain structure
   (chess from-to) is a *data-efficiency shortcut*, not an expressivity necessity.

Every clean win lives inside these; every honest failure traces to one of them. The
deepest contributions so far were **negative-sharpening** — they moved a wall and said
exactly where it now is. That is the genre to extend.

## 3. THE DEEP DIRECTIONS — the actual brief (pick 2–3 and go all the way)

Ranked by architectural depth × tractability. Each is theory-grounded and reuses real
modules. These are *architecture*-deepening, not tweaks.

### D1 — Close the IBF loop end-to-end (the unifying move). **Highest leverage.**
Today representation (Route A), value (oracle/outcome), selection (search), adaptation
(discrepancy), and meta (self-improve) are **separate modules**. The deep architecture is
**one closed online loop**: a single IBF agent that, *while acting*, runs
sense→select→act→**MODIFY(discrepancy)**→adapt(`k`) over a **learned simulation state**
*and* a **learned value landscape**, self-modifying online. Chess is predictive-only
today (the honest gap in §3.0); fuse the §1 acting unit with the §3 representation. Build
it on chess or a tractable world (K+R endgame). **Target:** an agent that learns
representation *and* value *and* policy in one discrepancy-driven loop, with
strength-rising and discrepancy-to-oracle-falling measured **online** (not in separate
batch phases). This is the single biggest "make it one machine" step.

### D2 — §2.2-general, done right (the deepest *representation* result available).
§8.10 left a precise target: a **noise-robust** learner that crosses the estimation
barrier. Build either (a) an **ALERGIA-style statistical state-merging** (RPNI + a
Hoeffding/χ² compatibility test, so more data *helps* instead of over-splitting), or
(b) a **learned factored / register PSR** simulation, and apply it to **chess atomic
tokens** — recover the board **without being given from-to**, pushing legal-set purity
toward Route A's 0.998. **Metrics:** purity + a *linear board-state probe* (Othello-GPT
style) on the learned state, against the from-to upper bound. Success here is a genuinely
new result: a prior-free, robust, learned world-model from atomic tokens.

### D3 — Emergent **non-linear** value (the strategy frontier; the user's "improve in all areas").
Oracle distillation (§3.19) works but is capped by the **material+PST linear** value
(residual ~89 cp = the oracle's mobility/king-safety it cannot express). Deepen: discover
**relational / non-linear board features prior-free from the occupancy state** — mobility
(`#occ-valid moves`), king-exposure (attacker counts on the king's neighbourhood),
piece-coordination (co-occurrence/defence graphs) — and **compose** them via the
network-coherence law `R = ΣR_i + ΣJ_ij R_pair` (Upgrade 6), distilled from the oracle.
**Target:** residual discrepancy ↓ and the gap-to-oracle in the arena closes (today the
distilled player *beats* the outcome-trained one but still loses to the oracle 0.00).

### D4 — Credit assignment: **coherence-valued TD** (the roadmap's #1 hard problem).
The modification ODE assigns credit *locally*; long horizons need a coherence-valued
temporal-difference. The seed is formal: `tdError'`, `td_is_modification_step`
(AGIFoundations §11) — *TD is literally an IBF modification step*. Build it: propagate
discrepancy backward over trajectories so value/policy learn from **delayed outcomes**
(win/loss), not just per-position oracle labels. This is the bridge from
imitation/distillation (D3) to genuine RL inside IBF, and it powers the online loop (D1).

### D5 — Formal deepening (if you are given a Lean toolchain).
`formal/AGIFoundations.lean` (32 defs/theorems) is provided but **not `lake`-built here**
(PyPI-only, no Mathlib). If you get a Lean env: build it, then prove the harder open ones
— **simulation-faithfulness ⟺ sufficiency on *stochastic* processes**; the
**estimation/horizon barrier as a sample-complexity theorem** (formalise §8.10); the
**credit-assignment convergence**; the **compressed-coherence-basis** bound
(`compression_fidelity_tradeoff'` is only the seed). Make theory and code co-evolve —
that is the distinctive strength of this program over pure-empirical ML.

### D6 — Scale (mostly blocked; frame it honestly).
acc@1 is **data-limited** (unsaturated at 5k games — needs a larger PGN slice via git);
strength is **eval-limited** (D3). The deepest scale problem is the open one:
**a learned basis of coherence functions** that is *sub-linear* in stored experience
(Gaussian-kernel δR is O(N); the Interface-Principle pruning §3.12 is a start). Lawvere
guarantees any compressed self-model is lossy — the question is *losing the right
information*.

## 4. The two named open theory problems (don't pretend these are solved)
- **Scalable representation** — a compressed coherence basis, sub-linear in experience.
- **Language / symbolic reasoning** — continuous coherence vs discrete compositional
  structure. Genuinely hard; connects to the foundations of the field.

## 5. Environment, norms, gotchas (will save you hours)
- **Network is PyPI-only.** No web fetch, no model API key usable from a subprocess, no
  apt. **Data and a real Stockfish binary arrive ONLY via git** (the user commits a
  slice/binary to the branch; recipe in `data/README.md`). The oracle in §3.19 is a
  pure-Python proxy *because* Stockfish isn't reachable — drop a UCI binary on the branch
  and it slots into `python-chess` trivially, upgrading D1/D3/D4.
- **LLM opponents** come via the **Agent tool** (`model:"haiku"`/`"fable"`), not raw API.
  A general subagent will deflect a "be a chess engine" prompt unless framed as a
  *measured best-effort benchmark* (see `chess_vs_llm.py`; Haiku is ~37% legal from FEN).
- **Branch `claude/happy-carson-3zhihq`** only. Commit + push after each unit; a stop-hook
  nags on a dirty tree. Long runs → background bash; **never sleep-poll**.
- **Deterministic agents collapse self-play to 2 repeated lines** — always `rand_open`
  the arena openings or your strength numbers are statistically empty (§3.18 footnote).
- `python-chess` is a parser / legality oracle / teacher — **never** a rule input to a
  prior-free model.

## 6. Reproduce / entry points
```bash
python -m mscn.tests | demo | benchmark                 # toy-model core
python -m mscn.agi_all                                   # 9 upgrades + sigma*
python -m mscn.agi_state_merging                         # the §8.10 frontier finding
python -m mscn.chess_mscn_player  --pgn data/elite_2024-01_5k.pgn   # end-to-end player + Maia/LLM bench
python -m mscn.chess_tournament   --pgn data/elite_2024-01_5k.pgn   # IBF strength tournament (quiescence/depth)
python -m mscn.chess_vs_llm       --emit /tmp/pos.json              # IBF-vs-Haiku positional head-to-head
python -m mscn.chess_oracle_train --pgn data/elite_2024-01_5k.pgn   # discrepancy-driven training vs oracle (D1/D3/D4 base)
```

## 7. Where to start (a concrete first week)
1. **D1 skeleton on K+R-vs-K**: a tractable world where the closed online loop is fully
   observable and the causal states are enumerable — get the single agent learning
   representation+value+policy online, discrepancy-driven, end to end.
2. **D2 ALERGIA upgrade** to `agi_state_merging.py` (statistical merge test) → re-run the
   counter/flag worlds (more data should now *help*), then chess atomic tokens.
3. **D3 emergent mobility feature** into `chess_oracle_train.py` → watch the residual
   discrepancy and the gap-to-oracle drop. This is the fastest "deepen the value" win.

Then write it up in ARCHITECTURE.md the same way everything else is: measured, with the
negative results in the open. Deepen the truth.
