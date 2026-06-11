# HANDOVER — after the Terrarium + IBF ULTRA session

You inherit the apparatus of HANDOVER-ULTRA.md, **executed**: the Terrarium
visual benchmark exists (7 episodes, scored + rendered), and the IBF ULTRA
agent exists (self-detected contexts + equipment policy), examined against
pre-registered criteria over THREE recorded rounds. This file is the truth
map of what happened, what stands, and what is genuinely open. Norms
unchanged: every claim measured, every negative reported, pre-register then
CI-grade, and **the show never outruns the measurements**.

## 0. Orient (~20 min)

```bash
git fetch origin <this branch> && git merge origin/<this branch>
pip install -U setuptools && pip install numpy scipy matplotlib cma chess zstandard pillow
python -m mscn.testall          # now 10 suites, ~30 min, must be GREEN
```

Read: `mscn/ARCHITECTURE.md` §11 (ULTRA: mechanism, three-round exam, 11.3
replication catch) and §12 (Terrarium episode table) · `mscn/ibf_ultra.py`
docstring (the detector's seven measured design iterations) ·
`terrarium_report/README.md` (the assembled show).

## 1. What was built

- **`mscn/terrarium.py`** — presentation-only render stack (terrain, signed
  particles, crystallization rings, trust halo = k_eff, option flags, fog,
  banners; PNG/GIF with hard size budgets). Self-check in the gate.
- **`mscn/terrarium_episodes.py`** — 7 episodes, each a pre-registered
  re-demonstration with novice numbers = backstage paired-CI means; panel
  JSON caching; `--assemble` writes `terrarium_report/README.md`;
  `--episode gate` is the testall suite. Failed registrations are STAGED,
  not hidden (E3, E5 are deliberately on stage as catches).
- **`mscn/ibf_ultra.py`** — `UltraASI(UnifiedASI)`: U-1 self-detected
  contexts (surge splits + recognition by ACTIVE LANDMARK PROBING) and U-2
  equipment policy (lean always; restarts never; planner on learned
  low-branching; crucible only on separated context clouds). `--smoke` in
  the gate; the full exam (`python -m mscn.ibf_ultra`) and `--matrix` are
  the heavy validations.

## 2. The measured verdicts (do not re-derive; cite)

| claim | verdict |
|---|---|
| U-1 split detection | **stands**: 11/12 lives detect the boundary within ~4 ticks; **0 false events in 24 stationary runs** (P3 MET) — the safety half is solid |
| U-1 re-binding | works in 8/12 lives @ ~18 ticks, **zero false binds**; P2 NOT MET as registered (10/12 bar) |
| U-1 savings repair (P1, the handover's ≥70% benchmark) | **NOT MET (−235%)**: missed recognitions are catastrophic on time-to-threshold; plus the substrate gap (even GIVEN the bell, unified earns +11 savings vs Scale-transplant's +29 — §10.2 reproduced) |
| U-1 asymptote (P4) | **MET**: A2 −0.105 [−0.587, +0.378] ns vs canonical Unified — what memory actually buys (§9.6) survives bell-free |
| U-2 matrix | no significantly harmful cell vs lean (8 regimes); directional cost everywhere (agg 3.04 vs 3.25); corridor LEVEL kept (2.91 vs 2.92) but attribution ns: **signed memory partially substitutes for planning in the corridor** (new mechanism fact) |
| E1/E2/E4 episodes | ancestors reproduced at CI grade (E4 to the digit: +1.411 vs +1.40*) |
| E3 | the tripwire worked: matched no-restart pair shows signed organ **costs** −0.41 [−0.68, −0.14] on the deceptive tail (NEW comparison, not the §10.2 protocol); evaluator arena re-demonstrated sig |
| E5 / V5 / 6.4 | **REPLICATION FAILURE (the fifth catch, §11.3)**: +0.54 [+0.01, +1.08] did not survive a fresh environment at identical code/seeds/protocol (−0.26 ns). 6.4 reverts to unsupported pending higher power |
| E6 | KRK 30k re-demo registered (mate ≥0.4, legal@1 ≥0.85, preserve ≥0.9); 150k numbers cited, never re-claimed |
| E7 | ULTRA unbelled grand tour; asymptote class held; carries P1's weakness in its own panel |

## 3. Traps added this session (each cost real time)

- **Anomalous ticks teach nothing permanent** — baseline, world-model cells,
  landmarks, sites: ALL must freeze during suspected boundaries (bounded by
  a timeout or drift worlds spiral). Four separate poisonings were measured
  before this rule was applied uniformly.
- **Passive recognition cannot work**: the current context's model adapts to
  whatever world is outside within ~3-20 ticks, so any "old model fits
  better" contrast closes. Recognition must PROBE (act), at precise points,
  against quarantined references.
- **Coarse cell references fake informativeness**: within-cell spread looks
  like model disagreement; point landmarks + per-context site estimates are
  required.
- **A CI lower bound that grazes zero is not a result** (V5: +0.01). Chaotic
  trajectories re-randomise under numpy/BLAS changes; treat environment as
  a seed dimension. If you assert on it, someone later inherits a red gate.
- The full `mscn.ibf_asi` V5 assert is now safety-class only; the verdict
  prints either way. Do not "fix" it back.

## 4. Open frontiers (named, not chased)

1. **Recognition recall** (the P1/P2 gap): landmark probes fail when an old
   context's champions are value-ambiguous against the new world (~1/3 of
   lives). Next moves: more/distributional landmarks, probe-time information
   maximisation, or recognising by the ENGINE's particle predictions rather
   than vmap landmarks. The savings economics of self-detection is the
   measurable target (re-run the §11.1 exam unchanged).
2. **The deceptive-regime cost of the signed organ under no-restart policy**
   (E3's −0.41): attribute signed-erosion vs local-k stickiness; maybe the
   equipment policy should re-enable restarts in deceptive-like regimes the
   agent can DETECT (high decoy count via multi-basin vmap structure?).
3. **6.4 / V5 higher-power study** (≥48 seeds × multiple environments)
   before any cooperation claim returns to the table.
4. Inherited frontiers unchanged: KRK perfect-defence technique gap (§9.6
   G4), factor-discovery trust-region bootstrap, Lean campaign (no toolchain
   here).

## 5. Entry points

```bash
python -m mscn.testall                                  # the 10-suite gate
python -m mscn.ibf_ultra [--smoke|--matrix]             # ULTRA exams
python -m mscn.terrarium_episodes --episode all --render  # rebuild the show (~2 h)
python -m mscn.terrarium_episodes --assemble            # re-write the report from cached panels
```

The apparatus now performs in public — with its failures on stage, which is
the only reason the wins are worth anything.
