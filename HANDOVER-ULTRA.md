# HANDOVER — build THE TERRARIUM (the visual showcase benchmark) and the IBF ULTRA AGENT

You are a successor session inheriting a mature, **honest**, fully-green research
codebase. Your brief is NOT to continue a to-do list — it is to build the
**demonstration layer**: a sophisticated visual environment in which the entire
apparatus performs, scored as a real benchmark, with every capability *visible
and explainable to a novice* — and the **IBF ULTRA AGENT** that performs in it.

The inherited contract, non-negotiable: **every claim measured, every negative
reported, pre-register before you run, CI-grade before you believe.** This
session's most valuable results were nulls and reversals caught by that
discipline. Hold the bar.

---

## 0. Orient fast (~15 min)

```bash
git checkout claude/ecstatic-goodall-dmb5qt && git pull
pip install -U setuptools && pip install numpy scipy matplotlib cma chess zstandard pillow
python -m mscn.testall          # the gate: 7 suites, ~15 min, must be green
```

Read, in order: `mscn/ARCHITECTURE.md` §9–§10 (the agent, the claim map, the
gauntlets, the unification) · `mscn/IBF_ASI_GAP.md` (spec audit) · the preprint
`(pre-print)information-as-alignment-v1.pdf` §4 (the classic engine — extract
text with pypdf; no PDF renderer here).

## 1. What you inherit (the one-page map)

**The law** (three coupled ODEs): motion along `k·∇R_eff`; modification
`δR' = α·D − μ·δR` with **signed** D (non-negative memory is the Thm-8a special
case — measured: it cannot suppress); responsiveness itself modified, locally
(`k_eff(x) = k₀ + δk(x)`).

**The organs** (all built, all validated):
- `mscn/ibf_asi.py` — **IBFASI**: the nine-stage embodied agent. Flags that
  matter: `model_planner` (+U8 options), `anneal_steps + horizon=1` = **lean
  mode**, `gate_contexts` (+`switch_context(id)`).
- `mscn/ibf_engine.py` — **PaperEngine**: the classic evaluator-corrector
  (signed particles, crystallization, crucible, context gating, `delta_k`).
- `mscn/ibf_unified.py` — **UnifiedASI**: engine-as-memory inside the ASI shell
  + local k. *Generalization without regression* (§10.2).
- KRK stack (`krk_world/closed_loop/value_general/gauntlet`) — the discrete
  showcase: exact tablebase, online learning curves.

**The measured map (don't re-derive — cite §9.x/§10.x):**

| capability | verdict |
|---|---|
| memory | the ONE universally significant stage (every regime, CI-starred) |
| context gating | the retention mechanism (forgetting 0.40→0.01; transplant +34.6 sig) |
| planning + option-commitment | binds ONLY in discrete/low-branching structure (corridor +1.40*; continuous: null both pre-registrations) |
| lean economics | unused machinery is the only generality tax (lean 3.25 > layer-1 3.13 aggregate) |
| signed corrections | necessary in evaluator arenas (Arena 1); directional-only as navigator suppression |
| crucible / verification | regime-conditional: want spatially separated contexts; harmful under shared input regions |
| dissolution (error-gated) | redundant with base decay at these timescales (honest null) |
| two-sided k restarts | the only stage with significantly HARMFUL cells |
| technique vs perfect defence | the killer: KRK mate 0.64→0.03 vs optimal defender — optimality gaps are fatal under pressure |

**Known traps (each cost hours; all documented in ARCHITECTURE):**
memory-homing silently undoes exploration (suspend warm jumps during
options/explore phases) · grouping **purity is invalid** on near-singleton
samples (a shuffled control scores 0.97; use MCC-of-changes) · replay amplifies
factorization errors; consistency objectives are **truth-blind below a
correctness threshold** · small-seed significance is a recurring mirage (4
catches; always paired CIs) · uint8 counters wrap · `&`-backgrounded shell jobs
don't notify (use run_in_background) · eval-budget matching is the norm; in
drifting worlds drive world-time per N senses for cross-agent fairness.

**Environment**: PyPI-only (no apt, no web, no GUI). Data/binaries arrive ONLY
via git. Long runs → background bash; never sleep-poll. Commit+push each unit;
a stop-hook nags on dirty trees. **Develop on the branch your session assigns
you** (do not assume this one).

---

## 2. THE BRIEF, part 1 — THE TERRARIUM (the visual showcase benchmark)

**Goal.** One coherent, living 2-D world — rendered, animated, narrated — in
which every validated capability is a *visible episode* a novice understands in
one sentence, AND which is simultaneously a real scored benchmark (seeds and
paired CIs backstage, simple scores front-stage). The demo layer sits ON TOP of
the validated science; it never replaces it.

**Rendering stack (no GUI):** matplotlib frames → PNG stills + animated GIFs
(`matplotlib.animation` with the `pillow` writer, or stitch frames with PIL).
Deliverables are FILES: a gallery directory + one auto-generated markdown/HTML
**story report** with embedded images, captions, and the novice scoreboard.
Keep GIFs small (≤ ~6 MB each, downsampled frames) — they get committed.

**The world (suggested design — adapt, but keep the mechanism↔feature map):**
a terrain heatmap (the coherence landscape) with living overlays —

| world feature | mechanism it makes visible | novice caption |
|---|---|---|
| **seasons** (the world's rules change, then return) | context gating / continual memory | "Winter came. The forgetful twin lost the map; ours kept it." |
| **fog & flicker** (observation noise, ripple drift) | de-noised memory (q-records) | "It can't trust its eyes, so it trusts its notes." |
| **earthquakes** (displacement + fast-mode memory damage) | consolidation, reflect/boost, recovery | "Knocked across the map, it walks straight home." |
| **mirage oases** (tall decoys; some marked by experience) | SIGNED memory: red ✗ marks on visited mirages | "It paints an X on every lie it has personally checked." |
| **the canyon + bridge** (a moat only lookahead crosses) | planner + option commitment | "It walks DOWN into the canyon because it knows what's beyond." |
| **a companion creature** (shares discoveries, ledgered) | reciprocity / cooperation (6.4) | "They trade maps — until one stops giving." |
| **the trust glow** (render k_eff(x) as halo brightness) | the third ODE, local responsiveness | "It moves boldly where its notes have never lied." |
| **the chess garden** (a KRK board corner, live arrows) | the discrete arc: rules learned from "no", mating curves | "Nobody taught it the rules. Watch the legal-move halo grow." |

**Episode scripts** (each ~30–60 s of GIF, each scored): (E1) Two Twins — full
vs amnesiac through fog+seasons; (E2) The Earthquake; (E3) The Mirage Field;
(E4) The Canyon; (E5) The Trade; (E6) The Chess Garden (legal@1 halo + mate-rate
curve animating over training); (E7) THE GRAND TOUR — one continuous run through
everything, the centrepiece.

**Scoring (two faces, one truth):** backstage = the existing standards (8+
seeds, eval parity, paired CIs, pre-registered expectations per episode —
mostly *re-demonstrations* of §9–§10 results, so expectations are already
known); front-stage = novice numbers ("treasure found 9/10; amnesiac 1/10")
that are EXACTLY the backstage means, never prettified. Where the science says
null (e.g., planner on open terrain), the showcase says so — an honest showcase
includes one "and here's what doesn't help, and why" panel. That panel builds
more trust than ten wins.

**Engineering notes:** build `mscn/terrarium.py` (world + overlays + recorder)
separate from `mscn/terrarium_episodes.py` (scripts + scoring + report
generator); reuse `ASIWorld`/`SwitchingWorld`/`MoatWorld` physics where
possible — the Terrarium may literally wrap them. Rendering must be OPTIONAL
(`--render`) so episodes double as headless CI checks. Frame budget: render
every Nth tick. Colour the memory particles by sign (green +, red −), size by
|v|, ring crystallized ones; draw the planner's current target as a flag and
the option path as a dotted line. The report generator writes
`terrarium_report/README.md` with embedded media + the scoreboard.

## 3. THE BRIEF, part 2 — the IBF ULTRA AGENT

The Terrarium's protagonist. ULTRA = the unified agent, completed with the one
capability everything this session points to but nobody built:

**U-1 (the new mechanism): SELF-DETECTED contexts and regimes.** Everything so
far receives `switch_context(id)` as a given signal (task-incremental, as in
the preprint). ULTRA must *detect*: a sustained surge in discrepancy/TD error
against crystallized memory (the phase.py monitor + the engine's D-histories
already carry the signal) ⇒ open a new context; recognition of an OLD context
(low discrepancy against a gated-out memory bank when probed) ⇒ re-bind to it.
Pre-register: self-detected gating recovers ≥ 70% of the given-signal
transplant's continual repair (+22 ticks savings benchmark), with zero false
context-splits on stationary worlds (measure the false-positive rate
explicitly). This is the genuinely novel result available here — the paper
names task-incremental as its simplification; ULTRA removes it.

**U-2 (composition): regime-aware equipment.** The claim map says every
mechanism has a habitat. ULTRA carries the equipment map as policy: lean
economics by default; planner+options engaged when the learned cell-model shows
low branching / barrier structure; crucible+verification enabled only when
detected contexts occupy separated regions of state space (measure overlap of
context particle clouds); restarts never (harmful cells). Each auto-engagement
must be logged and visible in the Terrarium (equipment icons lighting up — the
novice sees the agent *choosing its tools*).

**U-3 (acceptance, pre-register before building):** on the existing suites,
ULTRA ≥ lean on the open regimes, ≥ gated-ASI on G2 *without being told the
phases*, keeps corridor +sig, no significantly-harmful cell in the 8-regime
matrix; in the Terrarium, completes the Grand Tour with every episode's
backstage CI matching its §9/§10 ancestor result. Add ULTRA to `ibf_asi_fuzz`
(invariants must hold under self-detection too) and to `testall`.

**Explicitly out of scope** unless the user redirects: Lean proofs (no
toolchain), bigger chess data (arrives via git only), the KRK perfect-defence
technique gap and the factor-discovery trust-region bootstrap (named open
frontiers — mention, don't chase).

## 4. Suggested build order (a week-shape)

1. Terrarium world + renderer + E1 (Two Twins) end-to-end — proves the stack.
2. ULTRA U-1 (self-detected contexts) on the G2 benchmark — the science first,
   pre-registered, CI-graded; then its episode (E-seasons with no season bell).
3. Episodes E2–E5 (each: backstage score + GIF + caption).
4. The Chess Garden (E6) from the KRK stack's existing curves.
5. U-2 equipment policy + the Grand Tour (E7) + the story report generator.
6. Gate everything (`testall` extended), HANDOVER refresh, final push.

After each unit: commit, push, and write the honest section in ARCHITECTURE
(§11.x for ULTRA, §12.x for the Terrarium) — nulls included. If an episode's
demo contradicts a §9/§10 number, the DEMO is wrong until proven otherwise:
investigate, don't retune the science to fit the show.

## 5. Entry points you'll lean on

```bash
python -m mscn.testall                    # the gate (extend it with your units)
python -m mscn.ibf_asi | ibf_unified      # the agents + their validations
python -m mscn.ibf_asi_regimes            # the 8x7 claim map
python -m mscn.ibf_asi_gauntlet           # compound/continual/adversarial
python -m mscn.ibf_asi_benchmark          # vs layer-1/CMA/random + lean
python -m mscn.ibf_classic_vs_asi         # the two architectures, both arenas
python -m mscn.krk_closed_loop            # the discrete showcase curves
python -m mscn.ibf_engine                 # the classic lifecycle, ablated
```

The apparatus is one machine now. Your job is to let people *see* it — without
ever letting the show outrun the measurements.
