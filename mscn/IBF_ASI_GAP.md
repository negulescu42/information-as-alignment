# IBF-ASI spec — audit against THIS repository, and what was built

The IBF-ASI specification ("Specification of the Most Advanced IBF
Superintelligence") was checked clause-by-clause against the current architecture.
This note records (1) what the spec assumes that this repo does **not** contain,
(2) what already exists here for each spec component, (3) the **difference that was
built** — `mscn/ibf_asi.py`, one fully operational mechanism — and (4) what remains
open, including the honest negatives. Norm unchanged: every claim measured, every
null reported.

---

## 1. Spec premises vs. this repository (the honest discrepancy)

The spec's "Backing" column cites a Lean library this repository does not contain:

| Spec reference | Status here |
|---|---|
| `IBF.lean` (Axioms I–IV, Thms 1–11), `IBF/…` ≈45 verified modules | **Absent.** |
| `IBFSuperintelligence.lean` + `IBFSuperintelligence.capabilities` | **Absent.** |
| `ibf_superintelligence.py`, `ibf_core.py` | **Absent.** |
| Named declarations (`claim_memory`, `second_law_of_coherence`, `landauer_bound`, `IsReflexivelyCoherent`, `SelfModelTower`, `lojasiewicz_differential_inequality`, …) | **Absent.** |

What this repo's formal layer actually contains: **one file**,
`formal/AGIFoundations.lean` (36 declarations, provided by the theory team, not
lake-built in this PyPI-only sandbox). Of the spec's citations, only these map to
declarations that exist here: `planning_horizon_value_bound`,
`planning_diminishing_returns`, `planning_converges_if_gains_bounded`,
`transfer_preserves_superlevel/viability`, `transfer_coherence_lower_bound`,
`optimal_linear_allocation'`, `self_improvement_equilibrium_positive/_exceeds_half_gap`,
`temporal_abstraction_safety`, `macro_action_composition`, `capability_composition'`,
`IsAGICapable`, `tdError'`/`td_*`, `compression_fidelity_tradeoff'`,
`more_basis_less_error`, exploration/info-gain lemmas, RG lemmas.

The spec's ✅ rows are therefore read here as: *behaviourally validated in the mscn
Python system* (`mscn/tests.py` 24 checks + per-module measurements), not
Lean-verified in this repo. The Theorem-number references (Thm 1–11, Lawvere, phase
structure) correspond to the toy-model guarantees exercised by `tests.py`,
`selfmodel.py`, `phase.py` per `mscn/README.md`.

## 2. Component map (spec → existing mscn machinery)

| Spec §2.1 / §3 / §5 component | Existing module (validated separately) |
|---|---|
| `x, R̂, δR̂, k`, Boltzmann, Euler | `learner.py` (Layer-1 unit) |
| Multiscale `{δR̂ₛ, wₛ}` | `hierarchy.py`, `hierarchical_sigma.py` (σ*-ladder §3.15), `agi_hierarchical_sim.py` |
| PLAN (H-step lookahead) | `agi_planning.py` (U7: 0→100% on the trap corridor) |
| TRANSFER (replicate crystallised memory) | `agi_transfer.py` (U5: basins preserved 100%) |
| DISSOLVE (forget contradicted memory) | `nonstationary.py` (§3.14: **error**-gating, not count) |
| REFLECT (monitor, θ, self-correction) | `phase.py` (PhaseController, Zombie-Twin 100%) |
| Self-model + Lawvere obstruction | `selfmodel.py` (diagonal escape, capacity ceiling) |
| Γ allocation | `agi_selfimprove.py` (U9: cost-aware water-filling) |
| Interaction / society (L6) | `network.py`, `games.py` (EC-4), `mscn.py` (O(N·k_eff)) |
| Thermo/safety envelope (L7) | partial: decay/lifetime in `phase.py`; no Landauer/2nd-law module |

**The gap:** every stage existed as a separate validated module; **no single agent
ran all nine coupled**, none enforced the §8 runtime invariants, none emitted the §8
telemetry. That is the difference that was built.

## 3. What was built: `mscn/ibf_asi.py` (the §8 contract, operational)

One class, `IBFASI`, holding the full state tuple and executing all nine stages per
tick (see module docstring for the stage→module mapping), in the regime the spec
itself targets — noisy, multi-scale, **memory-requiring**, non-stationary, with
adversarial shocks. §8 contract compliance:

- **State** ✓ `(x, R̂(world), δR̂ₛ + wₛ, k, monitor, self-model table, Γ)`.
- **Step** ✓ the 9-stage cycle; discrete IBF/Euler updates.
- **Invariants** ✓ asserted live every tick: I1 R_eff non-decreasing along each
  autonomous ascent sub-segment; I2 δR̂ₛ ≥ 0 (pointwise basin expansion over the
  baseline); I3 below-θ excursions are bounded transients (measured); I4
  Σₛ|δR̂ₛ| ≤ Γ (capacity projection).
- **Telemetry** ✓ per tick: true/best coherence, monitor E, reflexive status,
  self-model error **and its conflation floor** (> 0, never claimed zero — the
  Lawvere gap reported, not denied), free energy −E − H(policy)/k, robustness
  radius max(E−θ, 0), eval count, k, memory mass.

Design corrections found *en route* (each measured before/after):
1. **Reinforce on RAW sensed improvement, never effective** — reinforcing R_eff
   self-manufactures traps (full agent went from *losing* to no-memory to beating
   it decisively). This is `learner.py`'s documented rule, re-confirmed at system level.
2. **Coarse scales fill only by consolidation** (crystallised, error-surviving fine
   centres promote upward; transfer-once) — direct coarse writes smear noise.
3. **Scale-stability must be physical** (RG: coarse modes slow): the testbed world
   has near-stable basin geography + fast-drifting fine ripples; adversarial shocks
   damage the **fast modes** (fine memory) — which is what consolidation is for.
4. **De-noised memory argmax** (per-centre quality EWMA `q`) — the noisy raw
   high-water mark is not a usable "best known".
5. **Reciprocity must be a ledger, not a window** (credit-based gating, EC-4), and
   transfer across *independently drifting* worlds without a morphism is
   misinformation (U5's lesson) — partners must share a world or a map.

## 4. Measured outcomes (CI-graded: paired per-seed differences, 95% t-intervals)

> The first version of this table reported 8-seed point means. CI-grading
> (`mscn/stats.py`, requested as testing upgrade #1) revised three of them — the
> revisions are the finding. Earlier values are kept in parentheses for the record.

| Validation | CI-graded result (16–24 seeds) |
|---|---|
| V1 integration | **Memory is decisively load-bearing and significant**: full−no-memory **+1.18 [+0.88, +1.47]** coherence, +0.21 (sig) reflexive viability. **Dissolution: null** −0.05 [−0.21, +0.12] (an 8-seed "+0.37, load-bearing" did **not** replicate) — consistent with §3.14: error-gating only ties a tuned fixed μ. Planning / scales / reflect: nulls. |
| V2 (6.1 analog) | **Directional, not significant**: +0.13 [−0.06, +0.31], 24 seeds (after fixing two real mechanism issues: warm jumps silently undoing restarts; ripple-dominated worlds being physically undiscriminable — at ripple 0.45 both arms tie *exactly*). The regime matrix adds: two-sided-k is significantly **harmful** under shocks (−0.30*) and in the moat (−0.15*). Theorem 6.1 stays open. |
| V3 (6.2 analog) | **Null, trending negative**: adaptive−uniform −0.23 [−0.51, +0.05]; uniform weights are the honest default. The allocation theorem (6.2) is not operationally discriminated by these worlds. |
| V4 (6.3 analog) | **Null both ways** (the earlier "costs ~3%" was also noise): +0.03 [−0.36, +0.42] coherence, −0.006 [−0.03, +0.02] viability. What stands: the conflation floor is measured, positive, and reported every tick — the agent never claims complete self-knowledge. |
| V5 (6.4 analog) | **CI-significant in the scarce-information regime** (3-D, narrow optimum, drift): cooperation−solo joint **+0.54 [+0.01, +1.08]**, both partners individually ahead; defection neither pays nor costs (+0.03 [−0.10, +0.16]). In 2-D the claim is a **null** (solo discovery is cheap; sharing is worthless) — both readings kept. |

### 4b. The testing apparatus (upgrades #1–3, built on request)

- **`stats.py`** — paired-CI machinery; all asserts now on CI bounds.
- **`ibf_asi_fuzz.py`** — property-based invariant fuzzing: 150+40 random
  configurations (dim 1–4, 0 decoys, zero noise/memory, tiny Γ, coupled/parasitic
  pairs); per-tick I1/I2/I4 + finiteness checks. Caught one real crash
  (`shock_point` with no decoys — fixed); zero violations since. I3 reported, not
  asserted (sub-critical configs legitimately live below θ).
- **`ibf_asi_regimes.py`** — the 6×6 regimes × mechanisms matrix, every cell a
  paired CI. Map: **memory significant in all six regimes** (+0.69…+0.91*);
  dissolution redundant with base decay; this plan operator (stochastic rollout on
  a noisy sensed field) buys nothing even in the moat — U7's planning win used BFS
  over a learned *discrete* simulation; two-sided-k harmful in 2/6 regimes.

## 5. Open (not claimable from here)

- The four 6.x **theorems** (global optimality, Pareto allocation, honest
  self-model, aligned interaction) — require the Lean campaign the spec describes,
  against a library this repo does not carry. No Lean toolchain (PyPI-only sandbox).
- The L7 thermodynamic backing (Landauer, 2nd law) — telemetry computes a free
  energy, but no conservation/dissipation law is verified here.
- 6.1–6.3 lack an operational *advantage* at CI grade (6.1 directional; 6.2/6.3
  null) — harder allocation-bound worlds might discriminate them; that is a
  measurement target, not a claim. 6.4's advantage exists and is regime-scoped
  (information scarcity), now CI-significant.
- ~~A planning operator worthy of the moat regime~~ **Done and measured
  (ARCHITECTURE §9.2): the pre-registered criterion was NOT met, and the failure
  maps a boundary** — a model-based planner (learned cell map, R_eff-scale
  frontier optimism, U3 arbitration, calibrated satiation) adds no CI-measurable
  value anywhere in continuous 2-D: shallow barriers yield to Boltzmann diffusion
  (planning unnecessary), deep barriers hide their prize (planning insufficient —
  needle-in-a-haystack). Planning binds in discrete, low-branching state spaces
  (the §3.18 chess search gains); the named follow-up is the discrete substrate
  (corridor / K+R-vs-K) inside the closed loop.

Run: `python -m mscn.ibf_asi` · `python -m mscn.ibf_asi_fuzz` ·
`python -m mscn.ibf_asi_regimes` (numpy+scipy; all asserts green).
