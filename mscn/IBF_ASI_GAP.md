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

## 4. Measured outcomes (8–12 seeds, eval-budget-matched)

| Validation | Result |
|---|---|
| V1 integration | **Memory and error-gated dissolution are decisively load-bearing**: no-memory 2.09 / reflexive-ok 0.78, no-dissolve 3.16, full **3.52 / ok 1.00**. Planning, extra scales, reflect: ties (honest nulls; the regimes where each wins are U7's trap corridor, U2/§3.15, and the Zombie-Twin resp.). |
| V2 (6.1 analog) | **Sharpened null**: with a noise-robust high-water agency signal, monotone-k self-limits and the two-sided reset adds nothing (0.42 vs 0.42). U3's measured win exists under the noisy per-tick k-ratchet, which the high-water rule removes at the source. Theorem 6.1 stays open. |
| V3 (6.2 analog) | Adaptive wₛ ≈ uniform ≈ single-scale (3.47–3.52): **honest null** at these scales; the allocation theorem (6.2) is not operationally discriminated by this world. |
| V4 (6.3 analog) | **Honest negative**: the floor-sized reserve costs ~3% steady-state coherence and does not buy faster shock recovery; reflexive viability unaffected (0.99 vs 1.00). What stands: the conflation floor is measured, positive, and reported every tick — the agent never claims complete self-knowledge. |
| V5 (6.4 analog) | **Supported operationally**: cooperation Pareto-beats solo (joint 7.18 vs 6.77; both partners individually better), and under drift **defection does not pay** (B cooperative 3.61 > B parasitic 3.48) because the reciprocity ledger withholds and stale gifts fade. |

## 5. Open (not claimable from here)

- The four 6.x **theorems** (global optimality, Pareto allocation, honest
  self-model, aligned interaction) — require the Lean campaign the spec describes,
  against a library this repo does not carry. No Lean toolchain (PyPI-only sandbox).
- The L7 thermodynamic backing (Landauer, 2nd law) — telemetry computes a free
  energy, but no conservation/dissipation law is verified here.
- 6.2 and 6.3 lack even an operational *advantage* in the tested regime (nulls
  above) — a harder allocation-bound world might discriminate them; that is a
  measurement target, not a claim.

Run: `python -m mscn.ibf_asi` (numpy only, ~4 min; all asserts green).
