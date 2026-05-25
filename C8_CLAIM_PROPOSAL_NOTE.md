# Supervisor note — Proposal to add Claim 8 to the LLM companion paper

**From:** Radu (LLM substrate work) → Radu (foundation paper author / LLM
companion lead)
**Subject:** D5b/D6/D7 findings + recommendation to elevate discovery
training to a numbered claim (C8)
**Companion to:** `AGENCY_DISCRETIZATION_NOTE.md` (Reading C engine fix)
**Status:** Recommendation pending decision

---

## TL;DR

Three experiments (D5b, D6, D7 — all complete) demonstrate that
**IBF-over-LLM can autonomously extend its compiled knowledge through
interaction**, on the same substrate that supports the existing C1–C7
claims, without modifying the base model:

- **D5b**: Discovery training on a scaffolded A→C closure produces
  emergent generalization on held-out paraphrases (test_AC 0.000 → 1.000
  in 4 epochs).
- **D6**: With the supervisor-prescribed Reading C engine fix
  (history-sufficiency agency gate), the agency channel engages exactly
  as Postulate 1 predicts — k_eff at A→C queries traces a U-shape
  (5.0 → 2.58 → 3.23) tracking D-variance dynamics.
- **D7**: With the kernel scaffold structurally removed (BC↔AC C-answer
  text at mean 8.09 σ_op — true de novo regime per up-front diagnostic),
  discovery training still produces emergence at ep 1 for both α and β.
  β's k_eff at AC traces a full U-shape independently: 4.63 → 2.48 →
  3.77 over 60 epochs. **`de_novo_emergence_both` verdict — discovery
  works without representational overlap.**

**Recommendation:** Elevate this to a numbered claim — C8: *discovery-driven
extension of compiled knowledge* — and structure the LLM companion paper
around C1–C8 rather than C1–C7. This converts the paper from "domain
port of the chess result" to "domain port plus new architectural
capability validated by two independent ablations (D6 α-vs-β; D7
with-vs-without scaffold)."

---

## Existing C1–C7 (relisted for reference)

From `(IBF)Companion-LLM-Durable-Alignment.ipynb` cell 4
("Claim-to-cell map"):

| Claim | Statement (short) | Headline cells |
|---|---|---|
| **C1** | Local durable alignment without editing base-model weights | § 8 (canonical lifecycle), § 26 (LoRA durability), § 30 (IBF benchmark runner) |
| **C2** | Truth-maintenance lifecycle on a single substrate (install / revise / remove / rollback / retain) | § 8, § 12, § 13, § 25, § 30, § 35 |
| **C3** | Override strong local priors while preserving locality | § 14, § 15, § 19, § 20 |
| **C4** | Compiled semantic structure durably enforced (and revisable) | § 22, § 24 (with § 23 as scope diagnostic) |
| **C5** | Durable alignment under base-model evolution (LoRA test) | § 26 |
| **C6** | IBF is not reducible to kNN or RAG | § 32, § 33, § 34 |
| **C7** | Cross-model mechanism generality (Qwen replication) | § 36, § 37 |

All seven are validated by the canonical paper-run artifacts already in
`mmlu_ibf_out/`. None of them requires discovery training; supervised
δR install + the compiled-surface architecture (§ 24) is sufficient.

C8, if adopted, would be the **eighth and most ambitious** claim.

---

## Proposed C8 — Discovery-driven extension of compiled knowledge

**Statement (short):**

> *IBF can autonomously extend its compiled knowledge through probabilistic
> exploration and environmental reward, finding alignment edges the
> compiler missed via Boltzmann action selection under agency-modulated
> responsiveness, without modifying base-model weights and with each
> update inheriting the lifecycle properties (install/revise/retract) of
> the supervised installation path. The mechanism is robust to absence
> of representational scaffolding between trained and target items
> (D7 de novo regime, mean 8.09 σ_op separation).*

**Headline cells:**
- § 24b-D5b — emergent A→C closure under scaffolded discovery training
- § 24b-D6 — α vs β ablation validating Reading C engine fix; agency
  channel engages as Postulate 1 predicts
- § 24b-D7 — de novo emergence without kernel scaffolding (mean 8.09 σ_op
  separation, confirmed broken scaffold)

**Paper role:**
- **Architectural completion**: Compiled closure (C4) handles the
  deductive case; discovery training (C8) handles the inductive case.
  Together they constitute a complete alignment substrate, not just a
  knowledge cache.
- **Cascade transfer from chess**: validates Memory + Agency →
  Intelligence on a fourth domain (after chess, RRW, CIFAR).
- **L1 resolution**: the no-emergent-closure limitation becomes a
  complementary scope statement rather than a hard wall — closure
  emerges *through interaction*, not from static δR readout.

---

## Empirical evidence in hand

### D5b — Scaffolded discovery training (paper-grade)

**Artifact:** `mmlu_ibf_out/fi_agency_channel_d5b_discovery.json`

**Setup.** Fork canonical engine; supervised AB+BC install via standard
δR pipeline (3 epochs × 3 reps); 40 epochs of Boltzmann discovery on
A→C queries with environmental reward; evaluation on held-out transitive
paraphrases.

**Results.**

| group | pre-discovery | post-discovery | Δ |
|---|---:|---:|---:|
| test_AB | 1.000 | 1.000 | +0.000 |
| test_BC | 0.875 | 1.000 | **+0.125** (positive backward transfer) |
| **test_AC_heldout** | **0.000** | **1.000** | **+1.000** |

Discovery takes the system from "cannot answer A→C" to "answers A→C
correctly on held-out paraphrases" in 4 epochs. AB preserved; BC
improved (positive BT, mirrors chess's BT_A = +35.4 cp finding).
Agency stayed at k_eff = k_0 throughout, so the active mechanism is
Boltzmann + value channel alone.

### D6 — α vs β agency gate ablation (validates Reading C)

**Artifact:** `mmlu_ibf_out/fi_agency_channel_d6_alpha_vs_beta.json`

**Setup.** Two engines forked from canonical, identical Phase 0
supervised install of AB+BC, identical item ordering during Phase 1
discovery. Difference: α uses status-quo engine semantics; β has
`_update_agency` and `delta_k` patched to gate on
`len(c.D_history) ≥ N_AGENCY_MIN` (default 20) instead of
`is_crystallized()`. Per Reading C from the previous note.

**Results.**

| | α (status quo) | β (Reading C) |
|---|---:|---:|
| test_AC pre→post | 0.000 → 1.000 | 0.000 → 1.000 |
| test_BC pre→post | 0.875 → 1.000 | 0.875 → 1.000 |
| **k_eff at AC: trajectory** | **flat at 5.000** | **5.000 → 4.66 → 2.58 → 3.23** |
| n_agency with history ≥ 20 | 2176 | 2177 |

β's k_eff at A→C queries traces a **U-shape**: drops as D-variance
spikes from Boltzmann exploration, then recovers as the system
converges on C and D becomes consistent. This is exactly the
variance-derived responsiveness modulation Postulate 1 prescribes,
with the dynamics tracking the learning process in real time. **α's
k_eff stays at exactly k_0 = 5.000 throughout 40 epochs**, confirming
that the engine's crystallization-gated discretization structurally
silences the agency channel in this regime.

Endpoint accuracy is identical because the value channel saturates at
ep 1 — same regime-dependence pattern as CIFAR (paper § 7.3 P8). But
the **mechanism is empirically validated**: agency engages on the LLM
substrate when properly gated.

### D7 — De novo emergence without kernel scaffolding (complete)

**Artifact:** `mmlu_ibf_out/fi_agency_channel_d7_de_novo.json`

**Setup.** Same as D6, but AC items use a new `C_for_AC` field that
describes the target outcome in lexically distinct (semantically
equivalent) text. Example for chain_approval:
- BC training C: `"release blocked until review completes"`
- AC discovery/test C: `"deployment suspended pending verification"`

**Scaffold-distance diagnostic (run at cell start):**

| chain | BC-C ↔ AC-C distance | σ_op multiples | scaffold? |
|---|---:|---:|---|
| chain_approval | 57.32 | 7.89 | BROKEN |
| chain_restricted_exposure | 69.03 | 9.51 | BROKEN |
| chain_credential_exposure | 47.63 | 6.56 | BROKEN |
| chain_change_control | 53.68 | 7.39 | BROKEN |
| chain_patient_discharge | 61.64 | 8.49 | BROKEN |
| chain_contract_transfer | 63.43 | 8.74 | BROKEN |
| chain_claim_exclusion | 67.85 | 9.34 | BROKEN |
| chain_data_access | 49.33 | 6.79 | BROKEN |

**Mean: 8.09 σ_op.** The 3.03 σ_op activation radius is far exceeded;
this is a true de novo regime — no kernel scaffolding between BC-trained
centers and AC test items.

**Final results (60 epochs, ~57 min runtime).**

| group | α pre | α post | β pre | β post |
|---|---:|---:|---:|---:|
| test_AB | 1.000 | 1.000 | 1.000 | 1.000 |
| test_BC | 0.875 | 1.000 | 0.875 | 1.000 |
| **test_AC_de_novo** | **0.000** | **1.000** | **0.000** | **1.000** |

**k_eff at AC trajectory:**
- α: **flat at 5.000** throughout
- β: full U-shape — **start 4.63, minimum 2.48 (ep 20), recovery to 3.77 (ep 60)**, range [2.48, 4.63]

**β agency w̄ trajectory:** 3.288 → 3.062 over 60 epochs (D-variance
slowly accumulating at canonical FI-region agency centers from
discovery cross-talk).

**Verdict:** `de_novo_emergence_both` — discovery training works under
both α and β even with no representational overlap between trained and
target items. test_BC also improved 0.875 → 1.000 in both runs
(positive backward transfer reproduced under de novo regime).

**The four most important observations from D7:**

1. **Pre-discovery baseline = 0.000 for both engines** — scaffold-distance
   diagnostic confirms AC's new C-text is truly out of kernel reach of
   BC's installed value centers. Clean de novo starting point.

2. **Both α and β reach test_AC = 1.000 at ep 1** and hold through ep 60.
   The Boltzmann + reward + value-channel-accumulation mechanism is
   robust to scaffold removal. Discovery builds new value centers at
   the new C-position **from scratch** in a single epoch.

3. **β's k_eff at AC traces a complete U-shape independently** of D6's
   chains: 4.63 → 2.48 → 3.77. The "drop then recover" pattern that
   Postulate 1 predicts (high D-variance during exploration, low
   D-variance after convergence) reproduces on this independent
   geometric configuration. **Reading C is validated twice across two
   different chain regimes.**

4. **α's k_eff at AC stays at exactly k_0 = 5.000** for all 60 epochs.
   The engine's crystallization-gated discretization structurally
   silences the agency channel here too. Same diagnostic as in D6.

---

## What this means for C8

D7 is **stronger than the partial trajectory hinted at**, and stronger
than I'd predicted before it ran. Four substantive findings:

**(a) Discovery training is mechanism-robust across geometric regimes.**
D5b worked with kernel scaffold (mean BC-C ↔ AC-C distance ≈ 0).
D7 works without it (mean 8.09 σ_op, all chains BROKEN per
diagnostic). The Boltzmann + reward mechanism doesn't depend on
representational overlap between training and target items. It builds
new value structure from scratch when needed.

**(b) Agency engagement is reproducible across regimes.** β's k_eff
U-shape in D6 happens again in D7 with full minimum-then-recovery
visible across 60 epochs. The Reading C engine fix produces the same
dynamics on completely different chain geometry. This is independent
confirmation, not just a single-experiment artifact.

**(c) Positive backward transfer reproduces in the de novo regime.**
Both D5b/D6 (scaffolded) and D7 (de novo) show test_BC going
0.875 → 1.000 during AC discovery. The mechanism is robust to
geometric distance; the BC improvement is real, not a scaffolding
artifact.

**(d) The endpoint-redundancy of agency in this regime is now confirmed
across TWO independent experiments AND TWO chain geometries.** D6 and
D7 both show: β engages (full U-shape k_eff), endpoint identical to α.
This is the regime-dependence prediction holding cleanly — agency is
observable but redundant when value saturates fast.

---

## Why C8 makes the paper stronger

**(1) Differentiation from the foundation paper.** Without C8, the
LLM companion is "the chess result transfers to LLMs." With C8, it is
"the chess result transfers AND adds a new capability the foundation
paper did not demonstrate" — emergent compositional extension on a
substrate that has both a deductive (compiler) and inductive
(discovery) path running on the same δR field.

**(2) Cleanly resolves L1.** The no-emergent-closure limitation
becomes a *complementary scope statement* with C4 and C8 as the two
paths: compiled closure handles edges the rule-corpus knows; discovery
handles edges the rule-corpus missed. Stronger framing than "we have a
workaround."

**(3) Expanded comparison frame.** C1–C7 position IBF against the
knowledge-editing literature (MEMIT/SERAC/GRACE/WISE). C8 expands the
frame to RLHF/DPO/online-alignment literature too — IBF as a
non-parametric, locally-reversible, no-backprop substrate for both
deductive and inductive alignment updates. Two competitive positions
instead of one.

**(4) Validates the Memory + Agency → Intelligence cascade on a
fourth substrate.** Chess, RRW, CIFAR + LLM closure. The
regime-dependence prediction holds: agency engages but its endpoint
contribution depends on whether the value channel has headroom to
saturate quickly. LLM closure becomes a new regime ("agency engages,
value saturates, endpoint redundant") — extending the
regime-dependence table from three to four entries with cleanly
characterized dynamics.

---

## Risks and recommended framing

**(1) Reviewers will ask: "isn't this just RL?"** Yes structurally,
no architecturally. The IBF substrate makes the RL-style updates
**local, reversible, non-destructive, non-backprop**. Each δR
perturbation inherits the install/revise/retract lifecycle guarantees.
The paper should frame this explicitly: *"Discovery training is RL on
an IBF substrate, which inherits the substrate's lifecycle properties
— every discovery-driven update is a local δR modification that the
Crucible can later dissolve, that compiled closure can later override,
and that base-model evolution (LoRA test, C5) leaves intact."* That's
the alignment-relevant difference from standard RL.

**(2) Reading C is a new engine modification.** The paper must
either (a) reference the supervisor note as documenting an engine
correction adopted concurrently, or (b) carry the agency-discretization
story as part of the LLM companion's methodological contribution.
Recommend (b) — it's a substantive contribution: *"We refined the
agency-channel discretization from crystallization-gated to
history-sufficiency-gated to faithfully implement Postulate 1's
parallel-equation framing on small-data substrates, validated by the
α vs β ablation in § 24b-D6 with independent reproduction in
§ 24b-D7 across a different chain configuration."*

**(3) The discovery-training section is a real addition to scope.**
Adds maybe two paper sections plus an appendix. Worth the cost if C8
makes it.

**(4) The "isn't C7 enough?" attack.** C7 says the *mechanism*
(cascade) transfers across base models. C8 says the mechanism produces
*new capability* (autonomous extension) on the LLM substrate that
wasn't demonstrated on chess. They're different claims; both should
remain.

---

## Engine-level recommendation (carry over from previous note)

Pending your confirmation:

1. **Adopt history-sufficiency gating in cell 9.** Patch
   `_update_agency` and `delta_k` to gate on
   `len(c.D_history) ≥ n_agency_min` (default 20). Add `eta_k_cryst`
   parameter (default 0.005, parallel to `eta_cryst` on the value side).
   Rationale documented in `AGENCY_DISCRETIZATION_NOTE.md`; empirical
   validation in `fi_agency_channel_d6_alpha_vs_beta.json` (D6) and
   independent reproduction in `fi_agency_channel_d7_de_novo.json`
   (D7 across a different chain geometry).

2. **Pre-merge: re-run chess/RRW/CIFAR with the patched engine** to
   confirm within-noise behavior in those regimes. Expected result:
   chess preserves the 19,054 vs 30 dissolution asymmetry and +90.2 cp
   advantage; RRW preserves the negative-agency outcome; CIFAR
   preserves the neutral-agency outcome. All within-noise compared to
   the original implementation.

3. **Resave `canonical_engine.pkl` under corrected semantics**,
   version-bumped (e.g., add `engine_version: "2.0-history_gate"` to
   the artifact metadata).

---

## What I need from you

1. **Approve C8 as a numbered claim** (or, conservatively, agree to
   include the discovery-training section without elevating to a
   numbered claim — "Option B" from the strategy discussion).

2. **Approve the engine patch** (per the previous note) given that D6
   AND D7 both empirically validate Reading C across different chain
   geometries. Re-running chess/RRW/CIFAR is non-trivial; need your
   call on prioritization.

3. **Approve the paper-side framing** for discovery training:
   *"discovery training is RL on an IBF substrate, inheriting the
   substrate's lifecycle properties"* — or your preferred alternative.
   Important to get right because it's how reviewers will categorize
   the contribution.

4. **Sanity-check the regime taxonomy refinement**:

| Regime | Agency develops? | Agency contribution to endpoint |
|---|---|---|
| Chess (paper § 7.2) | yes | **decisive (+8.3 cp)** |
| RRW (paper § 7.1) | yes | **harmful** |
| CIFAR (paper § 7.3) | no (no trajectory dependence) | neutral |
| **LLM closure (D6/D7 β)** | **yes (U-shape k_eff trajectory)** | **observable but endpoint-redundant in saturated regime** |

Do you agree with adding the fourth row, and with the characterization
("observable but endpoint-redundant" vs your preferred phrasing)?

---

## Bottom line

The empirical work for C8 is **complete**: D5b, D6, and D7 all finished
with positive results across two independent ablations (D6 α-vs-β on
the agency channel; D7 with-vs-without scaffold on the value channel).

The recommendation is to elevate it to a numbered claim and frame the
LLM companion paper around C1–C8 rather than C1–C7. The paper becomes
meaningfully stronger: from "domain port of the chess result" to
"domain port plus new architectural capability validated by two
independent ablations across different chain geometries."

If you'd rather keep the paper tight at C1–C7, the empirical results
can still go into a "Beyond Compiled Closure" section without elevation,
and C8 becomes a future-work direction. That's the safe path. The
ambitious path (C8 as numbered claim) is supported by the evidence.

Your call.
