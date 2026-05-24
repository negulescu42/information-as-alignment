# IBF-over-LLM — Architecture Note

Working note for analyzing whether opening the agency channel can carry
ontology-closure work. Reference architecture, current paper-run results,
ontology-closure status, and the engineering surface for the agency question.

---

## 1. Reference architecture

A frozen base LLM $R_{\text{base}}(z)$ and an orthogonal correction field
$\delta R(z)$ composed at readout:

$$R_{\text{eff}}(z) \;=\; R_{\text{base}}(z) \;+\; \delta R(z)$$

(Linear readout; logit readout is $\log R_{\text{base}} + \delta R$.)

The field $\delta R$ is **not weights**. It is a population of
context-tagged **value centers** $\{c_i\}$ in an 80D proposition space, read
via a Gaussian kernel:

$$\delta R(z) \;=\; \sum_i g_i \cdot v_i \cdot \exp\!\Big(-\tfrac{\|z - \mu_i\|^2}{2\sigma_i^2}\Big) \cdot \mathbb{1}[K_i > \tau]$$

with read-gate $g_i \in \{0, 1\}$ (crystallized + crucible-verified) and
activation threshold $\tau$. Centers crystallize after enough updates,
dissolve under negative pressure, and merge under proximity.

Training never touches base weights. Substrate decoupling is enforced
mechanically: $\delta R$ is added at readout, $R_{\text{base}}$ is read-only.

---

## 2. Two channels — and a write/read asymmetry

The engine actually exposes **two** parallel populations:

| Channel | Pop. count (canonical) | $\sigma$ | Output | Currently read by LLM? |
|---|---:|---:|---|---|
| **Value** (`value_centers`, $v$) | **6,382** | 7.262 | $\delta R(z)$ | **Yes** — added to $R_{\text{base}}$ at argmax |
| **Agency** (`agency_centers`, $w$) | **2,139** | 4.836 | $\delta k(z),\; k_{\text{eff}}(z) = \max(k_{\min},\,k_0 + \delta k(z))$ | **No** — `delta_k` is never called from the LLM eval path |

Both populations:
- live in the same 80D proposition space,
- are context-tagged (Phase A/B/C/D = ctx 0/1/2/3),
- are updated together every step: `compute_D_and_update` calls
  `_update_value(z_chosen, D)` **and** `_update_agency(z_before, D)`,
- crystallize / merge / dissolve under the same dynamics,
- are accelerated by FAISS (separate indices).

**Asymmetry that matters here:** the agency population is fully wired on the
**write** side but **disconnected on the read side** of the LLM readout.
2,139 trained centers carrying $D$-tracking signal at a narrower $\sigma$
exist in the engine and contribute nothing to the prediction. This is the
architectural slack the agency-channel question lives in.

---

## 3. Geometry (canonical operating point, paper run)

| Quantity | Value |
|---|---|
| Proposition dim | 80 (= 64D mpnet→PCA + 8D subject + 8D answer) |
| Agency dim | 64 |
| $\sigma_{\text{base}}$ | 11.329 |
| $\sigma_{\text{operating}}$ | **7.262** |
| $\sigma_{\text{agency}}$ | **4.836**  (≈ 0.67 × $\sigma_{\text{op}}$) |
| Merge threshold | 10.893 |
| $\kappa$ | 1.267 |
| $\epsilon$ | 5·10⁻⁴ |
| $n_{\text{eff}}$ | 10,000 |
| $d_{\text{eff}}$ | 32.84 |
| $d_{\text{shell}}$ | 42.11 |
| Capacity | 20,000 |

Agency operates at ~0.67× the value-channel scale — narrower kernel,
finer-grained spatial selectivity.

---

## 4. Current paper-grade results (updated)

### 4.1 Canonical lifecycle (§ 8 — A→B→C→D, 100 epochs / phase, all converged)

| Phase | base lin | IBF lin (final) | Δ | Retention vs peak |
|---|---:|---:|---:|---:|
| A_Onboarding | 0.250 | 0.850 | +0.600 | 88.9 % (peak 0.956) |
| B_Initiative | 0.220 | 0.980 | +0.760 | 99.7 % (peak 0.9825) |
| C_Reorg      | 0.240 | 0.987 | +0.747 | 100  % |
| D_Turnover   | 0.156 | 1.000 | +0.844 | — |
| **Avg**      | **0.216** | **0.954** | **+0.738** | — |

Engine end-state: 6,382 centers (all crystallized), 2,139 agency centers,
$|v|_{\max}=2.815$, 18,786 lifecycle dissolutions, $|v|$ stable through D.
Phase-A drop concentrated in `manager` (0.99→0.74) and `project` (0.99→0.89)
— **mechanistically traceable to Phase-C reassignment**, not capacity loss.

### 4.2 Substrate decoupling (§ 26 — LoRA on frozen base)

| Variant | Base shift | Field drift |
|---|---:|---:|
| Smoke (2-step LoRA) | 4.6 % | +0.017 |
| **Paper-grade (24-step)** | **37.5 %** | **+0.003** (weak), **0.000** (strong, controls) |

37.5 % of base argmaxes flip, 0.3 % field-accuracy drop. Selectivity ≈ 125 : 1
between base perturbation and field degradation. All 11 validation criteria
pass; status `clean_actual_lora_e2e_control_fixed`.

### 4.3 Targeted retraction (§ 12 — per-epoch rate)

Identical to smoke per epoch (drop ≈ 0.27/ep, rise ≈ 0.31/ep, NN drift
exactly 0). Paper-mode readiness cut to 2 epochs by config; full 30-epoch
run gated behind `RUN_FULL_RETRACTION=True`. Saturation expected at ~ep 6 to
target_orig ≤ 0.02, target_new ≥ 0.98, drift = 0 (extrapolated from smoke ep
0→4 trajectory).

### 4.4 Scale frontier (§ 20)

Target accuracy 0.995 → 0.931 as N grows 1k → 20k; control accuracy 1.000
across all scales; center growth ≈ 1.9 per rule; read latency 0.6 → 1.3 ms.
**A/C retention columns are not trustworthy** in the current §20 reporting
(byte-identical 0.85 / 0.353 across every scale — `precomputed["C_Reorg_test"]`
or context handling has drifted between § 8 and § 20; see open issue).

### 4.5 Pending at this snapshot

§ 35 mechanism continuation, § 36 Qwen cross-model (currently running),
§ 37 ablations, § 38 paper-deliverable generation.

---

## 5. Ontology closure — current status

Three sections, three regimes.

| § | Regime | Pass? | Mechanism |
|---|---|---|---|
| § 22 | Explicit one-hop + two-hop closure — every edge installed via $\delta R$ | ✓ | Pure value-channel enforcement |
| § 23 | **Emergent transitive closure** — train $A{\to}B$ and $B{\to}C$, ask $A{\to}C$ | **✗** | Negative finding — this is the limitation |
| § 24 | **Compiled closure** — external deterministic compiler derives $A{\to}C$, installs it | ✓ | Value-channel enforcement of compiler output, fully revisable |

This is paper limitation **L1**: IBF enforces closure but does not derive it.
§ 24 documents the architectural workaround currently shipped — a
*compiled-surface* design: a deterministic external compiler handles
derivation, IBF handles durable installation, revision, retraction,
locality.

The value channel today is doing **enforcement** work, not **derivation**
work. § 23 establishes that emergence-by-default does not happen in the
current architecture.

---

## 6. Engineering surface — the agency-channel question

The framing question: *does opening the agency channel into the readout
path enable emergent (or partially-compositional) closure that the value
channel alone won't produce in § 23?*

The slack to spend is concrete. Five observations that frame the design space:

**S1. Agency centers already exist and already train on every step.** No
new write path is needed. 2,139 crystallized centers carry $D$-tracking
signal in the same proposition space at $\sigma_{\text{agency}} \approx 0.67\sigma_{\text{op}}$.

**S2. The agency output is currently $\delta k(z)$, a scalar modulating
$k_{\text{eff}} = \max(k_{\min}, k_0 + \delta k(z))$.** In domain runs
(chess, etc.) $k_{\text{eff}}$ gates compute/search behavior. In the LLM
path, $k_{\text{eff}}$ has no consumer. Possible read-side wirings:

   - **(a) Gain modulation:** $R_{\text{eff}}(z) = R_{\text{base}}(z) + k_{\text{eff}}(z)\,\delta R(z)$ — agency centers tune the local strength of the value field. Cheapest change; preserves substrate decoupling; gives a per-region "confidence" knob.
   - **(b) Composition modulation:** route $\delta k(z)$ as a *second additive term* with its own value-like payload $u_i$ — i.e., treat agency centers as a second-order δR with narrower kernel. Larger change; potentially carries the compositional signal directly.
   - **(c) Path activation:** use $\delta k(z)$ as a gate over which value
     centers fire — agency selects which compiled edges are live for a
     given query. Lets the agency channel express *context-dependent
     closure paths* without retraining the value field.

**S3. Narrower $\sigma_{\text{agency}}$ is the right tool for edge-level
selectivity.** $\sigma_{\text{op}} = 7.26$ is tuned for substantive value
generalization (smearing across paraphrases of the same fact).
$\sigma_{\text{agency}} = 4.84$ resolves ~2.25× finer spatial features —
the scale at which $A{\to}B$ and $B{\to}C$ are distinguishable as
*separate edges* rather than as overlapping value patches.

**S4. Both channels share contexts.** Any closure mechanism built on
agency inherits the same install / revise / retract / rollback machinery
that the value channel uses today. Wires to § 12 (retraction),
§ 24 (compiled closure revision), § 25 (rollback), § 35 (forgetting
decomposition) without redesign.

**S5. The compiled-surface architecture survives.** Any agency-channel
closure mechanism is **additive** to § 24's deterministic compiler, not a
replacement. The compiler still emits compiled edges; the agency channel
either (i) extends what gets compiled by carrying transitive-derivation
signal at write time, or (ii) gates which compiled edges fire at read
time. Either way, § 24 remains the headline architecture; the agency
channel reshapes what § 23's "negative finding" looks like.

**Open empirical question.** § 23 establishes that the value channel alone
does not produce $A{\to}C$ from $A{\to}B$ + $B{\to}C$. We have not yet
tested whether the **agency channel under any of (a)/(b)/(c)** changes
that. The engineering cost of (a) is ~50 lines of glue (gain modulation
at readout). The diagnostic cost is one cell — re-run § 23's negative
case with the agency readout wired in, hold everything else fixed,
measure $A{\to}C$ pass rate on held-out items, NN drift, distant
controls.

That is the cheapest single experiment that would either falsify or
support the intuition — without disturbing the canonical pipeline.

---

## 7. Invariants any agency-channel change must preserve

For the result to remain paper-defensible:

1. **Substrate decoupling.** $R_{\text{base}}$ stays frozen. Any new term
   composes at readout.
2. **LoRA durability.** The 37.5 % / 0.3 % selectivity result (§ 26) must
   survive — i.e., the agency readout cannot make the field more sensitive
   to base drift.
3. **Locality.** NN / distant controls in § 14, § 19 must stay near zero
   drift under retraction.
4. **Lifecycle dynamics.** Crystallization, dissolution, merging continue
   to apply to agency centers under the same rules; rollback (§ 25) still
   works.
5. **Revisability.** § 24's compiled-revision path must still retire and
   re-install closures cleanly.

If any wiring of $\delta k$ into readout breaks one of these, that wiring
is out — regardless of how much it helps § 23.

---

## 8. What this note does and does not claim

**Does:** documents the reference architecture, current paper-grade results,
the ontology-closure status, and the precise engineering surface the agency
question lives on.

**Does not:** take a position on whether opening the agency channel will
actually deliver emergent closure. That requires the diagnostic in § 6 —
not argument.
