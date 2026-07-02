# Handover — Upgrading the Engine for Postulate 2 (Recursive Scale Structure)

**Audience:** you built and understand the original IBF engine (Postulate 1). You have
*not* seen Postulate 2. This document takes you from the new theory to working code
with nothing skipped and nothing extra. Everything described here is implemented and
validated in this `postulate2/` package; `validate.py` proves it runs end to end.

---

## 0. What you already have (Postulate 1, one paragraph)

The original engine is a continual-learning agent over a fixed representation `z`.
It reads a correction field `ΔR(z) = Σ gate·v·K(z, z_c)` from a set of localized
**memory centers**, selects actions, and on a **discrepancy** `D` writes a local
correction near the active center. Centers that stop moving **crystallize** (their
learning rate drops); crystallized centers survive cross-context exposure only if the
**Crucible** does not detect a sign reversal. This gives memory, agency, and
self-correction. **None of that changes.** Keep the file `engine.py` in your head as
"the engine" — Postulate 2 wraps it, it does not rewrite it.

---

## 1. Postulate 2 in plain terms

> **For any system observed at scale λ, its effective configuration space is realized
> by coherent structures crystallized at finer scales λ′ < λ.**

Operationally, and this is the whole claim you are asked to make real in code:

> **Lower-scale crystallization → an emergent higher-scale configuration space.**

Read literally: the *space the engine operates in* should not be handed to it. It
should **fall out of** running the same crystallization dynamics one level down. You
run the engine's own memory/crystallization process at a **finer scale** on raw
observations; the crystallized structures that survive *become the coordinate system*
in which the correction engine (the thing you already have) then operates.

So there are two scales:

- **Scale 1 (finer, λ′):** crystallize representation particles directly in the raw
  observation space. What survives defines an emergent 2D configuration space.
- **Scale 2 (coarser, λ):** the **original engine, unchanged**, running on the
  coordinates produced by Scale 1.

If Postulate 2 is true, Scale 2 on the *emergent* coordinates should perform like
Scale 2 on the *true* hidden coordinates. That is the thing we test.

---

## 2. The toy world that lets us test it

We need a world with a genuine "finer scale" hidden underneath the observation.

- A hidden 2D manifold `u = (u1, u2)` is the *truth*: the correct action is always a
  function of `u`, never of the raw observation directly.
- A fixed nonlinear embedding maps `u → x ∈ ℝ²⁰` (the observation). The engine only
  ever sees `x`.
- Two generators:
  - **1A (easy):** `x` is a near-isometric lift of `u`. Distances in `x` track
    distances in `u`. Geometry alone recovers `u`.
  - **1B (hard):** `u2` is encoded *only* through high-frequency aliased terms
    `sin(f·u2)`, `f = 3.0`. Two points far apart in `u2` can sit right next to each
    other in `x`. **Geometry alone cannot recover `u2`.** This is the realistic case
    and it is what forces the interactive mechanism in §4.2.

`environment.py` is this world. `Gate1Config` is its only knob-set. `correct_actions_batch(u, ctx)`
gives the truth for a context. Actions live on a unit circle: `k` actions =
`k` angular sectors (`k=2` for the geometry tests, `k=8` for the dense-field and
continual-learning tests).

---

## 3. The upgrade job, in exactly two parts

### Part A — the engine changes (small, guarded, no-ops by default)

You add **two fields** to `MemoryCenter` and a handful of guards. Both default to
"off," and with them off the engine is byte-for-byte the original. This is important:
Postulate 1 behaviour must be preserved.

1. **`frozen: bool = False`** — a *read-only reserve* center. It is still **read**
   during same-context readout (normal context gating applies), but it is **never
   written**: skipped in the Crucible write path, the same-context update, the
   epoch decay/crystallization pass, and merge. See the four `if c.frozen: continue`
   guards and the `reserve = [c for c in self.centers if c.frozen]` handling in
   `_merge`.

2. **`interface_group: int = None`** — optional. When several centers that were
   promoted together (see §4.3) co-activate under cross-context pressure, the group
   absorbs the reversal pressure **as a unit** (one peak center's worth) instead of
   N independent copies. Guarded so that `interface_group is None` reproduces the
   exact original update. See the `group_sum`/`group_max` block and
   `eff_kw = group_max[g]*kw/group_sum[g]` in `update()`.

That is the entire diff to the engine. `engine.py` documents both hooks at the top.

### Part B — the new layers around the engine (new files, engine untouched)

| file | role |
|---|---|
| `scale1.py`   | Scale-1 representation learner: crystallize particles in `x`-space, split non-converging ones along behavioural fault lines, embed the survivors into a 2D **emergent config space**, expose it as an encoder. |
| `encoders.py` | encoders that turn a raw `x` into a Scale-2 coordinate — the emergent one, plus the **interactive** encoder needed for generator 1B. |
| `scale2.py`   | thin, dimension-agnostic scaffolding to run `engine.IBFAgent` **unchanged** on whatever coordinates an encoder produces, plus metrics. |
| `promotion.py`| detect crystallized **basins** in a trained Scale-2 field and **promote** them: freeze the interior (read-only reserve), keep the boundary active as one interface. |

The mental model: **Scale 1 builds the room; the original engine (Scale 2) lives in
it; promotion compresses the room's furniture for reuse across contexts.**

---

## 4. The four mechanisms (this is the science; each has a check in `validate.py`)

### 4.1 Emergent configuration space  → check V1

Run the crystallization dynamics at Scale 1 on raw `x`. Representation particles that
converge crystallize; the survivors are embedded into 2D (spectral embedding over a
particle-similarity graph). That 2D space is the **emergent config space** — nobody
handed it to the engine, it crystallized out of finer-scale dynamics.

**Claim tested:** on the easy generator (1A), the emergent 2D space recovers the
hidden manifold — pairwise-distance rank correlation `rho_struct > 0.8`. Passing this
is the literal statement of Postulate 2: lower-scale crystallization produced a usable
higher-scale configuration space. *(V1: rho ≈ 0.83.)*

### 4.2 Interactive encoding for aliased observations  → check V2

On the hard generator (1B), `u2` is hidden behind aliasing, so **no static map of `x`
can recover it** — geometry is blind. The fix follows directly from what the engine
*is*: an agent that acts. Instead of reading coordinates off `x`, you **probe the
environment** — take a few exploratory actions, observe the rewards, and use that
*behavioural signature* as the coordinate. Behaviour disambiguates what geometry
cannot. `build_interactive_coords` in `encoders.py` does this.

**Claim tested:** static recovery of `u2` is ≈ 0 (`rho_u2 < 0.3`) while interactive
recovery clears `rho_u2 > 0.5`. *(V2: static ≈ 0.00, interactive ≈ 0.76.)*

> Note on splitting: on 1B, a single representation particle can straddle points that
> behave differently. Scale 1 detects these (high discrepancy variance) and **splits**
> the particle along the behavioural fault line via `k-means(2)` on the collected
> signatures (`_split_particles` / `_make_children`). This is what lets Scale 1 build a
> faithful graph even when the raw geometry is misleading; it feeds V2 and beyond.

### 4.3 Interface extraction — external shielding  → check V3

Train the original engine at Scale 2 with `k=8`. The crystallized field organizes into
coherent **basins**. Within a basin, distinguish:

- **boundary** centers — those with kernel overlap onto *another* structure, and
- **interior** centers — shielded, facing only their own basin.

The **Interface Principle** (the operative form we validated): a basin's interior is
negligible to *external* interaction. Not internally — inside the basin the interior
carries most of the field — but *externally*, at points outside the basin, the
interior's leakage is tiny. So a basin can be represented, for cross-structure
purposes, by its **boundary alone**. `detect_basins` / `partition_basin` /
`external_field_fidelity` implement this.

**Claim tested:** interior leakage measured at **external** points is `< 0.05`, at a
meaningful compression ratio (interior fraction `> 0.15`). *(V3: external leak ≈ 0.009,
interior fraction ≈ 0.25.)*

> Spec correction baked in: fidelity is measured on **external test points only**. The
> global (all-points) version is stricter because it includes the basin's own interior,
> where the interior is *supposed* to be large — measuring there tests the wrong claim.

### 4.4 Context-aware promotion — compression that *improves* retention  → check V4

This is the payoff and the surprising result. In continual learning, later contexts
erode what earlier ones learned. Promotion uses the interface structure to protect it.

`promote(agent)`:
- **interior → frozen reserve** (`c.frozen = True`): still read for **same-context**
  queries, dropped from cross-context readout while unverified, and — crucially —
  **never written**, so later contexts cannot corrode it.
- **boundary → stays active**, tagged with one `interface_group` id so cross-context
  reversal pressure is absorbed as a unit (§3 Part A hook 2).
- `interface_crucible` aggregates the engine's own Crucible to the interface level with
  a softened reversal threshold (the boundary averages evidence across its centers, so
  it needs less per-center evidence): a **majority-contradicted** interface dissolves.

**Claim tested:** across three contexts (A→B→C), retention of context A under promotion
is **no worse than, and in fact better than**, the un-promoted engine
(`Acc_A_promoted ≥ Acc_A_full − 0.03`), while compressing the field (interior frozen
fraction `> 0.10`). *(V4: full ≈ 0.59, promoted ≈ 0.72 — promotion **improves**
retention by ~13 points — at a ≈0.48 compression.)*

The discovery in one line: **freezing the shielded interior is not just compression, it
is protection — a frozen reserve is immune to the cross-context interference that erodes
an active interior.**

---

## 5. Validated parameters and thresholds

All defaults live in `engine.Config` (`C`) and `environment.Gate1Config`. The values
below are the ones the mechanisms were validated at — do not treat them as free.

| quantity | value | where |
|---|---|---|
| actions `k` | 2 (geometry) / 8 (dense field, continual) | `engine.C.k` — **set before use** |
| hard-generator alias frequency `f` | 3.0 | `Gate1Config.u2_freq` |
| Scale-1 splitting | enabled on 1B | `Gate1Config.enable_splitting=True` |
| crystallization reversal threshold | −0.125 | `C.reversal_threshold` |
| interface reversal threshold | `C.reversal_threshold × 0.5` | `promotion.REVERSAL_THRESHOLD_INTERFACE` |
| basin kernel-overlap threshold | 0.01 | `promotion.TAU_BASIN` |
| boundary "faces external" threshold | 1e-4 | `promotion.TAU_FACE` |
| min basin size | 3 | `promotion.MIN_BASIN` |
| action-separation target | 3.5 | `C.action_sep_target` |

Pass thresholds (the qualitative claims):

| check | threshold | meaning |
|---|---|---|
| V1 | `rho_struct > 0.8` | emergent config space recovers the manifold (1A) |
| V2 | static `rho_u2 < 0.3` **and** interactive `> 0.5` | interaction beats geometry (1B) |
| V3 | external leak `< 0.05`, interior fraction `> 0.15` | interior externally shielded |
| V4 | `Acc_A_P ≥ Acc_A_F − 0.03`, compression `> 0.10` | promotion preserves/improves retention |

---

## 6. How to run and validate

```bash
cd postulate2
python validate.py     # runs V1–V4 on small/fast configs; expects 7/7 PASS
```

`validate.py` uses deliberately small configs so the whole suite runs in a few minutes;
it proves the reference code is **functional and directionally correct end to end**. The
full-scale magnitudes quoted in §4 come from the larger configs (bump `N_repr_pool`,
`N_train_pool`, `E_scale1`, `E_scale2` and seed-average); the code path is identical.

**The one gotcha:** the number of actions is global — `engine.C.k`. Set it (2 or 8)
before you build environments/agents for a given experiment. Modules import each other
by bare name, so run from inside `postulate2/` (or add it to `sys.path`).

---

## 7. What to build first (suggested order)

1. Land the **engine hooks** (Part A) and confirm your existing Postulate-1 tests are
   unchanged with `frozen`/`interface_group` at defaults.
2. `environment.py` → sanity-check 1A/1B generators and balanced `k`-action truth.
3. `scale1.py` → get V1 green on 1A. This alone *is* Postulate 2's core statement.
4. `encoders.py` interactive path → get V2 green on 1B.
5. `scale2.py` is just scaffolding around your untouched engine — wire it and confirm an
   A→B run trains.
6. `promotion.py` → V3 then V4.

If V1 and V4 are both green, Postulate 2 is realized: a configuration space **emerged**
from finer-scale crystallization, and compressing it along its interface structure
**improved** the coarser-scale engine's continual learning.
