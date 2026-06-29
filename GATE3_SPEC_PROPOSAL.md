# Gate 3 — Promotion & Recursive Interaction (SPEC PROPOSAL, awaiting approval)

> Status: **proposal for supervisor review.** Every gate to date ran against a
> supervisor-approved spec; this drafts Gate 3 in the same format so it can be
> approved, amended, or replaced. No promotion/recursion code has been written.

## What Gates 1–2 established (the foundation Gate 3 stands on)
- A trained **interactive two-scale system** (Gate 1D): Scale 1 representation
  particles (with splitting) → interactive encoder → Scale 2 correction engine,
  validated on the hard f=3.0 generator.
- That system's Scale 2 correction field, on a **dense (k=8) task**, forms
  **stable basins** whose **interior is externally shielded** (Interface
  Principle, Gate 2): each basin is compressible to its **boundary-only**
  representation with 0.3% external-field error and ~2pp behavioral cost.

So we now have, per basin, a verified **compressed operational unit**: the
boundary-only correction that reproduces the basin's external behavior within ε.

## What Gate 3 tests
The Interface Principle gives *compression*. **Promotion** asks whether each
compressed basin can become a **single operational unit at a higher scale**, and
**recursive interaction** asks whether a Scale-3 process can run correction
dynamics over those units — i.e. whether the crystallized Scale-2 structure
becomes the *configuration space* for Scale 3, recursively instantiating
Postulate 2 one level up.

Two claims, tested in order (Gate 3a gates 3b):

**Gate 3a — Promotion (interface nodes).** Replace each basin with one **interface
node**: a compact descriptor (e.g. boundary centroid + aggregate correction
amplitude + agency `w` statistics + external-footprint bandwidth) that reproduces
the basin's external interaction within tolerance ε. Test: a system whose Scale 2
field is the set of interface nodes (one per basin) instead of the full center
population behaves within ε of the full system on the Gate 1D test set.

**Gate 3b — Recursive interaction.** Treat the interface nodes as the
**observation/configuration space for a Scale 3** correction process: define a
higher-scale task over *combinations/sequences* of basins (see "task" below), and
test whether the existing v1 engine, running on a representation built from the
interface nodes, learns it — and whether a *third-scale* crystallization emerges.

## Proposed mechanism (promotion)
For each basin B with boundary set ∂B (from Gate 2):
```
interface_node(B) = {
   z*      : external-effective center (boundary centroid weighted by |v|),
   sigma*  : external-footprint bandwidth (fit so the node's single kernel
             matches the basin's external delta_R within epsilon),
   v*      : net external correction amplitude,
   w*      : aggregate agency (mean/var of boundary w),
   context : basin's dominant context_id,
}
```
A node's external field is `v* * K_{sigma*}(y, z*)`. Promotion is **lossless to
tolerance** iff `max_{y external} |delta_R_basin(y) - v* K(y,z*)| / max|delta_R| <
epsilon` (this reuses Gate 2's external-fidelity machinery, with a single-kernel
surrogate instead of boundary-only).

## Proposed Scale-3 task (recursive interaction) — needs supervisor steer
The cleanest higher-scale task that uses basins as units: each Scale-2 basin
corresponds to an action-sector region; a **Scale-3 episode** presents a *pair*
(or short sequence) of positions and asks for a relation between their basins
(e.g. "are they in adjacent sectors?" / "which sector dominates?"). The Scale-3
representation is built from the interface-node activations of the constituent
positions; the v1 engine runs correction dynamics on it. This keeps the engine
unchanged and tests whether promoted units support new dynamics.
*(Open question for you: is a relational pair-task the right Scale-3 probe, or do
you want sequence/composition, matching the RRW/chess framing — still without
implementing those domains?)*

## Pass criteria (proposed, mirroring prior gates)
**Gate 3a (promotion):**
- external fidelity of node vs basin < 0.05 for every basin ≥5 interior;
- behavioral: |ACC_promoted − ACC_full| < 0.02 (median) on the Gate 1D test set;
- compression: #interface_nodes ≤ 0.25 × #crystallized_centers (one node per basin
  is ~8 / ~180 ≈ 4%, so this should pass comfortably if promotion is faithful).

**Gate 3b (recursion):**
- Scale-3 accuracy_gap < 0.15 vs an oracle Scale-3 representation (true basin ids);
- a third-scale crystallization forms (≥1 crystallized Scale-3 center per seed);
- ablation: Scale-3 on *shuffled* interface nodes fails (nodes carry real
  structure), 5 seeds.

## Controls
- promotion vs random-node baseline (random z*/sigma*): confirms node fidelity is
  earned;
- recursion vs raw-center baseline (Scale 3 on full Scale-2 centers, no promotion):
  does promotion *help or at least not hurt* Scale 3?;
- oracle Scale-3 (true basin ids) upper bound.

## What stays fixed
v1 Scale 2 engine, Scale 1 (splitting), interactive encoder, k=8 generator at
f=3.0, the Gate 2 basin/partition/external-fidelity code. Gate 3 adds only the
promotion step and the Scale-3 wrapper.

## Explicit non-goals
No RRW/chess/CIFAR/LLM domains. No changes to the validated engines. Recursion is
**one level** (Scale 2 → Scale 3); no unbounded recursion.

## Risks / open questions for the supervisor
1. **Single-kernel promotion may be too lossy** — a basin's external field may
   need 2–3 boundary kernels, not one. Acceptable to allow k_node ∈ {1,2,3}?
2. **Scale-3 task design** is the real degree of freedom (relational vs
   compositional). Your call sets the experiment.
3. **Density at Scale 3**: only ~8 basins/seed → ~8 interface nodes. A Scale-3
   field over 8 units may itself be too sparse (the Gate 2 k=2 problem one level
   up). May need more basins (more actions, or pooling seeds) for a meaningful
   Scale-3 field.

Awaiting your approval / amendments before any implementation.
