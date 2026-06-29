# Branch: `gate-2-interface-extraction`

## Goal
First operational test of the **Interface Principle** (machine-verified in Lean):
interior particles contribute negligibly to external interaction, so removing
them should preserve behavior. Gate 2 is **post-hoc analysis of the trained
Gate 1D interactive system at f=3.0** — no new training.

## Pipeline (`run_gate2.py`)
1. **Basin detection** — kernel-overlap graph on crystallized Scale 2 centers
   (edge if `exp(-d²/2σ_ij²) > tau_basin`), connected components ≥ 3 = basins.
2. **Boundary / interior partition** — a basin center is BOUNDARY if it has
   non-negligible overlap (`> tau_face`) with any center *outside* its basin
   (faces external structure); INTERIOR if shielded. (This resolves a degeneracy
   in the spec's literal rule: basins are connected components, so no center has
   a cross-component edge — the literal rule would label everything interior.)
3. **Field fidelity** — raw `δR = Σ vᵢ K(y, zᵢ)` at the test points, full basin
   vs boundary-only; relative error.
4. **Behavioral fidelity** — ACC with all centers (full) vs interior removed
   (compressed) vs oracle, on the Gate 1D test set.

```bash
python run_gate2.py 5
python summarize_gate2.py
```

## Result (5 seeds) — **GATE 2 FAIL**

| criterion | result | threshold | pass |
|---|---|---|---|
| field fidelity (basins ≥5 interior) | max rel-err **0.485** | < 0.05 | NO |
| behavioral fidelity (median \|ΔACC\|) | 0.004 (worst seed 0.023) | < 0.02 | YES |
| compression ratio (median interior) | **15.4%** | ≥ 20% | NO |

Every seed: ~53 crystallized centers → **2 basins** (~26 each), 0 singletons,
only **9–18% interior**. Removing interior changes the raw correction field by up
to 48–81%, yet behavioral accuracy barely moves (median ΔACC = 0.004).

## Interpretation
The 8-D correction field (6-D interactive coords + 2-D action embedding) is
**mostly boundary**: nearly every crystallized center has non-negligible overlap
with the other basin, so there is no meaningful *deep* interior. Two consequences:
- **Too little to compress** (15% < 20%), and
- the interior that exists is **not shielded** (field error up to 0.48–0.81) —
  the Interface Principle's negligible-interior prediction does **not** hold here.

Behavioral robustness is itself informative: even a ~48% change in the raw field
leaves accuracy intact, because action selection is argmax-robust (the correction
field has large decision margins). So compression is *behaviorally* safe but
*not* field-faithful, and there isn't enough interior for it to matter.

**This bounds the Interface Principle's operational applicability:** it requires
denser / higher-dimensional correction fields where genuine deep interior forms.
The present two-action toy is too sparse for interface extraction. An honest
scope finding, not a pipeline bug — matching the failure mode the spec
anticipated for this dimensionality.

## Status
```
field fidelity:       FAIL (interior not shielded, rel-err up to 0.48)
behavioral fidelity:  PASS at median (0.004); one seed breaches (0.023)
compression ratio:    FAIL (15.4% < 20%)
GATE 2:               FAIL  -> Gate 3 (promotion/recursion) does not proceed
```
No promotion/recursion implemented (Gate 3 awaits a Gate 2 pass, e.g. on a
higher-dimensional field).
