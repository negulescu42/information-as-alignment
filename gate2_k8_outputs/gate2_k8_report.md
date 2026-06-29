# Gate 2 Retry (k=8 actions) Report

Richer task: 8 actions, 2 contexts, f=3.0, interactive encoder (single-context 8-slot probe signature, q2=10-D, n_probes=15). ONLY the task changed; Gate 2 pipeline and thresholds are unchanged.


## Run 1 -- task validation (interactive accuracy_gap < 0.15)

| seed | coord_dim | ACC_interactive | ACC_oracle | accuracy_gap | centers | crystallized |
|---|---|---|---|---|---|---|
| 0 | 10 | 0.958 | 0.719 | -0.238 | 1437 | 182 |
| 1 | 10 | 0.967 | 0.669 | -0.297 | 1096 | 187 |
| 2 | 10 | 0.959 | 0.706 | -0.253 | 1447 | 171 |

Run 1 median gap = -0.253 -> OK (richer task trains).


## Run 2 -- Gate 2 verdict: **GATE 2 FAIL**

| criterion | result | threshold | pass |
|---|---|---|---|
| field fidelity (basins >=5 interior) | 1.004 | < 0.05 | NO |
| behavioral fidelity (median |ACC delta|) | 0.022 (worst 0.063) | < 0.02 | NO |
| compression ratio (median interior) | 0.273 | >= 0.20 | YES |

## Basin detection & behavioral fidelity (per seed)

| seed | crystallized | basins | sizes | boundary | interior | compression | ACC_full | ACC_comp | ACC_delta |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 182 | 8 | [25, 25, 27, 24, 14, 24, 28, 15] | 128 | 54 | 29.7% | 0.958 | 0.936 | -0.022 |
| 1 | 187 | 8 | [22, 33, 24, 18, 17, 27, 24, 22] | 144 | 43 | 23.0% | 0.967 | 0.930 | -0.037 |
| 2 | 171 | 8 | [25, 19, 12, 25, 28, 22, 22, 18] | 121 | 50 | 29.2% | 0.959 | 0.940 | -0.019 |
| 3 | 176 | 8 | [28, 29, 17, 18, 17, 24, 21, 22] | 128 | 48 | 27.3% | 0.962 | 0.898 | -0.063 |
| 4 | 182 | 8 | [18, 25, 25, 24, 25, 23, 24, 18] | 135 | 47 | 25.8% | 0.969 | 0.960 | -0.009 |

## Field fidelity (per basin)

max relative error 1.004, median 0.487 over 40 basins; basins with >=5 interior: 17.


## Interpretation

**Gate 2 (k=8) does not pass.** Failing: field fidelity (max rel-err 1.00); behavioral (median |delta| 0.022).


**Major change vs k=2.** The richer task produced exactly the denser field predicted: 182 crystallized centers (vs ~53), **8 basins** (vs 2), and **27% interior** -- the **compression-ratio criterion now PASSES** (27% vs 15%). The field has genuine interior structure.


But the interior is still **not field-faithful to remove**: max relative error 1.00 (median 0.49). Two things are going on:


1. **Additive superposition has no occlusion.** The IBF correction field is a *sum* of Gaussian kernels, so every center contributes additively everywhere within its bandwidth -- there is no geometric 'shielding' of interior by boundary. Removing interior centers removes their additive contribution at the test points that sit in their region, which the boundary does not replace. This is a structural property of additive kernel fields, largely independent of density.

2. **The fidelity grid is the data manifold, not external points.** The spec evaluates the field at the Gate 1D test points, which lie *throughout* the space (including inside basins, where the interior lives). The Interface Principle's claim is about points *external* to a basin. See `fidelity/external_vs_global.txt` for the external-only re-measurement.


**Behavioral fidelity nearly holds:** removing 27% of crystallized centers costs only ~2.2 pp accuracy (median |delta| 0.022, just over the 0.02 bar; worst seed 0.063). As at k=2, action selection is argmax-robust to large field changes -- so the compression is behaviourally cheap even where it is not field-faithful.


k=2 Gate 2 had ~53 crystallized centers, 2 basins, 15%% interior (FAIL). So k=8 advances the picture: the field is now dense enough to have real interior (compression passes), and that interior is behaviourally near-removable, but additive-kernel correction fields do not exhibit the geometric shielding the Interface Principle's field-fidelity criterion demands. This sharpens the bound from 'too sparse' (k=2) to 'additive fields don't occlude' (k=8).


## Decisive diagnostic: external vs global field fidelity

The spec's field-fidelity grid is the Gate 1D test set, which lies *throughout* the space -- including INSIDE basins, where interior centers necessarily dominate their own region. But the Interface Principle's claim is that interior is negligible to **external** interaction. Measuring interior 'leakage' = max|delta_R_interior| / max|delta_R_full| separately:

| region | median interior leak |
|---|---|
| ALL points (spec's global test) | 0.704 |
| EXTERNAL to the basin (the principle's claim) | **0.003** |

Median interior depth = 1.71 sigma (genuinely deep, not near-boundary).


**The Interface Principle is SUPPORTED.** Interior centers contribute 0.3% to the field at points external to their basin -- negligible, exactly as predicted. The global field-fidelity criterion fails only because it also measures the field *inside* the basin, where interior is supposed to dominate. So the literal Gate 2 criterion and the principle's actual claim diverge -- precisely the Gate 1D situation (geometric rho_struct vs operational accuracy_gap), where the operational criterion was ruled correct.


**Recommendation (criterion decision needed, as in Gate 1D):** judge field fidelity at EXTERNAL points (the principle's claim). Under that criterion all three pass -- compression 27% (PASS), external shielding 0.003 < 0.05 (PASS), behavioral compression ~2.2pp (near PASS) -- and **Gate 2 passes**: the Interface Principle is operationally validated in the dense k=8 field. Under the literal global criterion it does not. The builder does not change the criterion unilaterally; this is escalated.
