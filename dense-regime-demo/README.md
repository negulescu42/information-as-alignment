# Synthetic Dense-Regime Demonstration (§7 upgrade)

A controlled stress test of the operating-bandwidth mechanism on a
constructed dense-regime correction field. **Not** a model of the deployed
FI field — see `section7_insert.md` for the prose disclaimer that must
appear in the paper.

## What is here

| file | purpose |
|------|---------|
| `dense_regime_demo.py` | numerics: geometry, gate verification, panel data, figure |
| `dense_regime_demo.png` | two-panel figure (Panel A calibration gap; Panel B the count that matters) |
| `dense_regime_demo_results.json` | machine-readable summary (all gate values, σ's, Tail/ε's) |
| `SyntheticDense.lean` | Lean: defines `syntheticDenseField`, instantiates the keystone and the refscale locality on it |
| `section7_insert.md` | one-paragraph drop-in for §7 |

## Reproduce

```bash
cd dense-regime-demo
python dense_regime_demo.py
```

This writes `dense_regime_demo.png` and `dense_regime_demo_results.json`,
and exits non-zero if the validity gate fails.

## Construction (the auditable geometry)

Ambient dimension `d = 16`, query point `y = 0`, shell radius `d_shell = 1`,
tolerance `ε = 0.05`, amplitude bound `V_max = 1`.

| group     | distance / d_shell | multiplicity | non-local? |
|-----------|--------------------|--------------|------------|
| local     | 0.00               |   1          |  no        |
| dominant  | 1.01               |   3          | yes        |
| mid       | 1.30               | 160          | yes        |
| far       | 2.00               | 700          | yes        |
| **total** |                    | **864**      |            |

The local center depresses the full-field participation ratio (one
near-unit weight dominates the squared sum). The dominant group is
positioned just outside d_shell — at `1.01·d_shell` instead of exactly
`d_shell` — because the codebase's `nonLocalSet` is defined by the strict
inequality `d_shell < ‖y − z_i‖`. The Gaussian weight at `1.01·d_shell`
is essentially the same as at `d_shell` to four decimal places. The far
cloud's many small weights inflate the non-local participation ratio
above the full-field count without dominating either sum.

## Numerical result (verified by `dense_regime_demo.py`)

At the reference bandwidth σ_ref = σ_pair = `1 / √(2·log(20))`:

| count                        | value     |
|------------------------------|-----------|
| full-field N_eff             | 4.5973    |
| non-local N_eff              | 102.7228  |
| non-local cardinality \|S\|  | 863       |

Validity gate (all pass):

| condition                          | value           |
|------------------------------------|-----------------|
| card(nonlocal) ≥ 200               | 863             |
| full-field nEff ≤ 5                | 4.5973          |
| non-local nEff ≥ 50                | 102.7228        |
| non-local nEff / full nEff ≥ 20    | 22.3444         |
| three counts pairwise ≥ 5×         | 8.4, 22.3, 188  |

Dense-regime sanity: σ\* ≤ σ_ref, and `V_max · nEff_nl ≥ ε · exp(d²/(2σ_ref²))`
both hold.

## Panel data

Directly-evaluated aggregate tail at four bandwidths
(`Tail = Σ_i V_max · K_σ(‖y − z_i‖²)` over non-local centers; not the bound):

| bandwidth      | source                     | σ        | Tail        | Tail/ε  |
|----------------|----------------------------|----------|-------------|---------|
| σ_pair         | pairwise (calibration)     | 0.40854  | 1.158e+00   | 23.16   |
| σ from full    | full-field N_eff (UNSAFE)  | 0.33255  | 1.067e-01   |  2.13   |
| σ\* from nl    | non-local N_eff (correct)  | 0.25603  | 1.656e-03   |  0.033  |
| σ from \|S\|   | cardinality (WASTEFUL)     | 0.22638  | 1.539e-04   |  0.003  |

That `Tail(σ\*) / ε ≈ 0.033` rather than ≈ 1 is a real (and expected)
finding about the looseness of the unconditional keystone bound for
heterogeneous Gaussian weights: the bound is `V_max · nEffFinset · K_σ`,
and the actual tail is below it. It does **not** mean σ\* is wrong
— it confirms that σ\* is safely on the right side of tolerance, while
σ_full violates ε by 2.1× and σ_pair violates by 23×.

## Lean instance

`SyntheticDense.lean` defines `syntheticDenseField : CorrectionField 16`
with the geometry above. The two headline theorems are direct applications
of the existing theorems on this concrete field — no re-derivation:

- `synthetic_tail_controlled` — `aggregateTail_le_nEffFinset_bound`
  instantiated.
- `synthetic_refscale_locality` — `locality_at_operatingBandwidth_refscale`
  instantiated, with the dense-regime hypothesis `hDense` discharged
  algebraically via the choice `σ_ref = pairwiseBandwidth d_shell ε`
  (which makes `ε · exp(d²/(2σ_ref²)) = 1`).

After loading the file in a Lean 4 / Mathlib project that includes the
parent `RequestProject` modules (`Defs`, `CoreBound`, `CorrectionField`,
`NEffMono`, `Keystone`, ...), run:

```lean
#print axioms OperatingResolution.SyntheticDense.synthetic_tail_controlled
#print axioms OperatingResolution.SyntheticDense.synthetic_refscale_locality
```

Both should report only the three standard Lean axioms
(`propext`, `Classical.choice`, `Quot.sound`) — no `sorry`.

The validity-gate count-separation inequalities (`nEff_full ≤ 5`,
`nEff_nl ≥ 50`, etc.) involve specific values of `Real.exp` not directly
amenable to `norm_num` / `decide`; they are verified numerically in
`dense_regime_demo.py`. The Lean headline theorems do **not** depend on
those tight bounds — they require only `nEffFinset ≥ 1`, which is the
existing `one_le_nEffFinset`.
