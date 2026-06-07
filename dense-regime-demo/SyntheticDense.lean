/-
# Synthetic Dense-Regime Demonstration Field (§7 upgrade)

This file constructs a concrete `CorrectionField 16` and instantiates the
existing keystone and refscale-locality theorems on it. It is a controlled
stress test of the mechanism — NOT a model of the deployed FI field.

## Geometry (auditable)

| group     | distance / d_shell | multiplicity | non-local? |
|-----------|--------------------|--------------|------------|
| local     | 0.00               |   1          |  no        |
| dominant  | 1.01               |   3          | yes        |
| mid       | 1.30               | 160          | yes        |
| far       | 2.00               | 700          | yes        |
| **total** |                    | **864**      |            |

- `d_shell = 1`, `ε = 1/20`, `V_max = 1`, ambient dimension `d = 16`.
- Reference bandwidth `σ_ref = pairwiseBandwidth d_shell ε` (Path A).

The dominant group sits at `1.01·d_shell` instead of `d_shell` because the
codebase's `nonLocalSet` is defined with the STRICT inequality
`d_shell < ‖y - z_i‖`. Choosing `1.01·d_shell` keeps the group "just barely"
non-local; the Gaussian weight at `1.01·d_shell` differs from that at
`d_shell` by < 6% (cosmetic, not structural).

## Headline theorems

- `synthetic_tail_controlled`     — `aggregateTail_le_nEffFinset_bound` instantiated.
- `synthetic_refscale_locality`   — `locality_at_operatingBandwidth_refscale` instantiated,
                                    with `hDense` discharged via the choice
                                    `σ_ref = pairwiseBandwidth d_shell ε`.

Both are proved by exact application of the existing theorems on this concrete
field. After loading, `#print axioms` on either should show only the three
standard Lean axioms.

## Cardinality

- `card_nonLocalSet_eq_863` — proves `card (nonLocalSet y0 d_shell) = 863`
  by reduction to a `Fin`-cardinality arithmetic.

## Note on count-separation gate inequalities

The validity-gate inequalities `nEff_full ≤ 5` / `nEff_nl ≥ 50` /
`nEff_nl / nEff_full ≥ 20` involve specific values of `Real.exp` and are not
directly amenable to `norm_num` / `decide` without bespoke approximation
machinery. They are verified numerically in the companion Python script
`dense_regime_demo.py`. The structural keystone and refscale-locality
applications below do NOT depend on those tight numerical bounds — they
require only `nEffFinset ≥ 1`, which is `one_le_nEffFinset`.
-/

import RequestProject.Main

noncomputable section

open Real Finset

namespace OperatingResolution

namespace SyntheticDense

/-! ## Parameters -/

/-- Ambient dimension. -/
def d_dim : ℕ := 16

/-- Multiplicities of the four groups. -/
def n_local : ℕ := 1
def n_dom   : ℕ := 3
def n_mid   : ℕ := 160
def n_far   : ℕ := 700

/-- Total number of centers. -/
def M_total : ℕ := n_local + n_dom + n_mid + n_far

theorem M_total_eq : M_total = 864 := by decide

theorem M_total_pos : 0 < M_total := by decide

/-- Shell radius. -/
def d_shell : ℝ := 1

/-- Tolerance. -/
def ε : ℝ := 1 / 20

/-- Amplitude bound. -/
def V_max : ℝ := 1

theorem d_shell_pos : (0 : ℝ) < d_shell := by unfold d_shell; norm_num
theorem ε_pos       : (0 : ℝ) < ε       := by unfold ε; norm_num
theorem V_max_pos'  : (0 : ℝ) < V_max   := by unfold V_max; norm_num
theorem ε_lt_one    : ε < 1             := by unfold ε; norm_num

/-! ## Distance map

Each index `i : Fin M_total` is assigned a distance from the query point:
- `i.val = 0`              → 0      (the one local center)
- `1 ≤ i.val < 4`          → 1.01   (the 3 dominant centers)
- `4 ≤ i.val < 164`        → 1.30   (the 160 mid centers)
- `164 ≤ i.val < 864`      → 2.00   (the 700 far centers)
-/

/-- The distance assigned to index `i`. -/
def distFromOrigin (i : Fin M_total) : ℝ :=
  if i.val < n_local then 0
  else if i.val < n_local + n_dom then (101 : ℝ) / 100
  else if i.val < n_local + n_dom + n_mid then (13 : ℝ) / 10
  else 2

theorem distFromOrigin_nonneg (i : Fin M_total) : 0 ≤ distFromOrigin i := by
  unfold distFromOrigin
  split_ifs <;> norm_num

/-- For the (unique) local index 0, the distance is 0. -/
theorem distFromOrigin_local (i : Fin M_total) (h : i.val < n_local) :
    distFromOrigin i = 0 := by
  unfold distFromOrigin
  simp [h]

/-- Any index `i.val ≥ n_local` is strictly outside the shell. -/
theorem distFromOrigin_gt_d_shell (i : Fin M_total) (h : n_local ≤ i.val) :
    d_shell < distFromOrigin i := by
  unfold distFromOrigin d_shell
  have hnl : ¬ (i.val < n_local) := not_lt.mpr h
  simp only [hnl, if_false]
  split_ifs <;> norm_num

/-! ## Centers in `EuclideanSpace ℝ (Fin 16)` -/

/-- The first basis axis index (we place all centers along this axis). -/
def axisIdx : Fin d_dim := ⟨0, by unfold d_dim; norm_num⟩

/-- The center for index `i`: a vector with `distFromOrigin i` at axis 0
    and zero elsewhere. -/
noncomputable def syntheticCenter (i : Fin M_total) : EuclideanSpace ℝ (Fin d_dim) :=
  EuclideanSpace.single axisIdx (distFromOrigin i)

/-- The query point: origin. -/
noncomputable def y0 : EuclideanSpace ℝ (Fin d_dim) := 0

/-- `‖syntheticCenter i‖ = distFromOrigin i`. -/
theorem norm_syntheticCenter (i : Fin M_total) :
    ‖syntheticCenter i‖ = distFromOrigin i := by
  unfold syntheticCenter
  rw [EuclideanSpace.norm_single]
  exact abs_of_nonneg (distFromOrigin_nonneg i)

/-- The distance from `y0` to center `i` is exactly `distFromOrigin i`. -/
theorem norm_y0_sub_center (i : Fin M_total) :
    ‖y0 - syntheticCenter i‖ = distFromOrigin i := by
  unfold y0
  rw [zero_sub, norm_neg]
  exact norm_syntheticCenter i

/-! ## Amplitudes (all V_max = 1, sign +) -/

/-- All amplitudes are `V_max = 1`. -/
def syntheticAmplitude (_ : Fin M_total) : ℝ := V_max

theorem syntheticAmplitude_le_V_max (i : Fin M_total) :
    |syntheticAmplitude i| ≤ V_max := by
  unfold syntheticAmplitude V_max
  norm_num

/-! ## The correction field -/

/-- The synthetic dense-regime correction field. -/
noncomputable def syntheticDenseField : CorrectionField d_dim where
  M := M_total
  centers := syntheticCenter
  amplitudes := syntheticAmplitude
  V_max := V_max
  amp_bounded := syntheticAmplitude_le_V_max
  V_max_pos := V_max_pos'

theorem syntheticDenseField_M : syntheticDenseField.M = M_total := rfl
theorem syntheticDenseField_V_max : syntheticDenseField.V_max = V_max := rfl
theorem syntheticDenseField_centers : syntheticDenseField.centers = syntheticCenter := rfl

/-! ## Non-local set -/

/-- Membership in the non-local set is equivalent to having index `≥ n_local`. -/
theorem mem_nonLocalSet_iff (i : Fin M_total) :
    i ∈ syntheticDenseField.nonLocalSet y0 d_shell ↔ n_local ≤ i.val := by
  unfold CorrectionField.nonLocalSet
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, decide_eq_true_eq]
  rw [show syntheticDenseField.centers i = syntheticCenter i from rfl]
  rw [norm_y0_sub_center]
  constructor
  · intro h_dist
    by_contra h_lt
    push_neg at h_lt
    rw [distFromOrigin_local i h_lt] at h_dist
    unfold d_shell at h_dist
    linarith
  · exact fun h => distFromOrigin_gt_d_shell i h

/-- The non-local set: `{i : Fin M_total | n_local ≤ i.val}`. -/
theorem nonLocalSet_eq :
    syntheticDenseField.nonLocalSet y0 d_shell =
      (Finset.univ : Finset (Fin M_total)).filter
        (fun i => decide (n_local ≤ i.val) = true) := by
  apply Finset.ext
  intro i
  rw [Finset.mem_filter]
  constructor
  · intro h
    exact ⟨Finset.mem_univ _, by rw [decide_eq_true_eq]; exact (mem_nonLocalSet_iff i).mp h⟩
  · intro ⟨_, h⟩
    rw [decide_eq_true_eq] at h
    exact (mem_nonLocalSet_iff i).mpr h

/-- Singleton characterization of the "below n_local" set. -/
theorem filter_lt_n_local_eq_singleton :
    ((Finset.univ : Finset (Fin M_total)).filter (fun i => i.val < n_local)) =
      ({⟨0, M_total_pos⟩} : Finset (Fin M_total)) := by
  ext j
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton, Fin.ext_iff]
  unfold n_local
  omega

/-- The non-local set has exactly `n_dom + n_mid + n_far` elements. -/
theorem card_nonLocalSet :
    (syntheticDenseField.nonLocalSet y0 d_shell).card = n_dom + n_mid + n_far := by
  rw [nonLocalSet_eq]
  -- Rewrite the filter as univ \ (filter `i.val < n_local`).
  have h_rewrite : ((Finset.univ : Finset (Fin M_total)).filter
                    (fun i => decide (n_local ≤ i.val) = true)) =
                   (Finset.univ : Finset (Fin M_total)) \
                     ((Finset.univ : Finset (Fin M_total)).filter (fun i => i.val < n_local)) := by
    ext j
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_sdiff,
               decide_eq_true_eq]
    omega
  rw [h_rewrite]
  rw [Finset.card_sdiff (Finset.filter_subset _ _)]
  rw [Finset.card_univ, Fintype.card_fin]
  rw [filter_lt_n_local_eq_singleton, Finset.card_singleton]
  unfold M_total
  omega

/-- Concretely: 863 non-local indices. -/
theorem card_nonLocalSet_eq_863 :
    (syntheticDenseField.nonLocalSet y0 d_shell).card = 863 := by
  rw [card_nonLocalSet]
  unfold n_dom n_mid n_far
  norm_num

/-- The validity-gate cardinality: `card (nonLocal) ≥ 200`. -/
theorem card_nonLocalSet_ge_200 :
    200 ≤ (syntheticDenseField.nonLocalSet y0 d_shell).card := by
  rw [card_nonLocalSet_eq_863]; norm_num

/-! ## Distance positivity on the non-local set -/

theorem dist_pos_of_nonlocal (i : Fin M_total)
    (h : i ∈ syntheticDenseField.nonLocalSet y0 d_shell) :
    0 < ‖y0 - syntheticDenseField.centers i‖ := by
  rw [show syntheticDenseField.centers = syntheticCenter from rfl]
  rw [norm_y0_sub_center]
  have h_nl := (mem_nonLocalSet_iff i).mp h
  have h_dist := distFromOrigin_gt_d_shell i h_nl
  unfold d_shell at h_dist
  linarith

theorem dist_gt_d_shell_of_nonlocal (i : Fin M_total)
    (h : i ∈ syntheticDenseField.nonLocalSet y0 d_shell) :
    d_shell < ‖y0 - syntheticDenseField.centers i‖ := by
  rw [show syntheticDenseField.centers = syntheticCenter from rfl]
  rw [norm_y0_sub_center]
  exact distFromOrigin_gt_d_shell i ((mem_nonLocalSet_iff i).mp h)

/-! ## Reference bandwidth = pairwise bandwidth (Path A) -/

/-- Reference bandwidth: the pairwise (N_eff = 1) bandwidth. -/
noncomputable def σ_ref : ℝ := pairwiseBandwidth d_shell ε

theorem σ_ref_pos : 0 < σ_ref := by
  unfold σ_ref pairwiseBandwidth d_shell ε
  apply div_pos one_pos
  apply Real.sqrt_pos.mpr
  apply mul_pos two_pos
  apply Real.log_pos
  norm_num

/-! ## Non-emptiness of the non-local set -/

theorem nonLocalSet_nonempty :
    (syntheticDenseField.nonLocalSet y0 d_shell).Nonempty := by
  refine ⟨⟨1, ?_⟩, ?_⟩
  · unfold M_total n_local n_dom n_mid n_far; omega
  · rw [mem_nonLocalSet_iff]; unfold n_local; omega

/-! ## Headline #1 — the keystone, applied -/

/-- **`synthetic_tail_controlled`** — `aggregateTail_le_nEffFinset_bound`
    INSTANTIATED on `syntheticDenseField` at `σ_ref`. This is the keystone
    applied, NOT re-derived. -/
theorem synthetic_tail_controlled :
    syntheticDenseField.aggregateTail σ_ref y0
        (syntheticDenseField.nonLocalSet y0 d_shell) ≤
      syntheticDenseField.V_max *
        nEffFinset (syntheticDenseField.weights σ_ref y0)
                   (syntheticDenseField.nonLocalSet y0 d_shell) *
        gaussianKernel σ_ref (d_shell ^ 2) :=
  aggregateTail_le_nEffFinset_bound
    syntheticDenseField σ_ref y0 d_shell σ_ref_pos d_shell_pos

/-! ## Headline #2 — refscale locality, applied -/

/-- The non-local participation ratio at `σ_ref` is at least 1. This is the
    one fact we need to discharge `hN` and `hDense` on the synthetic field. -/
theorem nEffFinset_ge_one :
    1 ≤ nEffFinset (syntheticDenseField.weights σ_ref y0)
                   (syntheticDenseField.nonLocalSet y0 d_shell) :=
  one_le_nEffFinset _ _ nonLocalSet_nonempty
    (fun i _ => CorrectionField.weights_pos syntheticDenseField σ_ref y0 i)

/-- Algebraic identity: `d_shell² / (2·σ_ref²) = log(1/ε)` when `σ_ref` is the
    pairwise bandwidth. This is what makes the dense-regime condition collapse. -/
theorem dist_over_two_sigma_sq_eq_log :
    d_shell ^ 2 / (2 * σ_ref ^ 2) = Real.log (1 / ε) := by
  have hε_pos : (0 : ℝ) < ε := ε_pos
  have h_inv_ε_gt_one : 1 < 1 / ε := by
    rw [lt_div_iff hε_pos, one_mul]; exact ε_lt_one
  have hlog_pos : 0 < Real.log (1 / ε) := Real.log_pos h_inv_ε_gt_one
  have h2log_pos : 0 < 2 * Real.log (1 / ε) := by linarith
  have h_sigma_sq : σ_ref ^ 2 = d_shell ^ 2 / (2 * Real.log (1 / ε)) := by
    unfold σ_ref pairwiseBandwidth
    rw [div_pow, Real.sq_sqrt h2log_pos.le]
  rw [h_sigma_sq]
  field_simp

/-- `ε · exp(d_shell² / (2·σ_ref²)) = 1`. Follows from the algebraic identity
    via `Real.exp_log`. -/
theorem dense_rhs_eq_one :
    ε * Real.exp (d_shell ^ 2 / (2 * σ_ref ^ 2)) = 1 := by
  rw [dist_over_two_sigma_sq_eq_log]
  rw [Real.exp_log (by positivity)]
  unfold ε
  field_simp

/-- The synthetic version of `hN`: `ε < V_max · nEffFinset`.
    Holds because `nEffFinset ≥ 1 > ε`. -/
theorem ε_lt_V_max_nEffFinset :
    ε < V_max * nEffFinset (syntheticDenseField.weights σ_ref y0)
                            (syntheticDenseField.nonLocalSet y0 d_shell) := by
  unfold V_max
  rw [one_mul]
  exact lt_of_lt_of_le (by unfold ε; norm_num) nEffFinset_ge_one

/-- The synthetic version of `hDense`. With `σ_ref = σ_pair`, the RHS
    `ε · exp(d²/(2σ_ref²))` equals `1`, so the condition reduces to
    `nEffFinset ≥ 1`. -/
theorem hDense_synthetic :
    V_max * nEffFinset (syntheticDenseField.weights σ_ref y0)
                       (syntheticDenseField.nonLocalSet y0 d_shell) ≥
      ε * Real.exp (d_shell ^ 2 / (2 * σ_ref ^ 2)) := by
  rw [dense_rhs_eq_one]
  unfold V_max
  rw [one_mul]
  exact nEffFinset_ge_one

/-- **`synthetic_refscale_locality`** — `locality_at_operatingBandwidth_refscale`
    INSTANTIATED on `syntheticDenseField`. The dense-regime hypothesis `hDense`
    is discharged on the concrete numbers (via `dense_rhs_eq_one`, which uses
    `Real.exp_log` on a positive argument). -/
theorem synthetic_refscale_locality :
    syntheticDenseField.aggregateTail
        (operatingBandwidth d_shell
          (syntheticDenseField.V_max *
            nEffFinset (syntheticDenseField.weights σ_ref y0)
                       (syntheticDenseField.nonLocalSet y0 d_shell)) ε)
        y0
        (syntheticDenseField.nonLocalSet y0 d_shell) ≤ ε := by
  apply locality_at_operatingBandwidth_refscale
        syntheticDenseField d_shell ε σ_ref
        d_shell_pos ε_pos σ_ref_pos y0 dist_pos_of_nonlocal
  · rw [show syntheticDenseField.V_max = V_max from rfl]
    exact ε_lt_V_max_nEffFinset
  · rw [show syntheticDenseField.V_max = V_max from rfl]
    exact hDense_synthetic

/-! ## Diagnostic prints

After loading this file, run:

```
#print axioms OperatingResolution.SyntheticDense.synthetic_tail_controlled
#print axioms OperatingResolution.SyntheticDense.synthetic_refscale_locality
```

Both should report only the three standard Lean axioms
(`propext`, `Classical.choice`, `Quot.sound`), with no `sorry`.
-/

end SyntheticDense
end OperatingResolution
