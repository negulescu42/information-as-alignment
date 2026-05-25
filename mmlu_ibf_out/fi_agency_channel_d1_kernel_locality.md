# § 24b-D1 — Kernel-locality diagnostic

- engine: `arm_transitive['engine']  (§ 23 transitive arm, post AB+BC install)`
- value centers (ctx=0): **3880**
- agency centers crystallized (ctx=0): **1303**
- σ_op = 7.2621; σ_agency = 4.8361; activation_threshold = 0.01
- active radius (value):  22.039  (≈ 3.03·σ_op)
- active radius (agency): 14.677  (≈ 3.03·σ_agency)

## Verdict

- **VALUE:**  AC items sit ≥1σ further from value centers than AB/BC items (AC med 5.68σ vs AB 1.42σ, BC 2.68σ) — this is the kernel-locality story
- **AGENCY:** agency-active fraction is near zero across ALL groups (AB 0.0%, AC 0.0%) — crystallized agency centers do not overlap the closure dataset at all

## Per-group distances (in units of σ)

| group | n | val σ-min | val σ-mean | val σ-med | val %inside | ag σ-min | ag σ-mean | ag %inside |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test_AB | 24 | 0.00 | 2.19 | 1.42 | 56.2% | 4.27 | 4.78 | 0.0% |
| test_BC | 24 | 0.00 | 2.38 | 2.68 | 50.0% | 4.14 | 4.69 | 0.0% |
| test_AC_transitive | 24 | 3.91 | 5.56 | 5.68 | 0.0% | 4.19 | 4.84 | 0.0% |
| ordinary_AC_control | 24 | 3.25 | 5.47 | 5.56 | 0.0% | 4.24 | 4.66 | 0.0% |
| near | 180 | 4.15 | 5.50 | 5.53 | 0.0% | 3.73 | 4.33 | 0.0% |
| distant | 180 | 4.43 | 5.84 | 5.81 | 0.0% | 4.02 | 4.37 | 0.0% |
