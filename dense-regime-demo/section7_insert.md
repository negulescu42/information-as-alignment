# §7 paragraph insert — Synthetic dense-regime demonstration

The deployed FI field is in the sparse regime (non-local participation
ratio of roughly 2 across all observed query points), so it reproduces the
pairwise calibration but cannot exhibit the operating-bandwidth correction.
To make the correction visible we constructed a synthetic dense-regime
field: 864 Gaussian centers in ℝ^16, with one center at the origin
(query point), 3 dominant centers just outside the shell at distance
1.01·d_shell, 160 mid-distance centers at 1.30·d_shell, and 700 far centers
at 2.00·d_shell, with all amplitudes set to V_max = 1, d_shell = 1, and
ε = 0.05.

Measured at the pairwise reference bandwidth, the three candidate effective
counts are well-separated: full-field N_eff = 4.60, non-local N_eff =
102.72, and non-local cardinality |S| = 863, comfortably clearing all five
validity-gate conditions (notably non-local N_eff / full N_eff ≈ 22.3).
The construction is a controlled stress test of the mechanism, not a
model of the deployed instance.

The directly-evaluated aggregate tail (real sum over the real centers at
their real distances, not the bound checking itself) tells the story
quantitatively. At the pairwise bandwidth, Tail / ε = 23.16, far above
tolerance — the §2 anomaly in miniature. When the bandwidth is set from
each candidate count by the operating formula, the three regimes separate
cleanly: from full-field N_eff one obtains Tail / ε = 2.13 (UNSAFE — the
count is too small, the bandwidth too wide, and the tail overshoots ε
by more than 2×); from cardinality one obtains Tail / ε = 0.003 (WASTEFUL
— the count is too large, the bandwidth needlessly tight, and
generalization is sacrificed); from the non-local participation ratio
one obtains Tail / ε = 0.033, comfortably under tolerance (ON-TOLERANCE —
the bound that built σ* is conservative for heterogeneous Gaussian
weights, but it is honest: the directly-measured tail respects ε).
Only the non-local participation ratio is the count that lands the tail
where the theory promises; the other two err in opposite directions.

This closes the demonstrability gap (the correction is visible somewhere)
but not the deployment gap (the deployed FI field remains sparse; a real
deployed dense field is still future work, as noted in §8).
