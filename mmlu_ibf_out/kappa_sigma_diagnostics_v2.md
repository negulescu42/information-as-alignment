# Cell 8b — κ / σ field geometry diagnostics V2

Diagnostic only. σ was fixed by Cell 5b and used by Cell 8.

## Geometry

| d_eff | d_shell | sigma_base | kappa_base | sigma_operating | kappa_operating | N_active | N_budget | eps_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32.8442 | 42.1090 | 11.3290 | 1.9768 | 7.2621 | 1.2672 | 6382 | 10000 | 0.0005 |

## Candidate geometries

| name | source | sigma | kappa | sigma_over_base | K_shell | N_budget_times_K | N_active_times_K | implied_N_at_eps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_pairwise | Cell 5b pairwise | 11.3290 | 1.9768 | 1.0000 | 0.0010 | 10.0000 | 6.3820 | 0.5000 |
| operating | Cell 5b aggregate | 7.2621 | 1.2672 | 0.6410 | 5.00e-08 | 0.0005 | 0.0003 | 10000.0 |
| required_for_N_active | actual active centers | 7.3611 | 1.2844 | 0.6498 | 7.83e-08 | 0.0008 | 0.0005 | 6382.0 |
| eps_0.01 | N=10000, eps=0.01 | 8.0108 | 1.3978 | 0.7071 | 1.00e-06 | 0.0100 | 0.0064 | 500.0000 |
| eps_0.025 | N=10000, eps=0.025 | 8.2905 | 1.4466 | 0.7318 | 2.50e-06 | 0.0250 | 0.0160 | 200.0000 |
| eps_0.05 | N=10000, eps=0.05 | 8.5226 | 1.4871 | 0.7523 | 5.00e-06 | 0.0500 | 0.0319 | 100.0000 |
| eps_0.1 | N=10000, eps=0.1 | 8.7754 | 1.5312 | 0.7746 | 1.00e-05 | 0.1000 | 0.0638 | 50.0000 |
| eps_0.2 | N=10000, eps=0.2 | 9.0521 | 1.5795 | 0.7990 | 2.00e-05 | 0.2000 | 0.1276 | 25.0000 |
| N_3000 | N=3000, eps=0.0005 | 7.5370 | 1.3151 | 0.6653 | 1.67e-07 | 0.0017 | 0.0011 | 3000.0 |
| N_5000 | N=5000, eps=0.0005 | 7.4166 | 1.2941 | 0.6547 | 1.00e-07 | 0.0010 | 0.0006 | 5000.0 |
| N_10000 | N=10000, eps=0.0005 | 7.2621 | 1.2672 | 0.6410 | 5.00e-08 | 0.0005 | 0.0003 | 10000.0 |
| N_20000 | N=20000, eps=0.0005 | 7.1168 | 1.2418 | 0.6282 | 2.50e-08 | 0.0002 | 0.0002 | 20000.0 |
| N_50000 | N=50000, eps=0.0005 | 6.9376 | 1.2105 | 0.6124 | 1.00e-08 | 1.00e-04 | 6.38e-05 | 50000.0 |

## Radial shell audit

| relation | n_pairs | p01 | p05 | p10 | p50 | p90 | mean | std | d_shell_reference | fraction_le_d_shell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | 1200000 | 43.4554 | 48.8683 | 51.4559 | 60.5703 | 70.3524 | 60.7245 | 7.5125 | 42.1090 | 0.0069 |
| same_context | 517933 | 42.1942 | 48.4450 | 51.1975 | 60.5843 | 70.4709 | 60.6591 | 7.7325 | 42.1090 | 0.0098 |
| cross_context | 682067 | 44.3723 | 49.1838 | 51.6496 | 60.5603 | 70.2579 | 60.7741 | 7.3407 | 42.1090 | 0.0047 |

## Custom exact read σ diagnostic

Diagnostic only; not σ selection.

| gating | candidate | sigma | kappa | avg_log | avg_lin |
| --- | --- | --- | --- | --- | --- |
| same_context | base_pairwise | 11.3290 | 1.9768 | 0.6867 | 0.9067 |
| same_context | operating | 7.2621 | 1.2672 | 0.6878 | 0.9022 |
| same_context | required_for_N_active | 7.3611 | 1.2844 | 0.6878 | 0.9022 |
| same_context | eps_0.025 | 8.2905 | 1.4466 | 0.6878 | 0.9022 |
| same_context | eps_0.1 | 8.7754 | 1.5312 | 0.6867 | 0.9022 |
| same_context_or_verified | base_pairwise | 11.3290 | 1.9768 | 0.6867 | 0.9067 |
| same_context_or_verified | operating | 7.2621 | 1.2672 | 0.6878 | 0.9022 |
| same_context_or_verified | required_for_N_active | 7.3611 | 1.2844 | 0.6878 | 0.9022 |
| same_context_or_verified | eps_0.025 | 8.2905 | 1.4466 | 0.6878 | 0.9022 |
| same_context_or_verified | eps_0.1 | 8.7754 | 1.5312 | 0.6867 | 0.9022 |
| all | base_pairwise | 11.3290 | 1.9768 | 0.6511 | 0.8544 |
| all | operating | 7.2621 | 1.2672 | 0.6556 | 0.8556 |
| all | required_for_N_active | 7.3611 | 1.2844 | 0.6556 | 0.8556 |
| all | eps_0.025 | 8.2905 | 1.4466 | 0.6544 | 0.8556 |
| all | eps_0.1 | 8.7754 | 1.5312 | 0.6544 | 0.8556 |

## Percolation diagnostic

| sigma_name | sigma | threshold_multiplier | n_nodes | n_edges | n_components | largest_component | largest_component_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| operating | 7.2621 | 1.0000 | 6382 | 592 | 5790 | 2 | 0.0003 |
| operating | 7.2621 | 1.5000 | 6382 | 592 | 5790 | 2 | 0.0003 |
| operating | 7.2621 | 2.0000 | 6382 | 592 | 5790 | 2 | 0.0003 |
| operating | 7.2621 | 3.0000 | 6382 | 610 | 5773 | 3 | 0.0005 |
| base_pairwise | 11.3290 | 1.0000 | 6382 | 592 | 5790 | 2 | 0.0003 |
| base_pairwise | 11.3290 | 1.5000 | 6382 | 592 | 5790 | 2 | 0.0003 |
| base_pairwise | 11.3290 | 2.0000 | 6382 | 679 | 5712 | 4 | 0.0006 |
| base_pairwise | 11.3290 | 3.0000 | 6382 | 17401 | 424 | 5700 | 0.8931 |

## Flags

- `operating_budget_check`: `True`
- `base_violates_aggregate_budget`: `True`
- `current_sigma_safe_for_active_N`: `True`
- `fixed_sigma_conservative_for_active_N`: `True`
- `base_percolates_at_3sigma`: `True`
- `operating_avoids_giant_component_at_3sigma`: `True`
- `percolation_largest_fraction_ratio_base_over_operating`: `1900.0000000000002`
