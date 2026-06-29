# Gate 1D -- Run 5 (official 10-seed pass) + agency control

Generator 1B, f=3.0 (static x20->u2 ceiling ~0.04) | n_probes=5 | seeds=10 | **pass criterion: accuracy_gap < 0.15 (median AND all seeds)**; rho_struct reported, not gating.


## Per-seed (interactive, random probes -- the official encoder)

| seed | rho_struct | rho_u1 | rho_u2 | ACC | ACC_oracle | accuracy_gap | gap < 0.15 |
|---|---|---|---|---|---|---|---|
| 0 | 0.704 | 0.994 | 0.779 | 0.972 | 0.924 | -0.049 | YES |
| 1 | 0.679 | 0.995 | 0.677 | 0.928 | 0.886 | -0.042 | YES |
| 2 | 0.680 | 0.993 | 0.794 | 0.963 | 0.881 | -0.082 | YES |
| 3 | 0.668 | 0.994 | 0.766 | 0.963 | 0.931 | -0.032 | YES |
| 4 | 0.692 | 0.994 | 0.773 | 0.969 | 0.907 | -0.063 | YES |
| 5 | 0.691 | 0.992 | 0.728 | 0.961 | 0.881 | -0.079 | YES |
| 6 | 0.711 | 0.994 | 0.732 | 0.929 | 0.916 | -0.014 | YES |
| 7 | 0.701 | 0.994 | 0.733 | 0.971 | 0.867 | -0.104 | YES |
| 8 | 0.698 | 0.995 | 0.696 | 0.978 | 0.894 | -0.083 | YES |
| 9 | 0.676 | 0.992 | 0.778 | 0.961 | 0.908 | -0.053 | YES |
| **median** | 0.691 | 0.994 | 0.750 | 0.963 | 0.901 | -0.058 | YES |

## VERDICT: **GATE 1D PASS**

accuracy_gap < 0.15 on **all 10 seeds**: YES; on the **median** (-0.058): YES.


Postulate 2 is validated in this computational instantiation, with the specific finding that it requires **interactive** (not static) representation formation. The emergent configuration space supports Scale 2 correction dynamics that match or exceed oracle-provided coordinates (median ACC 0.963 vs oracle 0.901).


## Informed vs random probes (cross-scale agency)

Informed probes choose actions by a Boltzmann policy over the scout Scale 2 agent's corrections (k_eff*R_eff), instead of uniform random.

| metric | random | informed | informed better? |
|---|---|---|---|
| median ACC | 0.963 | 0.960 | no |
| median accuracy_gap | -0.058 | -0.058 | YES |
| median rho_u2 | 0.750 | 0.754 | YES |
| median rho_struct | 0.691 | 0.682 | no |

Informed ACC > random ACC on 6 / 10 seeds.

Informed probing did not beat random in aggregate here; for k=2 actions with deterministic reward, uniform probing already covers the behavioral slots well.


## Probe budget (from Run 3)

rho_u2 reaches 0.76 by 3 probes and 0.83 by 20 (see `figures/fig_probe_sweep.png`): the inference-time cost of interactive representation formation is ~3 interactions.
