# Gate 2 Report -- Basin Detection & Interface Extraction

Trained Gate 1D interactive system, f=3.0, post-hoc (no new training). 5 seeds. tau_basin=0.01, tau_face=0.0001.


## VERDICT: **GATE 2 FAIL**

| criterion | result | threshold | pass |
|---|---|---|---|
| field fidelity (rel-err, basins >=5 interior) | 0.485 | < 0.05 | NO |
| behavioral fidelity (median |ACC_comp-ACC_full|) | 0.004 (worst seed 0.023) | < 0.02 | YES |
| compression ratio (median interior fraction) | 0.154 | >= 0.20 | NO |

## Basin detection (per seed)

| seed | crystallized | basins | sizes | singleton/pair | boundary | interior | compression |
|---|---|---|---|---|---|---|---|
| 0 | 55 | 2 | [29, 26] | 0 | 50 | 5 | 9.1% |
| 1 | 53 | 2 | [27, 26] | 0 | 44 | 9 | 17.0% |
| 2 | 57 | 2 | [29, 28] | 0 | 47 | 10 | 17.5% |
| 3 | 53 | 2 | [25, 28] | 0 | 48 | 5 | 9.4% |
| 4 | 52 | 2 | [27, 25] | 0 | 44 | 8 | 15.4% |

## Behavioral fidelity (per seed)

| seed | ACC_full | ACC_compressed | ACC_delta | ACC_oracle |
|---|---|---|---|---|
| 0 | 0.972 | 0.970 | -0.002 | 0.924 |
| 1 | 0.928 | 0.931 | +0.003 | 0.886 |
| 2 | 0.963 | 0.940 | -0.023 | 0.881 |
| 3 | 0.963 | 0.959 | -0.004 | 0.931 |
| 4 | 0.969 | 0.962 | -0.007 | 0.907 |
| **median** | 0.963 | 0.959 | -0.004 | 0.907 |

## Field fidelity (per basin, all seeds)

max relative error 0.807, median 0.413 over 10 basins; basins with >=5 interior: 3.


## Interpretation

**Gate 2 does not pass.** Failing criteria: compression ratio (15.4% interior < 20%); field fidelity (3 basins with >=5 interior; max rel-error 0.485).


Note: behavioral fidelity is essentially preserved -- median |ACC_compressed - ACC_full| = 0.004 (worst seed 0.023). Removing interior centers barely moves accuracy because action selection is argmax-robust to the field change. The binding failures are structural: too little interior (compression 15.4%) and that interior is not shielded (field error up to 0.81).


The trained correction field at this dimensionality (8-D z = 6-D interactive coords + 2-D action embedding) is **mostly boundary**: nearly every crystallized center has non-negligible kernel overlap with another basin, so only a small fraction sit in shielded interior. Behavioral accuracy is robust to removing that interior (ACC delta within 2pp -- action selection is argmax-robust), but (a) there is too little interior to constitute meaningful compression and (b) the interior that exists is not well shielded (field relative-error 0.09-0.81), so the Interface Principle's negligible-interior prediction does not hold at this scale.


This bounds the Interface Principle's operational applicability: it requires denser / higher-dimensional correction fields where genuine deep interior forms. In the present toy the field is too sparse for interface extraction to matter -- an honest scope finding, not a pipeline bug.
