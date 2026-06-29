# Branch: `gate-1d-interactive-encoder`

## Goal
Gate 1C proved discrepancy-driven splitting makes crystallized signatures encode
the aliased coordinate (the information is *inside* the system), but a static
observation→representation map cannot get it *out* at test time: supervised
x20→u2 ceiling is ~0.04 at u2_freq=3.0. Gate 1D tests whether an **interactive
encoder** — one that takes a few exploratory actions and observes rewards before
committing to a representation — can recover what the static encoder cannot.

## Method (`gate1_encoders.py`)
```
q2 = [ static geometric projection(x20) , interactive probe signature ]
```
The probe signature takes `n_probes` exploratory (context, action) interactions,
observes the binary reward of each, and records the local behavioral pattern.
The hidden `u` is used only to query the environment for rewards (to interact),
never as a coordinate. Candidate A (raw reward over both contexts) → 4-dim
signature → `q2` is 6-dim. Scale 1 (splitting), the Scale 2 engine, and the 1B
generator are unchanged; Scale 2 sigma is recalibrated for the new dimension.

## Run
```bash
python run_gate1d.py 5          # Runs 1 (f=1.5), 2 (f=3.0), 3 (probe sweep)
python summarize_gate1d.py      # -> gate1d_outputs/gate1d_report.md + figures/
```

## Result (5 seeds, f=3.0 — the hard case, static ceiling ~0.04)

| encoder | dim | rho_struct | rho_u2 | ACC | accuracy_gap |
|---|---|---|---|---|---|
| static | 2 | 0.585 | −0.011 | 0.742 | +0.150 |
| **interactive** | 6 | 0.680 | **0.773** | **0.963** | **−0.049** |
| oracle | 2 | 1.000 | 1.000 | 0.907 | 0.000 |

Probe-budget sweep (f=3.0): ρ_u2 rises 0.61 → 0.83 as n_probes goes 1 → 20;
ρ_struct plateaus ~0.72 (`figures/fig_probe_sweep.png`).

## Verdict

| criterion | interactive | threshold | verdict |
|---|---|---|---|
| leading: ρ_u2 | 0.773 | > 0.50 | **PASS** |
| accuracy_gap | −0.049 | < 0.15 | **PASS** |
| ρ_struct | 0.680 | > 0.80 | FAIL |

**Interactive encoding recovers the aliased coordinate that no static encoder
can** (ρ_u2 0.77 vs static ~0, supervised ceiling 0.04), and the resulting
representation is **operationally usable**: Scale 2 matches/beats the oracle
(accuracy_gap negative). The strict geometric ρ_struct stays ~0.68 because the
probe signature encodes *behavioral* structure (which action wins where) — highly
task-useful but not pairwise-isometric to the hidden manifold. The two metrics
genuinely disagree, and the operational one (accuracy_gap) passes.

**Transparency note.** The probe signature is strongly task-informative — it
directly observes which action is correct in each context — which is why
interactive ACC ≥ oracle. This is legitimate under the interactive paradigm (the
encoder is allowed a few reward probes at encode time, exactly as the Scale 1
signatures are formed from interaction), but it means the headline is "the
representation becomes usable through interaction," not "the geometry is
recovered isometrically." Candidate B (10-D action-reward hash) was not run:
Candidate A already cleared the leading indicator, and with deterministic binary
reward the hash carries no information beyond Candidate A's sign-bits, so it
cannot raise ρ_struct.

## Status against the plan
```
Leading indicator (rho_u2 > 0.5):                 PASS
Operational pass (rho_u2 > 0.5 AND gap < 0.15):   PASS at f=3.0
Strict pass (rho_struct > 0.8 AND gap < 0.15):    NOT met (rho_struct ~0.68)
Run 5 (10-seed, gated on rho_struct > 0.8):       NOT triggered
```

## Theoretical significance
Interaction recovers what a static snapshot cannot. This supports the reading
that **Postulate 2 requires interactive representation formation**: the
higher-scale configuration space is discovered through a trajectory of
lower-scale interactions, not read off a snapshot. The probe sweep quantifies the
inference-time cost: ρ_u2 is already 0.76 at 3 probes and 0.83 at 20 — a few
interactions suffice.

No interface nodes, promotion, recursion, RRW/chess.
