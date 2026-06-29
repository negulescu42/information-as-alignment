# Gate 1 — Recursive Scale Structure in the Two-Scale Toy Model

Gate 1 tests one claim of **Postulate 2 (Recursive Scale Structure)**:

> lower-scale crystallization → emergent higher-scale configuration space

Concretely: can **Scale 1** IBF dynamics, operating in a 20D *observed* space,
produce an emergent 2D representation that recovers the true hidden manifold
well enough for the existing **Scale 2** v1 correction engine to learn on top of
it?

Gate 1 implements **only** this. No interfaces, no promotion, no recursion, no
RRW / chess / CIFAR / LLM. (See `04 -- interface compression status` in the
task spec.)

## Pipeline

```
hidden 2D u  --nonlinear embed-->  observed 20D x20
   x20  --Scale 1 IBF representation particles + crystallization-->  emergent q_hat(x) in R^2
   q_hat  --existing v1 IBFAgent (Scale 2)-->  A/B correction dynamics
```

The Scale 1 learner never sees the hidden coordinates `u`. They are used only
for (1) generating the environment, (2) the oracle baseline, (3) evaluation /
plot diagnostics.

## Files

| file | role |
|---|---|
| `ibf_v1_engine.py` | v1 engine, extracted verbatim from `(IBF)Toy-Model.ipynb` (Config, MemoryCenter, **IBFAgent**, ToyEnvironment, ToyEncoder, ToyBaseEvaluator, PassiveAgent, calibration, evaluate, run_toy_experiment). This is the **unchanged Scale 2 correction engine**. |
| `gate1_environment.py` | `Gate1Config`, `TwoScaleToyEnvironment` (hidden 2D manifold → fixed nonlinear 20D embedding; truth from `u`). |
| `gate1_encoders.py` | `Oracle2DEncoder`, `Raw20DEncoder`, `PCA2DEncoder`, `Random2DEncoder`, `EmergentScaleEncoder`. |
| `scale1_representation.py` | `Scale1RepresentationParticle`, `Scale1RepresentationLearner` (fit / crystallize / build_particle_graph / extract_embedding_2d / make_encoder). |
| `run_gate1.py` | Phase 4 runner: env → Scale 1 → rho_struct → Scale 2 on every condition → per-seed metrics + figures. |
| `gate1_plots.py` | per-seed diagnostic figures (spec 13.1–13.4). |
| `summarize_gate1.py` | Phase 5 aggregation → CSVs, report, accuracy-gap bar chart (13.5), PASS/FAIL/INCONCLUSIVE. |
| `run_regression.py` | Phase 0 regression: confirms the refactored engine still reproduces the original 2D toy model. |

## Exact commands

```bash
# 0. dependencies
pip install numpy scipy scikit-learn matplotlib pandas

# 1. Phase-0 regression (original 2D toy model still works)
python run_regression.py

# 2. quick smoke test (3 seeds, small pools — ~1 min)
python run_gate1.py --config smoke
python summarize_gate1.py

# 3. development run (seeds 0..9, full pools)
python run_gate1.py --config dev
python summarize_gate1.py

# 4. final report run (seeds 0..19)
python run_gate1.py --config final
python summarize_gate1.py

# custom sizes
python run_gate1.py --seeds 0 1 2 --n-repr 1500 --n-train 800 --n-test 800 --e1 30 --e2 25
```

Outputs land in `gate1_outputs/`:

```
gate1_outputs/gate1_results.json     # raw per-seed records (run_gate1.py)
gate1_outputs/gate1_per_seed.csv     # per-seed table (summarize_gate1.py)
gate1_outputs/gate1_summary.csv      # aggregate + conclusion
gate1_outputs/gate1_report.md        # human-readable report
gate1_outputs/figures/*.png          # figures 13.1–13.5
```

## Pass / fail condition

Gate 1 passes **only if both** hold (per seed, and on the aggregate median):

```
rho_struct > 0.8                          (manifold recovery)
ACC_emergent >= ACC_oracle - 0.15         (accuracy gap <= 0.15)
```

- `rho_struct` = Spearman correlation of pairwise distances between emergent
  `q_hat` and true `u` on held-out test points (rotation/scale/reflection
  invariant, hence pairwise-distance based).
- `ACC_gate = 0.5 * (Acc_A_after_B + Acc_B_after_B)` for each encoder; the gap is
  `ACC_oracle - ACC_emergent`.

## Conditions and controls

| condition | encoder coords | purpose |
|---|---|---|
| oracle | true `u` (2D) | target baseline |
| **emergent** | crystallized-particle diffusion embedding `q_hat(x20)` | the Gate 1 candidate |
| raw 20D | `x20` | is the task trivial in raw space? |
| PCA 2D | PCA(2) of `x20` | does plain geometry already solve it? |
| random 2D | random projection of `x20` | is the metric too weak? |
| no-crystallization ablation | embedding from transient (non-crystallized) particles | does crystallization matter? |
| shuffled-signature control | behavioral signatures shuffled before graph embedding | does the learned behavioral structure contribute? |

## Conclusion

The machine-generated conclusion (PASS / FAIL / INCONCLUSIVE) and the exact
numbers are written to `gate1_outputs/gate1_report.md` and
`gate1_outputs/gate1_summary.csv` after a run. See that report for the
authoritative result.

### Gate 1A vs causal validation

This generator is **geometrically easy**: the no-crystallization ablation and
the shuffled-signature control also recover the manifold above ρ = 0.8, i.e. a
geometry-only baseline passes too. So a pass here is a **Gate 1A pass** (the
pipeline runs end-to-end and clears the bar) — it is **not** causal validation
that lower-scale *crystallization* induces the configuration space. The
summarizer labels it accordingly.

To actually stress the causal claim, use the **Gate 1B** stress generator, where
geometry-only recovery is insufficient (u2 is encoded only through
high-frequency aliased terms). See `GATE1B_README.md` and:

```bash
python gate1b_geometry_check.py 5          # show geometry-only fails on 1B
python run_gate1.py --generator 1B --config dev
python summarize_gate1.py gate1b_outputs
```
