# postulate2 — Recursive Scale Structure reference package

Self-contained, validated implementation of the Postulate-2 upgrade to the IBF engine.
Start with **[HANDOVER.md](HANDOVER.md)** for the theory→code story; this README is the
operational reference.

## Layout

```
postulate2/
├── HANDOVER.md      theory → code, self-contained (read this first)
├── README.md        this file
├── engine.py        Postulate-1 correction engine + two guarded hooks (frozen, interface_group)
├── environment.py   two-scale toy world: hidden 2D manifold u → 20D observation x
├── scale1.py        Scale-1 learner: crystallize particles → emergent 2D config space
├── encoders.py      encoders over the config space, incl. the interactive encoder (1B)
├── scale2.py        run the (unchanged) engine on any representation; metrics
├── promotion.py     basin detection + context-aware interface promotion
└── validate.py      end-to-end functional validation (V1–V4)
```

## Run

```bash
cd postulate2
python validate.py
```

Expected: `7/7 checks passed`. Runtime is a few minutes (small configs).

Modules import each other by bare name — run from inside `postulate2/`, or
`sys.path.insert(0, ".../postulate2")` first (see the top of `validate.py`).

## The one global you must set

The number of actions is `engine.C.k`. Set it **before** building environments/agents:

```python
import engine
engine.C.k = 2   # geometry checks (V1, V2)
# or
engine.C.k = 8   # dense-field / continual-learning checks (V3, V4)
```

## Minimal usage

```python
import engine; engine.C.k = 2
from environment import TwoScaleToyEnvironment, Gate1Config
import scale1, scale2

cfg = Gate1Config(generator="1A", k=2, N_repr_pool=800, N_test=500,
                  E_scale1=20, sigma_x_scale=0.7)
env = TwoScaleToyEnvironment(0, cfg)

# Scale 1: crystallize an emergent 2D configuration space
learner = scale1.Scale1RepresentationLearner(cfg, 0); learner.fit(env)
enc = learner.make_encoder(np.eye(2), learner.get_crystallized_particles())

# quality of the emergent space
q = enc.encode_observation_batch(env.test_A_x20)
print(scale2.manifold_rho(q, env.test_A_u2))     # ~0.83 on 1A  → Postulate 2 core claim
```

For the hard generator use `generator="1B", enable_splitting=True` and build coordinates
with `encoders.build_interactive_coords(...)` instead of the static encoder. See
`validate.py` V2–V4 for complete, runnable examples of every path.

## What passes, and at what magnitude

| check | claim | small-config result | threshold |
|---|---|---|---|
| V1 | emergent config space recovers the manifold (1A) | rho ≈ 0.83 | > 0.8 |
| V2 | interaction beats geometry on aliased u2 (1B) | static ≈ 0.00, interactive ≈ 0.76 | < 0.3 / > 0.5 |
| V3 | basin interior is externally shielded | leak ≈ 0.009, interior ≈ 0.25 | < 0.05 / > 0.15 |
| V4 | promotion preserves/improves retention | full ≈ 0.59, promoted ≈ 0.72, comp ≈ 0.48 | ≥ F−0.03 / > 0.10 |

Full-scale runs reproduce larger magnitudes via the same code path — scale up
`N_repr_pool`, `N_train_pool`, `E_scale1`, `E_scale2` and average over seeds.

## Engine compatibility

`engine.py` is the original Postulate-1 engine plus two additions
(`MemoryCenter.frozen`, `MemoryCenter.interface_group`), both defaulting to off. With
the defaults, engine behaviour is identical to Postulate 1 — the hooks are no-ops until
`promotion.promote` opts in.
