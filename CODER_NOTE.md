# Note to the coder — start here

**Your handover is [`postulate2/HANDOVER.md`](postulate2/HANDOVER.md).** Read it first;
then `postulate2/README.md` for how to run. Everything is on branch
`claude/gate-1-recursive-scale-j80grv`.

## What you're getting

A self-contained theory→code package for the **Recursive Scale Structure (Postulate 2)**
upgrade to the IBF engine. It assumes you know only the original engine (Postulate 1).
It specifies exactly two guarded, default-off engine changes plus four thin layers, and
ships a `validate.py` that already passes 7/7 on small toy configs. That toy validation
is a **correctness harness, not the goal** — it proves the mechanisms are wired
correctly and directionally right.

## Your objective

Take this validated design and **upgrade the paper-grade engine**, then **redo the real
experiments and validate them** the way the paper does — i.e. on the actual benchmarks,
not the toy world:

- **Chess** (sequential-decision / continual setting)
- **CIFAR** (continual image classification, task/class-incremental)
- and the other benchmarks the paper reports.

Concretely, for each benchmark:
1. Port the two engine hooks (`frozen` reserve, `interface_group`) into the paper-grade
   engine — they must remain no-ops at their defaults so existing results are unchanged.
2. Provide the Scale-1 emergent config space and, where observations alias the latent
   (the realistic case), the **interactive** encoder — the toy `1B` generator is the
   stand-in for that regime; on real data the aliasing is intrinsic.
3. Run the **context-aware promotion** loop across the benchmark's task sequence and
   measure it against the un-promoted baseline.

## What to expect / what "success" looks like

The four toy checks tell you the four claims to reproduce at scale:

| toy check | claim to reproduce on chess / CIFAR / … |
|---|---|
| V1 | a configuration space **emerges** from finer-scale crystallization (not hand-designed features) |
| V2 | when the observation aliases the latent, **interaction/probing** recovers what static geometry can't |
| V3 | trained fields organize into basins whose interior is **externally shielded** → compressible to the boundary |
| V4 | **promotion preserves and typically improves continual-learning retention** while compressing the model |

V4 is the headline: on the toy it moved retention from 0.59 → 0.72. The paper-grade
target is the same signature — **backward transfer / retention no worse than baseline,
usually better, at a real compression ratio** — reported per benchmark with seed
averaging and the paper's standard metrics.

Keep the deliverable clean: port the mechanisms and the experiments, not the toy
scaffolding or any of the exploratory trial history.
