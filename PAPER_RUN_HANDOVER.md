# Paper Run Handover

**For**: a fresh Claude Code session
**From**: previous session that completed the full smoke run and the § 38 paper-deliverable generator
**Purpose**: launch the paper-grade run on the Runpod pod, monitor it, and produce paper-ready outputs

This document is self-contained. Read it before doing anything.

---

## 1. Project at a glance

**Paper**: *Durable Alignment via Orthogonal Memory in Large Language Models*
**Author**: Radu Negulescu, Informational Buildup Foundation (April 2026)
**Companion notebook**: `(IBF)Companion-LLM-Durable-Alignment.ipynb`

The notebook validates an orthogonal correction field over a frozen LLM as a durable alignment substrate. Eight headline claims (C1–C7 + scope C8 absorbed into L2), five limitations (L1–L5). Architecture: substrate decoupling + lifecycle on one substrate + locality/scale + compiled-surface for semantic structure + cross-model.

**Repo**: `github.com/negulescu42/information-as-alignment`
**Branch**: `claude/review-jupyter-notebook-8AU5y`
**Latest commit before handover**: `14699aa` (§ 38 paper-deliverable generator v3)

---

## 2. State at handover

Smoke run is **complete end-to-end**. All 7 claims are at 🟢 paper-grade in smoke except **C5 (LoRA durability)** which is 🟡 because smoke used `LORA_STEPS=2`; paper needs `LORA_STEPS=24` for the headline 37.5% base-shift number.

Artifacts on the pod under `mmlu_ibf_out/`:
- Canonical engine pickle (~311 MB)
- All per-cell JSON / Markdown reports
- `mmlu_ibf_out/paper/paper_tables.md` (11 tables, real numbers)
- `mmlu_ibf_out/paper/abstract_numbers.json`
- `mmlu_ibf_out/paper/claims_status_final.md`

Smoke headline numbers (from § 38 output):

| Claim | Smoke value |
|---|---|
| C1 canonical avg lin | **0.973** (base 0.216, gain +0.756) |
| C2 selective deletion | target 1.0 → 0.0, others drift +0.000 |
| C2 forgetting A→D | −0.029 (dominant boundary B→C: −0.019) |
| C3 strong-prior override | 1.000 (base rate 1.0 → 0.0) |
| C3 scale 1k | target 0.995, control 1.000 |
| C4 ontology closure 23B 2-hop | **0.000** (does not emerge) |
| C4 ontology closure 23C 2-hop | **0.875** (installs when trained) |
| C4 compiled closure initial A→C | **1.000** |
| C4 compiled closure revised A→D | **1.000** (old A→C retired) |
| C5 LoRA (2-step smoke) | base shift 4.6%, weak drift +0.017 ← needs 24-step paper |
| C6 IBF native vs kNN/RAG manual | IBF native=1.000 / kNN+RAG manual=1.000 |
| C6 RAG CounterFact revision | 0.500 (fails even with oracle refresh) |
| C7 Qwen vs Mistral | 5/6 metrics within ±0.01, Phase B gap −0.265 |

---

## 3. Repo structure and key files

```
information-as-alignment/                   # repo root
├── (IBF)Companion-LLM-Durable-Alignment.ipynb   # the notebook (102 cells, 40 sections, 8 parts)
├── runpod_setup.sh                              # one-shot pod provisioning
├── PAPER_RUN_HANDOVER.md                        # this file
├── information-as-alignment-v1.pdf              # IBF theory paper (companion, already published)
├── (pre-print)information-as-alignment-v1.pdf   # pre-print version
└── (other domain notebooks: chess, CIFAR, RRW, Toy-Model)
```

On the pod (Runpod with RTX 5090 / similar, network volume at `/workspace`):

```
/workspace/information-as-alignment/         # pod's clone of the repo
└── mmlu_ibf_out/                            # untracked: all artifacts go here
    ├── canonical_engine.pkl                 # ~311 MB
    ├── canonical_training_results.json
    ├── selective_deletion.json
    ├── fi_strong_prior_pilot.json
    ├── fi_ontology_closure_23bc.json
    ├── fi_compiled_ontology_closure_cell24b.json
    ├── forgetting_diagnostic_report.json
    ├── actual_lora_e2e_durability_control_fixed_report.json
    ├── cross_model_generality_qwen2_1_5b.json
    ├── standard_benchmarks/
    │   └── results/
    │       ├── benchmark_ibf_durable_lifecycle.json
    │       ├── benchmark_knn_durable_lifecycle.json
    │       ├── benchmark_rag_durable_lifecycle.json
    │       ├── benchmark_comparison_report.json
    │       ├── failure_mode_analysis.json
    │       ├── counterfact_paraphrase_geometry_audit.json
    │       ├── counterfact_sigma_sweep_diagnostic.json
    │       └── counterfact_paraphrase_anchor_diagnostic.json
    ├── ablations/
    │   └── ablation_results_final.json
    ├── counterfact.json                     # downloaded from rome.baulab.info (50 MB)
    ├── zsre_mend_eval.json                  # downloaded from rome.baulab.info (30 MB)
    └── paper/                               # § 38 outputs
        ├── paper_tables.md
        ├── abstract_numbers.json
        └── claims_status_final.md
```

---

## 4. Notebook structure

102 cells, 40 sections, 8 thematic Parts. Single toggle in **cell 1** switches smoke ↔ paper.

```
Part 0   Reading guide / glossary / claim-to-cell map  (cells 0–4)
Part I   Setup, data, representation                   §1–§7
Part II  IBF engine + canonical lifecycle              §8–§13
Part III Strong priors + local alignment               §14–§17
Part IV  Locality, scale, compiled closure             §18–§24
Part V   Durability: state + base-model evolution      §25–§26
Part VI  Standard benchmarks + baselines               §27–§35
Part VII Cross-model generality                        §36–§37
Part VIII Final audit + reproducibility                §38–§40
```

**§ 38** is the paper-deliverable generator (v3) — reads all artifacts, emits paper tables / abstract numbers / claims status.

---

## 5. Pre-paper-run checklist

Before launching the paper run, **verify all of these on the pod**:

### 5.1 Datasets are real, not synthetic

```bash
cd /workspace/information-as-alignment
ls -lh mmlu_ibf_out/counterfact.json mmlu_ibf_out/zsre_mend_eval.json
```

Both files should exist (~50 MB and ~30 MB respectively). They were downloaded from `rome.baulab.info` during smoke. If missing, re-download:

```bash
mkdir -p mmlu_ibf_out
wget -O mmlu_ibf_out/counterfact.json https://rome.baulab.info/data/dsets/counterfact.json
wget -O mmlu_ibf_out/zsre_mend_eval.json https://rome.baulab.info/data/dsets/zsre_mend_eval.json
```

Also verify `datasets` package + HF fetch capability:

```bash
python3 -c "from datasets import load_dataset; ds = load_dataset('NeelNanda/counterfact-tracing'); print(f'HF works: {len(ds[\"train\"])} rows')"
```

Should print `HF works: 21919 rows`. If not, HF auth or network issue — investigate before paper run launch.

### 5.2 GPU is clean

```bash
nvidia-smi
```

Memory should be near-empty (<1 GB used). Kill any stale processes (`kill -9 <PID>`) if needed. Paper run holds Mistral-7B + Qwen2-1.5B + IBF engine in memory simultaneously for parts of the run.

### 5.3 Repo is up to date

```bash
cd /workspace/information-as-alignment
git fetch origin
git log --oneline -3
```

Top should be `14699aa § 38 v3 paper-deliverable generator`. If not, `git pull --ff-only`.

### 5.4 No stale Jupyter kernels

In JupyterLab: **Kernel → Shut Down All Kernels** (sidebar → "Running" panel). Fresh state for the paper run.

---

## 6. Paper run launch sequence

### 6.1 Open the notebook fresh

In JupyterLab: close any open notebook tab, then **File → Open** `(IBF)Companion-LLM-Durable-Alignment.ipynb`.

### 6.2 Set paper-mode flags in cell 1

The run-mode toggle cell (cell 1, after the title) has five commented-out flags. **Uncomment all five**:

```python
CANONICAL_TRAINING_MODE     = "paper"
SCALE_CAPACITY_MODE         = "paper"
IBF_DURABLE_BENCHMARK_MODE  = "paper"
KNN_DURABLE_BENCHMARK_MODE  = "paper"
RAG_DURABLE_BENCHMARK_MODE  = "paper"
```

### 6.3 Add a scratch cell for paper-specific overrides

Insert a new code cell **immediately after cell 1** with these overrides:

```python
# Paper-grade overrides (smoke caps were too aggressive for benchmark cells)
IBF_DURABLE_INSTALL_EPOCHS = 10        # benchmark install epochs (smoke: 6)
IBF_PARAPHRASE_AUDIT_INSTALL_EPOCHS = 10
Q29_INSTALL_EPOCHS = 10                # § 31b CounterFact σ sweep
R29_INSTALL_EPOCHS = 10                # § 31c paraphrase anchor

# Critical for C5 — substrate decoupling story needs meaningful base perturbation
CONTROL_FIXED_LORA_STEPS = 24          # was 2 in smoke; 24 gave 37.5% base shift originally

# Optional: enable RUN_FULL_* flags for ontology cells if you want max-epoch paper-grade
# RUN_FULL_ONTOLOGY_CLOSURE = True     # § 22
# RUN_FULL_GRAPH_CLOSURE = True        # § 23
# RUN_FULL_COMPILED_CLOSURE = True     # § 24
# RUN_FULL_LOCAL_ALIGNMENT = True      # § 17
# RUN_FULL_LOCALITY_BLEED = True       # § 19 (already True by default; check)

# Enable real HF dataset fetch (already True by default after b79b64a)
ALLOW_HF_DATASET_FETCH = True

# Enable synthetic-benchmark inclusion if real datasets fail (defensive)
INCLUDE_SYNTHETIC_BENCHMARKS_IN_LIFECYCLE = True

print("Paper-mode config set:")
for var in ["CANONICAL_TRAINING_MODE", "SCALE_CAPACITY_MODE",
            "IBF_DURABLE_BENCHMARK_MODE", "KNN_DURABLE_BENCHMARK_MODE",
            "RAG_DURABLE_BENCHMARK_MODE", "IBF_DURABLE_INSTALL_EPOCHS",
            "CONTROL_FIXED_LORA_STEPS"]:
    print(f"  {var} = {globals().get(var)}")
```

Run that cell. Verify all five `*_MODE` show `'paper'` and `CONTROL_FIXED_LORA_STEPS` shows `24`.

### 6.4 Launch the run

**Kernel → Restart Kernel and Run All Cells**.

Expected wall clock: **~49 hours** on a single A100 / H100, plus **~19.9 hours** for Qwen replication in § 36 (paper § 11). Total ~70 hours wall clock.

### 6.5 Per-cell expected timing (paper mode)

| Cells | Description | Expected wall clock |
|---|---|---|
| § 1–§ 7 | Setup, representation, σ calibration | ~10 min |
| § 8 canonical training | A 300 / B 200 / C 200 / D 200 epochs | ~6–8 hours |
| § 9–§ 13 | Audits, retract, delete | ~30 min |
| § 14–§ 17 | Strong-prior, ontology shift, bridge, local alignment | ~3–4 hours |
| § 18–§ 19 | 1k rules, locality / bleed | ~1 hour |
| § 20 scale sweep | 1k / 3k / 5k / 10k / 20k / 50k with dynamic capacity | ~8–12 hours |
| § 21 long-horizon | 20 rounds × 5 epochs (~20 hours per the cell) | **~20 hours** |
| § 22 / § 23 / § 24 ontology trio | Compiled closure | ~30 min |
| § 25 forgetting | Analysis only | instant |
| § 26 LoRA durability | 24-step LoRA + re-extract R_base | ~30–60 min |
| § 27 external editor manifest | Capability probe | instant |
| § 28 dataset builder | Loads real CF + ZsRE | ~2 min |
| § 29 harness | Builds lifecycle tasks | ~1 min |
| § 30 IBF benchmark | 50 scenarios × 2 benchmarks × 10 epochs | ~30 min |
| § 31 / § 31b / § 31c | Paraphrase audit + σ sweep + anchor | ~15 min |
| § 32 kNN baseline | 50 scenarios | ~10 min |
| § 33 RAG baseline | 50 scenarios | ~30 min |
| § 34 comparison + § 35 failure-mode | Analysis | instant |
| § 36 Qwen replication | Fresh field over Qwen2-1.5B (full lifecycle) | **~19.9 hours** |
| § 37 mechanism continuation | Ablations | ~30 min |
| § 38 paper-deliverable generator | Reads artifacts, emits 3 files | <1 sec |
| § 39 / § 40 | Markdown | instant |

---

## 7. Monitoring the run

You can subscribe to PR activity or just check in periodically. The pod's JupyterLab will show progress per cell. Common things that can go wrong:

### 7.1 OOM during model loading (§ 4 base-model extraction)

Symptom: `CUDA out of memory` while loading Mistral-7B.

Response: another process is holding GPU memory. On pod terminal:
```bash
nvidia-smi
kill -9 <stale_pid>
```
Then in JupyterLab: **Kernel → Restart Kernel and Run All Cells**.

### 7.2 § 28 dataset loading falls back to synthetic

Symptom: § 28 prints `synthetic_fallback_only` for any benchmark.

Response: check `ALLOW_HF_DATASET_FETCH` is True. The `datasets` package must be installed. Real `counterfact.json` and `zsre_mend_eval.json` files must be in `mmlu_ibf_out/`. If `HF fetch skipped/failed: ALLOW_HF_DATASET_FETCH=False`, the variable was overridden somewhere — set it in a scratch cell and re-run § 28.

### 7.3 § 30 / § 31 / § 32 / § 33 produce 0.000 metrics

Symptom: lifecycle benchmark cells run in 0.0 seconds and produce zero everywhere.

Root cause: `durable_lifecycle_tasks.json` is empty (`benchmarks: {}`). Means § 29 harness filtered out all benchmarks because the records were marked `synthetic_fallback_only` AND `INCLUDE_SYNTHETIC_BENCHMARKS_IN_LIFECYCLE = False`.

Response: in a scratch cell, `INCLUDE_SYNTHETIC_BENCHMARKS_IN_LIFECYCLE = True`. Re-run § 29 then § 30 onwards.

### 7.4 § 31 / § 31b / § 31c CounterFact missing

Symptom: cell prints `Skipping CounterFact ... — CounterFact not present in durable_lifecycle_tasks.json`.

Response: a graceful skip; the cell wrote a stub artifact and the rest of the pipeline continues. To fix properly: § 28 needs to produce a real CounterFact record set; verify via `cat mmlu_ibf_out/standard_benchmarks/standard_benchmark_records.md` (should show `standard_records_ready`, not `synthetic_fallback_only`).

### 7.5 § 37 IndexError on `rb_revision_train`

Should NOT happen — the fix is in (commit `b3efc79` / `497a3ff`). If it does, verify the cell has this block:

```python
if "revision_train" in globals() and "make_revision_prior" in globals():
    _new_rb_revision_train = make_revision_prior(revision_train)
    if _new_rb_revision_train.shape[0] != rb_revision_train.shape[0]:
        print(f"  [§ 37 fix] Regenerated rb_revision_train: ...")
        rb_revision_train = _new_rb_revision_train
        ...
```

If missing, re-pull from origin.

### 7.6 Kernel dies mid-run (most common pain point)

JupyterLab's "Revert Notebook to Saved" KILLS the kernel. So does "Restart Kernel."

If kernel dies, you lose all in-memory state. Recovery: **Kernel → Restart Kernel and Run All Cells** from the top. The pickled artifacts on disk (canonical_engine.pkl etc.) survive, so cells that load-or-skip will be fast.

To preserve cell outputs across git pulls: **commit before pulling**:
```bash
git add "(IBF)Companion-LLM-Durable-Alignment.ipynb"
git commit -m "Snapshot mid-run outputs"
git pull --rebase origin claude/review-jupyter-notebook-8AU5y
```

---

## 8. After the paper run completes

### 8.1 Verify all cells produced clean output

In JupyterLab, scroll the whole notebook. Every cell should show its expected output (look for ✓ complete markers). Any `Error in input` or `Traceback` means that cell failed and needs investigation.

### 8.2 Run § 38 paper-deliverable generator (if not auto-run)

If "Run All Cells" did its job, § 38 already ran at the end. Otherwise click it manually. Check the console output:

```
CLAIMS STATUS (with headline numbers):
  C1 🟢  Canonical avg lin = ...
  C2 🟢  Delete: target ... → ..., ...
  C3 🟢  Strong-prior ...
  C4 🟢  ...
  C5 🟢  LoRA (24 steps): base shift ...% ...  ← should now be green, not yellow
  C6 🟢  ...
  C7 🟢  ...
```

If C5 is still 🟡, check `actual_lora_e2e_durability_control_fixed_report.json` → `config.lora_steps` should be 24, not 2.

### 8.3 Commit results to origin

```bash
cd /workspace/information-as-alignment
git add "(IBF)Companion-LLM-Durable-Alignment.ipynb"
git add mmlu_ibf_out/                                       # ALL artifacts, including paper/
git commit -m "Paper-grade run results — RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)"
git push origin claude/review-jupyter-notebook-8AU5y
```

This bakes the displayed outputs into the notebook AND pushes all JSON / pickle artifacts. The 311 MB canonical engine pickle is the largest single file; git LFS may be needed if not already configured.

### 8.4 Hand off the three paper deliverables

The paper writer needs these three files (now in the repo at `mmlu_ibf_out/paper/`):
- `paper_tables.md` — 11 paper-section tables with real numbers
- `abstract_numbers.json` — abstract figures with provenance
- `claims_status_final.md` — claim verdicts + paper-quotable interpretations + narrative findings

---

## 9. Paper structure (locked, per supervisor)

The writer's structure (with our amendments):

```
1. Abstract
2. Scope, Contribution, and Claims
   2.1 Alignment as persistence, not specification
   2.2 Central claim
   2.3 Supporting claims (use 3-tier framing: substrate / geometry / generality)
   2.4 Non-claims and boundaries
3. Why This Matters Now
4. Future Industries: A World That Needs Local Alignment
5. Architecture: A Correction Field Over a Frozen Model
   5.1 Forward-reference to IBF theory paper (one sentence)
   5.2 Frozen base distribution
   5.3 Orthogonal correction field
   5.4 Proposition space and local centers
   5.5 Discrepancy-driven update dynamics
   5.6 Expanded link to the IBF theory paper
6. Experimental Protocol
7. A Truth-Maintenance Lifecycle in One Correction Field   ← C1 + C2
   (open with: "Install, revise, retract, delete, retain — five homogeneous operations
    on the same correction field." This is C2's architectural punchline.)
8. Where Corrections Live   ← C3 (geometric — strong-prior + locality + scale)
   8.1 Strong-prior override
   8.2 Locality and bleed
   8.3 Scale and post-σ operating geometry
9. When Facts Have Consequences   ← C4 (compiled-surface architecture)
   (lead with the constructive answer; reference §23 diagnostic; generalize to compiled
    paraphrase anchors §31c as the same architectural pattern.)
10. When the Ground Moves   ← C5 (substrate decoupling under base-model evolution)
   (lead with the §26 headline: 37.5% base shift / 0.3% effective drift.)
11. The Correction Field on a Different Model   ← C7 (cross-model, with L3 caveat)
12. Why This Is Not Retrieval   ← C6 (architectural, not score-based)
13. Limits   ← L1–L5
14. Conclusion
```

Three writer edits we agreed on:
1. **§ 7 opening sentence** ("five homogeneous operations") — explicitly states C2's architectural claim
2. **§ 9 generalization paragraph** that connects § 24 compiled closure and § 31c compiled paraphrase under one "compiled-surface" pattern
3. **§ 13 L2 expansion** with the geometric quantification (ZsRE 4.6 vs CounterFact 16.7 z-distance, σ frontier, anchor compiler)

---

## 10. Critical things to NOT do

1. **Don't run "Restart Kernel" mid-run** unless you intend to lose hours of in-memory state. Recovery requires re-running from § 1.
2. **Don't use `git push --force` on `main`** — main is the public branch. Use the dev branch (`claude/review-jupyter-notebook-8AU5y`) for all in-progress work. Open a PR to merge into main only when ready to publish.
3. **Don't trust JupyterLab's "Reload Notebook from Disk" silently** — if the menu item is greyed out, JupyterLab is holding unsaved changes. Use "Revert Notebook to Saved" only as last resort (it kills the kernel).
4. **Don't pull without committing first** — Jupyter's autosaved cell outputs will block the pull. Either commit first or `git checkout -- <file>` to discard.
5. **Don't push the entire `mmlu_ibf_out/` without checking what's there** — the canonical_engine.pkl is 311 MB. Large pickles are fine to keep on the pod but may need git LFS to push.
6. **Don't change the §1 run-mode toggle's structure** — many cells inherit smoke-vs-paper behavior from `globals().get("CANONICAL_TRAINING_MODE", "smoke")`. Breaking the toggle breaks the whole notebook.

---

## 11. Known design decisions baked into the notebook

### 11.1 Smoke = pipeline validation, paper = headline numbers

Smoke is intentionally short (~2-3 hours). It validates every cell runs end-to-end. Paper-mode reuses the same cells with longer epoch caps, real datasets, full scale sweep, 24-step LoRA, full long-horizon.

### 11.2 The §27 interim audit was DELETED, downstream renumbered

In commit `9567750`, § 27 (a 954-line JSON-dumping audit) was deleted entirely and § 28–§ 41 renumbered to § 27–§ 40. Don't reference the old numbering.

### 11.3 § 38 v3 is the paper deliverable cell

Produces `paper_tables.md` + `abstract_numbers.json` + `claims_status_final.md` from artifacts. Reads only — no training. Adjust the cell's `dig()` paths if you change any artifact schema.

### 11.4 GRACE / SERAC stubs were DELETED

Per supervisor (L4 — external editors deferred), GRACE and SERAC stubs (formerly § 32b / § 32c) were removed in commit `b79b64a`. Comparison is IBF vs kNN vs RAG only.

### 11.5 §21 long-horizon stubs in smoke

In smoke mode, § 21 writes a stub artifact and skips the 20h experiment. Paper mode runs the full experiment. Same pattern for § 21b VMAX readout. Triggered by `CANONICAL_TRAINING_MODE == "smoke"`.

### 11.6 Compiled-surface architecture is the L1 answer

§ 23 confirms transitive closure does NOT emerge automatically. § 24 provides the constructive answer: deterministic closure compiler + IBF enforcement. § 31c extends this to paraphrase anchors. Frame as **scope discipline**, not failure ("compiled semantic structure can be durably enforced").

### 11.7 The cross-model story is fresh-field, NOT zero-shot transfer

§ 36 trains a fresh δR field over Qwen2-1.5B. It does NOT take the Mistral-trained δR and run it on Qwen. L3 caveat must be explicit everywhere C7 appears.

---

## 12. Roles

- **Radu Negulescu** — PI. Decides on scope, framing, paper structure. Email: `radu@ibf.xyz`.
- **Supervisor** — gave the D1–D5 decisions (kept §22+§23+§24 ontology trio, deferred external editors, etc.) and the paper-framing rules ("not 'IBF does not reason' as headline", "compiled semantic structure can be durably enforced", "fresh-field cross-model, not zero-shot transfer").
- **Companion writer** — drafting the paper from the notebook + the three § 38 deliverables. Has access to the 14-section structure we agreed on.
- **You (new Claude session)** — execute the paper run, monitor, handle errors, produce final deliverables, support the writer.

---

## 13. Quick-reference commit history

| Commit | What |
|---|---|
| `14699aa` | § 38 v3 paper-deliverable generator (correct schemas, 11 tables) |
| `a37a450` | § 38 v2 first attempt (had schema-guess bugs — superseded) |
| `497a3ff` | § 37 fix re-applied on top of user's output commit |
| `a4350f4` | (user push) smoke run outputs through § 36 |
| `b3efc79` | § 37 fix: regenerate rb_revision_train (first attempt — replaced by 497a3ff) |
| `b79b64a` | Remove GRACE/SERAC stubs; re-enable HF fetch default; fix self-ref |
| `9113c22` | § 29 harness: smoke-aware default for synthetic-benchmark inclusion |
| `a8c2104` | § 31b / § 31c: graceful skip when CounterFact missing |
| `9567750` | Delete § 27 entirely; renumber 28..41 → 27..40 |
| `da8ad7e` | Fix four cells (§ 19, § 20, § 21, § 21b) that ignored or undershot smoke |
| `ecc2dfd` | Smoke max_epochs=2 universal + § 20/§ 20b merge with dynamic capacity |
| `f08748a` | Phase 2 readability pass (8 Parts, 41 sections, claim-to-cell map) |
| `4ee2bbe` | Phase 1 structural cleanup |
| `ae198b4` | Add LLM companion notebook as baseline |

---

## 14. If you're stuck

1. **Read this document end-to-end first** before any action.
2. **Check `claims_status_final.md`** for the latest run's state.
3. **Check the latest commit's commit message** for the most recent intent.
4. **Don't restart the kernel without need** — it costs hours.
5. **Don't push to `main`** — push only to the dev branch.
6. **Ask Radu** if a framing decision is ambiguous — supervisor's rules are explicit and non-negotiable.

Good luck with the paper run.

— Previous session
