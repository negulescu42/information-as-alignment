# Paper Run — Launch Runbook

One-page operator checklist. Companion to `PAPER_RUN_HANDOVER.md` (read that once
end-to-end before your first run). This page is the cheat sheet for every
subsequent launch.

**Target**: paper-grade run of `(IBF)Companion-LLM-Durable-Alignment.ipynb` on
the Runpod pod. **Wall clock**: ~49h main + ~19.9h Qwen § 36 ≈ **70h**.

---

## 0. Pod prerequisites (once per fresh pod)

- GPU ≥ 24 GB VRAM (A100 / H100 / A6000 48 GB / L40S / RTX 5090)
- Network volume mounted at `/workspace`, ≥ 100 GB free
- JupyterLab reachable

---

## 1. Pre-flight (always — runs in ~2 min)

```bash
cd /workspace/information-as-alignment
git pull --ff-only origin claude/review-jupyter-notebook-8AU5y
git log --oneline -1            # expect: 6b7a38e or later
bash runpod_setup.sh            # idempotent: installs deps, wgets CF + ZsRE
nvidia-smi                      # expect: <1 GB used. Kill stragglers if not.
ls -lh mmlu_ibf_out/counterfact.json mmlu_ibf_out/zsre_mend_eval.json
# expect: ~50 MB and ~30 MB
python3 -c "from datasets import load_dataset; \
  print(len(load_dataset('NeelNanda/counterfact-tracing')['train']))"
# expect: 21919
```

**If any line above fails, stop and fix before launching.**
HF auth/network failure here = synthetic fallback in § 28 = silent C6 corruption.

---

## 2. Notebook setup (in JupyterLab)

1. **Shut down all kernels**: sidebar → Running → "Shut Down All".
2. **File → Open** `(IBF)Companion-LLM-Durable-Alignment.ipynb`.
3. **Edit cell 1** — uncomment all five `*_MODE` flags:

   ```python
   CANONICAL_TRAINING_MODE     = "paper"
   SCALE_CAPACITY_MODE         = "paper"
   IBF_DURABLE_BENCHMARK_MODE  = "paper"
   KNN_DURABLE_BENCHMARK_MODE  = "paper"
   RAG_DURABLE_BENCHMARK_MODE  = "paper"
   ```

   **Do not** insert a scratch override cell — paper-mode defaults are correct
   for `IBF_DURABLE_INSTALL_EPOCHS` (10), `CONTROL_FIXED_LORA_STEPS` (24), and
   `INCLUDE_SYNTHETIC_BENCHMARKS_IN_LIFECYCLE` (False). Pre-setting the last
   one to True will pollute C6 with synthetic records.

4. **Kernel → Restart Kernel and Run All Cells**.
5. Confirm cell 1 prints `'paper'` for all five flags.

---

## 3. Monitor (drop in every few hours)

| Cells | Cumulative wall clock | What to glance at |
|---|---|---|
| § 1–§ 7 | +10 min | σ calibration prints, no tracebacks |
| § 8 | +6–8 h | A 300 / B 200 / C 200 / D 200 epochs done |
| § 14–§ 17 | +12 h | Strong-prior + local-alignment numbers stable |
| § 20 | +20 h | Scale frontier through 50k |
| § 21 | +40 h | Long-horizon — 20 rounds × 5 epochs |
| § 26 | +41 h | LoRA: `config.lora_steps = 24`, base shift ~37%, drift <1% |
| § 28 | +41 h | **Source = paper-grade, NOT `synthetic_fallback_only`** |
| § 30–§ 33 | +43 h | Each benchmark > 0.0 metrics (0.000 = harness empty) |
| § 36 | +63 h | Qwen2-1.5B fresh field |
| § 38 | +70 h | Three deliverables written to `mmlu_ibf_out/paper/` |

---

## 4. Failure quick-ref (symptom → action)

| Symptom | First action | Where in handover |
|---|---|---|
| `CUDA out of memory` on § 4 base load | `nvidia-smi` → `kill -9 <pid>` → Restart & Run All | §7.1 |
| § 28 prints `synthetic_fallback_only` for any benchmark | Verify `ALLOW_HF_DATASET_FETCH=True`, `datasets` installed, both JSONs present in `mmlu_ibf_out/`. **Do not just set `INCLUDE_SYNTHETIC_BENCHMARKS_IN_LIFECYCLE=True`** — fix the root cause first | §5.1, §7.2 |
| § 30/§ 31/§ 32/§ 33 finish in 0.0s with 0.000 metrics | `durable_lifecycle_tasks.json` is empty. Only as last resort: set `INCLUDE_SYNTHETIC_BENCHMARKS_IN_LIFECYCLE=True` in a scratch cell, re-run § 29 onwards | §7.3 |
| § 31/§ 31b/§ 31c print `Skipping CounterFact ...` | Graceful skip; § 28 didn't ship real CounterFact. Check `standard_benchmark_records.md` | §7.4 |
| § 37 `IndexError on rb_revision_train` | Should not occur (fix in commit `497a3ff`). If it does, re-pull | §7.5 |
| Kernel dies mid-run | Kernel → Restart Kernel and Run All. Pickled artifacts on disk survive | §7.6 |

**Never** click "Revert Notebook to Saved" or "Restart Kernel" mid-run unless
you accept losing in-memory state.

---

## 5. Post-run (after § 38 finishes)

```bash
# 5a. Inspect deliverables
cat mmlu_ibf_out/paper/claims_status_final.md   # expect all C1–C7 🟢
ls mmlu_ibf_out/paper/                          # 3 files
ls mmlu_ibf_out/                                # all artifacts present
# C5 sanity:
python3 -c "import json; \
  d = json.load(open('mmlu_ibf_out/actual_lora_e2e_durability_control_fixed_report.json')); \
  print('lora_steps =', d['config']['lora_steps'])"   # expect 24

# 5b. Commit results — DEV BRANCH ONLY, never push to main
git add "(IBF)Companion-LLM-Durable-Alignment.ipynb" mmlu_ibf_out/
git commit -m "Paper-grade run results — RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)"
git push origin claude/review-jupyter-notebook-8AU5y
```

If push fails on size (`canonical_engine.pkl` is ~311 MB), configure git LFS
or push artifacts separately.

---

## 6. Hard "don'ts"

1. **Never** `git push --force` to `main`.
2. **Never** restart kernel mid-run without intent.
3. **Never** pull without first committing notebook (autosaved outputs block ff).
4. **Never** edit the cell-1 toggle structure (downstream cells read via `globals().get`).
5. **Never** pre-set `INCLUDE_SYNTHETIC_BENCHMARKS_IN_LIFECYCLE = True` in paper mode.

---

When in doubt, re-read `PAPER_RUN_HANDOVER.md` end-to-end. This page is the
shortcut, not the source of truth.
