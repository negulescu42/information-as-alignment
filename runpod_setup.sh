#!/usr/bin/env bash
# Runpod setup for the IBF companion LLM notebook.
#
# Usage:
#   bash runpod_setup.sh
#
# Idempotent: safe to re-run after a pod restart. The HuggingFace cache
# and the repo clone live on /workspace, so they survive pod stop/start
# as long as the network volume is reattached.
#
# Recommended pod:
#   - GPU: A6000 48 GB / A100 40 GB / L40S / H100 (any with >= 24 GB VRAM)
#   - Image: any recent RunPod PyTorch 2.x + JupyterLab template
#   - Network volume mounted at /workspace, >= 100 GB

set -euo pipefail

REPO_URL="https://github.com/negulescu42/information-as-alignment.git"
REPO_DIR="/workspace/information-as-alignment"
BRANCH="claude/review-jupyter-notebook-8AU5y"

# HuggingFace cache on the persistent volume.
mkdir -p /workspace/.cache/huggingface
export HF_HOME=/workspace/.cache/huggingface
if ! grep -q "HF_HOME=" ~/.bashrc 2>/dev/null; then
    echo "export HF_HOME=/workspace/.cache/huggingface" >> ~/.bashrc
fi

# Repo clone / fetch.
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

# Optional git identity (override with env vars if you prefer).
git config user.name  "${GIT_AUTHOR_NAME:-Radu Negulescu}"
git config user.email "${GIT_AUTHOR_EMAIL:-radu@ibf.xyz}"

# Notebook deps not in the base PyTorch image.
pip install --quiet sentence-transformers faiss-cpu scipy scikit-learn datasets peft

# Pre-download paper-grade benchmark datasets (used by § 28 dataset builder).
# Guarded so re-running the script doesn't re-download on a persistent volume.
mkdir -p mmlu_ibf_out
if [ ! -f mmlu_ibf_out/counterfact.json ]; then
    wget -q -O mmlu_ibf_out/counterfact.json https://rome.baulab.info/data/dsets/counterfact.json
    echo "Downloaded counterfact.json"
fi
if [ ! -f mmlu_ibf_out/zsre_mend_eval.json ]; then
    wget -q -O mmlu_ibf_out/zsre_mend_eval.json https://rome.baulab.info/data/dsets/zsre_mend_eval.json
    echo "Downloaded zsre_mend_eval.json"
fi

# Sanity prints.
echo
echo "===== Sanity ====="
python - <<'PY'
import torch
print(f"torch         : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")
    print(f"VRAM (GB)     : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
import transformers, sentence_transformers, faiss, sklearn, scipy, datasets, peft
print(f"transformers  : {transformers.__version__}")
print(f"sent-transf.  : {sentence_transformers.__version__}")
print(f"faiss         : {faiss.__version__}")
print(f"sklearn       : {sklearn.__version__}")
print(f"scipy         : {scipy.__version__}")
print(f"datasets      : {datasets.__version__}")
print(f"peft          : {peft.__version__}")
PY

echo
echo "Setup complete. Open in JupyterLab:"
echo "  $REPO_DIR/(IBF)Companion-LLM-Durable-Alignment.ipynb"
echo
echo "Default run mode is smoke (~30 min on A100). For the paper-grade"
echo "run, uncomment the five MODE flags in cell 1 (top run-mode cell)."
