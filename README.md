# information-as-alignment

Code, configurations, and result artifacts for the paper **_Information as Structural Alignment: A Dynamical Theory of Continual Learning_**.

This repository is the paper made executable. It contains the toy model, the three validation domains, the saved result artifacts behind the reported numbers, and the minimum instructions needed to inspect or reproduce the main experiments.

---

## What is here

this release contains four layers:

- **Toy model** — the full seven-step mechanism made visible in 2D
- **RRW** — the controlled mechanism-confirmation domain
- **Chess** — the strategic external-oracle domain
- **CIFAR-100** — the high-dimensional benchmark domain

together, they map directly onto the paper's validation arc:

1. mechanism visibility
2. mechanism confirmation
3. emergent behavior under genuine structure
4. scaling and non-destructiveness

---

## Repository layout

```text
.
├── README.md
├── INSTRUCTIONS.md
├── IBF_Paper.pdf
│
├── (IBF)Toy-Model.ipynb
├── (IBF)Domain-I-RRW.ipynb
├── (IBF)Domain-II-Chess.ipynb
├── (IBF)Domain-III-CIFAR-100.ipynb
│
├── paper_results.json
├── results_seeds.json
└── ... additional result artifacts produced by the notebooks
```

---

## What reproduces what

### Toy model
reproduces:
- Section 2 / Figures 1–8 / the mini-ablation table

notebook:
- `(IBF)Toy-Model.ipynb`

### RRW
reproduces:
- Section 7.1 / Table 1 / five-seed mechanism confirmation / RRW evaluation-bandwidth sweep

notebook:
- `(IBF)Domain-I-RRW.ipynb`

### Chess
reproduces:
- Section 7.2 / Table 2 / independent Stockfish evaluation / post-training readout sweep / seed-level replication artifacts / agency / crucible diagnostics

notebook:
- `(IBF)Domain-II-Chess.ipynb`

primary saved artifacts:
- `paper_results.json`
- `results_seeds.json`

### CIFAR-100
reproduces:
- Section 7.3 / Table 3 / ablation outputs / weak-head analysis / Class-IL evaluation

notebook:
- `(IBF)Domain-III-CIFAR-100.ipynb`

---

## Reference compute environment

the reported runs were produced on the following reference pod:

- **GPU:** 1 × NVIDIA RTX 5090
- **vCPU:** 21 (`AMD EPYC 9354 32-Core Processor`)
- **Memory:** 125 GB
- **Container disk:** 30 GB
- **Observed pod uptime during the main run window:** 4w 4d

this is the reference environment for the runtime estimates below.

---

## Runtime estimates

approximate wall-clock times on the reference pod:

| Experiment | Estimated runtime | Notes |
|---|---:|---|
| Toy model | < 1 min | full notebook |
| RRW | ~30 min | 5 seeds, Table 1 |
| Chess | ~60 h | main run + ablations + sweep |
| CIFAR-100 | ~75 h | main run + ablations + weak-head analysis |

notes:

- runtimes are approximate
- cached assets, local I/O, and checkpoint reuse affect wall-clock time
- smoke checks are substantially cheaper than full paper runs
- !!! the notebooks are useful even without full reruns because they contain embedded reported outputs

---

## Software requirements

recommended environment:

- Python 3.12+
- Jupyter Notebook or JupyterLab
- PyTorch 2.x
- torchvision
- numpy
- scipy
- scikit-learn
- matplotlib
- python-chess

---

## External dependencies

### Stockfish

the chess notebook requires **Stockfish 16**.

examples:

```bash
# Ubuntu / Debian
sudo apt install stockfish

# macOS
brew install stockfish
```

if Stockfish is not available on `PATH`, edit the engine path inside the notebook.

### CIFAR-100

the CIFAR-100 notebook uses the torchvision loader and can download the dataset automatically if needed.

### Chess PGN database

the chess notebook expects an elite-game PGN file at the path specified inside the notebook.

recommended source:
- a Lichess elite PGN dump or equivalent elite-game corpus: https://database.lichess.org/

---

## Quick start

if you want the fastest sanity check, run the notebooks in this order:

1. `(IBF)Toy-Model.ipynb`
2. `(IBF)Domain-I-RRW.ipynb`
3. `(IBF)Domain-II-Chess.ipynb`
4. `(IBF)Domain-III-CIFAR-100.ipynb`

that order is deliberate:
- the toy model shows every mechanism in the open
- RRW confirms the mechanism under controlled contradiction
- chess shows the mechanism under real strategic structure
- CIFAR-100 shows the same substrate surviving at larger scale

---

## Reproduction guide

### 1. Toy model
purpose:
- inspect the full mechanism in the smallest visible setting

expected outputs:
- the seven-step lifecycle
- generated figures corresponding to Section 2
- mini-ablation table

### 2. RRW
purpose:
- reproduce the controlled mechanism-confirmation domain

expected outputs:
- Table 1 values
- five-seed summary
- RRW bandwidth sweep
- No-Agency / No-Crystallization / No-Crucible outputs

### 3. Chess
purpose:
- reproduce the strategic domain under independent Stockfish evaluation

expected outputs:
- Table 2 comparison
- behavioral advantage under external-oracle evaluation
- post-training readout sweep
- seed-level replication artifacts
- agency and Crucible diagnostics

recommended workflow:
- inspect the saved result artifacts first
- inspect the notebook outputs second
- rerun only if you want end-to-end reconstruction

### 4. CIFAR-100
purpose:
- reproduce the high-dimensional continual-learning validation

expected outputs:
- main Task-IL benchmark outputs
- ablation outputs
- weak-head analysis
- Class-IL evaluation

recommended workflow:
- verify the saved outputs first
- then run the notebook sections required for the specific comparison you want to inspect

---

### !!! Notebook outputs
the notebooks themselves contain embedded reported outputs. in many cases, inspecting the notebook output is enough to audit the paper-facing result without rerunning the full experiment.

---

## Recommended audit path

if your goal is to verify the paper efficiently:

1. read `IBF_Paper.pdf`
2. inspect `(IBF)Toy-Model.ipynb`
3. inspect RRW notebook outputs and confirm Table 1
4. inspect the chess notebook outputs and confirm Table 2 / seed replication / sweep
5. inspect CIFAR notebook outputs and confirm Table 3 / ablations / weak-head result

this is the fastest serious path through the release.

---

## Reproducibility notes

- the notebooks are the primary entry points for inspection and reproduction
- the paper's reported numbers are tied to notebook outputs and saved artifacts
- chess combines:
  - multi-seed headline results for the main behavioral and backward-transfer metrics
  - a designated run for the finer mechanism comparisons
- CIFAR combines:
  - the main benchmark outputs
  - ablations
  - the weaker-head diagnostic used to test whether corrections remain informative below the strong-baseline regime

---

## Intended use

This repository is meant for:

- inspection of the reported experiments
- reproduction of the main paper results
- audit of the saved result artifacts
- further research on the IBF engine and its domain instantiations

---

## Paper citation

if you use this repository, please cite:

**Information as Structural Alignment: A Dynamical Theory of Continual Learning**  
Radu Negulescu  
The Informational Buildup Foundation  
April 2026

---

## Contact

for questions related to the paper or the repository:

**radu@ibf.xyz**

---

## License

Code in this repository is licensed under **Apache-2.0**.

Documentation, paper text, figures, and result artifacts are licensed under **CC BY 4.0**, unless otherwise noted.
