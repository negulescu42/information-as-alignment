[![arXiv](https://img.shields.io/badge/arXiv-2604.07108-b31b1b.svg)](https://arxiv.org/abs/2604.07108)

# REPRODUCIBILITY

this file gives the shortest path from paper claim to artifact to code.

start with inspection. rerun only what you actually need.

## FASTEST AUDIT PATH

read `information-as-alignment-v1.pdf`.

open `(IBF)Toy-Model.ipynb` and inspect the outputs.

open `(IBF)Domain-I-RRW.ipynb` and confirm the RRW summary behind table 1.

inspect `paper_results.json` and `results_seeds.json`.

open `(IBF)Domain-II-Chess.ipynb` and inspect the chess outputs.

open `(IBF)Domain-III-CIFAR-100.ipynb` and inspect the CIFAR outputs.

that path is enough to audit the paper without immediately committing to long reruns.

## EXTERNAL REQUIREMENTS

the chess notebook requires **stockfish 16** and a strong-game PGN archive.

stockfish install examples:

```bash
# ubuntu / debian
sudo apt install stockfish

# macOS
brew install stockfish
```

for the PGN archive, use the public lichess database:

https://database.lichess.org/

download a suitable archive and point the notebook to your local PGN file.

the CIFAR-100 notebook uses the torchvision dataset loader and can download CIFAR automatically if needed.

## WHAT REPRODUCES WHAT

the toy model notebook reproduces section 2, figures 1–8, and the mini-ablation table.

the RRW notebook reproduces section 7.1, table 1, the five-seed mechanism confirmation, and the RRW evaluation-bandwidth sweep.

the chess notebook reproduces section 7.2, table 2, the readout sweep, the seed-level replication artifacts, and the agency / Crucible diagnostics.

the CIFAR-100 notebook reproduces section 7.3, table 3, the ablations, the class-IL evaluation, and the weak-head analysis.

## RERUN ORDER

if you want to rerun everything from smallest to largest, use this order:

1. `(IBF)Toy-Model.ipynb`
2. `(IBF)Domain-I-RRW.ipynb`
3. `(IBF)Domain-II-Chess.ipynb`
4. `(IBF)Domain-III-CIFAR-100.ipynb`

this order mirrors the paper’s logic: mechanism visibility first, then controlled validation, then strategic emergence, then high-dimensional scaling.

## TRAINING ENVIRONMENT & RUNTIMES

all reported experiments were developed and executed on a rented cloud compute instance provided by **RunPod**. specifically, we provisioned a single-GPU pod running a standard PyTorch + JupyterLab template. the notebooks provided in this repository were executed directly within that Jupyter environment without requiring any custom containerization or complex cluster setups.

if you wish to perfectly replicate our environment on RunPod (or a similar cloud provider), the reference pod specifications were:

- 1 × RTX 5090
- 21 vCPU (`AMD EPYC 9354 32-Core Processor`)
- 125 GB system memory
- 30 GB container disk space

approximate wall-clock estimates on this reference hardware:

| experiment | estimate |
|---|---:|
| toy model | < 1 min |
| RRW | ~30 min |
| chess | ~50 h |
| CIFAR-100 | ~75 h |

## ARTIFACT-FIRST CHECKS

for chess, inspect `paper_results.json` first. this is the fastest way to verify the main comparison outputs without rerunning the full notebook.

then inspect `results_seeds.json` for the seed-level chess replication results.

for RRW, the notebook output itself is the fastest audit path.

for CIFAR-100, inspect the notebook summaries before attempting a full rerun.

## TROUBLESHOOTING

if stockfish is not found, install it or point the notebook to the binary explicitly.

if the chess PGN is not found, download a Lichess archive and update the path variable in the notebook.

if CIFAR does not load, rerun the dataset-loading cell and check local cache permissions.

if runtime is too long, do not start with chess or CIFAR from scratch. inspect the saved artifacts and notebook outputs first.

## CONTACT

for questions related to the paper or repository:

**radu@ibf.xyz**
