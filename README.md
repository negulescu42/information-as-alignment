# information-as-alignment

code, configurations, and result artifacts for **information as structural alignment: a dynamical theory of continual learning**, including the toy model, RRW, chess, and CIFAR-100 experiments.

## OVERVIEW

this repository is the code and artifact companion to the paper. it contains the toy model, the three validation domains, and the saved outputs needed to inspect, audit, or reproduce the reported results.

the repository is organized as a research release, not as a production package. the fastest path is therefore to inspect the saved artifacts first, then the notebook outputs, and only then rerun the experiments you actually care about.

## REPOSITORY LAYOUT

```text
.
├── README.md
├── REPRODUCIBILITY.md
├── information-as-alignment.pdf
│
├── (IBF)Toy-Model.ipynb
├── (IBF)Domain-I-RRW.ipynb
├── (IBF)Domain-II-Chess.ipynb
├── (IBF)Domain-III-CIFAR-100.ipynb
│
├── RRW-paper-results.json
├── chess-results-seeds.json
├── chess-paper-results.json
└── CIFAR-paper-results.json
```

the four notebooks correspond to the toy model, RRW, chess, and CIFAR-100. the json artifacts contain the saved outputs used for the paper-facing comparisons, especially in chess.

## WHAT IS IN HERE

the toy model reproduces the full mechanism lifecycle in two dimensions.

RRW reproduces the controlled mechanism-confirmation domain and the five-seed summary behind table 1.

chess reproduces the independent-oracle strategic domain, including the main comparison table, the readout sweep, seed replication artifacts, and the agency / Crucible diagnostics.

CIFAR-100 reproduces the high-dimensional continual-learning domain, including the main benchmark results, ablations, class-IL evaluation, and the weak-head analysis.

## COMPUTE ENVIRONMENT

the reported runs were produced on a reference pod with:

- 1 × RTX 5090
- 21 vCPU (`AMD EPYC 9354 32-Core Processor`)
- 125 GB memory
- 30 GB container disk

this is the reference environment for the runtime estimates below. you do not need this exact setup to inspect the repository, but the full chess and CIFAR runs were produced on hardware of this class.

## RUNTIME

approximate wall-clock estimates on the reference pod:

| experiment | estimate | notes |
|---|---:|---|
| toy model | < 1 min | full notebook |
| RRW | ~30 min | 5 seeds, table 1 |
| chess | ~65 h | full run suite |
| CIFAR-100 | ~75 h | main run + ablations + weak-head |

these are practical estimates, not promises. checkpoint reuse, local I/O, cached assets, and external binaries all affect total time.

## DEPENDENCIES

recommended environment:

- python 3.12+
- jupyter notebook or jupyterlab
- pytorch 2.x
- torchvision
- numpy
- scipy
- scikit-learn
- matplotlib
- python-chess

the chess notebook also requires **stockfish 16**.

install examples:

```bash
# ubuntu / debian
sudo apt install stockfish

# macOS
brew install stockfish
```

the CIFAR-100 dataset is handled through `torchvision` and can be downloaded automatically if needed.

for chess data, use the public lichess database:

https://database.lichess.org/

download a suitable PGN archive and point the notebook to your local file.

## QUICK START

if you want the fastest understanding of the mechanism, start with:

1. `(IBF)Toy-Model.ipynb`
2. `(IBF)Domain-I-RRW.ipynb`

if you want the fastest audit of the paper, read `REPRODUCIBILITY.md` and inspect the saved artifacts before attempting any long reruns.

## RESULT ARTIFACTS

`paper_results.json` is the main chess artifact. it contains the outputs used for the main chess comparison table and the paper-facing centipawn comparisons.

`results_seeds.json` contains the chess seed-replication artifact, including the seed-level behavioral and backward-transfer summaries.

the notebooks themselves also contain embedded reported outputs. in many cases, direct inspection of notebook output is enough to verify the relevant claim.

## PAPER

this repository accompanies:

**information as structural alignment: a dynamical theory of continual learning**  
radu negulescu  
the informational buildup foundation  
april 2026

## LICENSE

code in this repository is licensed under **apache-2.0**.

documentation, paper text, figures, and result artifacts are licensed under **CC BY 4.0**, unless otherwise noted.
