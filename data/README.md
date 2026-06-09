# data/ — drop real PGNs here

The sandbox network allows only PyPI, so the runner cannot fetch games from any
URL (lichess, nikonoel, RunPod proxy all return 403). The working channel is
**git**: commit a PGN to the branch and it appears in the container on pull.

Keep it git-friendly: commit a **slice** (a few thousand games, < ~25 MB), not a
full monthly dump. From the machine that has the file:

```bash
# first 5000 games into a small file
awk 'BEGIN{g=0} /^\[Event /{g++} g>5000{exit} {print}' \
    workspace/lichess_elite_2024-01.pgn > data/elite_2024-01_5k.pgn

git add data/elite_2024-01_5k.pgn
git commit -m "data: lichess elite 2024-01 sample (5k games)"
git push origin claude/happy-carson-3zhihq
```

Then run all three stages on the real games:

```bash
python -m mscn.chess_strategy --pgn data/elite_2024-01_5k.pgn
```

`.pgn.zst` is also accepted (handled by `load_pgn`). If you'd rather not slice,
enable an allowed network policy for the host instead and the runner can stream
it directly.
