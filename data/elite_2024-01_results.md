# Emergent chess on REAL lichess Elite games — results

**Data:** `data/elite_2024-01_5k.pgn` — first 5,000 games of the Lichess Elite
Database, 2024-01 (rated games, players 2200+; Elo 2314–2950, median ~2556;
average 89 plies/game). Model sees **only opaque move tokens** (no board, pieces,
rules, strategy). `python-chess` is used only to parse and as a legality oracle.

Run: `python -m mscn.chess_strategy --pgn data/elite_2024-01_5k.pgn`
(trained on the stronger half: 2,126 games, 1,826 move-token vocab.)

## Stage 1 — emergent rules (legal-move prediction, no rules given)

| phase   | IBF legal@1 | legal@5 | legal-mass | unigram@1 | random@1 | acc@1 (real move) |
|---------|------------:|--------:|-----------:|----------:|---------:|------------------:|
| opening |       85.0% |   99.2% |      51.2% |     31.4% |    1.5%  |             38.6% |
| midgame |       52.0% |   88.9% |      31.9% |      4.1% |    1.9%  |             20.4% |
| endgame |       17.9% |   44.7% |      10.3% |      0.2% |    1.5%  |              5.2% |
| all     |       31.6% |   58.7% |      18.8% |      4.4% |    1.6%  |             11.7% |

Real data is *much* stronger than the synthetic proxy in the opening/midgame: the
model predicts the **actual human move 38.6% of the time in the opening** (it
learned opening theory from tokens alone) and keeps 85% legal@1. The "all"
average is dragged down by the long Elite endgames (avg 89, max 314 plies): novel
late positions are exactly where a fixed-order associative memory cannot track
board state — the gap analysed in `NOTE-state-tracking-gap.md`.

## Stage 2 — emergent board geometry

Procrustes disparity **0.10** vs the true board; **71%** of true king-adjacencies
recovered as nearest neighbours. The 8×8 grid emerges from real human play with
no spatial prior (see `chess_board_emergence_real.png`).

## Stage 3 — emergent strategy (real Elo + results)

| metric | value |
|---|---|
| coherence by Elo tercile | low(~2459) 0.042 · mid(~2569) 0.043 · high(~2790) **0.070** |
| corr(move-coherence, Elo) | **+0.137** (weak; Elite Elo band is narrow) |
| move quality (cp, 1-ply settled) | model-top **−486** vs random **−700** vs actual **−54** |
| corr(white_coh − black_coh, result) | **+0.003** (≈ null) |

Coherence rises for the very strongest players (top tercile clearly above), but
the overall Elo correlation is weak because every Elite player is strong (small
spread). The model's preferred move beats random but is far below human choices.
The **outcome correlation is essentially zero**: among uniformly strong players,
move-coherence does not pick the winner — a clean, honest null (contrast the
skill-stratified proxy, where coherence-difference predicted the winner at +0.31
precisely *because* the players differed in strength).

## Takeaway

On real human games the full arc holds: **rules emerge** (and the model even
learns opening theory — 39% exact-move match in the opening), **board geometry
emerges** (disparity 0.10), and **coherence tracks the very top of the strength
range**. The two honest limits are unchanged: deep/novel endgame positions
(state-tracking ceiling) and discriminating *between equally strong* players.
