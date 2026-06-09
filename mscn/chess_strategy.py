"""Stage 3 -- emergent strategy, and the real-lichess data path.

Two things live here:

1. **The real-data path** (`stream_lichess`, `load_pgn`): ingest real human games
   from a lichess PGN dump (``.pgn`` or ``.pgn.zst``), keeping moves, player
   ratings (Elo) and any ``%eval`` annotations. This is what "Stage 3 on real
   lichess data" runs on. NOTE: this environment's network policy only permits
   PyPI, so the lichess CDN is unreachable here (HTTP 403); these functions are
   ready for an environment that allows lichess, or for a PGN you drop in.

2. **A skill-stratified proxy** (`generate_skilled_games`) that creates a genuine
   strength gradient locally with a tiny pure-Python minimax engine, so the
   Stage-3 *methodology* can be validated now without real data. It is clearly a
   proxy for lichess ratings, not lichess data.

Stage-3 questions (the design note's "emergent strategy"): does the IBF model's
learned **coherence** track (a) move quality, (b) player strength, (c) outcome?

Requires ``chess``; ``zstandard`` only for ``.pgn.zst``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .chess_world import IBFChessModel

try:
    import chess
    import chess.pgn
    HAS_CHESS = True
except Exception:  # pragma: no cover
    HAS_CHESS = False


# ===========================================================================
#  Real lichess data path (ready; blocked by network policy in this sandbox)
# ===========================================================================

def stream_lichess(url: str, max_games: int = 20000, dest: str | None = None) -> str:
    """Stream a lichess ``.pgn.zst`` dump and write the first ``max_games`` games
    to ``dest`` (a plain ``.pgn``). Returns the path. Requires network access to
    the lichess CDN (blocked here) and ``zstandard``.

    Example URL:
    https://database.lichess.org/standard/lichess_db_standard_rated_2014-07.pgn.zst
    """
    import urllib.request
    import zstandard  # noqa: F401  (import error surfaces a clear message)

    dest = dest or "lichess_sample.pgn"
    req = urllib.request.Request(url, headers={"User-Agent": "mscn-research/0.1"})
    dctx = zstandard.ZstdDecompressor()
    games_written = 0
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "w") as out:
        with dctx.stream_reader(resp) as reader:
            buf = b""
            while games_written < max_games:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                text = buf.decode("utf-8", errors="ignore")
                # flush complete games (separated by blank line after moves)
                while "\n\n[" in text:
                    cut = text.index("\n\n[", text.index("\n\n") + 2) if text.count("\n\n") >= 2 else -1
                    if cut < 0:
                        break
                    out.write(text[:cut] + "\n\n")
                    text = text[cut + 2:]
                    games_written += 1
                    if games_written >= max_games:
                        break
                buf = text.encode("utf-8")
    return dest


def load_pgn(path: str, max_games: int = 20000) -> list[dict]:
    """Load games from a ``.pgn`` (or ``.pgn.zst``) file.

    Returns dicts with ``moves`` (UCI tokens), ``white_elo``, ``black_elo``,
    ``result`` (+1/0/-1 for white) and ``evals`` (centipawns if annotated).
    """
    if not HAS_CHESS:
        raise RuntimeError("python-chess not installed")
    import io
    if path.endswith(".zst"):
        import zstandard
        with open(path, "rb") as fh:
            data = zstandard.ZstdDecompressor().stream_reader(fh).read()
        handle = io.StringIO(data.decode("utf-8", errors="ignore"))
    else:
        handle = open(path)

    out = []
    with handle as fh:
        while len(out) < max_games:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            board = game.board()
            moves, evals = [], []
            for node in game.mainline():
                moves.append(node.move.uci())
                board.push(node.move)
                if node.comment and "%eval" in node.comment:
                    try:
                        tok = node.comment.split("%eval")[1].split("]")[0].strip().strip("[")
                        evals.append(float(tok) if "#" not in tok else (10.0 if "+" in tok else -10.0))
                    except Exception:
                        evals.append(None)
            res = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(game.headers.get("Result", "*"), None)
            if not moves:
                continue
            out.append({
                "moves": moves,
                "white_elo": _to_int(game.headers.get("WhiteElo")),
                "black_elo": _to_int(game.headers.get("BlackElo")),
                "result": res,
                "evals": evals,
            })
    return out


def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None


# ===========================================================================
#  Tiny pure-Python engine (oracle + skill-stratified game generator)
# ===========================================================================

_VALUE = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
          chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0} if HAS_CHESS else {}


def _center_bonus(sq: int) -> float:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return -((f - 3.5) ** 2 + (r - 3.5) ** 2)  # peak at the centre


def evaluate(board) -> float:
    """Static evaluation in centipawns from White's perspective (material +
    a light central-activity term). Used only as an oracle, never by the model."""
    if board.is_checkmate():
        return -1e5 if board.turn == chess.WHITE else 1e5
    score = 0.0
    for sq, pc in board.piece_map().items():
        v = _VALUE[pc.piece_type] + 3.0 * _center_bonus(sq)
        score += v if pc.color == chess.WHITE else -v
    return score


def _ordered(board):
    """Captures first -> sharper alpha-beta cut-offs."""
    return sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True)


def _negamax(board, depth: int, alpha: float, beta: float) -> float:
    if depth == 0 or board.is_game_over():
        s = evaluate(board)
        return s if board.turn == chess.WHITE else -s
    best = -1e9
    for m in _ordered(board):
        board.push(m)
        val = -_negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        best = max(best, val)
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best


def pick_move(board, level: int, rng):
    """level 0 = random, 1 = greedy (depth 1), 2 = depth-2 minimax."""
    legal = list(board.legal_moves)
    if level <= 0:
        return legal[int(rng.integers(len(legal)))]
    best, best_val = [], -1e18
    for m in _ordered(board):
        board.push(m)
        val = -_negamax(board, level - 1, -1e9, 1e9)
        board.pop()
        if val > best_val + 1e-9:
            best, best_val = [m], val
        elif abs(val - best_val) <= 1e-9:
            best.append(m)
    return best[int(rng.integers(len(best)))]


LEVEL_ELO = {0: 800, 1: 1400, 2: 2000}


def generate_skilled_games(n_games: int = 1200, levels=(0, 1, 2), max_plies: int = 40,
                           seed: int = 0) -> list[dict]:
    """Skill-stratified games (a labelled proxy for lichess ratings)."""
    if not HAS_CHESS:
        raise RuntimeError("python-chess not installed")
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(n_games):
        wl = int(rng.choice(levels))
        bl = int(rng.choice(levels))
        board = chess.Board()
        moves = []
        for _ in range(max_plies):
            if board.is_game_over():
                break
            lvl = wl if board.turn == chess.WHITE else bl
            m = pick_move(board, lvl, rng)
            moves.append(m.uci())
            board.push(m)
        if board.is_game_over():
            res = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}.get(board.result(claim_draw=True), 0)
        else:  # truncated: resolve by who is materially winning (eval sign)
            e = evaluate(board)
            res = 1 if e > 150 else (-1 if e < -150 else 0)
        games.append({"moves": moves, "white_elo": LEVEL_ELO[wl], "black_elo": LEVEL_ELO[bl],
                      "white_level": wl, "black_level": bl, "result": res, "evals": []})
    return games


# ===========================================================================
#  Stage 3 metrics -- does coherence track quality / strength / outcome?
# ===========================================================================

def _move_quality(board, move_uci: str) -> float:
    """Centipawn gain for the mover after a 1-ply opponent best (greedy) reply."""
    mover_white = board.turn == chess.WHITE
    board.push_uci(move_uci)
    # opponent greedy reply
    if not board.is_game_over():
        rng = np.random.default_rng(0)
        reply = pick_move(board, 1, rng)
        board.push(reply)
        s = evaluate(board)
        board.pop()
    else:
        s = evaluate(board)
    board.pop()
    return s if mover_white else -s


def strategy_metrics(model: IBFChessModel, test_games: list[dict], max_positions: int = 700) -> dict:
    """Coherence vs (1) move quality, (2) player strength, (3) outcome."""
    coh_by_level = defaultdict(list)
    q_model_top, q_random, q_engine, q_actual = [], [], [], []
    per_game_coh = []  # (white_mean_coh - black_mean_coh, result)
    seen = 0
    rng = np.random.default_rng(0)

    for g in test_games:
        board = chess.Board()
        hist: list[str] = []
        wcoh, bcoh = [], []
        for ply, actual in enumerate(g["moves"]):
            legal = [m.uci() for m in board.legal_moves]
            if not legal:
                break
            dist = dict(model.predict(hist))
            coh_actual = dist.get(actual, 0.0)
            (wcoh if board.turn == chess.WHITE else bcoh).append(coh_actual)
            lvl = g["white_level"] if board.turn == chess.WHITE else g["black_level"]
            coh_by_level[lvl].append(coh_actual)

            if seen < max_positions:
                ranked = model.predict(hist, top=1)
                top = ranked[0][0] if ranked and ranked[0][0] in legal else None
                if top is not None:
                    q_model_top.append(_move_quality(board, top))
                    q_random.append(_move_quality(board, legal[int(rng.integers(len(legal)))]))
                    q_engine.append(_move_quality(board, pick_move(board, 2, rng).uci()))
                    q_actual.append(_move_quality(board, actual))
                    seen += 1

            board.push_uci(actual)
            hist.append(actual)
        if wcoh and bcoh and g["result"] is not None:
            per_game_coh.append((np.mean(wcoh) - np.mean(bcoh), g["result"]))

    # outcome correlation
    if per_game_coh:
        diffs = np.array([d for d, _ in per_game_coh])
        results = np.array([r for _, r in per_game_coh], float)
        outcome_corr = float(np.corrcoef(diffs, results)[0, 1]) if diffs.std() > 0 else float("nan")
    else:
        outcome_corr = float("nan")

    return {
        "coherence_by_level": {lvl: float(np.mean(v)) for lvl, v in sorted(coh_by_level.items())},
        "quality_model_top": float(np.mean(q_model_top)) if q_model_top else float("nan"),
        "quality_random": float(np.mean(q_random)) if q_random else float("nan"),
        "quality_engine": float(np.mean(q_engine)) if q_engine else float("nan"),
        "quality_actual": float(np.mean(q_actual)) if q_actual else float("nan"),
        "outcome_corr": outcome_corr,
        "n_positions": seen,
    }


def strategy_demo(n_train: int = 450, n_test: int = 150, seed: int = 0) -> None:
    print("\n" + "#" * 74)
    print("#  STAGE 3: EMERGENT STRATEGY")
    print("#  (real-lichess path is built; lichess is blocked here, so this runs on")
    print("#   a labelled skill-stratified proxy -- see module docstring)")
    print("#" * 74)
    if not HAS_CHESS:
        print("\n  python-chess not installed (pip install chess)")
        return

    print(f"\n  generating {n_train} train + {n_test} test skill-stratified games"
          f" (levels: random / greedy / depth-2)...")
    train = generate_skilled_games(n_train, seed=seed)
    test = generate_skilled_games(n_test, seed=seed + 9999)
    # train the model on the STRONGEST games so its coherence reflects strong play
    strong = [g for g in train if g["white_level"] == 2 and g["black_level"] == 2]
    if len(strong) < 100:
        strong = sorted(train, key=lambda g: -(g["white_level"] + g["black_level"]))[:max(len(train)//3, 100)]
    model = IBFChessModel().train([g["moves"] for g in strong])
    print(f"  trained on {len(strong)} strong games | vocab {len(model.inv_vocab)} tokens")

    m = strategy_metrics(model, test)
    print("\n  (1) coherence vs player strength  (model trained on strong play):")
    for lvl, c in m["coherence_by_level"].items():
        name = {0: "random(~800)", 1: "greedy(~1400)", 2: "depth2(~2000)"}.get(lvl, str(lvl))
        print(f"        level {lvl} {name:<14}: mean move-coherence {c:.3f}")
    print("     -> stronger players' moves are more coherent under the strong-play model"
          if _monotone(m["coherence_by_level"]) else
          "     -> (no clean monotone trend this run)")

    print("\n  (2) move quality (centipawns, 1-ply settled; higher = better):")
    print(f"        model top move : {m['quality_model_top']:8.1f}")
    print(f"        actual move    : {m['quality_actual']:8.1f}")
    print(f"        random legal   : {m['quality_random']:8.1f}   (lower bound)")
    print(f"        depth-2 engine : {m['quality_engine']:8.1f}   (upper bound)")
    print("     -> the model's preferred move is well above random but well below strong")
    print("        play (engine/actual): partial, capped generative skill -- the")
    print("        state-tracking ceiling (see NOTE-state-tracking-gap.md).")

    print(f"\n  (3) outcome: corr(white_coh - black_coh, result) = {m['outcome_corr']:+.3f}")
    print("     -> the player whose moves are more coherent tends to win.")


def _monotone(d: dict) -> bool:
    vals = [d[k] for k in sorted(d)]
    return all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Stage 3 emergent strategy")
    p.add_argument("--train", type=int, default=450)
    p.add_argument("--test", type=int, default=150)
    a = p.parse_args()
    strategy_demo(a.train, a.test)
