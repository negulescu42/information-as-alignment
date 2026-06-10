"""Head-to-head arena -- measure real playing strength (referee-judged).

acc@1 (matching strong-human moves) plateaus for the non-neural prior-free models;
move-quality is confounded by the material oracle. To measure strength *honestly*
we play agents against each other with python-chess as **referee** (legality +
game result only -- never given to the models): each agent proposes its ranked
moves, the referee plays its highest-ranked *legal* move, and the actual game
outcome decides who is stronger.

`see_value(...)` adds static-exchange-evaluation tactics on the emergent piece
values (resolve capture sequences via the occupancy state's attackers), so the
value agent grabs material soundly and avoids hanging pieces. We then check whether
that makes it a stronger *player* than the context-only (move-frequency) agent.
"""

from __future__ import annotations

import numpy as np

try:
    import chess
    HAS_CHESS = True
except Exception:  # pragma: no cover
    HAS_CHESS = False


# ---------------------------------------------------------------------------
#  Static exchange evaluation on the emergent piece values
# ---------------------------------------------------------------------------

def _lva(ag, occ, sq, side):
    best, bv = None, 1e18
    for tid in ag._to_tokens.get(sq, ()):
        f = ag.sim._from[ag.inv_vocab[tid]]
        if f in occ and ag.owner.get(occ[f], 0) == side and ag.val[occ[f]] < bv:
            bv, best = ag.val[occ[f]], f
    return best


def _see(ag, occ, sq, side, depth=0):
    if sq not in occ or depth > 8:
        return 0.0
    f = _lva(ag, occ, sq, side)
    if f is None:
        return 0.0
    captured = ag.val[occ[sq]]
    o2 = dict(occ); o2[sq] = o2.pop(f)
    return max(0.0, captured - _see(ag, o2, sq, 1 - side, depth + 1))


def see_move_value(ag, occ, tok, mover):
    """Material consequence of a move: capture gain (SEE-resolved) minus the
    opponent's best SEE threat on the resulting position."""
    f, t = ag.sim._from[tok], ag.sim._to[tok]
    o2 = dict(occ); gain = 0.0
    if t in o2 and ag.owner.get(o2[t], 0) != mover:
        captured = ag.val[o2[t]]; o2[t] = o2.pop(f)
        gain = captured - _see(ag, o2, t, 1 - mover)
    else:
        o2[t] = o2.pop(f, f)
    threat = 0.0
    for sp in o2:
        if ag.owner.get(o2[sp], 0) == mover:
            s = _see(ag, o2, sp, 1 - mover)
            if s > threat:
                threat = s
    return gain - threat


# ---------------------------------------------------------------------------
#  Move functions (agent -> chosen legal uci)
# ---------------------------------------------------------------------------

def context_move(ag, hist, legal):
    for tok, _ in ag.sim.vom.predict(hist):
        if tok in legal:
            return tok
    return next(iter(legal))


def value_see_move(ag, hist, legal, beta=2.0, n_cand=24):
    occ = {l: (tg[1] if isinstance(tg, tuple) else tg) for l, tg in ag.sim.replay(hist).items()}
    occset = set(occ)
    mover = len(hist) % 2
    ctx = [(t, p) for t, p in ag.sim.vom.predict(hist) if ag.sim._from.get(t, -1) in occset][:n_cand]
    best, best_sc = None, -1e18
    for tok, p in ctx:
        if tok not in legal:
            continue
        sc = np.log(p + 1e-9) + beta * see_move_value(ag, occ, tok, mover)
        if sc > best_sc:
            best_sc, best = sc, tok
    return best if best is not None else context_move(ag, hist, legal)


# ---------------------------------------------------------------------------
#  Arena
# ---------------------------------------------------------------------------

def play_game(white_fn, black_fn, max_plies=160, seed=0, rand_open=0):
    rng = np.random.default_rng(seed)
    board = chess.Board()
    hist = []
    for _ in range(rand_open):            # random (but legal) opening so games are distinct
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        mv = legal[int(rng.integers(len(legal)))].uci()
        board.push_uci(mv); hist.append(mv)
    for ply in range(max_plies):
        if board.is_game_over():
            break
        legal = {m.uci() for m in board.legal_moves}
        fn = white_fn if board.turn == chess.WHITE else black_fn
        mv = fn(hist, legal)
        if mv not in legal:
            mv = list(legal)[int(rng.integers(len(legal)))]
        board.push_uci(mv); hist.append(mv)
    if board.is_checkmate():
        return -1 if board.turn == chess.WHITE else 1     # side just mated loses
    # otherwise decide by material (referee), prior-free outcome proxy
    v = sum((1 if p.color == chess.WHITE else -1) *
            {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}[p.piece_type]
            for p in board.piece_map().values())
    return 1 if v > 1 else (-1 if v < -1 else 0)


def arena(fn_a, fn_b, n_games=40, max_plies=160, rand_open=0) -> dict:
    """A vs B, alternating colours. Returns A's score (win=1, draw=0.5). With
    ``rand_open`` random opening plies the games are distinct (deterministic agents
    otherwise collapse to 2 repeated lines, making the score statistically empty)."""
    wins = draws = losses = 0
    for g in range(n_games):
        if g % 2 == 0:
            r = play_game(fn_a, fn_b, max_plies, seed=g, rand_open=rand_open)
            res = r
        else:
            r = play_game(fn_b, fn_a, max_plies, seed=g, rand_open=rand_open)
            res = -r
        if res > 0:
            wins += 1
        elif res < 0:
            losses += 1
        else:
            draws += 1
    n = n_games
    return {"A_wins": wins, "draws": draws, "B_wins": losses,
            "A_score": (wins + 0.5 * draws) / n}
