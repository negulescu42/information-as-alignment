"""A pure-Python reference chess engine -- the training ORACLE (Stockfish-proxy).

The user's program: train the IBF chess player by **reducing the discrepancy to a
strong oracle** (Postulate IV, `δR' = α·discrepancy − μ·δR`). The true Stockfish binary
is unavailable in this PyPI-only sandbox (no binary, apt/network blocked), so the oracle
here is a self-contained alpha-beta engine with a real evaluation (material + piece-
square + mobility + king safety) and quiescence. It is **far stronger than the IBF's
emergent material+PST eval** -- a meaningful teacher -- though not Stockfish-strength;
labelled honestly as a proxy. It uses `python-chess` for rules (it is the *teacher*; the
IBF model is never given rules -- it only sees the oracle's evaluations/moves).

`OracleEngine.evaluate(board)` -> cp (White's perspective); `best_move(board, depth)`;
`eval_for_mover(board)` -> cp from the side-to-move's perspective (the IBF training target).
"""

from __future__ import annotations

import chess

# material (centipawns)
_VAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

# compact piece-square tables (White's view, a1=0 .. h8=63), midgame-ish
_PST = {
    chess.PAWN: [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, -20, -20, 10, 10, 5,
                 5, -5, -10, 0, 0, -10, -5, 5, 0, 0, 0, 20, 20, 0, 0, 0,
                 5, 5, 10, 25, 25, 10, 5, 5, 10, 10, 20, 30, 30, 20, 10, 10,
                 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0],
    chess.KNIGHT: [-50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 5, 5, 0, -20, -40,
                   -30, 5, 10, 15, 15, 10, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30,
                   -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 10, 15, 15, 10, 0, -30,
                   -40, -20, 0, 0, 0, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50],
    chess.BISHOP: [-20, -10, -10, -10, -10, -10, -10, -20, -10, 5, 0, 0, 0, 0, 5, -10,
                   -10, 10, 10, 10, 10, 10, 10, -10, -10, 0, 10, 10, 10, 10, 0, -10,
                   -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 5, 10, 10, 5, 0, -10,
                   -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -10, -10, -10, -10, -20],
    chess.ROOK: [0, 0, 0, 5, 5, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5,
                 5, 10, 10, 10, 10, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    chess.QUEEN: [-20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 5, 0, 0, 0, 0, -10,
                  -10, 5, 5, 5, 5, 5, 0, -10, 0, 0, 5, 5, 5, 5, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5,
                  -10, 0, 5, 5, 5, 5, 0, -10, -10, 0, 0, 0, 0, 0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20],
    chess.KING: [20, 30, 10, 0, 0, 10, 30, 20, 20, 20, 0, 0, 0, 0, 20, 20,
                 -10, -20, -20, -20, -20, -20, -20, -10, -20, -30, -30, -40, -40, -30, -30, -20,
                 -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30],
}


class OracleEngine:
    def __init__(self, depth: int = 3, qcap: int = 4, mobility_w: float = 2.0,
                 kingsafe_w: float = 6.0) -> None:
        self.depth = depth
        self.qcap = qcap
        self.mobility_w = mobility_w
        self.kingsafe_w = kingsafe_w

    # ----- evaluation (cp, White's perspective) -----
    def evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -100000 if board.turn == chess.WHITE else 100000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        score = 0.0
        for sq, pc in board.piece_map().items():
            v = _VAL[pc.piece_type]
            idx = sq if pc.color == chess.WHITE else chess.square_mirror(sq)
            v += _PST[pc.piece_type][idx]
            score += v if pc.color == chess.WHITE else -v
        # mobility (cheap, side-to-move agnostic estimate)
        wm = board.legal_moves.count() if board.turn == chess.WHITE else 0
        b2 = board.copy(stack=False); b2.turn = not board.turn
        om = b2.legal_moves.count()
        mob = (wm - om) if board.turn == chess.WHITE else (om - wm)
        score += self.mobility_w * mob
        # king safety: enemy attackers adjacent to each king
        for color in (chess.WHITE, chess.BLACK):
            ks = board.king(color)
            if ks is None:
                continue
            atk = 0
            for sq in board.attacks(ks) | {ks}:
                atk += len(board.attackers(not color, sq))
            score += (-self.kingsafe_w * atk) if color == chess.WHITE else (self.kingsafe_w * atk)
        return score

    def eval_for_mover(self, board: chess.Board) -> float:
        s = self.evaluate(board)
        return s if board.turn == chess.WHITE else -s

    # ----- search -----
    def _ordered(self, board):
        return sorted(board.legal_moves, key=lambda m: board.is_capture(m), reverse=True)

    def _quiesce(self, board, alpha, beta, qd):
        stand = self.eval_for_mover(board)
        if stand >= beta:
            return beta
        if stand > alpha:
            alpha = stand
        if qd >= self.qcap:
            return alpha
        for m in board.legal_moves:
            if not board.is_capture(m):
                continue
            board.push(m)
            v = -self._quiesce(board, -beta, -alpha, qd + 1)
            board.pop()
            if v >= beta:
                return beta
            if v > alpha:
                alpha = v
        return alpha

    def _negamax(self, board, depth, alpha, beta):
        if board.is_game_over():
            return self.eval_for_mover(board)
        if depth == 0:
            return self._quiesce(board, alpha, beta, 0)
        best = -1e9
        for m in self._ordered(board):
            board.push(m)
            v = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def best_move(self, board: chess.Board, depth: int | None = None) -> str:
        depth = self.depth if depth is None else depth
        best, best_v = None, -1e9
        for m in self._ordered(board):
            board.push(m)
            v = -self._negamax(board, depth - 1, -1e9, 1e9)
            board.pop()
            if v > best_v:
                best_v, best = v, m
        return best.uci() if best else next(iter(m.uci() for m in board.legal_moves))

    def move_fn(self, depth: int | None = None):
        """An arena-compatible move function (hist, legal) -> uci, replaying hist."""
        def fn(hist, legal):
            board = chess.Board()
            for u in hist:
                try:
                    board.push_uci(u)
                except Exception:
                    break
            mv = self.best_move(board, depth)
            return mv if mv in legal else next(iter(legal))
        return fn


if __name__ == "__main__":
    eng = OracleEngine(depth=3)
    b = chess.Board()
    print("startpos best move (depth 3):", eng.best_move(b))
    b.push_uci("e2e4"); b.push_uci("e7e5")
    print("after 1.e4 e5 eval (cp, white):", eng.evaluate(b), "| best:", eng.best_move(b))
