"""K+R-vs-K world: exact rules engine + retrograde DTM tablebase (the ORACLE).

This is TEACHER/REFEREE infrastructure (like `chess_oracle.py`): the rules and the
tablebase are never given to the learning agent -- the agent sees opaque squares,
proposes moves, and receives accept/reject + terminal signals only
(`krk_closed_loop.py`). Hand-coding the rules here (instead of python-chess) buys
numpy-vectorised episode speed and a solvable 524k-state tablebase; correctness is
established by EXACT cross-validation against python-chess on sampled positions
(legal-move sets, mate/stalemate flags) plus Bellman-consistency checks on the
solved table, and the known KRK bound (maximum DTM = 16 moves) is verified.

State: (wk, bk, wr, stm) -> idx = ((wk*64 + bk)*64 + wr)*2 + stm; stm 0 = white.
DTM[idx] = plies to mate under optimal play by BOTH sides (white minimises, black
maximises), 0 = black is currently mated, INF (stored -1) = not won.

Run: ``python -m mscn.krk_world``   (builds + validates + caches data/krk_dtm.npz).
"""

from __future__ import annotations

import os

import numpy as np

N_SQ = 64
N_STATES = 64 * 64 * 64 * 2
INF = np.int16(32000)

_KING_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def encode(wk: int, bk: int, wr: int, stm: int) -> int:
    return ((wk * 64 + bk) * 64 + wr) * 2 + stm


def decode(idx: int) -> tuple[int, int, int, int]:
    stm = idx & 1
    idx >>= 1
    wr = idx % 64
    idx //= 64
    bk = idx % 64
    return idx // 64, bk, wr, stm


# ---------------------------------------------------------------------------
#  Vectorised geometry over all 64^3 placements
# ---------------------------------------------------------------------------

def _king_dest_table() -> np.ndarray:
    """[64, 8] destination square per king offset, -1 off-board."""
    t = np.full((64, 8), -1, dtype=np.int32)
    for s in range(64):
        r, f = divmod(s, 8)
        for j, (dr, df) in enumerate(_KING_OFFS):
            rr, ff = r + dr, f + df
            if 0 <= rr < 8 and 0 <= ff < 8:
                t[s, j] = rr * 8 + ff
    return t


def _adjacent_table() -> np.ndarray:
    """[64, 64] bool: squares are king-adjacent (excludes equality)."""
    a = np.zeros((64, 64), dtype=bool)
    kd = _king_dest_table()
    for s in range(64):
        for d in kd[s]:
            if d >= 0:
                a[s, d] = True
    return a


KING_DEST = _king_dest_table()
ADJ = _adjacent_table()


def _rook_attack(wr: np.ndarray, target: np.ndarray, blocker: np.ndarray) -> np.ndarray:
    """Vectorised: does a rook on `wr` attack `target`, with a single possible
    `blocker` (a king) that blocks only if strictly between on the shared line?"""
    wr_r, wr_f = wr // 8, wr % 8
    t_r, t_f = target // 8, target % 8
    b_r, b_f = blocker // 8, blocker % 8
    same_r = (wr_r == t_r) & (wr != target)
    same_f = (wr_f == t_f) & (wr != target)
    # blocker strictly between, on the same line
    blk_r = same_r & (b_r == wr_r) & (np.minimum(wr_f, t_f) < b_f) & (b_f < np.maximum(wr_f, t_f))
    blk_f = same_f & (b_f == wr_f) & (np.minimum(wr_r, t_r) < b_r) & (b_r < np.maximum(wr_r, t_r))
    return (same_r & ~blk_r) | (same_f & ~blk_f)


class KRKTables:
    """All successor tables + terminal flags + the solved DTM, vectorised."""

    def __init__(self) -> None:
        wk, bk, wr = np.meshgrid(np.arange(64), np.arange(64), np.arange(64),
                                 indexing="ij")
        self.wk = wk.ravel().astype(np.int32)      # [64^3] placement arrays
        self.bk = bk.ravel().astype(np.int32)
        self.wr = wr.ravel().astype(np.int32)
        P = self.wk.size                           # 262144 placements

        distinct = ((self.wk != self.bk) & (self.wk != self.wr)
                    & (self.bk != self.wr))
        kings_ok = ~ADJ[self.wk, self.bk]
        self.bk_checked = _rook_attack(self.wr, self.bk, self.wk) & distinct
        self.legal_w = distinct & kings_ok & ~self.bk_checked   # white to move
        self.legal_b = distinct & kings_ok                       # black to move

        # ----- white successors: [P, 8 king slots + 28 rook slots] -> bk-stm
        # placement id of the successor, -1 = illegal slot
        self.w_succ = np.full((P, 36), -1, dtype=np.int32)
        for j in range(8):                         # wK moves
            d = KING_DEST[self.wk, j]
            ok = (d >= 0) & self.legal_w
            ok &= (d != self.wr) & (d != self.bk) & ~ADJ[np.maximum(d, 0), self.bk]
            nid = (np.maximum(d, 0) * 64 + self.bk) * 64 + self.wr
            self.w_succ[:, j] = np.where(ok, nid, -1)
        slot = 8
        for (dr, df) in _ROOK_DIRS:                # wR rays
            r, f = self.wr // 8, self.wr % 8
            open_ray = self.legal_w.copy()
            for step in range(1, 8):
                rr, ff = r + dr * step, f + df * step
                on = (rr >= 0) & (rr < 8) & (ff >= 0) & (ff < 8)
                d = np.where(on, rr * 8 + ff, 0)
                hit_own = on & (d == self.wk)
                hit_bk = on & (d == self.bk)
                ok = open_ray & on & ~hit_own & ~hit_bk
                nid = (self.wk * 64 + self.bk) * 64 + d
                self.w_succ[:, slot] = np.where(ok, nid, -1)
                open_ray &= on & ~hit_own & ~hit_bk   # blockers stop the ray
                slot += 1

        # ----- black successors: [P, 8] -> wk-stm placement id;
        # -2 = rook captured (terminal draw), -1 = illegal
        self.b_succ = np.full((P, 8), -1, dtype=np.int32)
        for j in range(8):
            d = KING_DEST[self.bk, j]
            ok = (d >= 0) & self.legal_b
            dd = np.maximum(d, 0)
            ok &= (dd != self.wk) & ~ADJ[dd, self.wk]
            cap = ok & (dd == self.wr)             # capture the rook -> KK draw
            # non-capture: destination must not be attacked by the rook
            safe = ~_rook_attack(self.wr, dd, self.wk)
            ok_nc = ok & ~cap & safe
            nid = (self.wk * 64 + dd) * 64 + self.wr
            self.b_succ[:, j] = np.where(ok_nc, nid, np.where(cap, -2, -1))

        self.b_has_move = (self.b_succ != -1).any(axis=1)
        self.b_mate = self.legal_b & self.bk_checked & ~self.b_has_move
        self.b_stale = self.legal_b & ~self.bk_checked & ~self.b_has_move
        self.dtm_w, self.dtm_b = self._solve()

    # ----- retrograde solve (value iteration over plies) -----
    def _solve(self) -> tuple[np.ndarray, np.ndarray]:
        P = self.wk.size
        dtm_b = np.full(P, INF, dtype=np.int16)
        dtm_b[self.b_mate] = 0
        dtm_w = np.full(P, INF, dtype=np.int16)
        w_valid = self.w_succ >= 0
        w_succ0 = np.maximum(self.w_succ, 0)
        b_valid = self.b_succ >= 0
        b_draw = self.b_succ == -2                 # rook capture = escape hatch
        b_succ0 = np.maximum(self.b_succ, 0)
        b_movable = self.legal_b & self.b_has_move
        for _ in range(80):
            # white to move: 1 + min over moves of black's dtm
            s = np.where(w_valid, dtm_b[w_succ0], INF)
            new_w = (1 + s.min(axis=1).astype(np.int32)).astype(np.int16)
            new_w[~self.legal_w] = INF
            new_w = np.minimum(new_w, np.int16(INF))
            # black to move: lost only if EVERY option leads to a white win
            t = np.where(b_valid, dtm_w[b_succ0], np.int32(-1))
            all_lost = (np.where(b_valid, dtm_w[b_succ0], 0) < INF).all(axis=1) \
                & ~b_draw.any(axis=1) & b_movable
            new_b = np.where(all_lost, 1 + t.max(axis=1), INF).astype(np.int16)
            new_b[self.b_mate] = 0
            changed = (new_w < dtm_w).any() or (new_b < dtm_b).any()
            dtm_w = np.minimum(dtm_w, new_w)
            dtm_b = np.minimum(dtm_b, new_b)
            if not changed:
                break
        return dtm_w, dtm_b

    # ----- scalar-state interface (referee + oracle) -----
    def pid(self, wk: int, bk: int, wr: int) -> int:
        return (wk * 64 + bk) * 64 + wr

    def is_legal(self, wk, bk, wr, stm) -> bool:
        p = self.pid(wk, bk, wr)
        return bool(self.legal_w[p] if stm == 0 else self.legal_b[p])

    def white_moves(self, wk, bk, wr) -> list[tuple[str, int, int]]:
        """[(piece 'K'/'R', from, to)] legal white moves."""
        p = self.pid(wk, bk, wr)
        out = []
        for j in range(36):
            s = self.w_succ[p, j]
            if s < 0:
                continue
            nwk, nbk, nwr = s // 4096, (s // 64) % 64, s % 64
            out.append(("K", wk, nwk) if nwk != wk else ("R", wr, nwr))
        return out

    def black_moves(self, wk, bk, wr) -> list[tuple[int, int]]:
        """[(to, succ_pid or -2 for rook capture)]."""
        p = self.pid(wk, bk, wr)
        out = []
        for j in range(8):
            s = self.b_succ[p, j]
            if s == -1:
                continue
            d = KING_DEST[bk, j]
            out.append((int(d), int(s)))
        return out

    def dtm(self, wk, bk, wr, stm) -> int:
        p = self.pid(wk, bk, wr)
        v = self.dtm_w[p] if stm == 0 else self.dtm_b[p]
        return -1 if v >= INF else int(v)


_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "krk_dtm.npz")
_TABLES: KRKTables | None = None


def tables() -> KRKTables:
    global _TABLES
    if _TABLES is None:
        _TABLES = KRKTables()
    return _TABLES


# ---------------------------------------------------------------------------
#  Validation: EXACT agreement with python-chess on sampled positions
# ---------------------------------------------------------------------------

def _to_board(wk, bk, wr, stm):
    import chess
    b = chess.Board(None)
    b.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    b.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
    b.turn = chess.WHITE if stm == 0 else chess.BLACK
    return b


def validate(n_samples: int = 1500, seed: int = 0) -> dict:
    import chess
    T = tables()
    rng = np.random.default_rng(seed)
    checked = mates = stales = 0
    # targeted terminal-flag validation (random placements rarely hit mates)
    for flag, pool in (("mate", np.flatnonzero(T.b_mate)),
                       ("stale", np.flatnonzero(T.b_stale))):
        for p in rng.choice(pool, size=min(200, pool.size), replace=False):
            wk, bk, wr = int(p) // 4096, (int(p) // 64) % 64, int(p) % 64
            b = _to_board(wk, bk, wr, 1)
            assert b.is_checkmate() == (flag == "mate"), f"{flag} flag at {p}"
            assert b.is_stalemate() == (flag == "stale"), f"{flag} flag at {p}"
            mates += (flag == "mate")
            stales += (flag == "stale")
    for _ in range(n_samples):
        wk, bk, wr = [int(x) for x in rng.integers(0, 64, 3)]
        stm = int(rng.integers(0, 2))
        b = _to_board(wk, bk, wr, stm)
        pc_legal_pos = b.is_valid()
        ours = T.is_legal(wk, bk, wr, stm)
        assert ours == pc_legal_pos, f"position legality mismatch {(wk, bk, wr, stm)}"
        if not ours:
            continue
        checked += 1
        pc_moves = {(m.from_square, m.to_square) for m in b.legal_moves}
        if stm == 0:
            mine = {(f, t) for _, f, t in T.white_moves(wk, bk, wr)}
        else:
            mine = {(bk, t) for t, _ in T.black_moves(wk, bk, wr)}
        assert mine == pc_moves, \
            f"move-set mismatch at {(wk, bk, wr, stm)}: ours^pc = {mine ^ pc_moves}"
        if stm == 1:
            p = T.pid(wk, bk, wr)
            assert bool(T.b_mate[p]) == b.is_checkmate(), f"mate flag {(wk, bk, wr)}"
            assert bool(T.b_stale[p]) == b.is_stalemate(), f"stale flag {(wk, bk, wr)}"
    return {"checked": checked, "mates_seen": mates, "stales_seen": stales}


def bellman_check(n_samples: int = 4000, seed: int = 1) -> dict:
    """Internal consistency of the solved table: every won WTM state has a move
    decreasing DTM by exactly 1; every lost BTM state's best defence achieves
    exactly its DTM; plus the known KRK bound (max DTM = 16 white moves)."""
    T = tables()
    rng = np.random.default_rng(seed)
    won = np.flatnonzero(self_legal := (T.legal_w & (T.dtm_w < INF)))
    sample = rng.choice(won, size=min(n_samples, won.size), replace=False)
    for p in sample:
        succ = T.w_succ[p]
        best = min(int(T.dtm_b[s]) for s in succ[succ >= 0])
        assert best + 1 == int(T.dtm_w[p]), f"WTM Bellman violated at {p}"
    lost = np.flatnonzero(T.legal_b & (T.dtm_b < INF) & ~T.b_mate)
    sample = rng.choice(lost, size=min(n_samples, lost.size), replace=False)
    for p in sample:
        succ = T.b_succ[p]
        ok = succ[succ >= 0]
        assert (T.b_succ[p] != -2).all(), f"lost BTM had a rook-capture escape {p}"
        worst = max(int(T.dtm_w[s]) for s in ok)
        assert worst + 1 == int(T.dtm_b[p]), f"BTM Bellman violated at {p}"
    max_plies = int(T.dtm_w[T.legal_w & (T.dtm_w < INF)].max())
    frac_won = float((T.dtm_w[T.legal_w] < INF).mean())
    return {"max_dtm_plies": max_plies, "max_dtm_moves": (max_plies + 1) // 2,
            "wtm_won_fraction": frac_won}


def main() -> None:
    import time
    print("\n" + "#" * 74)
    print("#  KRK WORLD -- exact rules engine + retrograde DTM tablebase (oracle)")
    print("#" * 74)
    t0 = time.perf_counter()
    T = tables()
    print(f"\n  built successor tables + solved DTM over {T.wk.size * 2} states "
          f"in {time.perf_counter() - t0:.1f}s")
    v = validate()
    print(f"  python-chess cross-validation: {v['checked']} sampled positions, "
          f"EXACT match on")
    print(f"  position legality, full move sets, mate/stalemate flags "
          f"({v['mates_seen']} mates, {v['stales_seen']} stalemates in sample)")
    b = bellman_check()
    print(f"  Bellman consistency: PASS on sampled won/lost states")
    print(f"  max DTM = {b['max_dtm_plies']} plies = {b['max_dtm_moves']} moves "
          f"(known KRK bound: 16)")
    print(f"  white-to-move won fraction: {b['wtm_won_fraction']:.3f} "
          f"(KRK is almost always won)")
    assert b["max_dtm_moves"] == 16, "the solved table must reproduce the known bound"
    assert b["wtm_won_fraction"] > 0.95
    np.savez_compressed(_CACHE, dtm_w=T.dtm_w, dtm_b=T.dtm_b)
    print(f"  cached DTM to {os.path.normpath(_CACHE)}\n")


if __name__ == "__main__":
    main()
