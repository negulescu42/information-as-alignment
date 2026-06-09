"""Route A, full -- a simulation-faithful recurrent state (reconstruct the board).

The theorem (`formal/CausalStates.lean`) says a recurrent state is predictively
sufficient iff it **simulates** the process: `σ(G(z,a)) = T(σ(z), a)`. For chess the
sufficient state is the board. We reconstruct it from move tokens *prior-free*:

A move token in from-to form is an **occupancy transfer** — it moves whatever
occupies one opaque location onto another. Replaying transfers reconstructs the
board (occupancy + per-piece *lineage* tags) with **no grid, no piece types, no
rules** given. This is `G` = occupancy transfer and `σ` = identity on the board;
the only structure used beyond atomic tokens is that a move has a source and a
destination location (the "from-to pairs" encoding) — geometry and piece *roles*
still must emerge (Stage 2 shows the grid does).

Because two histories that reach the same physical board collapse to the same state
(transpositions merge), the reconstructed board is a sufficient statistic: it
determines the legal-move set. We verify that directly (purity → 1) and use the
state as a legality mask on the variable-order predictor.

Caveats (minor occupancy inaccuracies, stated honestly): castling moves only the
king token (the rook transfer is implicit), en-passant removes a pawn off the
destination square, and promotions change piece type — none are special-cased, so
purity is ≈1 rather than exactly 1.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .chess_recurrent import RecurrentIBFChessModel


class SimStateChessModel:
    def __init__(self, max_depth: int = 6) -> None:
        self.loc_id: dict[str, int] = {}
        self.start_occ: set[int] = set()
        self.vom = RecurrentIBFChessModel(max_depth=max_depth)
        self.unigram = Counter()
        self.inv_vocab: list[str] = []
        self.vocab: dict[str, int] = {}
        self._from: dict[str, int] = {}   # token -> source location id (opaque)
        self._to: dict[str, int] = {}     # token -> destination location id (opaque)

    def _loc(self, sq: str) -> int:
        if sq not in self.loc_id:
            self.loc_id[sq] = len(self.loc_id)
        return self.loc_id[sq]

    def _parse(self, tok: str) -> tuple[int, int]:
        if tok not in self._from:
            self._from[tok] = self._loc(tok[0:2])
            self._to[tok] = self._loc(tok[2:4])
        return self._from[tok], self._to[tok]

    def train(self, games: list[list[str]]) -> "SimStateChessModel":
        # context coherence (variable-order) for the next-move scores
        self.vom.train(games)
        self.vocab, self.inv_vocab, self.unigram = self.vom.vocab, self.vom.inv_vocab, self.vom.unigram
        # learn the start occupancy: a location used as a source before ever being a
        # destination (in a game) was occupied at the start -> emergent start squares.
        for g in games:
            placed: set[int] = set()
            for tok in g:
                f, t = self._parse(tok)
                if f not in placed:
                    self.start_occ.add(f)
                placed.add(t)
        return self

    # ----- the recurrent simulation-faithful state -----
    def replay(self, tokens: list[str]) -> dict[int, tuple]:
        """z = board occupancy: location -> lineage tag. G = occupancy transfer."""
        occ: dict[int, tuple] = {loc: ("start", loc) for loc in self.start_occ}
        for tok in tokens:
            if tok not in self._from:        # unseen token: still parse its squares
                self._parse(tok)
            f, t = self._from[tok], self._to[tok]
            tag = occ.pop(f, ("start", f))   # whatever occupies f (capture overwrites t)
            occ[t] = tag
        return occ

    def state_key(self, tokens: list[str]):
        return frozenset(self.replay(tokens).items())

    # ----- prediction: variable-order coherence masked by occupancy -----
    def predict(self, history_tokens: list[str], top: int | None = None) -> list[tuple[str, float]]:
        occ = set(self.replay(history_tokens).keys())
        ranked = self.vom.predict(history_tokens)        # (token, prob), descending
        masked = [(tok, p) for tok, p in ranked if self._from.get(tok, -1) in occ]
        if not masked:                                   # fallback: occ-valid global frequents
            masked = [(self.inv_vocab[i], c) for i, c in self.unigram.most_common()
                      if self._from.get(self.inv_vocab[i], -1) in occ]
        z = sum(p for _, p in masked) or 1.0
        out = [(tok, p / z) for tok, p in masked]
        return out[:top] if top else out

    def prob_of(self, history_tokens: list[str], token: str) -> float:
        return dict(self.predict(history_tokens)).get(token, 0.0)

    def move_salience(self) -> dict[str, float]:
        return {self.inv_vocab[i]: c for i, c in self.unigram.items()}
