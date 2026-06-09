"""Position-value coherence layer -- climbing a coherence landscape over boards.

The board-conditioned agent showed that *association* over the board doesn't give
strong play: ranking needs position **value**. This layer learns a coherence
function over the sufficient board state -- `R(z)` = how winning a position is --
from game **outcomes**, then selects the move whose resulting board has highest
coherence (Boltzmann-`k`). That is the IBF gradient-flow / basin formulation
(MSCN Layer-1) applied to the *position* rather than a 1-D landscape.

Everything stays prior-free. The board `z` is the reconstructed occupancy+lineage
state (`chess_simstate.py`). Coherence is `R(z) = Σ_pieces sign(owner)·val[lineage]`
-- a learned material value, where:
  * a piece's **owner side** is inferred from move parity (a lineage first moved on
    an even ply is White's), and
  * each piece's **value** `val[lineage]` is learned by regressing board material
    onto the game result (Widrow-Hoff / IBF discrepancy modification).
So **piece values emerge from outcomes** -- no values, types, rules, or board
given. Move selection does a 1-ply material lookahead (the opponent's best capture)
on this learned value, i.e. it climbs the learned coherence landscape.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .chess_simstate import SimStateChessModel

# start-square -> piece type, used ONLY to *interpret* the emergent values (never by
# the model). a1..h1 = R N B Q K B N R ; a2..h2 = pawns ; mirror for black.
_BACK = ["R", "N", "B", "Q", "K", "B", "N", "R"]


def _piece_type(sqname: str) -> str:
    file = "abcdefgh".index(sqname[0])
    rank = int(sqname[1])
    if rank in (1, 8):
        return _BACK[file]
    if rank in (2, 7):
        return "P"
    return "?"


class ValueIBFAgent:
    def __init__(self, lr: float = 0.02, k: float = 1.5, lookahead: int = 1,
                 epochs: int = 4) -> None:
        self.sim = SimStateChessModel(max_depth=2)
        self.lr = lr
        self.k = k
        self.lookahead = lookahead
        self.epochs = epochs
        self.val: dict[int, float] = defaultdict(float)     # lineage -> value
        self.owner: dict[int, int] = {}                     # lineage -> side (0 white, 1 black)
        self.loc_name: dict[int, str] = {}                  # loc id -> square name (for interpretation)

    def _occ_after(self, occ: dict[int, int], tok: str) -> dict[int, int]:
        f, t = self.sim._from[tok], self.sim._to[tok]
        o = dict(occ)
        o[t] = o.pop(f, f)
        return o

    def position_value(self, occ: dict[int, int], mover: int) -> float:
        return sum(self.val[p] * (1.0 if self.owner.get(p, 0) == mover else -1.0)
                   for p in occ.values())

    def train(self, games: list[dict]) -> "ValueIBFAgent":
        moves_only = [g["moves"] for g in games]
        self.sim.train(moves_only)
        for sq, i in self.sim.loc_id.items():
            self.loc_name[i] = sq

        # (1) infer each piece's owner side from move parity (first time it moves)
        votes: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for g in moves_only:
            occ = {loc: loc for loc in self.sim.start_occ}
            for i, tok in enumerate(g):
                f = self.sim._from[tok]
                votes[occ.get(f, f)][i % 2] += 1
                occ[self.sim._to[tok]] = occ.pop(f, f)
        self.owner = {p: (0 if v[0] >= v[1] else 1) for p, v in votes.items()}

        # (2) learn piece values by regressing the FINAL board material onto the
        #     result (Widrow-Hoff = IBF discrepancy modification of the value
        #     coherence). The final position carries the decisive material imbalance;
        #     regressing over *all* positions instead weights pieces by capture
        #     frequency (pawns) rather than importance and mis-ranks the queen.
        finals = []
        for g in games:
            r = g.get("result")
            if r is None:
                continue
            occ = {loc: loc for loc in self.sim.start_occ}
            for tok in g["moves"]:
                occ[self.sim._to[tok]] = occ.pop(self.sim._from[tok], self.sim._from[tok])
            finals.append((set(occ.values()), r))
        for _ in range(self.epochs):
            for alive, r in finals:
                pred = sum(self.val[p] * (1.0 if self.owner.get(p, 0) == 0 else -1.0)
                           for p in alive)
                err = r - pred
                for p in alive:
                    self.val[p] += self.lr * err * (1.0 if self.owner.get(p, 0) == 0 else -1.0)
        return self.build_index()

    # ----- move selection: climb the value landscape -----
    def _opp_best_capture(self, occ: dict[int, int], opp: int) -> dict[int, int]:
        """Opponent plays its highest-value capture (1-ply material lookahead)."""
        best, best_gain = None, 0.0
        occset = set(occ)
        for tloc in occset:                              # one of the mover's pieces
            if self.owner.get(occ[tloc], 0) == opp:
                continue
            gain = self.val[occ[tloc]]
            if gain <= best_gain:
                continue
            for tid in self._to_tokens.get(tloc, ()):    # an opponent token capturing onto tloc
                f = self.sim._from[self.inv_vocab[tid]]
                if f in occset and self.owner.get(occ[f], 0) == opp:
                    best_gain, best = gain, self.inv_vocab[tid]
                    break
        return self._occ_after(occ, best) if best is not None else occ

    def predict(self, history_tokens: list[str], top: int | None = None,
                n_cand: int = 20) -> list[tuple[str, float]]:
        """Rank by resulting-position value *among context-plausible moves*.

        Pure value-greedy fails (the occupancy state has no movement legality, so it
        chases impossible high-value captures). Restricting to the context model's
        plausible occupancy-valid moves keeps choices legal; ranking them by the
        learned position value (with a 1-ply material lookahead) gives sound play.
        """
        occ = {loc: (tag[1] if isinstance(tag, tuple) else tag)
               for loc, tag in self.sim.replay(history_tokens).items()}
        occset = set(occ)
        mover = len(history_tokens) % 2
        ctx = [t for t, _ in self.sim.vom.predict(history_tokens)
               if self.sim._from.get(t, -1) in occset][:n_cand]
        if not ctx:
            return []
        scores = []
        for tok in ctx:
            o1 = self._occ_after(occ, tok)
            if self.lookahead >= 1:
                o1 = self._opp_best_capture(o1, 1 - mover)
            scores.append(self.position_value(o1, mover))
        scores = np.array(scores)
        logits = self.k * (scores - scores.max())
        p = np.exp(logits); p /= p.sum()
        order = np.argsort(-p)
        out = [(ctx[i], float(p[i])) for i in order]
        return out[:top] if top else out

    def prob_of(self, history_tokens, token):
        return dict(self.predict(history_tokens)).get(token, 0.0)

    def move_salience(self):
        return self.sim.move_salience()

    @property
    def inv_vocab(self):
        return self.sim.inv_vocab

    @property
    def vocab(self):
        return self.sim.vocab

    @property
    def unigram(self):
        return self.sim.unigram

    def build_index(self) -> "ValueIBFAgent":
        self._from_tokens: dict[int, list[int]] = defaultdict(list)
        self._to_tokens: dict[int, list[int]] = defaultdict(list)
        for tok in self.sim._from:
            self._from_tokens[self.sim._from[tok]].append(self.sim.vocab[tok])
            self._to_tokens[self.sim._to[tok]].append(self.sim.vocab[tok])
        return self

    # ----- emergent piece values (interpretation) -----
    def emergent_values(self) -> dict[str, float]:
        by_type: dict[str, list[float]] = defaultdict(list)
        for lineage, v in self.val.items():
            name = self.loc_name.get(lineage)
            if name:
                by_type[_piece_type(name)].append(abs(v))
        return {t: float(np.mean(vs)) for t, vs in by_type.items() if vs}
