"""Deeper search on the emergent value -- more strength, then a lever for acc@1.

The context model is itself a *learned move generator* (`sim.vom.predict(hist)`
gives plausible occupancy-valid moves at any node, implicitly respecting movement
patterns). So we can run genuine **negamax / alpha-beta search**: the context model
generates moves at each node, the emergent position value (material + piece-square,
`PositionalIBFAgent`) evaluates leaves, and the occupancy state transfers pieces
(`G`). No hand-coded movement rules.

Two uses:
* **strength** -- `best_move_search` picks the move with the best minimax value
  (deeper tactics than 1-ply SEE);
* **acc@1** -- `predict` mixes the human move-prior with the search value
  `score(m) = log P_context(m) + β·minimax_value(m)`, so deeper, sounder lookahead
  can nudge predictions toward the strong (often forcing) human move.
"""

from __future__ import annotations

import numpy as np

from .chess_positional_ibf import PositionalIBFAgent


class SearchIBFAgent(PositionalIBFAgent):
    def __init__(self, depth: int = 2, branch: int = 6, beta: float = 1.0, **kw) -> None:
        super().__init__(beta=beta, **kw)
        self.depth = depth
        self.branch = branch

    def _moves(self, hist, occ, side):
        occset = set(occ)
        out = []
        for tok, p in self.sim.vom.predict(hist):
            f = self.sim._from.get(tok, -1)
            if f in occset and self.owner.get(occ[f], 0) == side:
                out.append(tok)
                if len(out) >= self.branch:
                    break
        return out

    def _negamax(self, hist, occ, side, depth, alpha, beta):
        if depth == 0:
            return self.position_value(occ, side)
        moves = self._moves(hist, occ, side)
        if not moves:
            return self.position_value(occ, side)
        best = -1e18
        for m in moves:
            v = -self._negamax(hist + [m], self._occ_after(occ, m), 1 - side,
                               depth - 1, -beta, -alpha)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def _root(self, hist):
        occ = {l: (t[1] if isinstance(t, tuple) else t)
               for l, t in self.sim.replay(hist).items()}
        side = len(hist) % 2
        cands = []
        for tok, p in self.sim.vom.predict(hist):
            f = self.sim._from.get(tok, -1)
            if f in set(occ) and self.owner.get(occ[f], 0) == side:
                cands.append((tok, p))
                if len(cands) >= max(self.branch, 12):
                    break
        scored = []
        for tok, p in cands:
            v = -self._negamax(hist + [tok], self._occ_after(occ, tok), 1 - side,
                               self.depth - 1, -1e18, 1e18)
            scored.append((tok, p, v))
        return scored

    # ----- strength: best minimax move -----
    def best_move_search(self, hist, legal):
        scored = [(t, v) for t, p, v in self._root(hist) if t in legal]
        if not scored:
            for t, _ in self.sim.vom.predict(hist):
                if t in legal:
                    return t
            return next(iter(legal))
        return max(scored, key=lambda kv: kv[1])[0]

    # ----- acc@1: context prior nudged by search value -----
    def predict(self, history_tokens, top=None, n_cand=None):
        scored = self._root(history_tokens)
        if not scored:
            return []
        out = [(t, np.log(p + 1e-9) + self.beta * v) for t, p, v in scored]
        out.sort(key=lambda kv: -kv[1])
        return out[:top] if top else out
