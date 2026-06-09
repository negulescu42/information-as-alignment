"""Positional coherence layer -- material + emergent piece-square value.

The material-value agent played soundly but un-humanly (acc@1 dropped) because it
ignored *position*. This layer enriches the coherence landscape over board states
with an emergent **piece-square table** (PST): `pst[piece-lineage, location]` = how
much having that piece on that square correlates with winning, learned from game
**outcomes** (no positional knowledge given). Position value becomes

    R(z) = Σ_pieces sign(owner) · ( val[lineage] + pst[lineage, location] ).

Crucially, move selection is **not** pure value-greedy (that crashes human-matching
and chases impossible captures). It mixes the context model's human move-prior with
the value *gradient* — the IBF agency picking coherence-improving moves weighted by
how plausible they are:

    score(m) = log P_context(m)  +  β · ( R(z'_m) − R(z) )      (1-ply material lookahead)

This preserves human-like move-matching while nudging toward value-improving moves.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .chess_value_ibf import ValueIBFAgent, _piece_type


class PositionalIBFAgent(ValueIBFAgent):
    def __init__(self, lr: float = 0.02, lr_pst: float = 0.01, beta: float = 1.0,
                 k: float = 1.5, lookahead: int = 1, epochs: int = 4,
                 pst_sample_every: int = 4, pst_epochs: int = 2) -> None:
        super().__init__(lr=lr, k=k, lookahead=lookahead, epochs=epochs)
        self.lr_pst = lr_pst
        self.beta = beta
        self.pst_sample_every = pst_sample_every
        self.pst_epochs = pst_epochs
        self.pst: dict[tuple, float] = defaultdict(float)   # (lineage, loc) -> positional value

    def train(self, games: list[dict]) -> "PositionalIBFAgent":
        super().train(games)        # sim, owner, material val, index
        # learn the piece-square table from sampled positions vs outcome
        for _ in range(self.pst_epochs):
            for g in games:
                r = g.get("result")
                if r is None:
                    continue
                occ = {loc: loc for loc in self.sim.start_occ}
                for i, tok in enumerate(g["moves"]):
                    occ[self.sim._to[tok]] = occ.pop(self.sim._from[tok], self.sim._from[tok])
                    if i % self.pst_sample_every == 0:
                        pred = sum(self.pst[(occ[l], l)] * (1.0 if self.owner.get(occ[l], 0) == 0 else -1.0)
                                   for l in occ)
                        err = r - pred
                        for l in occ:
                            s = 1.0 if self.owner.get(occ[l], 0) == 0 else -1.0
                            self.pst[(occ[l], l)] += self.lr_pst * err * s
        return self

    def position_value(self, occ: dict[int, int], mover: int) -> float:
        return sum((1.0 if self.owner.get(occ[l], 0) == mover else -1.0)
                   * (self.val[occ[l]] + self.pst[(occ[l], l)]) for l in occ)

    def predict(self, history_tokens: list[str], top: int | None = None,
                n_cand: int = 20) -> list[tuple[str, float]]:
        occ = {loc: (tag[1] if isinstance(tag, tuple) else tag)
               for loc, tag in self.sim.replay(history_tokens).items()}
        occset = set(occ)
        mover = len(history_tokens) % 2
        ctx = [(t, p) for t, p in self.sim.vom.predict(history_tokens)
               if self.sim._from.get(t, -1) in occset][:n_cand]
        if not ctx:
            return []
        base = self.position_value(occ, mover)
        scored = []
        for tok, p in ctx:
            o1 = self._occ_after(occ, tok)
            if self.lookahead >= 1:
                o1 = self._opp_best_capture(o1, 1 - mover)
            gain = self.position_value(o1, mover) - base
            scored.append((tok, np.log(p + 1e-9) + self.beta * gain))
        scored.sort(key=lambda kv: -kv[1])
        out = [(tok, s) for tok, s in scored]
        return out[:top] if top else out

    # ----- interpretation: did positional structure emerge? -----
    def pawn_advancement_value(self) -> dict[int, float]:
        """Mean PST value of pawns by rank (do advanced pawns score higher?)."""
        by_rank: dict[int, list[float]] = defaultdict(list)
        for (lineage, loc), v in self.pst.items():
            ln, cn = self.loc_name.get(lineage), self.loc_name.get(loc)
            if ln and cn and _piece_type(ln) == "P":
                by_rank[int(cn[1])].append(v if self.owner.get(lineage, 0) == 0 else -v)
        return {r: float(np.mean(vs)) for r, vs in sorted(by_rank.items()) if vs}
