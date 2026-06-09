"""Board-conditioned IBF agent -- coherence over the sufficient board state.

Route A gave us a *sufficient* recurrent state (the reconstructed board,
`chess_simstate.py`). This agent conditions the IBF coherence on that state: it
learns a **board-conditioned modification** `δR(piece, destination)` — for each
piece (tracked prior-free by its *lineage*, i.e. the location it started on) it
accumulates coherence over the destinations it moves to — with the full IBF-unit
dynamics (online discrepancy reinforcement `δR' = α·max(T_ext−R_eff,0) − μ·δR`,
selective-retention decay, adaptive responsiveness `k`/agency).

Prediction mixes three signals, all masked to occupancy-valid moves:
* the move-context coherence (variable-order, good in the opening),
* the **board-conditioned** piece→destination coherence (generalises across move
  orders / transpositions — the new signal the board enables),
* the occupancy mask (only pieces that exist can move).

`λ` mixes context vs board-conditioned. `λ=0` recovers the sim-state-masked
variable-order model; raising `λ` adds the board signal. We retest next-move
accuracy and move quality (the strategy signal the board enables).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .chess_simstate import SimStateChessModel


class BoardConditionedIBFAgent:
    def __init__(self, max_depth: int = 6, alpha: float = 0.5, mu: float = 0.02,
                 k: float = 2.0, k_adapt: float = 0.001, k_max: float = 8.0,
                 T_ext: float = 4.0, lam: float = 0.5) -> None:
        self.sim = SimStateChessModel(max_depth)
        self.alpha, self.mu, self.T_ext, self.lam = alpha, mu, T_ext, lam
        self.k, self.k0, self.k_adapt, self.k_max = k, k, k_adapt, k_max
        self.pd: dict[int, dict[int, float]] = defaultdict(dict)   # lineage -> {to_loc: δR}
        self.from_tokens: dict[int, list[int]] = defaultdict(list)  # from_loc -> [token_id]
        self._agency = 0
        self._steps = 0

    @property
    def inv_vocab(self):
        return self.sim.inv_vocab

    @property
    def vocab(self):
        return self.sim.vocab

    @property
    def unigram(self):
        return self.sim.unigram

    def train(self, games: list[list[str]]) -> "BoardConditionedIBFAgent":
        self.sim.train(games)
        # index tokens by their (opaque) source location
        for tok, fid in self.sim._from.items():
            self.from_tokens[fid].append(self.sim.vocab[tok])
        # online board-conditioned modification pass (full IBF-unit dynamics)
        for g in games:
            occ = {loc: loc for loc in self.sim.start_occ}   # location -> lineage (start loc)
            for tok in g:
                f, t = self.sim._from[tok], self.sim._to[tok]
                p = occ.get(f, f)                            # moving piece's lineage
                d = self.pd[p]
                old = d.get(t, 0.0)
                disc = max(self.T_ext - old, 0.0)
                self._steps += 1
                if old >= 0.5 * self.T_ext:                  # agency: already coherent
                    self.k = min(self.k + self.k_adapt, self.k_max)
                    self._agency += 1
                d[t] = max(old + self.alpha * disc - self.mu * old, 0.0)
                if self.mu > 0 and len(d) > 1:               # selective retention
                    for tt in d:
                        if tt != t:
                            d[tt] *= (1.0 - self.mu)
                occ[t] = occ.pop(f, f)                       # G: occupancy transfer
        return self

    def _board_dist(self, occ: dict[int, int], occset: set[int]) -> dict[int, float]:
        """P_pd(move | board): board-conditioned piece→destination, occupancy-masked."""
        scores: dict[int, float] = {}
        for floc in occset:
            p = occ[floc]
            dd = self.pd.get(p)
            if not dd:
                continue
            for tid in self.from_tokens.get(floc, ()):
                v = dd.get(self.sim._to[self.inv_vocab[tid]], 0.0)
                if v > 0:
                    scores[tid] = v
        z = sum(scores.values())
        return {m: v / z for m, v in scores.items()} if z > 0 else {}

    def predict(self, history_tokens: list[str], top: int | None = None) -> list[tuple[str, float]]:
        occ = {loc: tag[1] if isinstance(tag, tuple) else tag
               for loc, tag in self.sim.replay(history_tokens).items()}
        occset = set(occ.keys())
        # context distribution (variable-order), occupancy-masked
        ctx = {self.vocab[t]: p for t, p in self.sim.vom.predict(history_tokens)
               if t in self.vocab and self.sim._from.get(t, -1) in occset}
        zc = sum(ctx.values())
        if zc > 0:
            ctx = {m: p / zc for m, p in ctx.items()}
        board = self._board_dist(occ, occset)
        cands = set(ctx) | set(board)
        mixed = {m: (1 - self.lam) * ctx.get(m, 0.0) + self.lam * board.get(m, 0.0) for m in cands}
        zt = sum(mixed.values())
        if zt <= 0:                                          # fallback: occ-valid frequents
            mixed = {self.vocab[t]: c for t, c in self.unigram.most_common()
                     if self.sim._from.get(t, -1) in occset}
            zt = sum(mixed.values()) or 1.0
        ranked = sorted(mixed.items(), key=lambda kv: -kv[1])
        out = [(self.inv_vocab[m], p / zt) for m, p in ranked]
        return out[:top] if top else out

    def prob_of(self, history_tokens: list[str], token: str) -> float:
        return dict(self.predict(history_tokens)).get(token, 0.0)

    def move_salience(self) -> dict[str, float]:
        return self.sim.move_salience()

    def agency_rate(self) -> float:
        return self._agency / max(self._steps, 1)
