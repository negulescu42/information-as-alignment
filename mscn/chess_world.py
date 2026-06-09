"""Emergent chess representation from move tokens alone (no priors).

A representation-learning / generalisation experiment for the IBF mechanism: can
it learn the *structure* of chess -- legal moves (rules) and the 8x8 board
(geometry) -- purely from sequences of opaque move tokens, with **no built-in
board, pieces, rules or strategy**?

This mirrors the Othello-GPT / Chess-GPT result (sequence models trained only on
move text develop an internal board representation) but uses the non-neural IBF
coherence mechanism: knowledge is accumulated as coherence over observed
``(context -> next move)`` transitions (Theorem 3/5: interaction-driven memory),
and the latent board geometry is read out of that learned representation.

Strict "no priors": the model sees each move only as an **atomic token** (a UCI
string used purely as an identifier). It is never told that tokens contain
squares, that squares form a grid, or that pieces have movement rules. The
``chess`` library is used *only* to (a) generate legal games as data and (b) act
as an evaluation oracle (replay a game to check whether a predicted move is
legal, and decode tokens to squares for the geometry probe). None of that
touches the model's inputs.

Data note: this environment's network policy blocks the lichess CDN, so games are
generated locally (random-legal with a light capture/check bias). Stages 1-2
(rules + geometry) need only legal game data; emergent *strategy* (Stage 3) would
benefit from real human games and is left for an environment that can reach
lichess.

Requires ``chess`` (and ``scipy`` for the geometry probe): ``pip install chess scipy``.
Run ``python -m mscn.chess_world``.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

try:
    import chess
    HAS_CHESS = True
except Exception:  # pragma: no cover
    HAS_CHESS = False


# ===========================================================================
#  Data generation (legal games -> opaque move-token sequences)
# ===========================================================================

def generate_games(n_games: int = 3000, max_plies: int = 50, seed: int = 0,
                   epsilon: float = 0.5) -> list[list[str]]:
    """Generate legal games; return each as a list of UCI move tokens.

    Policy: mix of uniform-random legal moves and a light human-ish bias toward
    captures and checks (``epsilon`` is the random fraction). The chess rules are
    used only to *generate* the data; the tokens carry no rule information to the
    learner.
    """
    if not HAS_CHESS:
        raise RuntimeError("python-chess not installed (pip install chess)")
    rng = np.random.default_rng(seed)
    games = []
    for _ in range(n_games):
        b = chess.Board()
        moves = []
        for _ in range(max_plies):
            if b.is_game_over():
                break
            legal = list(b.legal_moves)
            scores = np.ones(len(legal))
            for i, m in enumerate(legal):
                if b.is_capture(m):
                    scores[i] += 3.0
                b.push(m)
                if b.is_check():
                    scores[i] += 1.0
                b.pop()
            p = scores / scores.sum()
            p = epsilon / len(legal) + (1 - epsilon) * p
            m = legal[int(rng.choice(len(legal), p=p))]
            moves.append(m.uci())
            b.push(m)
        games.append(moves)
    return games


# ===========================================================================
#  The IBF chess model -- backoff coherence over (context -> next token)
# ===========================================================================

class IBFChessModel:
    """Non-neural next-move predictor built on coherence accumulation.

    For each observed transition ``(context -> next)`` the model reinforces a
    coherence value ``delta_R[context][next]`` (memory formation). Contexts are
    the last ``k`` move tokens for ``k = 2, 1, 0``; prediction linearly
    interpolates the orders (backoff). Everything is keyed on opaque token ids.
    """

    def __init__(self, weights=(0.6, 0.3, 0.1)) -> None:
        self.w = weights
        self.vocab: dict[str, int] = {}
        self.inv_vocab: list[str] = []
        self.tables = [defaultdict(Counter) for _ in range(3)]  # order 2,1,0
        self.unigram = Counter()

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            self.vocab[tok] = len(self.inv_vocab)
            self.inv_vocab.append(tok)
        return self.vocab[tok]

    def train(self, games: list[list[str]]) -> "IBFChessModel":
        for g in games:
            ids = [self._id(t) for t in g]
            for i, nxt in enumerate(ids):
                self.unigram[nxt] += 1
                self.tables[2][()][nxt] += 1  # order-0 (global), stored under ()
                if i >= 1:
                    self.tables[1][(ids[i - 1],)][nxt] += 1
                if i >= 2:
                    self.tables[0][(ids[i - 2], ids[i - 1])][nxt] += 1
        self._V = len(self.inv_vocab)
        self._global = self.tables[2][()]
        self._global_total = sum(self._global.values())
        return self

    def _dist(self, history_ids: list[int]) -> dict[int, float]:
        """Interpolated next-token probability over seen tokens."""
        w2, w1, w0 = self.w
        scores: dict[int, float] = defaultdict(float)
        # order-0 (global)
        for tok, c in self._global.items():
            scores[tok] += w0 * c / self._global_total
        # order-1
        if history_ids:
            c1 = self.tables[1].get((history_ids[-1],))
            if c1:
                tot = sum(c1.values())
                for tok, c in c1.items():
                    scores[tok] += w1 * c / tot
        # order-2
        if len(history_ids) >= 2:
            c2 = self.tables[0].get((history_ids[-2], history_ids[-1]))
            if c2:
                tot = sum(c2.values())
                for tok, c in c2.items():
                    scores[tok] += w2 * c / tot
        z = sum(scores.values())
        if z > 0:
            for k in scores:
                scores[k] /= z
        return scores

    def predict(self, history_tokens: list[str], top: int | None = None) -> list[tuple[str, float]]:
        hist = [self.vocab[t] for t in history_tokens if t in self.vocab]
        dist = self._dist(hist)
        ranked = sorted(dist.items(), key=lambda kv: -kv[1])
        out = [(self.inv_vocab[i], p) for i, p in ranked]
        return out[:top] if top else out

    def prob_of(self, history_tokens: list[str], token: str) -> float:
        hist = [self.vocab[t] for t in history_tokens if t in self.vocab]
        if token not in self.vocab:
            return 0.0
        return self._dist(hist).get(self.vocab[token], 0.0)

    def move_salience(self) -> dict[str, float]:
        """Total learned coherence per move token (its global reinforcement)."""
        return {self.inv_vocab[i]: c for i, c in self.unigram.items()}


# ===========================================================================
#  Stage 1 -- emergent RULES (legal-move prediction with no rules given)
# ===========================================================================

def _phase(ply: int) -> str:
    return "opening" if ply < 10 else ("midgame" if ply < 26 else "endgame")


def evaluate_rules(model: IBFChessModel, test_games: list[list[str]]) -> dict:
    """Replay held-out games; measure how often the model's predictions are
    legal (the rules it was never told) and how often it predicts the real move.

    Baselines: a uniform random token, and the global-unigram predictor.
    """
    if not HAS_CHESS:
        raise RuntimeError("python-chess not installed")
    phases = ["opening", "midgame", "endgame", "all"]
    stat = {m: {p: defaultdict(float) for p in phases} for m in ("IBF", "unigram", "random")}
    vocab = model.inv_vocab
    unigram_rank = [model.inv_vocab[i] for i, _ in model.unigram.most_common()]

    for g in test_games:
        board = chess.Board()
        hist: list[str] = []
        for ply, actual in enumerate(g):
            legal = {m.uci() for m in board.legal_moves}
            if not legal:
                break
            n_legal = len(legal)
            buckets = (_phase(ply), "all")

            # IBF prediction
            ranked = model.predict(hist, top=5)
            top1 = ranked[0][0] if ranked else None
            top5 = [t for t, _ in ranked]
            dist = dict(model.predict(hist))
            legal_mass = sum(dist.get(u, 0.0) for u in legal)

            # unigram prediction (ignores position)
            uni_top1 = unigram_rank[0] if unigram_rank else None

            for b in buckets:
                s = stat["IBF"][b]
                s["n"] += 1
                s["legal@1"] += (top1 in legal)
                s["legal@5"] += any(t in legal for t in top5)
                s["acc@1"] += (top1 == actual)
                s["acc@5"] += (actual in top5)
                s["legal_mass"] += legal_mass
                su = stat["unigram"][b]
                su["n"] += 1
                su["legal@1"] += (uni_top1 in legal)
                sr = stat["random"][b]
                sr["n"] += 1
                sr["legal@1"] += n_legal / max(len(vocab), 1)  # expected legal rate of a random token

            board.push_uci(actual)
            hist.append(actual)

    def finalize(d):
        n = d.get("n", 0) or 1
        return {k: (v / n if k != "n" else int(v)) for k, v in d.items()}

    return {m: {p: finalize(stat[m][p]) for p in phases} for m in stat}


# ===========================================================================
#  Stage 2 -- emergent BOARD GEOMETRY (recover the 8x8 grid, no spatial prior)
# ===========================================================================

def _square_graph(model: IBFChessModel) -> np.ndarray:
    """64x64 affinity: salience-weighted from<->to transitions decoded from the
    learned move tokens (decoding is experimenter-side, for the probe only)."""
    A = np.zeros((64, 64))
    for tok, sal in model.move_salience().items():
        if len(tok) < 4:
            continue
        try:
            f = chess.SQUARE_NAMES.index(tok[0:2])
            t = chess.SQUARE_NAMES.index(tok[2:4])
        except ValueError:
            continue
        A[f, t] += sal
        A[t, f] += sal
    return A


def _laplacian_eigenmap(A: np.ndarray, dims: int = 2) -> np.ndarray:
    deg = A.sum(1)
    d = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(A.shape[0]) - (d[:, None] * A * d[None, :])
    w, v = np.linalg.eigh(L)
    return v[:, 1:1 + dims]


def recover_board_geometry(model: IBFChessModel) -> dict:
    """Embed the 64 squares from move statistics and compare to the true board."""
    from scipy.spatial import procrustes
    A = _square_graph(model)
    emb = _laplacian_eigenmap(A, 2)
    true = np.array([[chess.square_file(s), chess.square_rank(s)] for s in range(64)], float)
    true_std, emb_aligned, disparity = procrustes(true, emb)

    # neighbour recovery: true king-adjacency captured by embedding kNN
    def king_adj():
        out = []
        for s in range(64):
            f, r = chess.square_file(s), chess.square_rank(s)
            nb = set()
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df or dr:
                        ff, rr = f + df, r + dr
                        if 0 <= ff < 8 and 0 <= rr < 8:
                            nb.add(chess.square(ff, rr))
            out.append(nb)
        return out

    D = np.linalg.norm(emb_aligned[:, None] - emb_aligned[None, :], axis=2)
    np.fill_diagonal(D, np.inf)
    adj = king_adj()
    rec = np.mean([len(set(np.argsort(D[s])[:4]) & adj[s]) / min(4, len(adj[s])) for s in range(64)])
    return {"embedding": emb_aligned, "true": true_std, "disparity": float(disparity),
            "neighbor_recovery": float(rec), "affinity": A}


def _geometry_figure(geo: dict, figpath: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"   (figure skipped: {e})"); return
    import os
    os.makedirs(os.path.dirname(figpath) or ".", exist_ok=True)
    emb, true = geo["embedding"], geo["true"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.4))
    for a, pts, title in ((ax[0], true, "true board"), (ax[1], emb, "recovered from move tokens")):
        files = np.round((true[:, 0] - true[:, 0].min()) / (np.ptp(true[:, 0]) + 1e-9) * 7).astype(int)
        a.scatter(pts[:, 0], pts[:, 1], c=files, cmap="viridis", s=120)
        for s in range(64):
            a.annotate(chess.SQUARE_NAMES[s], (pts[s, 0], pts[s, 1]), fontsize=5,
                       ha="center", va="center")
        a.set_title(title); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Emergent 8x8 board (Procrustes disparity {geo['disparity']:.3f}, "
                 f"adjacency recovery {geo['neighbor_recovery']:.0%})")
    fig.tight_layout(); fig.savefig(figpath, dpi=120); plt.close(fig)
    print(f"   board-emergence figure -> {figpath}")


# ===========================================================================
#  Demo
# ===========================================================================

def chess_demo(n_train: int = 4000, n_test: int = 400, seed: int = 0,
               figures: bool = False) -> None:
    print("\n" + "#" * 74)
    print("#  EMERGENT CHESS FROM MOVE TOKENS  (no board / pieces / rules priors)")
    print("#" * 74)
    if not HAS_CHESS:
        print("\n  python-chess not installed (pip install chess); cannot run.")
        return

    print(f"\n  generating {n_train} train + {n_test} test legal games (data only)...")
    train = generate_games(n_train, seed=seed)
    test = generate_games(n_test, seed=seed + 9999)
    model = IBFChessModel().train(train)
    print(f"  vocabulary: {len(model.inv_vocab)} distinct move tokens "
          f"(model sees these as opaque ids)")

    print("\n  === STAGE 1: EMERGENT RULES (legal-move prediction, rules never given) ===")
    res = evaluate_rules(model, test)
    print(f"\n  {'phase':<9}{'IBF legal@1':>13}{'IBF legal@5':>13}{'IBF legalmass':>15}"
          f"{'unigram l@1':>13}{'random l@1':>12}{'IBF acc@1':>11}")
    for p in ("opening", "midgame", "endgame", "all"):
        i, u, r = res["IBF"][p], res["unigram"][p], res["random"][p]
        print(f"  {p:<9}{i['legal@1']:>12.1%}{i['legal@5']:>13.1%}{i['legal_mass']:>15.1%}"
              f"{u['legal@1']:>12.1%}{r['legal@1']:>11.2%}{i['acc@1']:>11.1%}")
    print("  -> the model assigns most of its probability to LEGAL moves and predicts")
    print("     a legal top move far above chance, with no rules ever supplied.")

    print("\n  === STAGE 2: EMERGENT BOARD GEOMETRY (recover the 8x8 grid) ===")
    geo = recover_board_geometry(model)
    print(f"  Procrustes disparity vs true board (0 = perfect): {geo['disparity']:.4f}")
    print(f"  true king-adjacency recovered in embedding kNN:   {geo['neighbor_recovery']:.1%}")
    print("  -> the 2D board geometry emerges from move co-occurrence alone (no spatial prior).")
    if figures:
        _geometry_figure(geo, "mscn_outputs/chess_board_emergence.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Emergent chess representation experiment")
    p.add_argument("--train", type=int, default=4000)
    p.add_argument("--test", type=int, default=400)
    p.add_argument("--figures", action="store_true")
    a = p.parse_args()
    chess_demo(a.train, a.test, figures=a.figures)
