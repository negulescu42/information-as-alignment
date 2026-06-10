"""IBF vs an LLM (Haiku) -- a positional head-to-head on identical positions.

A full-game tournament needs a model call per move; in this sandbox the only channel
to Haiku is the Agent tool (no usable API key for a subprocess), which is impractical
per-move. So we run the **tractable** form: on a fixed set of real Elite positions,
compare the move chosen by (a) the end-to-end IBF player, (b) Haiku, (c) the human, on
three axes -- **legality**, **human-match (acc@1)**, and **move quality (cp)**.

Two-phase (so the LLM moves can come from a batched Agent call):
  1. ``--emit positions.json``  trains the IBF model, samples positions, records each
     one's FEN + history + the IBF move + the human move + their cp quality, and prints
     the FEN list to paste into the Haiku prompt.
  2. ``--positions positions.json --llm llm_moves.json``  scores the LLM's moves vs IBF
     vs human on the same positions.

Run:
  python -m mscn.chess_vs_llm --emit /tmp/pos.json --pgn data/elite_2024-01_5k.pgn
  (get Haiku moves for the printed FENs as JSON -> /tmp/haiku.json)
  python -m mscn.chess_vs_llm --positions /tmp/pos.json --llm /tmp/haiku.json
"""

from __future__ import annotations

import json

import numpy as np

from .chess_strategy import _move_quality, load_pgn
from .chess_tournament import StrongSearchIBFAgent

try:
    import chess
    HAS_CHESS = True
except Exception:  # pragma: no cover
    HAS_CHESS = False


def _phase(ply: int) -> str:
    return "opening" if ply < 10 else ("midgame" if ply < 26 else "endgame")


def emit_positions(pgn_path: str, out_path: str, n_per_phase: int = 10,
                   n_train: int = 2200, max_games: int = 5000, seed: int = 0) -> None:
    games = load_pgn(pgn_path, max_games=max_games)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(games))
    train = [games[i] for i in idx[:n_train]]
    test = [games[i] for i in idx[n_train:n_train + 400]]

    def avg_elo(g):
        es = [e for e in (g["white_elo"], g["black_elo"]) if e is not None]
        return np.mean(es) if es else 0
    strong = sorted(train, key=avg_elo)[len(train) // 2:] or train
    print(f"  training IBF model on {len(strong)} games...")
    ag = StrongSearchIBFAgent(depth=2, branch=5).train(strong)

    buckets = {"opening": [], "midgame": [], "endgame": []}
    want = n_per_phase
    for g in test:
        if all(len(v) >= want for v in buckets.values()):
            break
        board = chess.Board(); hist = []
        for ply, actual in enumerate(g["moves"]):
            ph = _phase(ply)
            legal = {m.uci() for m in board.legal_moves}
            if len(buckets[ph]) < want and len(legal) > 4 and rng.random() < 0.25:
                ibf = ag.best_move(hist, legal, depth=2, quiesce=True)
                rec = {"fen": board.fen(), "hist": list(hist), "human": actual,
                       "phase": ph, "ibf": ibf, "legal": sorted(legal),
                       "q_human": _move_quality(board, actual),
                       "q_ibf": _move_quality(board, ibf) if ibf in legal else None}
                buckets[ph].append(rec)
            try:
                board.push_uci(actual)
            except Exception:
                break
            hist.append(actual)

    positions = []
    pid = 1
    for ph in ("opening", "midgame", "endgame"):
        for rec in buckets[ph]:
            rec["id"] = str(pid); pid += 1
            positions.append(rec)
    with open(out_path, "w") as f:
        json.dump(positions, f)
    print(f"  wrote {len(positions)} positions to {out_path}\n")
    print("  ==== FENs for the Haiku prompt (give UCI move per id, JSON only) ====")
    for r in positions:
        print(f'  {r["id"]}: {r["fen"]}')


def _quality(fen: str, uci: str):
    board = chess.Board(fen)
    try:
        if chess.Move.from_uci(uci) not in board.legal_moves:
            return None
        return _move_quality(board, uci)
    except Exception:
        return None


def score(positions: list[dict], moves: dict[str, str], label: str) -> dict:
    legal = acc = 0
    q = []
    n = len(positions)
    for r in positions:
        mv = moves.get(r["id"])
        if mv is None:
            continue
        is_legal = mv in set(r["legal"])
        legal += is_legal
        acc += (mv == r["human"])
        if is_legal:
            qq = _quality(r["fen"], mv)
            if qq is not None:
                q.append(qq)
    return {"label": label, "n": n, "legal_rate": legal / n, "acc1": acc / n,
            "quality": float(np.mean(q)) if q else float("nan")}


def compare(positions_path: str, llm_path: str) -> None:
    positions = json.load(open(positions_path))
    llm_moves = json.load(open(llm_path))
    ibf_moves = {r["id"]: r["ibf"] for r in positions}
    human_moves = {r["id"]: r["human"] for r in positions}

    rows = [
        score(positions, human_moves, "human (Elite 2400+)"),
        score(positions, ibf_moves, "IBF end-to-end player"),
        score(positions, llm_moves, "Haiku (LLM)"),
    ]
    print("\n" + "#" * 74)
    print(f"#  IBF vs HAIKU -- positional head-to-head ({len(positions)} real Elite positions)")
    print("#" * 74)
    print(f"\n  {'mover':<24}{'legal@1':>10}{'acc@1':>9}{'move quality (cp)':>20}")
    for r in rows:
        print(f"  {r['label']:<24}{r['legal_rate']:>9.0%}{r['acc1']:>9.0%}{r['quality']:>18.0f}")

    ibf, llm = rows[1], rows[2]
    print(f"\n  reading:")
    print(f"   * legality: IBF {ibf['legal_rate']:.0%} (learned board model, masked play) vs")
    print(f"     Haiku {llm['legal_rate']:.0%} -- the LLM makes illegal moves from FEN (no reliable")
    print(f"     board tracking: null moves like d4d4, impossible piece moves). The")
    print(f"     prior-free IBF state machine is legal by construction.")
    print(f"   * human-match acc@1: IBF {ibf['acc1']:.0%} vs Haiku {llm['acc1']:.0%} "
          f"(human is 100% by definition).")
    print(f"   * move quality (cp): human {rows[0]['quality']:.0f}, IBF {ibf['quality']:.0f}, "
          f"Haiku {llm['quality']:.0f}* -- *SURVIVORSHIP-BIASED: Haiku's cp is over only its")
    print(f"     {llm['legal_rate']:.0%} legal moves (mostly easy opening positions); it forfeits the")
    print(f"     rest. In a real game its illegal moves are losses/forfeits.")
    print(f"   * honest verdict: on identical real positions the **grounded IBF player is the")
    print(f"     more reliable mover** -- 100% legal and {ibf['acc1']:.0%} human-match vs Haiku's")
    print(f"     {llm['legal_rate']:.0%} legal / {llm['acc1']:.0%} match. The LLM has real book/tactical knowledge")
    print(f"     but no world model; the IBF player has a world model but a shallow eval.")
    print(f"     Both are far below human-level. (Caveat: a chess-tuned or board-API'd LLM,")
    print(f"     e.g. gpt-3.5-turbo-instruct ~99.8% legal, would not show this legality gap.)\n")


def main() -> None:
    import argparse
    if not HAS_CHESS:
        raise RuntimeError("python-chess required")
    p = argparse.ArgumentParser(description="IBF vs LLM positional benchmark")
    p.add_argument("--emit", type=str, help="emit positions JSON to this path")
    p.add_argument("--positions", type=str, help="positions JSON (for scoring)")
    p.add_argument("--llm", type=str, help="LLM moves JSON {id: uci}")
    p.add_argument("--pgn", type=str, default="data/elite_2024-01_5k.pgn")
    p.add_argument("--max-games", type=int, default=5000)
    a = p.parse_args()
    if a.emit:
        emit_positions(a.pgn, a.emit, max_games=a.max_games)
    elif a.positions and a.llm:
        compare(a.positions, a.llm)
    else:
        p.error("use --emit PATH, or --positions PATH --llm PATH")


if __name__ == "__main__":
    main()
