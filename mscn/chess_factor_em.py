"""D2c follow-up -- the EM REPLAY-CONSISTENCY loop: refine the discovered
factorization against a PRIOR-FREE internal objective.

3.20 ended with the gradient demonstrated: the self-inconsistency rate (fraction
of replay transfers whose source location is not currently occupied) separates the
true / discovered / shuffled factorizations at 0.017 / 0.081 / 0.369 -- an internal
signal that needs no oracle. This module closes the loop: alternate REPLAY
(E-step: run the occupancy simulation with the current factorization, collecting
miss statistics) with three targeted refinements (M-step), and measure whether
descending the internal objective moves the GROUND-TRUTH metrics (evaluation-only,
held-out): exact-factorization accuracy and the occupancy change-MCC.

The three M-moves target the measured error modes:
  * MERGE fragmented locations -- fragmentation causes handoff misses (a piece
    arrives at fragment A, the next departure is labelled from fragment B); the
    co-occupied-at-miss tallies propose merges, mutual-best + support-gated,
    vetoed by the exact (A) cannot-links (which also block the degenerate
    'one immortal location' collapse of the raw objective);
  * RE-ASSIGN the from-location of high-miss tokens, only within a bounded
    evidence set (locations co-occupied at that token's occurrences with
    same-side last movers), never a global argmax;
  * ASSIGN unfactored tokens (26% of the stream) the same way -- skipped
    transfers corrupt the state for every other token, so coverage growth
    compounds.

PRE-REGISTERED question (the unit's claim is the answer, whichever way): does
prior-free objective descent improve ground truth? Asserted: the objective falls
and the held-out change-MCC does not degrade; the magnitude of improvement is
reported honestly.

Run: ``python -m mscn.chess_factor_em``   (numpy + scipy + python-chess; ~8 min).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .chess_factor_discovery import (FactorDiscovery, FactoredSimState,
                                     factor_accuracy, inconsistency_rate,
                                     occupancy_probe, true_factor)
from .chess_strategy import load_pgn


class EMRefiner:
    def __init__(self, disc: FactorDiscovery, games: list[dict]) -> None:
        self.from_loc = dict(disc.from_loc)
        self.to_loc = dict(disc.to_loc)
        self.n_locations = disc.n_locations
        self.games = games
        # exact (A) cannot-links between TOKENS (same-player +2 pairs)
        self.forbid: defaultdict = defaultdict(set)
        for g in games:
            mv = g["moves"]
            for i in range(len(mv) - 2):
                self.forbid[mv[i]].add(mv[i + 2])
                self.forbid[mv[i + 2]].add(mv[i])

    def factor(self, tok: str):
        f, t = self.from_loc.get(tok), self.to_loc.get(tok)
        return (f, t) if f is not None and t is not None else None

    # ----- E-step: replay with the current factorization, collect evidence -----
    def replay_stats(self) -> dict:
        miss = Counter()
        plays = Counter()
        merge_tally: Counter = Counter()      # (missing from-loc, occupied loc)
        cand_tally: defaultdict = defaultdict(Counter)   # token -> loc votes
        start_occ: set[int] = set()
        for g in self.games:                  # start squares under current F
            placed: set[int] = set()
            for tok in g["moves"]:
                ft = self.factor(tok)
                if ft is None:
                    continue
                if ft[0] not in placed:
                    start_occ.add(ft[0])
                placed.add(ft[1])
        self.start_occ = start_occ
        J = tot = 0
        for g in self.games:
            occ: dict[int, int] = {loc: -1 for loc in start_occ}   # loc -> last side
            for i, tok in enumerate(g["moves"]):
                side = i % 2
                ft = self.factor(tok)
                if ft is None:
                    # clean evidence for unfactored tokens: same-side-occupied locs
                    for loc, s in occ.items():
                        if s == side or s == -1:
                            cand_tally[tok][loc] += 1
                    continue
                f, t = ft
                plays[tok] += 1
                tot += 1
                if f not in occ:
                    J += 1
                    miss[tok] += 1
                    for loc, s in occ.items():            # merge/reassign evidence
                        if s == side or s == -1:
                            merge_tally[(f, loc)] += 1
                            cand_tally[tok][loc] += 1
                else:
                    cand_tally[tok][f] += 2               # support the current pick
                occ.pop(f, None)
                occ[t] = side
        return {"J": J / max(tot, 1), "miss": miss, "plays": plays,
                "merge": merge_tally, "cand": cand_tally}

    def snapshot(self) -> tuple[dict, dict]:
        return dict(self.from_loc), dict(self.to_loc)

    def restore(self, snap: tuple[dict, dict]) -> None:
        self.from_loc, self.to_loc = dict(snap[0]), dict(snap[1])

    def J_eval(self, n_games: int = 800) -> float:
        """The COMPARABLE internal objective: (misses + skips) / all token
        occurrences. A skipped (unfactored) transfer corrupts the state exactly
        like a miss, so it must cost the same -- otherwise coverage growth is
        unaccountable and J is incomparable across rounds (measured)."""
        start_occ: set[int] = set()
        gs = self.games[:n_games]
        for g in gs:
            placed: set[int] = set()
            for tok in g["moves"]:
                ft = self.factor(tok)
                if ft is None:
                    continue
                if ft[0] not in placed:
                    start_occ.add(ft[0])
                placed.add(ft[1])
        bad = tot = 0
        for g in gs:
            occ: set[int] = set(start_occ)
            for tok in g["moves"]:
                tot += 1
                ft = self.factor(tok)
                if ft is None:
                    bad += 1                      # skip = corruption = miss
                    continue
                f, t = ft
                if f not in occ:
                    bad += 1
                occ.discard(f)
                occ.add(t)
        return bad / max(tot, 1)

    # ----- M-moves -----
    def _loc_tokens(self) -> dict[int, set]:
        lt: defaultdict = defaultdict(set)
        for tok, l in self.from_loc.items():
            lt[l].add(tok)
        return lt

    def _merge_allowed(self, la: int, lb: int, loc_tokens: dict) -> bool:
        """(A) veto at the location level: no observed +2 pair may share a
        from-location after the merge."""
        ta, tb = loc_tokens.get(la, set()), loc_tokens.get(lb, set())
        return not any(self.forbid[x] & tb for x in ta)

    def merge_locations(self, merge_tally: Counter, max_merges: int = 25,
                        min_support: int = 10) -> int:
        """Mutual-best, support-gated, (A)-vetoed, and each merge is kept only
        if the comparable objective J actually improves (trial-and-revert)."""
        loc_tokens = self._loc_tokens()
        best_of: dict[int, tuple[int, int]] = {}
        for (fm, lo), c in merge_tally.items():
            if c >= min_support and c > best_of.get(fm, (0, 0))[1]:
                best_of[fm] = (lo, c)
        rev: dict[int, tuple[int, int]] = {}
        for (fm, lo), c in merge_tally.items():
            if c > rev.get(lo, (0, 0))[1]:
                rev[lo] = (fm, c)
        done = 0
        J = self.J_eval()
        for fm, (lo, c) in sorted(best_of.items(), key=lambda kv: -kv[1][1]):
            if done >= max_merges or fm == lo:
                continue
            if rev.get(lo, (None, 0))[0] != fm:          # mutual best only
                continue
            if not self._merge_allowed(fm, lo, loc_tokens):
                continue
            snap = self.snapshot()
            for tok in list(self.from_loc):
                if self.from_loc[tok] == fm:
                    self.from_loc[tok] = lo
            for tok in list(self.to_loc):
                if self.to_loc[tok] == fm:
                    self.to_loc[tok] = lo
            J2 = self.J_eval()
            if J2 < J - 1e-4:
                J = J2
                loc_tokens = self._loc_tokens()
                done += 1
            else:
                self.restore(snap)
        return done

    def reassign_tokens(self, stats: dict, miss_thresh: float = 0.3,
                        min_votes: int = 5) -> int:
        loc_tokens = self._loc_tokens()
        moved = 0
        for tok, m in stats["miss"].items():
            p = stats["plays"][tok]
            if p < 5 or m / p < miss_thresh:
                continue
            votes = stats["cand"][tok]
            if not votes:
                continue
            new, v = votes.most_common(1)[0]
            if v < min_votes or new == self.from_loc.get(tok):
                continue
            others = loc_tokens.get(new, set()) - {tok}
            if self.forbid[tok] & others:                  # (A) veto
                continue
            self.from_loc[tok] = new
            moved += 1
        return moved

    def assign_unfactored(self, stats: dict, min_votes: int = 30) -> int:
        loc_tokens = self._loc_tokens()
        added = 0
        for tok, votes in stats["cand"].items():
            if tok in self.from_loc or not votes:
                continue
            if self.to_loc.get(tok) is None:
                continue                                  # need a to-side anchor
            new, v = votes.most_common(1)[0]
            if v < min_votes:
                continue
            if self.forbid[tok] & (loc_tokens.get(new, set()) - {tok}):
                continue
            self.from_loc[tok] = new
            added += 1
        return added


def main(pgn: str = "data/elite_2024-01_5k.pgn", n_disc: int = 3500,
         n_eval: int = 800, rounds: int = 8) -> None:
    print("\n" + "#" * 74)
    print("#  D2c-EM -- replay-consistency refinement of the discovered factorization")
    print("#  (prior-free objective; ground truth tracked evaluation-only, held-out)")
    print("#" * 74)
    games = load_pgn(pgn, max_games=n_disc + n_eval)
    disc_g, eval_g = games[:n_disc], games[n_disc:]

    disc = FactorDiscovery().fit(disc_g)
    em = EMRefiner(disc, disc_g)

    def evaluate(label: str) -> dict:
        shim = type("D", (), {"factor": em.factor, "n_locations": 4096,
                              "from_loc": em.from_loc, "to_loc": em.to_loc})()
        acc = factor_accuracy(shim, eval_g)
        sim = FactoredSimState(em.factor).train_start(disc_g)
        probe = occupancy_probe(
            lambda h: {loc for loc, _ in sim.state_key(h)[0]},
            acc["mapping"], eval_g, max_prefixes=2500)
        inc = inconsistency_rate(sim, eval_g)
        n_loc = len(set(em.from_loc.values()) | set(em.to_loc.values()))
        return {"label": label, "acc": acc["acc_both"], "cov": acc["coverage"],
                "mcc": probe["mcc"], "inc": inc, "locs": n_loc}

    rows = [dict(evaluate("round 0 (3.20 baseline)"), J=float("nan"))]
    print(f"\n  {'round':<26}{'J(int)':>8}{'self-inc':>9}{'locs':>6}{'cover':>7}"
          f"{'acc':>7}{'MCC':>7}")
    r0 = rows[0]
    print(f"  {r0['label']:<26}{'--':>8}{r0['inc']:>9.3f}{r0['locs']:>6}"
          f"{r0['cov']:>7.3f}{r0['acc']:>7.3f}{r0['mcc']:>7.3f}")

    J_hist = [em.J_eval()]
    for rd in range(1, rounds + 1):
        stats = em.replay_stats()
        n_m = em.merge_locations(stats["merge"])
        # reassign and assign are accepted as guarded batches (trial-and-revert
        # on the comparable objective J, where a skip costs like a miss)
        n_r = n_a = 0
        for mover, tag in ((em.reassign_tokens, "r"), (em.assign_unfactored, "a")):
            snap = em.snapshot()
            J_before = em.J_eval()
            n = mover(stats)
            if n and em.J_eval() < J_before - 1e-4:
                if tag == "r":
                    n_r = n
                else:
                    n_a = n
            elif n:
                em.restore(snap)
        J_now = em.J_eval()
        J_hist.append(J_now)
        r = evaluate(f"round {rd} (m{n_m} r{n_r} a{n_a})")
        r["J"] = J_now
        rows.append(r)
        print(f"  {r['label']:<26}{J_now:>8.3f}{r['inc']:>9.3f}{r['locs']:>6}"
              f"{r['cov']:>7.3f}{r['acc']:>7.3f}{r['mcc']:>7.3f}")
        if n_m + n_r + n_a == 0:
            break

    # ----- attribution diagnostic (ORACLE-ASSISTED, diagnosis only, not part of
    # the loop): inject TRUE from-assignments for wrongly-assigned tokens; if J
    # barely moves, the objective itself -- not the search -- is the binding
    # limitation at token granularity.
    shim = type("D", (), {"factor": em.factor, "n_locations": 4096,
                          "from_loc": em.from_loc, "to_loc": em.to_loc})()
    mapping = factor_accuracy(shim, eval_g)["mapping"]
    sq2loc = {sq: loc for loc, sq in mapping.items()}
    wrong = [t for t in em.from_loc
             if mapping.get(em.from_loc[t]) != true_factor(t)[0]
             and true_factor(t)[0] in sq2loc]
    rng = np.random.default_rng(0)
    inject = list(rng.choice(wrong, size=min(60, len(wrong)), replace=False))
    snap = em.snapshot()
    J_before_inj = em.J_eval()
    for t in inject:
        em.from_loc[t] = sq2loc[true_factor(t)[0]]
    J_after_inj = em.J_eval()
    em.restore(snap)
    print(f"\n  [attribution diagnostic, oracle-assisted] injecting TRUE from-locs")
    print(f"  for {len(inject)} wrongly-assigned tokens: J {J_before_inj:.4f} -> "
          f"{J_after_inj:.4f} (delta {J_after_inj - J_before_inj:+.4f})")
    print(f"  -> {'the gradient EXISTS at token granularity (search problem)' if J_after_inj < J_before_inj - 0.003 else 'J is nearly flat under truth-injection: the OBJECTIVE, not the search, is the binding limit'}")

    first, last = rows[0], rows[-1]
    print(f"\n  verdict (the pre-registered question: does PRIOR-FREE objective")
    print(f"  descent improve GROUND TRUTH?):")
    print(f"   * internal objective J: {J_hist[0]:.3f} -> {J_hist[-1]:.3f}; "
          f"held-out self-inconsistency {first['inc']:.3f} -> {last['inc']:.3f}")
    print(f"   * exact factorization accuracy: {first['acc']:.3f} -> {last['acc']:.3f}"
          f"   coverage {first['cov']:.3f} -> {last['cov']:.3f}")
    print(f"   * occupancy change-MCC: {first['mcc']:.3f} -> {last['mcc']:.3f}"
          f"   locations {first['locs']} -> {last['locs']}")
    assert J_hist[-1] < J_hist[0] - 0.005, "the internal objective must descend"
    assert last["mcc"] > first["mcc"] - 0.02, \
        "objective descent must not degrade held-out ground truth"
    gain = last["mcc"] - first["mcc"]
    print(f"   -> {'GROUND TRUTH IMPROVES with internal descent' if gain > 0.05 else 'descent is real but ground-truth gain is ' + ('modest' if gain > 0 else 'absent')}"
          f" (MCC {'+' if gain >= 0 else ''}{gain:.3f}).")
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="EM replay-consistency refinement")
    p.add_argument("--pgn", type=str, default="data/elite_2024-01_5k.pgn")
    p.add_argument("--rounds", type=int, default=8)
    a = p.parse_args()
    main(pgn=a.pgn, rounds=a.rounds)
