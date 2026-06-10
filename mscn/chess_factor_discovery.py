"""D2c -- discover the FROM-TO factorization of chess move tokens from ATOMIC
token streams alone, then rebuild the board and measure sufficiency.

Route A's sufficient board state (ARCHITECTURE 3.5, legal-set purity 0.998) was
GIVEN the from-to decomposition of each token. The 8.10/8.11 arc sharpened why
that matters: structure buys estimability. This module asks the deepest remaining
representation question of the chess arc: is the structure ITSELF statistically
recoverable from atomic opaque tokens?

Signals (all measured on a dev slice first, thresholds then FROZEN; ground truth
is used for EVALUATION ONLY):
  * same-player +2 CONTINUATION pairs (x then u): enriched for to(x) = from(u)
    (a piece moves again) -- audited precision 0.76 at the frozen gate;
  * co-neighbourhood edges: tokens SHARING >= 3 continuation successors are
    same-TO (audited 0.977); sharing >= 3 predecessors are same-FROM (0.942);
  * two EXACT impossibility relations, 0 violations in 153k audited events,
    used as hard cannot-link constraints: (A) a +2 pair can never share a
    from-square (the mover's own square was just vacated/occupied by the enemy);
    (B) a +1 pair can never have from(v) = to(x) (the enemy cannot move the
    piece that just arrived).

Pipeline: continuation edges -> union-find to-classes (co-successor edges) and
(A)-constrained union-find from-classes (co-predecessor edges) -> bipartite
matching of to-classes to from-classes into LOCATIONS (continuation mass votes,
(B) vetoes) -> extension pass assigning remaining tokens by majority vote of
their bipartite links -> occupancy-transfer replay with the DISCOVERED
factorization (start squares discovered as in Route A) -> legal-set purity on
held-out games, against the suffix baseline, a shuffled-factorization control,
and the given-from-to Route A upper bound.

Protocol: thresholds frozen from the 1000-game dev audit (GATE=12, pair count
>= 3, ply-diversity >= 3, lift >= 8, shared >= 3); discovery on the first 3500
games; evaluation on held-out games 3500+.

Run: ``python -m mscn.chess_factor_discovery [--pgn data/elite_2024-01_5k.pgn]``
(numpy + scipy + python-chess; ~6 min).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .chess_strategy import load_pgn

GATE = 12          # skip the stereotyped opening plies when counting
MIN_C, MIN_DIV, MIN_LIFT, MIN_SHARED = 3, 3, 8.0, 3


# ---------------------------------------------------------------------------
#  Discovery (sees only opaque token ids; strings never inspected)
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self) -> None:
        self.p: dict = {}

    def find(self, a):
        self.p.setdefault(a, a)
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


class FactorDiscovery:
    def __init__(self) -> None:
        self.from_loc: dict[str, int] = {}
        self.to_loc: dict[str, int] = {}
        self.n_locations = 0
        self.stats: dict = {}

    def fit(self, games: list[dict]) -> "FactorDiscovery":
        n2: Counter = Counter()
        div2: defaultdict = defaultdict(set)
        n1_pairs: set = set()
        forbid_from: defaultdict = defaultdict(set)   # exact (A) cannot-links
        for g in games:
            mv = g["moves"]
            for i, x in enumerate(mv):
                if i + 1 < len(mv):
                    n1_pairs.add((x, mv[i + 1]))      # (B) vetoes, exact
                if i + 2 < len(mv):
                    u = mv[i + 2]
                    forbid_from[x].add(u)             # (A): from(u) != from(x)
                    forbid_from[u].add(x)
                    if i >= GATE:
                        n2[(x, u)] += 1
                        div2[(x, u)].add(i // 8)
        T2 = sum(n2.values())
        o2: Counter = Counter()
        r2: Counter = Counter()
        for (x, u), c in n2.items():
            o2[x] += c
            r2[u] += c
        cont = [(x, u) for (x, u), c in n2.items()
                if c >= MIN_C and len(div2[(x, u)]) >= MIN_DIV
                and c * T2 / (o2[x] * r2[u] + 1e-9) >= MIN_LIFT]
        succ: defaultdict = defaultdict(set)
        pred: defaultdict = defaultdict(set)
        for x, u in cont:
            succ[x].add(u)
            pred[u].add(x)

        # ----- robust agglomeration. A plain union-find cascades: at 0.977 edge
        # precision, ~30 wrong edges bridge true squares and transitive closure
        # chains them (measured: pair precision collapsed). Rule: two clusters
        # merge only on >= 2 INDEPENDENT edges (one wrong edge cannot bridge; a
        # confirming second between the same pair is a p^2 event) or one edge of
        # overwhelming support (shared >= 6), subject to the (A) cannot-links.
        def collect_edges(nbr) -> list[tuple]:
            ts = [t for t in nbr if len(nbr[t]) >= 2]
            E = []
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    sh = len(nbr[ts[i]] & nbr[ts[j]])
                    if sh >= MIN_SHARED:
                        E.append((ts[i], ts[j], sh))
            return E

        def robust_cluster(edges, forbid: dict | None) -> dict:
            cl: dict = {}
            for a, b, _ in edges:
                cl.setdefault(a, a)
                cl.setdefault(b, b)
            cforb: defaultdict = defaultdict(set)
            cmem: defaultdict = defaultdict(set)
            for t in cl:
                cmem[t] = {t}
                if forbid:
                    cforb[t] = set(forbid.get(t, ()))
            for _ in range(30):
                support: defaultdict = defaultdict(lambda: [0, 0])
                for a, b, sh in edges:
                    ca, cb = cl[a], cl[b]
                    if ca == cb:
                        continue
                    key = (ca, cb) if str(ca) < str(cb) else (cb, ca)
                    support[key][0] += 1
                    support[key][1] = max(support[key][1], sh)
                cand = [(k, n, mx) for k, (n, mx) in support.items()
                        if n >= 2 or mx >= 6]
                if not cand:
                    break
                cand.sort(key=lambda kv: -(kv[1] * 10 + kv[2]))
                merged_any = False
                for (ca, cb), _n, _mx in cand:
                    ra, rb = cl.get(ca, ca), cl.get(cb, cb)
                    ra = next(iter({cl[m] for m in cmem[ra]}), ra)
                    rb = next(iter({cl[m] for m in cmem[rb]}), rb)
                    if ra == rb:
                        continue
                    if forbid and ((cforb[ra] & cmem[rb]) or (cforb[rb] & cmem[ra])):
                        continue                      # exact (A) veto
                    for m in cmem[rb]:
                        cl[m] = ra
                    cmem[ra] |= cmem[rb]
                    cforb[ra] |= cforb[rb]
                    cmem.pop(rb, None)
                    merged_any = True
                if not merged_any:
                    break
            return cl

        to_cls = robust_cluster(collect_edges(succ), None)
        fr_cls = robust_cluster(collect_edges(pred), forbid_from)
        toks, toks_f = list(to_cls), list(fr_cls)

        # Continuation sets of DIFFERENT to-squares are disjoint by construction
        # (the successors' from-squares differ), so two to-classes sharing a
        # strongly-linked from-class partner are fragments of ONE location;
        # symmetrically for from-classes (respecting the exact (A) vetoes).
        def cls_forbidden(ra, rb) -> bool:
            ma, mb = members[ra] | {ra}, members[rb] | {rb}
            return bool((forb[ra] & mb) or (forb[rb] & ma))

        # ----- bipartite matching: to-class <-> from-class = one LOCATION -----
        W: Counter = Counter()
        for x, u in cont:
            if x in to_cls and u in fr_cls:
                W[(to_cls[x], fr_cls[u])] += 1
        vetoed = set()                                # (B): from(v) != to(x)
        for x, v in n1_pairs:
            if x in to_cls and v in fr_cls:
                vetoed.add((to_cls[x], fr_cls[v]))
        loc_of_to: dict = {}
        loc_of_fr: dict = {}
        self.n_locations = 0
        for (A, B), w in W.most_common():
            if (A, B) in vetoed or A in loc_of_to or B in loc_of_fr:
                continue
            loc_of_to[A] = loc_of_fr[B] = self.n_locations
            self.n_locations += 1
        for A in set(to_cls.values()):                # to-only locations
            if A not in loc_of_to:
                loc_of_to[A] = self.n_locations
                self.n_locations += 1
        for B in set(fr_cls.values()):                # from-only locations
            if B not in loc_of_fr:
                loc_of_fr[B] = self.n_locations
                self.n_locations += 1

        for t in to_cls:
            self.to_loc[t] = loc_of_to[to_cls[t]]
        for t in fr_cls:
            self.from_loc[t] = loc_of_fr[fr_cls[t]]

        # ----- extension pass: vote remaining tokens in via bipartite links ----
        for _ in range(2):
            for x in list(succ):
                if x not in self.to_loc:
                    votes = Counter(self.from_loc[u] for u in succ[x]
                                    if u in self.from_loc)
                    if votes and votes.most_common(1)[0][1] >= 2 and \
                            votes.most_common(1)[0][1] >= 0.6 * sum(votes.values()):
                        self.to_loc[x] = votes.most_common(1)[0][0]
            for u in list(pred):
                if u not in self.from_loc:
                    votes = Counter(self.to_loc[x] for x in pred[u]
                                    if x in self.to_loc)
                    if votes and votes.most_common(1)[0][1] >= 2 and \
                            votes.most_common(1)[0][1] >= 0.6 * sum(votes.values()):
                        self.from_loc[u] = votes.most_common(1)[0][0]

        self.stats = {"cont_edges": len(cont), "to_classes": len(set(to_cls.values())),
                      "from_classes": len(set(fr_cls.values())),
                      "locations": self.n_locations,
                      "factored": len(set(self.from_loc) & set(self.to_loc))}
        return self

    def factor(self, tok: str) -> tuple[int, int] | None:
        f, t = self.from_loc.get(tok), self.to_loc.get(tok)
        return (f, t) if f is not None and t is not None else None


# ---------------------------------------------------------------------------
#  Occupancy replay with a (discovered or given) factorization
# ---------------------------------------------------------------------------

class FactoredSimState:
    """Route A's occupancy transfer, parameterised by an arbitrary token
    factorization. Start occupancy: a location used as a source before ever
    being a destination is initially occupied (the 3.5 rule)."""

    def __init__(self, factor_fn) -> None:
        self.factor = factor_fn
        self.start_occ: set[int] = set()

    def train_start(self, games: list[dict]) -> "FactoredSimState":
        for g in games:
            placed: set[int] = set()
            for tok in g["moves"]:
                ft = self.factor(tok)
                if ft is None:
                    continue
                f, t = ft
                if f not in placed:
                    self.start_occ.add(f)
                placed.add(t)
        return self

    def state_key(self, tokens: list[str]) -> tuple:
        occ = {loc: ("s", loc) for loc in self.start_occ}
        skipped = 0
        for tok in tokens:
            ft = self.factor(tok)
            if ft is None:
                skipped += 1
                continue
            f, t = ft
            tag = occ.pop(f, ("s", f))
            occ[t] = tag
        return frozenset(occ.items()), skipped


# ---------------------------------------------------------------------------
#  Evaluation (ground truth used here ONLY)
# ---------------------------------------------------------------------------

def true_factor(tok: str) -> tuple[str, str]:
    return tok[0:2], tok[2:4]


def purity_eval(state_of, games: list[dict], max_prefixes: int = 6000) -> dict:
    """Group held-out positions by the model's state; purity = count-weighted
    fraction whose TRUE legal-move set equals their group's majority legal set."""
    import chess
    groups: defaultdict = defaultdict(Counter)
    n_pref = 0
    skipped_any = 0
    for g in games:
        board = chess.Board()
        hist: list[str] = []
        for tok in g["moves"]:
            try:
                board.push_uci(tok)
            except Exception:
                break
            hist.append(tok)
            if len(hist) < 6:
                continue
            key, skipped = state_of(hist)
            legal = frozenset(m.uci() for m in board.legal_moves)
            groups[key][legal] += 1
            n_pref += 1
            skipped_any += (skipped > 0)
            if n_pref >= max_prefixes:
                break
        if n_pref >= max_prefixes:
            break
    tot = correct = 0
    for c in groups.values():
        tot += sum(c.values())
        correct += c.most_common(1)[0][1]
    return {"purity": correct / max(tot, 1), "positions": tot,
            "groups": len(groups), "skipped_frac": skipped_any / max(n_pref, 1)}


def factor_accuracy(disc: FactorDiscovery, games: list[dict]) -> dict:
    """Occurrence-weighted from/to accuracy under the best location<->square
    bijection (Hungarian), plus bijection-free PAIR consistency (immune to
    fragmentation: are two same-square tokens co-clustered?)."""
    from scipy.optimize import linear_sum_assignment
    occ: Counter = Counter()
    for g in games:
        occ.update(g["moves"])
    toks = [t for t in occ if disc.factor(t) is not None]
    squares = sorted({s for t in toks for s in true_factor(t)})
    sq_id = {s: i for i, s in enumerate(squares)}
    agree = np.zeros((disc.n_locations, len(squares)))
    for t in toks:
        f, to = disc.factor(t)
        tf, tt = true_factor(t)
        agree[f, sq_id[tf]] += occ[t]
        agree[to, sq_id[tt]] += occ[t]
    r, c = linear_sum_assignment(-agree)
    mapping = {int(a): squares[int(b)] for a, b in zip(r, c)}
    both = w_tot = 0
    for t in toks:
        f, to = disc.factor(t)
        tf, tt = true_factor(t)
        ok = (mapping.get(f) == tf) and (mapping.get(to) == tt)
        both += occ[t] * ok
        w_tot += occ[t]
    coverage = sum(occ[t] for t in toks) / sum(occ.values())
    # pair consistency over the most frequent factored tokens
    top = sorted(toks, key=lambda t: -occ[t])[:400]
    tp = fp = fn = 0
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]
            same_d = disc.factor(a)[1] == disc.factor(b)[1]
            same_t = true_factor(a)[1] == true_factor(b)[1]
            tp += same_d and same_t
            fp += same_d and not same_t
            fn += same_t and not same_d
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"acc_both": both / max(w_tot, 1), "coverage": coverage,
            "n_factored_tokens": len(toks), "mapping": mapping,
            "pair_prec_to": prec, "pair_rec_to": rec}


def occupancy_probe(state_occ_fn, loc2sq, games: list[dict],
                    max_prefixes: int = 4000) -> dict:
    """The Othello-GPT-style probe: does the reconstructed state's occupancy
    track the TRUE board? Per held-out prefix, F1 between {mapped occupied
    locations} and {truly occupied squares}. Unlike grouping purity, an
    injective-but-wrong state CANNOT score here: its occupancy decorrelates
    from the board within a few plies."""
    import chess
    f1s: defaultdict = defaultdict(list)
    n = 0
    for g in games:
        board = chess.Board()
        hist: list[str] = []
        for tok in g["moves"]:
            try:
                board.push_uci(tok)
            except Exception:
                break
            hist.append(tok)
            if len(hist) < 6 or len(hist) % 3:
                continue
            pred = {loc2sq[l] for l in state_occ_fn(hist) if l in loc2sq}
            true = {chess.square_name(s) for s in board.piece_map()}
            inter = len(pred & true)
            f1 = 2 * inter / max(len(pred) + len(true), 1)
            bucket = "ply 6-20" if len(hist) <= 20 else \
                ("ply 21-40" if len(hist) <= 40 else "ply 41+")
            f1s[bucket].append(f1)
            # the DISCRIMINATIVE variant: Matthews correlation over the 64
            # change-indicators (occupancy changed vs the start position).
            # Plain F1 of change-SETS has a high random-overlap floor (the sets
            # are large); MCC sends a random factorization to ~0.
            start = {chess.square_name(s) for s in chess.Board().piece_map()}
            pc, tc = pred ^ start, true ^ start
            tp = len(pc & tc)
            fp = len(pc - tc)
            fn = len(tc - pc)
            tn = 64 - tp - fp - fn
            den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
            f1s["mcc"].append((tp * tn - fp * fn) / den if den > 0 else 0.0)
            n += 1
            if n >= max_prefixes:
                break
        if n >= max_prefixes:
            break
    out = {b: float(np.mean(v)) for b, v in f1s.items()}
    out["all"] = float(np.mean([x for b, v in f1s.items() if b != "mcc"
                                for x in v]))
    return out


def inconsistency_rate(sim: "FactoredSimState", games: list[dict],
                       cap: int = 300) -> float:
    """PRIOR-FREE internal diagnostic: fraction of replay transfers whose source
    location is not currently occupied. Needs no ground truth -- if it separates
    the factorizations, it is the self-supervised objective a future EM
    refinement can optimise."""
    bad = tot = 0
    for g in games[:cap]:
        occ = {loc for loc in sim.start_occ}
        for tok in g["moves"]:
            ft = sim.factor(tok)
            if ft is None:
                continue
            f, t = ft
            tot += 1
            if f not in occ:
                bad += 1
            occ.discard(f)
            occ.add(t)
    return bad / max(tot, 1)


def main(pgn: str = "data/elite_2024-01_5k.pgn", n_disc: int = 3500,
         n_eval: int = 800) -> None:
    print("\n" + "#" * 74)
    print("#  D2c -- FROM-TO FACTOR DISCOVERY from atomic tokens (no structure given)")
    print("#  thresholds frozen on the dev audit; evaluation on held-out games")
    print("#" * 74)
    games = load_pgn(pgn, max_games=n_disc + n_eval)
    disc_g, eval_g = games[:n_disc], games[n_disc:]
    print(f"\n  discovery on {len(disc_g)} games; evaluation on {len(eval_g)} held-out")

    disc = FactorDiscovery().fit(disc_g)
    s = disc.stats
    print(f"  continuation edges {s['cont_edges']}; to-classes {s['to_classes']}, "
          f"from-classes {s['from_classes']} -> {s['locations']} locations")

    acc = factor_accuracy(disc, eval_g)
    print(f"\n  factorization (held-out, occurrence-weighted):")
    print(f"   tokens factored {s['factored']}  |  stream coverage "
          f"{acc['coverage']:.3f}  |  from+to BOTH correct (best bijection): "
          f"{acc['acc_both']:.3f}")

    print(f"   pair consistency (to-square, top-400 tokens): precision "
          f"{acc['pair_prec_to']:.3f}  recall {acc['pair_rec_to']:.3f}")

    sim_d = FactoredSimState(disc.factor).train_start(disc_g)
    sim_true = FactoredSimState(lambda t: true_factor(t)).train_start(disc_g)
    rng = np.random.default_rng(0)
    toks_all = list({t for g in disc_g for t in g["moves"]})
    perm = dict(zip(toks_all, rng.permutation(len(toks_all))))
    sim_ctrl = FactoredSimState(
        lambda t: (int(perm[t]) % 64, (int(perm[t]) // 64) % 64)
        if t in perm else None).train_start(disc_g)

    def occ_fn(sim):
        return lambda h: {loc for loc, _tag in sim.state_key(h)[0]}

    sq_id = {s: s for g in eval_g for t in g["moves"] for s in true_factor(t)}
    print(f"\n  OCCUPANCY PROBE on held-out positions (F1 of reconstructed-vs-true")
    print(f"  board occupancy; an injective-but-wrong state CANNOT score here):")
    probes = {}
    for name, sim, l2s in (
            ("DISCOVERED factorization", sim_d, acc["mapping"]),
            ("given from-to (Route A bound)", sim_true, sq_id),
            ("shuffled control", sim_ctrl,
             {i: f"{'abcdefgh'[i % 8]}{i // 8 % 8 + 1}" for i in range(64)})):
        p = occupancy_probe(occ_fn(sim), l2s, eval_g)
        p["inconsistency"] = inconsistency_rate(sim, eval_g)
        probes[name] = p
        print(f"   {name:<30} change-MCC {p['mcc']:.3f}  rawF1 {p['all']:.3f}  "
              f"self-inconsistency {p['inconsistency']:.3f}")

    print(f"\n  legal-set purity (secondary; HONEST CAVEAT: at this sample size most")
    print(f"  states are singletons, so purity cannot distinguish a correct state")
    print(f"  from an injective-but-wrong one -- the shuffled control scores ~0.97;")
    print(f"  it only catches too-COARSE states like the suffix baseline):")
    for name, sim in (("DISCOVERED", sim_d), ("given from-to", sim_true)):
        r = purity_eval(sim.state_key, eval_g)
        print(f"   {name:<30} purity {r['purity']:.3f}  ({r['groups']} states /"
              f" {r['positions']} positions)")
    suffix = purity_eval(lambda h: (tuple(h[-2:]), 0), eval_g)
    print(f"   {'suffix-2 baseline':<30} purity {suffix['purity']:.3f}")

    d, t, c = (probes["DISCOVERED factorization"],
               probes["given from-to (Route A bound)"], probes["shuffled control"])
    print(f"\n  verdict (change-MCC is the discriminative number: raw F1 carries the")
    print(f"  shared start position and a large random-overlap floor):")
    print(f"   * change-MCC: discovered {d['mcc']:.3f} vs given-structure bound "
          f"{t['mcc']:.3f} and shuffled {c['mcc']:.3f}.")
    print(f"     The gap to Route A is "
          f"{'substantially closed' if d['mcc'] > 0.7 * t['mcc'] else 'narrowed but open'}.")
    print(f"   * the PRIOR-FREE self-inconsistency diagnostic separates the three")
    print(f"     ({d['inconsistency']:.3f} / {t['inconsistency']:.3f} / "
          f"{c['inconsistency']:.3f}) -- an internal signal a future EM refinement")
    print(f"     can optimise with no oracle (the named next mechanism).")
    print(f"   * STRUCTURAL READING: replay is an ERROR AMPLIFIER -- per-token")
    print(f"     factorization errors compound multiplicatively over a prefix, so")
    print(f"     board reconstruction demands near-perfect per-token accuracy.")
    print(f"     Partial recovery (pair precision {acc['pair_prec_to']:.2f}, coverage "
          f"{acc['coverage']:.2f}) buys only")
    print(f"     change-MCC {d['mcc']:.2f}: Route A's given structure was structurally")
    print(f"     necessary, not a convenience -- 8.10's lesson, now quantified at the")
    print(f"     functional level.")
    assert d["mcc"] > 0.05 and abs(c["mcc"]) < 0.05, \
        "discovered change-tracking must be real; the control must sit at zero"
    assert acc["pair_prec_to"] > 0.8, \
        "to-square co-clustering must stay high-precision"
    assert d["inconsistency"] < c["inconsistency"] - 0.1, \
        "the prior-free consistency signal must rank discovered above shuffled"
    print()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="from-to factor discovery")
    p.add_argument("--pgn", type=str, default="data/elite_2024-01_5k.pgn")
    a = p.parse_args()
    main(pgn=a.pgn)
