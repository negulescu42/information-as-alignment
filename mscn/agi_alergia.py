"""D2 -- STATISTICAL state-merging (ALERGIA/MDI-style): cross the estimation barrier.

ARCHITECTURE 8.10 sharpened the 2.2-general frontier to a precise residual: RPNI's
*exact* legal-set merge crosses the long-range counter (which bounded suffixes cannot),
but it is **brittle** -- on the stochastic flag world it over-splits and loses to
suffix-CSSR (0.84 vs 0.98), and on the counter MORE data over-splits it (1.00@50 ->
0.98@900). The named fix is a **statistical compatibility test** on the next-token
distributions instead of exact set equality.

A per-node Hoeffding test (classic ALERGIA, Carrasco & Oncina 1994) is not enough: on
the flag world the mode-distinguishing evidence lives in *rare-token subtrees* (a set
token starts ~2% of histories), so any single node pair is statistically marginal even
when the subtrees jointly scream. The honest fix is the MDI move (Thollard et al.
2000): pool the evidence **across the entire speculative fold** -- a G-test (log-
likelihood ratio against the pooled distribution) summed over every folded node pair,
accepted iff the total is below the chi-square critical value at the total degrees of
freedom. Two properties follow and are what we measure:

  * **noise-robustness** -- sampling fluctuations cannot veto a merge the way one
    stray observation flips an exact legal-set equality; identical states pool;
  * **data monotonicity** -- more data sharpens the G-test (separating genuinely
    different states, e.g. the counter phases via the recursive fold) instead of
    manufacturing exact-set mismatches. More data should now HELP, not over-split.

Plus EDSM-style ordering: the most-evidenced blue class is processed first, and a blue
merges into the *best-fitting* compatible red (lowest G per degree of freedom), not the
first compatible one. Same red-blue frame and unbounded recurrent fold as RPNI (cycles
= unbounded memory); only the merge decision changes from exact to statistical.

Run: ``python -m mscn.agi_alergia``   (numpy + scipy).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from scipy.stats import chi2

from .agi_simulation import (LearnedSimulation, counter_process, even_process,
                             flag_process, legal_set_purity, toggle_process,
                             _test_histories)
from .agi_state_merging import StateMergingSimulation


class AlergiaSimulation(StateMergingSimulation):
    """Statistical state-merging: RPNI red-blue + recursive fold, with the exact
    legal-set test replaced by a fold-pooled G-test (MDI-style) and EDSM ordering."""

    def __init__(self, delta: float = 0.01, min_pair: int = 4) -> None:
        super().__init__()
        self.delta = delta          # significance level of the merge test
        self.min_pair = min_pair    # a folded pair carries evidence only if both
                                    # sides have >= this many observations (the
                                    # G-test is anti-conservative below that)

    # ----- class-level pooled counts (replaces the exact legal-set summary) -----
    def _recompute(self) -> None:
        self.ccnt: dict[int, Counter] = defaultdict(Counter)
        self.ctrans: dict[int, dict[str, int]] = defaultdict(dict)
        for n in range(self.N):
            L = self._find(n)
            self.ccnt[L].update(self.nxt[n])
            for tok, ch in self.children[n].items():
                self.ctrans[L][tok] = self._find(ch)
        for L in {self._find(n) for n in range(self.N)}:
            self.ccnt.setdefault(L, Counter())

    def _g_pair(self, ca: Counter, cb: Counter) -> tuple[float, int]:
        """G-statistic (2*sum O*ln(O/E)) + degrees of freedom for one folded pair,
        testing 'both drawn from the pooled next-token distribution'."""
        na, nb = sum(ca.values()), sum(cb.values())
        if na < self.min_pair or nb < self.min_pair:
            return 0.0, 0                            # too little evidence on one side
        toks = set(ca) | set(cb)
        g = 0.0
        for t in toks:
            p = (ca[t] + cb[t]) / (na + nb)
            for o, n in ((ca[t], na), (cb[t], nb)):
                if o > 0:
                    g += 2.0 * o * math.log(o / (n * p))
        return g, max(len(toks) - 1, 0)

    # ----- speculative merge: fold + accumulate the global G-test -----
    def _try_merge(self, r: int, b: int):
        """Returns (plan, score) where score = G/df of the whole fold (lower = better
        fit), or None if the fold is statistically rejected."""
        local: dict[int, int] = {}
        gtrans: dict[int, dict[str, int]] = {}
        gcnt: dict[int, Counter] = {}
        G_total, df_total = 0.0, 0

        def lead(x: int) -> int:
            x = self._find(x)
            while local.get(x, x) != x:
                x = local[x]
            return x

        def trans_of(L: int) -> dict[str, int]:
            if L not in gtrans:
                gtrans[L] = dict(self.ctrans[L])
            return gtrans[L]

        def cnt_of(L: int) -> Counter:
            return gcnt[L] if L in gcnt else self.ccnt[L]

        stack = [(r, b)]
        while stack:
            x, y = stack.pop()
            A, C = lead(x), lead(y)
            if A == C:
                continue
            ca, cc = cnt_of(A), cnt_of(C)
            g, df = self._g_pair(ca, cc)
            G_total += g
            df_total += df
            local[A] = C
            gcnt[C] = ca + cc                        # pool the evidence
            tA, tC = trans_of(A), trans_of(C)
            for tok, tgt in tA.items():
                if tok in tC:
                    stack.append((tC[tok], tgt))
                else:
                    tC[tok] = tgt
            gtrans[C] = tC

        if df_total > 0 and G_total > chi2.isf(self.delta, df_total):
            return None                              # subtrees are jointly distinct
        score = G_total / df_total if df_total > 0 else 1.0   # null expectation
        return local, score

    # ----- fit: EDSM ordering (most-evidenced blue; best-fitting red) -----
    def fit(self, seqs: list[list[str]]) -> "AlergiaSimulation":
        self._build_apta(seqs)
        self.uf = list(range(self.N))
        self._recompute()
        red = [0]
        red_set = {0}
        while True:
            blue = []
            seen = set()
            for r in red:
                for tok in sorted(self.ctrans[r]):
                    c = self.ctrans[r][tok]
                    if c not in red_set and c not in seen:
                        seen.add(c); blue.append(c)
            blue = [self._find(b) for b in blue]
            blue = [b for b in dict.fromkeys(blue) if b not in red_set]
            if not blue:
                break
            b = max(blue, key=lambda x: sum(self.ccnt[x].values()))   # most evidence
            best_plan, best_score = None, float("inf")
            for r in red:
                r = self._find(r)
                if r == b:
                    continue
                out = self._try_merge(r, b)
                if out is not None and out[1] < best_score:
                    best_plan, best_score = out[0], out[1]
            if best_plan is not None:
                self._apply(best_plan)
                red = [self._find(x) for x in red]
                red_set = set(red)
            else:
                red.append(b); red_set.add(b)
        self.start_state = self._find(0)
        return self

    # ----- decode sigma: the class legal set is its observed support -----
    def legal_set(self, state: int):
        c = self.ccnt.get(state)
        if not c:
            return frozenset()
        tot = sum(c.values())
        return frozenset(t for t, v in c.items() if v >= max(1.0, 0.005 * tot))


# ===========================================================================
#  Evaluation: suffix-CSSR vs RPNI vs ALERGIA, + the data-monotonicity sweeps
# ===========================================================================

def evaluate3(world, n_train: int, length: int = 50, seed_train: int = 1) -> dict:
    train = world.generate(n_train, length, seed=seed_train)
    test = _test_histories(world.generate(80, length, seed=2))
    out = {"world": world.name, "n_true": world.n_true_states}
    for name, learner in (("suffix", LearnedSimulation(max_suffix=6, min_count=20)),
                          ("rpni", StateMergingSimulation()),
                          ("alergia", AlergiaSimulation())):
        m = learner.fit(train)
        out[f"{name}_states"] = m.n_states
        out[f"{name}_purity"] = legal_set_purity(m.state_of, test, world)
    return out


def main() -> None:
    print("\n" + "#" * 74)
    print("#  D2 -- STATISTICAL state-merging vs the 8.10 estimation barrier")
    print("#  (RPNI's fold kept; exact merge test -> fold-pooled G-test + EDSM order)")
    print("#" * 74)

    worlds = [even_process(), toggle_process(), flag_process(), counter_process(4)]
    seeds = (1, 3, 5)
    print(f"\n  states / legal-set purity at 400 training sequences "
          f"(mean over {len(seeds)} data seeds):\n")
    print(f"  {'world':<10}{'true':>5}{'suffix-CSSR':>16}{'RPNI (exact)':>16}{'statistical':>16}")
    res = {}            # world -> mean purities + max states over seeds
    for w in worlds:
        runs = [evaluate3(w, n_train=400, seed_train=s) for s in seeds]
        r = {"world": w.name, "n_true": w.n_true_states}
        for m in ("suffix", "rpni", "alergia"):
            r[f"{m}_purity"] = sum(x[f"{m}_purity"] for x in runs) / len(runs)
            r[f"{m}_states"] = max(x[f"{m}_states"] for x in runs)
        res[w.name] = r
        print(f"  {r['world']:<10}{r['n_true']:>5}"
              f"{r['suffix_states']:>8} / {r['suffix_purity']:.2f}"
              f"{r['rpni_states']:>8} / {r['rpni_purity']:.2f}"
              f"{r['alergia_states']:>8} / {r['alergia_purity']:.2f}")

    # the two named 8.10 failures, swept over data: does more data now HELP?
    print(f"\n  data sweeps (states / purity, seed 1) -- 8.10's two named failures:")
    sweeps = {}
    for w in (counter_process(4), flag_process()):
        print(f"\n      {w.name}:  {'#train':>8}{'RPNI':>16}{'statistical':>16}")
        sweeps[w.name] = {}
        for n in (50, 150, 400, 900):
            r = evaluate3(w, n_train=n)
            sweeps[w.name][n] = r
            print(f"             {n:>8}"
                  f"{r['rpni_states']:>8} / {r['rpni_purity']:.3f}"
                  f"{r['alergia_states']:>8} / {r['alergia_purity']:.3f}")

    cnt, flg = res["counter4"], res["flag"]
    sc, sf = sweeps["counter4"], sweeps["flag"]
    print(f"\n  reading:")
    print(f"   * NOISE-ROBUSTNESS (8.10 failure #1 fixed) -- on the stochastic flag")
    print(f"     world, RPNI's exact merge over-splits ({flg['rpni_purity']:.2f}); the fold-pooled")
    print(f"     G-test reaches {flg['alergia_purity']:.2f} with {flg['alergia_states']} states "
          f"(true: 2) -- pooling the WHOLE")
    print(f"     speculative fold gives the power a single rare-token node lacks, while")
    print(f"     sampling fluctuations no longer veto merges of identical states.")
    print(f"   * DATA MONOTONICITY (8.10 failure #2 fixed) -- counter: RPNI decays with")
    print(f"     data ({sc[50]['rpni_purity']:.2f}@50 -> {sc[900]['rpni_purity']:.2f}@900); statistical: "
          f"{sc[50]['alergia_purity']:.2f}@50 -> {sc[900]['alergia_purity']:.2f}@900")
    print(f"     ({sc[900]['alergia_states']} states). flag: RPNI {sf[50]['rpni_purity']:.2f}@50 -> "
          f"{sf[900]['rpni_purity']:.2f}@900; statistical {sf[50]['alergia_purity']:.2f} -> "
          f"{sf[900]['alergia_purity']:.2f}.")
    print(f"     More data sharpens the test instead of manufacturing exact-set mismatches.")
    print(f"   * MINIMALITY -- the learned machines are exact or near-exact everywhere:")
    print(f"     even {res['even']['alergia_states']} (true 2), toggle {res['toggle']['alergia_states']} "
          f"(true 2), flag {flg['alergia_states']} (true 2), counter {cnt['alergia_states']} (true 4),")
    print(f"     vs RPNI's {res['even']['rpni_states']}/{res['toggle']['rpni_states']}/"
          f"{flg['rpni_states']}/{cnt['rpni_states']} -- statistical merging is also the better compressor.")

    # assertions: both 8.10 failures fixed + the 8.10 crossing kept, multi-seed
    assert flg["alergia_purity"] > 0.97 and flg["alergia_purity"] > flg["rpni_purity"] + 0.05, \
        "statistical merge must fix the stochastic-flag brittleness (8.10 failure #1)"
    assert all(sweeps[w][n]["alergia_purity"] > 0.97 for w in sweeps for n in (150, 400, 900)), \
        "more data must not over-split the statistical merge (8.10 failure #2)"
    assert cnt["alergia_purity"] > 0.97, "must keep the 8.10 counter crossing"
    assert all(res[w]["alergia_states"] <= 2 * res[w]["n_true"] + 2 for w in res), \
        "learned machines must stay near-minimal"

    print(f"\n  Finding: replacing the exact merge with a statistical one (G-test pooled")
    print(f"  across the speculative fold, chi-square acceptance, EDSM ordering) removes")
    print(f"  BOTH named 8.10 residuals at once -- stochastic worlds merge correctly,")
    print(f"  more data helps, and the recovered machines are (near-)minimal -- while")
    print(f"  keeping the unbounded-memory crossing. On enumerable worlds the estimation")
    print(f"  barrier is now fully crossed; what remains for chess-from-atomic-tokens is")
    print(f"  COMBINATORIAL state growth, which needs a factored state, not a better")
    print(f"  merge test (see the factor-discovery study).\n")


if __name__ == "__main__":
    main()
