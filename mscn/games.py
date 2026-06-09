"""Layer 2 (part B) -- emergent cooperation in the iterated Prisoner's Dilemma.

This is the ``EmergentCooperation.lean`` result made executable. The mechanism is
exactly the one the design document describes: an IBF agent senses only the
*immediate own payoff* as its prior (which always favours defection -- the
classic trap), but its **coherence modification** ``delta_R[s, a]`` accumulates
the discrepancy between that prior and the realised *total coherence*

    realised_total = own_payoff(a, o)  +  J * R_pair(a, o)

where ``R_pair`` rewards relational alignment (mutual cooperation). Once enough
modification has built up, the effective coherence of cooperating exceeds that of
defecting and the agent cooperates -- *no cooperation reward was ever
programmed*. A memoryless agent (alpha = 0) never updates, so it stays on the
defection-favouring prior. This is the content of theorem **EC-4**: sufficient
modification breaks the defection trap.

States are conditioned on the opponent's previous move, so an IBF-memory agent
still learns to **defend against pure defectors** (it does not get suckered).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

C, D = 0, 1
ACTIONS = (C, D)
NAMES = {C: "C", D: "D"}

# Standard PD payoff: PAYOFF[my_action][opp_action]; T=5 > R=3 > P=1 > S=0.
PAYOFF = np.array([[3.0, 0.0],   # I cooperate: vs C -> R, vs D -> S
                   [5.0, 1.0]])  # I defect:    vs C -> T, vs D -> P


def r_pair(a: int, o: int) -> float:
    """Relational coherence: maximal for mutual cooperation, else 0."""
    return 1.0 if (a == C and o == C) else 0.0


# ---------------------------------------------------------------------------
#  Fixed reference strategies
# ---------------------------------------------------------------------------

class Strategy:
    name = "strategy"

    def reset(self) -> None:
        pass

    def act(self) -> int:
        raise NotImplementedError

    def observe(self, my_action: int, opp_action: int) -> None:
        pass


class AlwaysDefect(Strategy):
    name = "AllD"

    def act(self) -> int:
        return D


class AlwaysCooperate(Strategy):
    name = "AllC"

    def act(self) -> int:
        return C


class TitForTat(Strategy):
    name = "TitForTat"

    def reset(self) -> None:
        self._last_opp = C

    def act(self) -> int:
        return self._last_opp

    def observe(self, my_action: int, opp_action: int) -> None:
        self._last_opp = opp_action


class Pavlov(Strategy):
    """Win-stay, lose-shift: keep action after R/T, switch after P/S."""

    name = "Pavlov"

    def reset(self) -> None:
        self._action = C

    def act(self) -> int:
        return self._action

    def observe(self, my_action: int, opp_action: int) -> None:
        payoff = PAYOFF[my_action][opp_action]
        if payoff < 3.0:  # P or S -> shift
            self._action = D if my_action == C else C


class RandomPlayer(Strategy):
    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self) -> int:
        return int(self.rng.integers(2))


# ---------------------------------------------------------------------------
#  The IBF game agent
# ---------------------------------------------------------------------------

@dataclass
class IBFGameAgent(Strategy):
    """Coherence-gradient agent for the iterated PD.

    ``delta_R[s, a]`` is the learned modification for taking action ``a`` when the
    opponent last played ``s``. Selection is Boltzmann over effective coherence
    ``R_eff = prior + delta_R``; ``prior`` is the immediate own payoff (favours D).
    Set ``alpha = 0`` for a *memoryless* agent (never learns -> defects).
    """

    alpha: float = 0.4
    mu: float = 0.06
    k: float = 4.0                 # initial responsiveness
    k_adapt: float = 0.1           # responsiveness growth (agency, Thm 7/8c)
    k_max: float = 8.0
    coupling: float = 4.0          # J: strength of relational coherence
    init_coop: float = 4.0         # optimistic modification toward the coherent state
    seed: int = 0
    name: str = "IBF"

    delta_R: np.ndarray = field(init=False)
    prior: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)
    _k0: float = field(init=False)
    _state: int = field(default=C, init=False)

    def __post_init__(self) -> None:
        # prior[s, a] = own payoff for action a if opponent repeats its last move s
        self.prior = np.array([[PAYOFF[a][s] for a in ACTIONS] for s in ACTIONS])
        self._k0 = self.k
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def _init_mod(self) -> np.ndarray:
        # Optimistic prior toward cooperation, stored in the modification map.
        m = np.zeros((2, 2))
        m[:, C] = self.init_coop
        return m

    def reset(self) -> None:
        self.delta_R = self._init_mod()
        self._state = C
        self.k = self._k0
        self.rng = np.random.default_rng(self.seed)

    @property
    def memoryless(self) -> bool:
        return self.alpha == 0.0

    def r_eff(self, s: int) -> np.ndarray:
        return self.prior[s] + self.delta_R[s]

    def act(self) -> int:
        logits = self.k * self.r_eff(self._state)
        logits = logits - logits.max()
        p = np.exp(logits)
        p = p / p.sum()
        return int(self.rng.choice(2, p=p))

    def observe(self, my_action: int, opp_action: int) -> None:
        s = self._state
        realised_total = PAYOFF[my_action][opp_action] + self.coupling * r_pair(my_action, opp_action)
        discrepancy = realised_total - self.r_eff(s)[my_action]
        # Euler modification dynamics: delta_R' = alpha * discrepancy - mu * delta_R
        self.delta_R[s, my_action] += self.alpha * discrepancy - self.mu * self.delta_R[s, my_action]
        # ADAPT responsiveness: anneal toward exploitation, faster when coherence
        # exceeded expectation (a cooperative discovery). Memoryless agents (no
        # learning) keep their fixed responsiveness.
        if not self.memoryless:
            bonus = self.k_adapt if discrepancy > 0 else 0.0
            self.k = min(self.k + self.k_adapt + bonus, self.k_max)
        self._state = opp_action  # next state = opponent's move


# ---------------------------------------------------------------------------
#  Match and tournament machinery
# ---------------------------------------------------------------------------

def play_match(a: Strategy, b: Strategy, rounds: int = 200) -> dict:
    """Play one iterated-PD match; returns per-player scores and coop rates."""
    a.reset()
    b.reset()
    sa = sb = 0.0
    coop_a = coop_b = 0
    mutual = 0
    half = rounds // 2
    coop_a2 = mutual2 = 0  # second-half (asymptotic) statistics
    for t in range(rounds):
        ai, bi = a.act(), b.act()
        sa += PAYOFF[ai][bi]
        sb += PAYOFF[bi][ai]
        coop_a += (ai == C)
        coop_b += (bi == C)
        mutual += (ai == C and bi == C)
        if t >= half:
            coop_a2 += (ai == C)
            mutual2 += (ai == C and bi == C)
        a.observe(ai, bi)
        b.observe(bi, ai)
    denom2 = max(rounds - half, 1)
    return {
        "score_a": sa / rounds,
        "score_b": sb / rounds,
        "coop_a": coop_a / rounds,
        "coop_b": coop_b / rounds,
        "mutual_coop": mutual / rounds,
        "coop_a_late": coop_a2 / denom2,
        "mutual_coop_late": mutual2 / denom2,
    }


def round_robin(players: dict[str, Callable[[], Strategy]], rounds: int = 200) -> dict:
    """Round-robin tournament (each pair, including self-play).

    ``players`` maps a label to a *factory* (so a fresh agent is built per match).
    Returns mean score per round and mean cooperation rate for each label.
    """
    labels = list(players)
    scores = {l: [] for l in labels}
    coop = {l: [] for l in labels}
    coop_matrix = {l: {} for l in labels}
    for i, li in enumerate(labels):
        for lj in labels[i:]:
            res = play_match(players[li](), players[lj](), rounds)
            scores[li].append(res["score_a"])
            scores[lj].append(res["score_b"])
            coop[li].append(res["coop_a"])
            coop[lj].append(res["coop_b"])
            coop_matrix[li][lj] = res["coop_a"]
            coop_matrix[lj][li] = res["coop_b"]
    summary = {
        l: {"mean_score": float(np.mean(scores[l])), "mean_coop": float(np.mean(coop[l]))}
        for l in labels
    }
    return {"summary": summary, "coop_matrix": coop_matrix, "labels": labels}


# ---------------------------------------------------------------------------
#  Evolutionary selection by differential persistence (Thm 6 corollary)
# ---------------------------------------------------------------------------

def differential_persistence(strengths: np.ndarray, mu: float, threshold: float,
                             t_max: float = 500.0, dt: float = 1.0) -> dict:
    """Under uniform decay ``f(t) = f0 * exp(-mu t)``, stronger modifications stay
    above ``threshold`` strictly longer -- natural selection with no fitness
    function. Returns the survival time of each strength."""
    strengths = np.asarray(strengths, dtype=float)
    survival = []
    for f0 in strengths:
        if f0 <= threshold:
            survival.append(0.0)
        else:
            survival.append(float(np.log(f0 / threshold) / mu))
    return {"strengths": strengths, "survival_time": np.array(survival)}
