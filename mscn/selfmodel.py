"""Layer 3 (part B) -- bounded self-knowledge.

Three formal results from the self-knowledge files are made concrete here:

* **Lawvere obstruction** (LawvereSelfKnowledge.lean): no self-model is
  surjective. Given any ``model : S -> (S -> R)`` and a fixed-point-free
  ``g : R -> R``, the *diagonal* response ``d(s) = g(model(s)(s))`` is not in the
  image of ``model``. Perfect self-knowledge is therefore impossible -- and this
  holds at *every* depth of a self-model tower.
* **Self-knowledge ceiling** (QuantitativeSelfKnowledge.lean): with a finite
  introspection budget ``Gamma`` and a per-level base cost, the achievable tower
  depth is ``floor(Gamma / base_cost)``.
* **Consciousness-competence tradeoff** (ReflexiveCoherence.lean): a global
  capacity is shared between self-modelling and domain competence; spending more
  on introspection necessarily leaves less for the task.
"""

from __future__ import annotations

import numpy as np

ArrayI = np.ndarray


# ---------------------------------------------------------------------------
#  Lawvere diagonal obstruction
# ---------------------------------------------------------------------------

def diagonal_response(model: ArrayI, g) -> ArrayI:
    """The Lawvere diagonal ``d(s) = g(model[s, s])`` -- the response no state's
    self-model reproduces."""
    model = np.asarray(model)
    n = model.shape[0]
    return np.array([g(model[s, s]) for s in range(n)])


def is_surjective(model: ArrayI, target: ArrayI) -> bool:
    """Does some state's self-model row equal ``target``?"""
    model = np.asarray(model)
    return any(np.array_equal(model[s], target) for s in range(model.shape[0]))


def lawvere_obstruction(model: ArrayI, g) -> dict:
    """Construct the diagonal and verify it escapes the model (no surjection).

    ``model[s]`` is state ``s``'s predicted response profile over all queries;
    ``g`` is any fixed-point-free map on responses (e.g. boolean ``not``).
    """
    d = diagonal_response(model, g)
    # row s differs from d at position s, because d[s] = g(model[s,s]) != model[s,s]
    n = np.asarray(model).shape[0]
    differs_everywhere = all(d[s] != model[s, s] for s in range(n))
    return {
        "diagonal": d,
        "diagonal_is_representable": is_surjective(model, d),
        "differs_from_every_self_model": differs_everywhere,
        "perfect_self_knowledge_possible": is_surjective(model, d),
    }


def bool_not(b):
    return 1 - int(b)


# ---------------------------------------------------------------------------
#  Self-knowledge ceiling
# ---------------------------------------------------------------------------

def achievable_depth(gamma: float, base_cost: float) -> int:
    """Maximum introspection depth that fits in budget ``gamma`` (>= 0)."""
    if base_cost <= 0:
        raise ValueError("base_cost must be positive")
    return max(int(np.floor(gamma / base_cost)), 0)


def tower_costs(depth: int, base_cost: float) -> np.ndarray:
    """Cumulative cost of building a self-model tower up to each level."""
    return np.cumsum(np.full(depth, base_cost))


# ---------------------------------------------------------------------------
#  Consciousness-competence tradeoff
# ---------------------------------------------------------------------------

def competence_after_introspection(gamma: float, self_budget: float) -> float:
    """Domain capacity left after allocating ``self_budget`` to self-modelling."""
    return max(gamma - self_budget, 0.0)


def tradeoff_curve(gamma: float, base_cost: float, n: int = 11) -> dict:
    """Sweep introspection depth and report the domain capacity that remains."""
    depths = np.arange(0, achievable_depth(gamma, base_cost) + 1)
    self_budget = depths * base_cost
    domain = np.array([competence_after_introspection(gamma, s) for s in self_budget])
    return {"depth": depths, "self_budget": self_budget, "domain_capacity": domain}


# ---------------------------------------------------------------------------
#  A concrete (necessarily imperfect) self-model
# ---------------------------------------------------------------------------

class SelfModel:
    """A coarse predictor of an agent's own modification map.

    It buckets the modification values and predicts them; by the Lawvere
    obstruction the prediction is provably never perfect, yet -- as the
    Zombie-Twin experiment shows (see :mod:`mscn.phase`) -- an *imperfect*
    self-model still supports useful self-correction.
    """

    def __init__(self, resolution: int = 8) -> None:
        self.resolution = resolution
        self.table: dict[int, float] = {}

    def _bucket(self, x: float, scale: float) -> int:
        return int(np.clip(np.floor(x / max(scale, 1e-9) * self.resolution), 0, self.resolution - 1))

    def observe(self, key: float, value: float, scale: float) -> None:
        self.table[self._bucket(key, scale)] = value

    def predict(self, key: float, scale: float) -> float:
        return self.table.get(self._bucket(key, scale), 0.0)
