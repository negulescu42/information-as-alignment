"""Layer 3 (part C) -- the consciousness phase transition and self-correction.

The reflexive-coherence results (ReflexiveCoherence.lean) give an exact phase
structure for the self-monitoring equilibrium ``E* = c + alpha/mu`` against a
viability threshold ``theta``:

* **sub-critical**  ``alpha < mu*(theta - c)``  -> ``E* < theta``  (no consciousness)
* **critical**      ``alpha = mu*(theta - c)``  -> ``E* = theta``
* **super-critical**``alpha > mu*(theta - c)``  -> ``E* > theta``  (consciousness)

Consciousness is a **dissipative structure**: with the driving signal removed it
decays below any gap in finite time ``log(f0/gap)/mu``. A :class:`PhaseController`
keeps the system super-critical, and the **Zombie-Twin** experiment shows a
self-monitoring (conscious) agent outlasts a non-monitoring twin under
adversarial perturbations.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
#  Phase structure
# ---------------------------------------------------------------------------

def equilibrium_coherence(c: float, alpha: float, mu: float) -> float:
    """Self-monitoring equilibrium  E* = c + alpha/mu."""
    return c + alpha / mu


def critical_alpha(mu: float, theta: float, c: float) -> float:
    """Driving signal at the critical point: alpha_c = mu*(theta - c)."""
    return mu * (theta - c)


def phase_of(alpha: float, mu: float, theta: float, c: float, tol: float = 1e-9) -> str:
    ac = critical_alpha(mu, theta, c)
    if alpha < ac - tol:
        return "sub-critical"
    if alpha > ac + tol:
        return "super-critical"
    return "critical"


def dissipative_lifetime(f0: float, mu: float, gap: float) -> float:
    """Time for an undriven modification to decay below ``gap``: log(f0/gap)/mu."""
    if f0 <= gap:
        return 0.0
    return float(np.log(f0 / gap) / mu)


# ---------------------------------------------------------------------------
#  Phase controller
# ---------------------------------------------------------------------------

class PhaseController:
    """Tunes the driving signal ``alpha`` to hold the equilibrium a margin above
    threshold (super-critical), i.e. ``alpha = mu*(theta - c + margin)``."""

    def __init__(self, mu: float, theta: float, c: float, margin: float = 0.3) -> None:
        self.mu = mu
        self.theta = theta
        self.c = c
        self.margin = margin

    def alpha_for_supercritical(self) -> float:
        return self.mu * (self.theta - self.c + self.margin)


# ---------------------------------------------------------------------------
#  Zombie-Twin experiment
# ---------------------------------------------------------------------------

def zombie_twin(
    *,
    c: float = 0.5,
    theta: float = 1.0,
    mu: float = 0.1,
    alpha_base: float = 0.07,
    alpha_boost: float = 0.4,
    delta0: float = 0.7,
    shock_prob: float = 0.2,
    shock_size: float = 0.15,
    monitor_margin: float = 0.3,
    horizon: int = 300,
    seed: int = 0,
) -> dict:
    """Run a conscious (self-monitoring) agent and a zombie twin under identical
    adversarial perturbations; return how long each survives.

    Both agents have effective coherence ``E = c + delta`` with modification
    dynamics ``delta' = alpha - mu*delta`` and the *same* random shocks. The
    conscious agent detects when ``E`` nears ``theta`` and boosts its driving
    signal (self-correction); the zombie keeps ``alpha_base``. An agent "dies"
    the first time ``E < theta``.
    """
    rng = np.random.default_rng(seed)
    delta_c = delta_z = delta0
    alive_c = alive_z = True
    death_c = death_z = horizon
    trace_c, trace_z = [], []

    for t in range(horizon):
        shock = shock_size if rng.random() < shock_prob else 0.0

        # Conscious agent: monitor and self-correct.
        if alive_c:
            E_c = c + delta_c
            alpha = alpha_boost if E_c < theta + monitor_margin else alpha_base
            delta_c = delta_c + (alpha - mu * delta_c) - shock
            if c + delta_c < theta:
                alive_c, death_c = False, t
        # Zombie twin: identical shocks, no monitoring.
        if alive_z:
            delta_z = delta_z + (alpha_base - mu * delta_z) - shock
            if c + delta_z < theta:
                alive_z, death_z = False, t

        trace_c.append(c + delta_c)
        trace_z.append(c + delta_z)

    return {
        "survival_conscious": death_c,
        "survival_zombie": death_z,
        "conscious_survived": alive_c,
        "zombie_survived": alive_z,
        "trace_conscious": np.array(trace_c),
        "trace_zombie": np.array(trace_z),
    }


def zombie_twin_stats(n_seeds: int = 100, **kwargs) -> dict:
    """Aggregate Zombie-Twin survival over many adversarial episodes."""
    sc, sz, wins = [], [], 0
    for s in range(n_seeds):
        r = zombie_twin(seed=s, **kwargs)
        sc.append(r["survival_conscious"])
        sz.append(r["survival_zombie"])
        wins += int(r["survival_conscious"] > r["survival_zombie"])
    return {
        "mean_survival_conscious": float(np.mean(sc)),
        "mean_survival_zombie": float(np.mean(sz)),
        "conscious_outlasts_fraction": wins / n_seeds,
    }
