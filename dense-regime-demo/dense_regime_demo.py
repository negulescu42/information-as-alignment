"""
Synthetic Dense-Regime Demonstration (§7 upgrade).

A controlled stress test of the operating-bandwidth mechanism. NOT a model of
the deployed FI field — see the prose caption / §7 insert for the disclaimer.

This script:
  1. Defines the geometry (distance × multiplicity table) as a `def`s.
  2. Verifies the validity gate (5 quantitative conditions).
  3. Computes the three candidate bandwidths σ_full, σ_nl, σ_card from the
     three candidate effective counts (full-field nEff, |S|, non-local nEff).
  4. Directly evaluates the aggregate tail at each σ (the actual sum over
     real centers at real distances — NOT the bound checking itself).
  5. Generates the two-panel figure.
  6. Prints a JSON summary used by the Lean instance + §7 insert.

Run:  python dense_regime_demo.py
Outputs:  dense_regime_demo_results.json, dense_regime_demo.png
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------
# Geometry table  (the "auditable" part of the construction)
# ----------------------------------------------------------------------------
# d_shell = 1.  All centers are placed at one of four distances from y = 0.
# Dominant group is at 1.01·d_shell (just barely non-local; the codebase's
# nonLocalSet is the STRICT inequality `d_shell < distance`, so exactly
# d_shell would be excluded from the tail).
#
# (distance / d_shell,  multiplicity,  group label,  whether non-local)
GEOMETRY: list[tuple[float, int, str, bool]] = [
    (0.00, 1,   "local",    False),  # n_local = 1   (depresses full-field nEff)
    (1.01, 3,   "dominant", True),   # n_D     = 3   (boundary shell)
    (1.30, 160, "mid",      True),   # n_M     = 160 (moderate cloud)
    (2.00, 700, "far",      True),   # n_F     = 700 (far cloud)
]

D_SHELL  = 1.0
EPSILON  = 0.05
V_MAX    = 1.0
D_AMBIENT = 16  # ambient dimension (reported but irrelevant to the math)


# ----------------------------------------------------------------------------
# Core formulas
# ----------------------------------------------------------------------------
def gaussian_kernel(sigma: float, d_sq: float) -> float:
    """K_σ(d²) = exp(-d²/(2σ²))."""
    return math.exp(-d_sq / (2.0 * sigma * sigma))


def operating_bandwidth(d_shell: float, n_eff: float, eps: float) -> float:
    """σ* = d / √(2 · log(N_eff / ε))."""
    return d_shell / math.sqrt(2.0 * math.log(n_eff / eps))


def pairwise_bandwidth(d_shell: float, eps: float) -> float:
    """σ_pair = d / √(2 · log(1/ε)) = operatingBandwidth(d, 1, ε)."""
    return d_shell / math.sqrt(2.0 * math.log(1.0 / eps))


def n_eff(weights: np.ndarray) -> float:
    """Participation ratio (Σw)² / Σw²."""
    s1 = float(weights.sum())
    s2 = float((weights * weights).sum())
    if s2 == 0.0:
        return 0.0
    return s1 * s1 / s2


# ----------------------------------------------------------------------------
# Expand the geometry into a flat array of distances
# ----------------------------------------------------------------------------
def build_distances() -> tuple[np.ndarray, np.ndarray]:
    """Return (all_distances, nonlocal_mask)."""
    distances = []
    nonlocal_mask = []
    for d_ratio, mult, _label, is_nonlocal in GEOMETRY:
        distances.extend([d_ratio * D_SHELL] * mult)
        nonlocal_mask.extend([is_nonlocal] * mult)
    return np.array(distances), np.array(nonlocal_mask, dtype=bool)


# ----------------------------------------------------------------------------
# Compute everything
# ----------------------------------------------------------------------------
@dataclass
class Results:
    # geometry
    d_shell: float
    epsilon: float
    V_max: float
    ambient_dimension: int
    M_total: int
    n_local: int
    n_dom: int
    n_mid: int
    n_far: int
    card_nonlocal: int

    # bandwidths
    sigma_pair: float
    sigma_full: float    # operatingBandwidth from full-field nEff
    sigma_nl: float      # operatingBandwidth from non-local nEff   (= σ*)
    sigma_card: float    # operatingBandwidth from non-local cardinality

    # counts at σ_ref = σ_pair
    nEff_full_ref: float
    nEff_nl_ref: float
    card_count: int      # = card_nonlocal (the "|S|" count)

    # gate-pass booleans
    gate_card: bool
    gate_full: bool
    gate_nl: bool
    gate_ratio: bool
    gate_separations: bool

    # measured aggregate tails (directly evaluated sums, NOT the bound)
    tail_pair: float
    tail_full: float
    tail_nl: float       # at σ*
    tail_card: float

    # Tail / ε
    tail_pair_over_eps: float
    tail_full_over_eps: float
    tail_nl_over_eps: float
    tail_card_over_eps: float

    # dense-regime sanity check
    sigma_op_le_sigma_ref: bool
    hDense_holds: bool

    # Sweep data (for plotting)
    sweep_sigmas: list
    sweep_tails: list


def compute(distances: np.ndarray, nonlocal_mask: np.ndarray) -> Results:
    M = len(distances)
    nl_idx = np.where(nonlocal_mask)[0]
    card_nl = len(nl_idx)
    d_nl = distances[nl_idx]
    d_all = distances

    # reference bandwidth = pairwise bandwidth
    sigma_pair = pairwise_bandwidth(D_SHELL, EPSILON)

    # weights at σ_ref
    w_all_ref = np.exp(-(d_all ** 2) / (2.0 * sigma_pair ** 2))
    w_nl_ref  = np.exp(-(d_nl  ** 2) / (2.0 * sigma_pair ** 2))

    nEff_full_ref = n_eff(w_all_ref)
    nEff_nl_ref   = n_eff(w_nl_ref)
    card_count    = card_nl

    # candidate bandwidths
    sigma_full = operating_bandwidth(D_SHELL, V_MAX * nEff_full_ref, EPSILON)
    sigma_nl   = operating_bandwidth(D_SHELL, V_MAX * nEff_nl_ref,   EPSILON)
    sigma_card = operating_bandwidth(D_SHELL, V_MAX * card_count,    EPSILON)

    # measured aggregate tails: Σ_{i ∈ nonlocal} |v_i| · K_σ(‖y−z_i‖²)
    # v_i = +V_max = +1 for all i (per spec)
    def tail_at(sigma: float) -> float:
        return float(np.sum(V_MAX * np.exp(-(d_nl ** 2) / (2.0 * sigma ** 2))))

    tail_pair = tail_at(sigma_pair)
    tail_full = tail_at(sigma_full)
    tail_nl   = tail_at(sigma_nl)
    tail_card = tail_at(sigma_card)

    # gate checks
    gate_card  = card_nl >= 200
    gate_full  = nEff_full_ref <= 5.0
    gate_nl    = nEff_nl_ref >= 50.0
    gate_ratio = (nEff_nl_ref / nEff_full_ref) >= 20.0
    # pairwise separations of (nEff_full, nEff_nl, card) by ≥ 5×
    seps = [
        nEff_nl_ref / nEff_full_ref,
        card_count / nEff_nl_ref,
        card_count / nEff_full_ref,
    ]
    gate_separations = all(s >= 5.0 for s in seps)

    # dense-regime checks
    sigma_op_le_sigma_ref = sigma_nl <= sigma_pair
    # hDense: V_max * nEff_nl_ref ≥ ε · exp(d_shell²/(2·σ_ref²))
    rhs_dense = EPSILON * math.exp((D_SHELL ** 2) / (2.0 * sigma_pair ** 2))
    hDense_holds = (V_MAX * nEff_nl_ref) >= rhs_dense

    # σ-sweep for plotting
    sweep_sigmas = np.geomspace(sigma_card * 0.7, sigma_pair * 1.5, 400)
    sweep_tails  = np.array([tail_at(s) for s in sweep_sigmas])

    return Results(
        d_shell=D_SHELL,
        epsilon=EPSILON,
        V_max=V_MAX,
        ambient_dimension=D_AMBIENT,
        M_total=M,
        n_local=GEOMETRY[0][1],
        n_dom=GEOMETRY[1][1],
        n_mid=GEOMETRY[2][1],
        n_far=GEOMETRY[3][1],
        card_nonlocal=card_nl,
        sigma_pair=sigma_pair,
        sigma_full=sigma_full,
        sigma_nl=sigma_nl,
        sigma_card=sigma_card,
        nEff_full_ref=nEff_full_ref,
        nEff_nl_ref=nEff_nl_ref,
        card_count=card_count,
        gate_card=gate_card,
        gate_full=gate_full,
        gate_nl=gate_nl,
        gate_ratio=gate_ratio,
        gate_separations=gate_separations,
        tail_pair=tail_pair,
        tail_full=tail_full,
        tail_nl=tail_nl,
        tail_card=tail_card,
        tail_pair_over_eps=tail_pair / EPSILON,
        tail_full_over_eps=tail_full / EPSILON,
        tail_nl_over_eps=tail_nl   / EPSILON,
        tail_card_over_eps=tail_card / EPSILON,
        sigma_op_le_sigma_ref=sigma_op_le_sigma_ref,
        hDense_holds=hDense_holds,
        sweep_sigmas=sweep_sigmas.tolist(),
        sweep_tails=sweep_tails.tolist(),
    )


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(r: Results, path: Path) -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))

    sweep_sigmas = np.array(r.sweep_sigmas)
    sweep_tails  = np.array(r.sweep_tails)

    # ---- Panel A: calibration gap ----
    axA.loglog(sweep_sigmas, sweep_tails, color="black", lw=1.5, label="aggregate tail")
    axA.axhline(r.epsilon, color="grey", ls="--", lw=1, label=f"ε = {r.epsilon}")
    axA.axvline(r.sigma_pair, color="C3", ls=":", lw=1.5,
                label=f"σ_pair = {r.sigma_pair:.4f}")
    axA.axvline(r.sigma_nl, color="C0", ls=":", lw=1.5,
                label=f"σ* = {r.sigma_nl:.4f}")
    axA.scatter([r.sigma_pair], [r.tail_pair], color="C3", zorder=5, s=40)
    axA.scatter([r.sigma_nl], [r.tail_nl], color="C0", zorder=5, s=40)
    axA.annotate(f"Tail/ε = {r.tail_pair_over_eps:.1f}",
                 xy=(r.sigma_pair, r.tail_pair),
                 xytext=(8, 8), textcoords="offset points",
                 color="C3", fontsize=9)
    axA.annotate(f"Tail/ε = {r.tail_nl_over_eps:.3f}",
                 xy=(r.sigma_nl, r.tail_nl),
                 xytext=(8, -14), textcoords="offset points",
                 color="C0", fontsize=9)
    axA.set_xlabel("kernel bandwidth σ")
    axA.set_ylabel("aggregate tail  Σ |v| · K_σ(‖y−z‖²)")
    axA.set_title("Panel A — calibration gap\n"
                  f"pairwise overshoots ε by ≈ {r.tail_pair_over_eps:.0f}×; "
                  f"σ* lands the tail at/under ε")
    axA.grid(True, which="both", ls=":", alpha=0.3)
    axA.legend(loc="lower right", fontsize=8)

    # ---- Panel B: the count that matters ----
    axB.loglog(sweep_sigmas, sweep_tails, color="black", lw=1.5, label="aggregate tail")
    axB.axhline(r.epsilon, color="grey", ls="--", lw=1, label=f"ε = {r.epsilon}")

    axB.axvline(r.sigma_full, color="C1", ls=":", lw=1.5,
                label=f"σ from full nEff = {r.sigma_full:.4f}")
    axB.axvline(r.sigma_nl,   color="C0", ls=":", lw=1.5,
                label=f"σ* from non-local nEff = {r.sigma_nl:.4f}")
    axB.axvline(r.sigma_card, color="C2", ls=":", lw=1.5,
                label=f"σ from |S| = {r.sigma_card:.4f}")

    axB.scatter([r.sigma_full, r.sigma_nl, r.sigma_card],
                [r.tail_full, r.tail_nl, r.tail_card],
                color=["C1", "C0", "C2"], zorder=5, s=40)

    axB.annotate(f"unsafe\n(Tail/ε={r.tail_full_over_eps:.2f})",
                 xy=(r.sigma_full, r.tail_full), xytext=(8, 8),
                 textcoords="offset points", color="C1", fontsize=9)
    axB.annotate(f"on-tolerance\n(Tail/ε={r.tail_nl_over_eps:.3f})",
                 xy=(r.sigma_nl, r.tail_nl), xytext=(8, -22),
                 textcoords="offset points", color="C0", fontsize=9)
    axB.annotate(f"wasteful\n(Tail/ε={r.tail_card_over_eps:.4f})",
                 xy=(r.sigma_card, r.tail_card), xytext=(-100, 8),
                 textcoords="offset points", color="C2", fontsize=9)

    axB.set_xlabel("kernel bandwidth σ")
    axB.set_ylabel("aggregate tail")
    axB.set_title("Panel B — the count that matters\n"
                  "only non-local nEff lands the tail at/under tolerance")
    axB.grid(True, which="both", ls=":", alpha=0.3)
    axB.legend(loc="lower right", fontsize=8)

    # Inset table
    inset_text = (
        f"d = {r.ambient_dimension}, d_shell = {r.d_shell}, ε = {r.epsilon}\n"
        f"|S|        = {r.card_count}\n"
        f"nEff (full)= {r.nEff_full_ref:.3f}\n"
        f"nEff (nl)  = {r.nEff_nl_ref:.3f}"
    )
    axB.text(0.02, 0.98, inset_text, transform=axB.transAxes,
             fontsize=8, va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    fig.suptitle("Constructed dense-regime field — controlled stress test of the mechanism, "
                 "NOT a model of the deployed instance", fontsize=10, y=1.005)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Pretty-printers
# ----------------------------------------------------------------------------
def print_geometry_table() -> None:
    print()
    print("Geometry table (d_shell = {}, ε = {}, V_max = {}, d = {}):".format(
        D_SHELL, EPSILON, V_MAX, D_AMBIENT))
    print(f"  {'group':<10}{'dist/d_shell':>14}{'multiplicity':>14}{'is non-local':>14}")
    for d_ratio, mult, label, is_nl in GEOMETRY:
        print(f"  {label:<10}{d_ratio:>14.4f}{mult:>14}{str(is_nl):>14}")
    total = sum(g[1] for g in GEOMETRY)
    nl = sum(g[1] for g in GEOMETRY if g[3])
    print(f"  {'-'*52}")
    print(f"  {'total':<10}{'':<14}{total:>14}")
    print(f"  {'nonlocal':<10}{'':<14}{nl:>14}")
    print()


def print_results(r: Results) -> None:
    print("=" * 64)
    print("Validity gate (the experiment is void unless all hold):")
    print(f"  card(nonlocal) ≥ 200          : {r.card_nonlocal:>10}   {'OK' if r.gate_card else 'FAIL'}")
    print(f"  full_field_nEff ≤ 5           : {r.nEff_full_ref:>10.4f}   {'OK' if r.gate_full else 'FAIL'}")
    print(f"  nonlocal_nEff ≥ 50            : {r.nEff_nl_ref:>10.4f}   {'OK' if r.gate_nl else 'FAIL'}")
    print(f"  nonlocal_nEff/full_nEff ≥ 20  : {r.nEff_nl_ref/r.nEff_full_ref:>10.4f}   {'OK' if r.gate_ratio else 'FAIL'}")
    print(f"  three counts pairwise ≥ 5×    :              {'OK' if r.gate_separations else 'FAIL'}")
    print()
    print(f"Dense regime sanity: σ* ≤ σ_ref : {r.sigma_op_le_sigma_ref}")
    print(f"                    hDense holds : {r.hDense_holds}")
    print()
    print("=" * 64)
    print("The three counts (at σ_ref = σ_pair):")
    print(f"  full-field N_eff   = {r.nEff_full_ref:>10.4f}")
    print(f"  non-local |S|      = {r.card_count:>10d}")
    print(f"  non-local N_eff    = {r.nEff_nl_ref:>10.4f}")
    print()
    print("The three candidate bandwidths and the directly-evaluated tail:")
    fmt = "  {:<30} σ = {:.6f}    Tail = {:.6e}   Tail/ε = {:.4f}"
    print(fmt.format("from full N_eff (unsafe)", r.sigma_full, r.tail_full, r.tail_full_over_eps))
    print(fmt.format("from non-local N_eff  (σ*)", r.sigma_nl,   r.tail_nl,   r.tail_nl_over_eps))
    print(fmt.format("from non-local |S| (wasteful)", r.sigma_card, r.tail_card, r.tail_card_over_eps))
    print()
    print("Panel A reference:")
    print(fmt.format("at σ_pair (calibration)", r.sigma_pair, r.tail_pair, r.tail_pair_over_eps))
    print("=" * 64)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    print_geometry_table()
    distances, nonlocal_mask = build_distances()
    r = compute(distances, nonlocal_mask)
    print_results(r)

    here = Path(__file__).parent
    fig_path = here / "dense_regime_demo.png"
    json_path = here / "dense_regime_demo_results.json"

    make_figure(r, fig_path)
    print(f"Figure written: {fig_path}")

    # Drop sweeps from JSON to keep it readable
    serializable = {k: v for k, v in asdict(r).items()
                    if k not in ("sweep_sigmas", "sweep_tails")}
    json_path.write_text(json.dumps(serializable, indent=2))
    print(f"Results written: {json_path}")

    all_gates = (r.gate_card and r.gate_full and r.gate_nl
                 and r.gate_ratio and r.gate_separations)
    if not all_gates:
        print("\n*** GATE FAILED — geometry needs adjustment ***", file=sys.stderr)
        return 1
    if not (r.sigma_op_le_sigma_ref and r.hDense_holds):
        print("\n*** DENSE REGIME CONDITION FAILED — not in dense regime ***", file=sys.stderr)
        return 1

    print("\nAll gates passed. Construction is valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
