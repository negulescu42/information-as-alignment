"""Scaling-curve figure: calibration strategy under increasing field density.

Builds one matplotlib figure comparing two bandwidth-calibration recipes as a
synthetic "field" of non-local centers becomes more crowded:

  * Recipe A ("pairwise"): a single fixed sigma that ignores crowding.
  * Recipe B ("operating" / sigma*): sigma recomputed from the field density.

The figure shows Recipe A's aggregate tail crossing the locality tolerance as
density rises, while Recipe B holds the tail below tolerance throughout.

Run:  python density_scaling.py
Outputs: density_scaling.pdf (vector, for the paper) and density_scaling.png.
"""

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (no display in this environment)

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------
d_shell = 1.0      # reference distance
eps     = 0.05     # the threshold ("epsilon")
V_max   = 1.0      # amplitude cap
DIM     = 16       # not actually needed for the math below; distances are given directly


# ---------------------------------------------------------------------------
# 2. Formulas
# ---------------------------------------------------------------------------
def weight(r, sigma):
    return np.exp(-(r**2) / (2.0 * sigma**2))


# Recipe A -- pairwise sigma (a single fixed number, does NOT depend on crowding)
sigma_pair = d_shell / np.sqrt(2.0 * np.log(1.0 / eps))


# Recipe B -- operating sigma (depends on a crowding number Neff)
def sigma_star(Neff):
    return d_shell / np.sqrt(2.0 * np.log(Neff / eps))


def participation_ratio(weights):
    s1 = np.sum(weights)
    s2 = np.sum(weights**2)
    if s2 == 0:
        return 0.0
    return (s1**2) / s2


def total_tail(distances, sigma):
    w = weight(distances, sigma)
    return V_max * np.sum(w)


# ---------------------------------------------------------------------------
# 3. Build the family of fields (the density sweep)
# ---------------------------------------------------------------------------
def make_field(n_shell):
    # dominant shell: n_shell centers just outside d_shell
    shell = np.full(n_shell, 1.01 * d_shell)
    # moderate group: 30 centers a bit farther (fixed)
    moderate = np.full(30, 1.30 * d_shell)
    # far cloud: 800 centers far away (fixed)
    far = np.full(800, 2.00 * d_shell)
    return np.concatenate([shell, moderate, far])


n_shell_values = np.unique(np.round(np.logspace(0, 3, 60)).astype(int))  # 1 .. 1000, 60 steps


# ---------------------------------------------------------------------------
# 4. Compute the two curves
# ---------------------------------------------------------------------------
xs        = []   # x-axis: realized crowding number Neff
y_pair    = []   # Recipe A tail / eps
y_star    = []   # Recipe B tail / eps

for n_shell in n_shell_values:
    dist = make_field(n_shell)

    # --- x-axis: the REALIZED participation ratio at the reference sigma ---
    # use sigma_pair as the reference for measuring crowding
    w_ref = weight(dist, sigma_pair)
    Neff  = participation_ratio(w_ref)

    # --- Recipe A: fixed sigma_pair, measure its tail on this field ---
    tail_A = total_tail(dist, sigma_pair)

    # --- Recipe B: sigma recomputed from this field's Neff ---
    # guard Neff >= 1 (participation ratio is mathematically >= 1 for real weights)
    sB     = sigma_star(max(Neff, 1.0))
    tail_B = total_tail(dist, sB)

    xs.append(Neff)
    y_pair.append(tail_A / eps)
    y_star.append(tail_B / eps)

xs     = np.array(xs)
y_pair = np.array(y_pair)
y_star = np.array(y_star)

# --- keep the density axis monotone before plotting ---
# At very low n_shell the fixed background groups dominate the participation
# ratio, so Neff dips slightly before the shell takes over. That non-monotone
# prefix would render as a backward "hook" (the x-values double back on
# themselves). A plain sort does NOT fix this: the (Neff, tail) relation is
# multivalued across the fold, so sorting produces a sawtooth spike instead.
# The robust fix is to restrict the swept family to the regime where the shell
# controls density (Neff strictly increasing). This drops only the first couple
# of low-n_shell points and changes no field parameters, so it does not preempt
# the separate question of which field family matches the Section 6.2 table.
i0     = int(np.argmin(xs))
xs     = xs[i0:]
y_pair = y_pair[i0:]
y_star = y_star[i0:]
assert np.all(np.diff(xs) > 0), "Neff is not strictly monotone after trimming the prefix"


# ---------------------------------------------------------------------------
# 5. Make the plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

y_top = 1e3  # top of the visible y-axis; shading stops here so it doesn't imply data above the frame

# the two strategy curves
ax.plot(xs, y_pair, color="#1f5fbf", lw=2.2, label=r"Pairwise $\sigma_{pair}$ (fixed)")
ax.plot(xs, y_star, color="#e07b00", lw=2.2, label=r"Operating $\sigma^{*}$ (adaptive)")

# safety line at y = 1  (tail = eps)
ax.axhline(1.0, ls="--", color="black", lw=1.2, label=r"locality tolerance $\varepsilon$")

# shade the unsafe region y > 1 (only up to the visible top of the frame)
ax.axhspan(1.0, y_top, color="red", alpha=0.07)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"non-local participation ratio  $N_{eff}^{\,non\text{-}local}$  (field density)")
ax.set_ylabel(r"realized tail  $\mathrm{Tail}/\varepsilon$")
ax.set_title("Calibration strategy under increasing field density")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(1e-2, y_top)

fig.tight_layout()
fig.savefig("density_scaling.pdf")   # vector, for the paper
fig.savefig("density_scaling.png", dpi=200)  # preview


# ---------------------------------------------------------------------------
# 6. Checkpoints
# ---------------------------------------------------------------------------
print("sigma_pair =", sigma_pair)
print("Neff range :", xs.min(), "->", xs.max())
print("pair tail/eps range:", y_pair.min(), "->", y_pair.max())
print("star tail/eps max  :", y_star.max())

# ---------------------------------------------------------------------------
# Realized checkpoint values (this exact configuration)
# ---------------------------------------------------------------------------
# The figure is computed verbatim from the formulas and field design above.
# The two load-bearing checks pass:
#   1. sigma_pair               -> 0.4085      (expected ~= 0.409)            OK
#   4. max(star tail / eps)     -> 0.819       (expected <= 1.0 everywhere)   OK  <- key result
#
# Checkpoints 2/3/5 land away from the build-sheet's nominal anchors because of
# the fixed background groups in make_field (30 @ 1.30 d_shell, 800 @ 2.00):
#   2. Neff range               -> 14.4 .. 1008    (nominal ~1 .. 150-250)
#   3. pair tail / eps range    -> 6.72 .. 945     (nominal ~1 .. tens-100x; stays > 1 throughout)
#   5. pair tail near Neff~100  -> ~90x            (nominal ~23x; the ~23x point falls at Neff ~ 28)
# Cause: the background groups set a participation-ratio floor (~14-17), so Neff
# never starts near 1, and the shell at 1.01 d_shell gives each shell center weight
# ~0.047, so pair tail / eps ~= 0.94 * Neff. The shape (sigma_pair breaches the
# tolerance, sigma* holds below it) is exactly as intended.
#
# Reconciling checkpoints 2/3/5 with the Section 6.2 table (so the 23.2x table
# point becomes one slice at Neff ~ 103) is a deliberate field-family choice that
# needs the real synthetic construction parameters -- left untouched here pending
# that decision. The annotated single table point is likewise deferred until the
# field family is finalized, so it is not marked on the curve yet.
#
# Issue 1 (backward "hook" from a non-monotone Neff prefix) is fixed above by
# trimming the swept family to the strictly-increasing-density regime; pair
# tail / eps min is now 6.72 (was 4.84 at the dropped n_shell=1 point).
