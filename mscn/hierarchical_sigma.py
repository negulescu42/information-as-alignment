"""Roadmap 2.1 / 3.1 -- hierarchical Operating Resolution: the renormalisation-group
flow read as a *sequence* of operating bandwidths.

The Operating Resolution principle (``formal/``, ``OperatingResolution.lean``) gives
the optimal kernel bandwidth at **one** scale::

    sigma*  =  d_shell / sqrt( 2 * log( N_eff / eps ) )

The RG flow (``hierarchy.py``) coarse-grains a field level by level, suppressing
irrelevant small-scale structure. Roadmap 2.1/3.1 asks to read that flow as a
sequence ``sigma*_0, sigma*_1, sigma*_2, ...`` -- one operating bandwidth per
coarse-graining level -- so that coarse-graining is **not** an arbitrary block size
but carries its own *operating resolution* at every scale.

Why this is exact (not a re-derivation). Operating Resolution depends only on the
**local** geometry: ``d_shell`` (the local neighbourhood radius) and ``N_eff`` (the
local effective interference count). One RG step rescales the lattice spacing by the
coarse-graining ``factor`` but leaves the *local neighbour pattern* self-similar, so

* ``d_shell`` grows geometrically:  ``d_shell_ell ~ factor**ell * d_shell_0``;
* ``N_eff`` is ~scale-invariant (same local pattern), drifting down only at the
  coarsest levels where few centres remain.

Hence ``sigma*_ell`` rises monotonically by ~``factor`` per step: the RG semigroup
acts on the resolution as a pure rescaling. And the level whose ``sigma*_ell`` first
reaches the signal's correlation length is the **relevant** scale -- a principled RG
stopping rule (finer sigma* over-resolves noise = irrelevant operators; coarser
washes the signal out). This module computes the sequence from a field's centres
using the *same* ``operating_bandwidth`` machinery the chess kernel uses, and
validates the three predictions on a multi-scale field.

Run: ``python -m mscn.hierarchical_sigma``  (numpy only).
"""

from __future__ import annotations

import numpy as np

from .chess_kernel import operating_bandwidth
from .hierarchy import coarse_grain_1d, signal_to_noise

ArrayF = np.ndarray


def _shell_and_neff(coords: ArrayF, k: int) -> tuple[float, float]:
    """Local shell distance and effective interference count for centres ``coords``.

    Mirrors the chess-kernel "Path A" prescription (``chess_kernel.train``):
    ``d_shell`` = median k-th-nearest-neighbour distance (the local-neighbourhood
    radius); ``N_eff`` = participation ratio of the Gaussian kernel weights at the
    reference scale ``sigma_ref = d_shell``. This is the local density measure the
    Operating Resolution law reads -- it is bounded by ``k``, not the centre count.
    """
    coords = np.sort(np.asarray(coords, dtype=float))
    M = len(coords)
    kk = min(k, M - 1)
    if kk < 1:
        return 1.0, 1.0
    D = np.abs(coords[:, None] - coords[None, :])
    D.sort(axis=1)                       # row 0 is self (distance 0)
    nn = D[:, 1:kk + 1]                  # k nearest neighbours (exclude self)
    d_shell = float(np.median(nn[:, -1]))  # k-th NN radius (the shell)
    sref = d_shell + 1e-12
    w = np.exp(-(nn ** 2) / (2 * sref ** 2))    # kernel weights at the reference scale
    n_eff = float(np.mean(w.sum(1) ** 2 / (np.sum(w ** 2, axis=1) + 1e-12)))
    return d_shell, max(n_eff, 1.0)


def sigma_star_sequence(field: ArrayF, coords: ArrayF | None = None, factor: int = 2,
                        steps: int = 6, k: int = 16, eps: float = 0.01,
                        signal: ArrayF | None = None) -> list[dict]:
    """The operating-bandwidth sequence over RG levels.

    At each level ``ell`` the field (and its centre coordinates) are coarse-grained by
    ``factor``; we read ``(d_shell, N_eff)`` from the level's centres and set the
    operating bandwidth ``sigma*_ell``. If ``signal`` is given we also record the
    signal-to-noise ratio at that level (``hierarchy.signal_to_noise``).
    """
    field = np.asarray(field, dtype=float)
    coords = np.arange(field.size, dtype=float) if coords is None else np.asarray(coords, float)
    out: list[dict] = []
    f, c = field, coords
    for ell in range(steps + 1):
        d_shell, n_eff = _shell_and_neff(c, k)
        sigma = operating_bandwidth(d_shell, max(n_eff, eps * 1.001), eps)
        snr = signal_to_noise(f, signal) if signal is not None else float("nan")
        spacing = float(np.median(np.diff(np.sort(c)))) if c.size > 1 else float("nan")
        out.append({"level": ell, "size": int(f.size), "spacing": spacing,
                    "d_shell": d_shell, "n_eff": n_eff, "sigma": sigma, "snr": snr})
        if f.size < factor * 2:
            break
        f = coarse_grain_1d(f, factor)
        c = coarse_grain_1d(c, factor)     # block centroids -> spacing *= factor
    return out


def _build_field(n: int = 1024, lam_signal: float = 64.0, noise: float = 1.0,
                 seed: int = 0) -> tuple[ArrayF, ArrayF]:
    """A 1-D multi-scale field: a smooth signal of correlation length ``lam_signal``
    (lattice units) plus white noise (correlation length ~1). Returns (field, signal)."""
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    signal = np.sin(2 * np.pi * x / lam_signal) + 0.4 * np.sin(2 * np.pi * x / (2 * lam_signal))
    field = signal + rng.normal(0.0, noise, n)
    return field, signal


def main() -> None:
    factor, eps, lam, k = 2, 0.01, 64.0, 8
    field, signal = _build_field(n=1024, lam_signal=lam, noise=1.0, seed=0)
    seq = sigma_star_sequence(field, factor=factor, steps=8, k=k, eps=eps, signal=signal)
    # the "scaling regime": enough centres for a meaningful k-neighbourhood (size >= 4k).
    # Below that the lattice is exhausted (finite-size tail) and sigma* saturates -- the
    # correct signal that the RG flow has reached the system size, reported but not asserted.
    regime = [i for i, r in enumerate(seq) if r["size"] >= 4 * k]

    print("\n" + "#" * 72)
    print("#  HIERARCHICAL OPERATING RESOLUTION  (roadmap 2.1 / 3.1)")
    print("#  RG flow as a sequence of operating bandwidths sigma*_ell")
    print("#" * 72)
    print(f"\n  field: 1024-point 1-D, signal correlation length lambda = {lam:.0f} "
          f"lattice units + white noise;  RG factor = {factor},  eps = {eps}\n")
    print(f"  {'level':>5}{'#centres':>9}{'spacing':>9}{'d_shell':>9}"
          f"{'N_eff':>8}{'sigma*':>9}{'ratio':>7}{'S/N':>8}  regime")
    prev = None
    for r in seq:
        ratio = (r["sigma"] / prev) if prev else float("nan")
        tag = "scaling" if r["level"] in regime else "finite-size"
        print(f"  {r['level']:>5}{r['size']:>9}{r['spacing']:>9.2f}{r['d_shell']:>9.2f}"
              f"{r['n_eff']:>8.2f}{r['sigma']:>9.3f}"
              f"{('  -  ' if prev is None else f'{ratio:>6.2f}')}{r['snr']:>8.2f}  {tag}")
        prev = r["sigma"]

    sigmas = [r["sigma"] for r in seq]
    spacings = [r["spacing"] for r in seq]
    snrs = [r["snr"] for r in seq]
    # ratios within the scaling regime (consecutive in-regime levels)
    reg_ratios = [sigmas[i + 1] / sigmas[i] for i in regime[:-1] if i + 1 in regime]
    reg_sigmas = [sigmas[i] for i in regime]
    peak = int(np.argmax(snrs))

    print("\n  reading:")
    mono = all(b > a - 1e-12 for a, b in zip(reg_sigmas, reg_sigmas[1:]))
    print(f"   * sigma*_ell monotone increasing in the scaling regime : {mono}")
    print(f"   * per-step ratio sigma*_(l+1)/sigma*_l ~ RG factor {factor} : "
          f"median {np.median(reg_ratios):.2f}  (range {min(reg_ratios):.2f}-{max(reg_ratios):.2f})")
    print(f"     -> the RG semigroup acts on the operating resolution as a "
          f"~factor-{factor} rescaling (local N_eff ~ scale-invariant: "
          f"{np.mean([seq[i]['n_eff'] for i in regime]):.1f})")
    print(f"   * relevant scale: S/N peaks at level {peak} "
          f"(spacing {spacings[peak]:.0f}, sigma* = {sigmas[peak]:.2f}); signal lambda = {lam:.0f}")
    print(f"     -> the operating-resolution sequence brackets the signal scale; "
          f"finer levels resolve noise, coarser wash it out")
    print(f"   * finite-size tail (size < {4*k}): sigma* saturates -- the RG flow has "
          f"reached the system size (reported, not asserted)")

    # ---- guarantees (assert the three predictions, in the scaling regime) ----
    assert mono, "sigma* must increase under coarse-graining (Operating Resolution monotonicity)"
    assert all(0.6 * factor <= rr <= 1.6 * factor for rr in reg_ratios), \
        f"per-step sigma* ratio should track the RG factor {factor}: {reg_ratios}"
    assert 0 < peak < len(snrs) - 1, "S/N should peak at an interior (relevant) scale"
    # the relevant level's spacing should bracket the signal correlation length
    assert spacings[peak] <= lam <= 4 * spacings[peak], \
        "the relevant level's scale should bracket the signal correlation length"
    print("\n  all hierarchical-sigma* guarantees hold (scaling regime).\n")


if __name__ == "__main__":
    main()
