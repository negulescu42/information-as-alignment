"""
postulate2.promotion  --  interface extraction and context-aware promotion.

A trained Scale-2 correction field crystallizes into coherent BASINS. The
Interface Principle (machine-verified in Lean) says a basin's INTERIOR is
negligible to EXTERNAL interaction, so a basin can be compressed to its BOUNDARY
for cross-context use. Validated additions:

  detect_basins / partition_basin  -- kernel-overlap graph -> connected
      components (>=3 = basin); boundary = faces another structure, interior =
      shielded.
  external_field_fidelity          -- interior's leakage at points EXTERNAL to
      the basin (the Principle's actual claim), vs the global value.
  promote                          -- CONTEXT-AWARE: interior becomes a FROZEN
      same-context reserve (read for same-context queries, dropped from
      cross-context readout, never written); boundary stays active and is tagged
      as one interface unit.
  interface_crucible               -- aggregate the engine's Crucible to the
      interface level with a softened reversal threshold (interfaces need less
      per-center evidence, because the signal is averaged across the boundary).

The discovery: freezing the interior does not just compress -- it IMPROVES
continual-learning retention, because the frozen reserve is immune to the
cross-context interference that erodes an active interior.
"""

import numpy as np
from engine import C

TAU_BASIN = 0.01
TAU_FACE = 1e-4
MIN_BASIN = 3
REVERSAL_THRESHOLD_INTERFACE = C.reversal_threshold * 0.5   # softened for aggregation


# ---------------------------------------------------------------- basin detection
def build_overlap_graph(centers, sigmas, tau=TAU_BASIN):
    n = len(centers)
    Z = np.asarray(centers)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            sig = np.sqrt(sigmas[i] * sigmas[j])
            if np.exp(-np.sum((Z[i] - Z[j]) ** 2) / (2 * sig ** 2)) > tau:
                adj[i, j] = adj[j, i] = True
    return adj


def connected_components(adj):
    n = len(adj)
    comp = -np.ones(n, dtype=int)
    cid = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        stack = [s]; comp[s] = cid
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if comp[v] < 0:
                    comp[v] = cid; stack.append(v)
        cid += 1
    return comp


def partition_basin(Z, sigmas, comp, basin_ids, tau_face=TAU_FACE):
    """Boundary if a center has kernel overlap > tau_face with ANY center outside
    its basin (faces external structure); interior if shielded."""
    boundary, interior = [], []
    for i in basin_ids:
        faces = False
        for j in range(len(Z)):
            if comp[j] == comp[i]:
                continue
            sig = np.sqrt(sigmas[i] * sigmas[j])
            if np.exp(-np.sum((Z[i] - Z[j]) ** 2) / (2 * sig ** 2)) > tau_face:
                faces = True; break
        (boundary if faces else interior).append(i)
    return boundary, interior


def detect_basins(cryst_centers):
    """Return (Z, sigmas, comp, basins) where basins is a list of member-index lists."""
    Z = np.array([c.z for c in cryst_centers])
    sg = np.array([c.sigma for c in cryst_centers])
    comp = connected_components(build_overlap_graph(Z, sg))
    sizes = np.bincount(comp)
    basins = [list(np.where(comp == b)[0]) for b in range(len(sizes)) if sizes[b] >= MIN_BASIN]
    return Z, sg, comp, basins


# ---------------------------------------------------------------- field fidelity
def raw_delta_R(Z_eval, Z_c, sigmas, vs, idx):
    if len(idx) == 0:
        return np.zeros(len(Z_eval))
    Zc, sg, vv = Z_c[idx], sigmas[idx], vs[idx]
    d2 = (np.sum(Z_eval ** 2, 1)[:, None] + np.sum(Zc ** 2, 1)[None, :] - 2 * Z_eval @ Zc.T)
    return np.exp(-np.maximum(d2, 0) / (2 * sg[None, :] ** 2)) @ vv


def external_field_fidelity(Z, sg, vs, members, interior, Z_eval, tau=TAU_BASIN):
    """Interior leakage = max|delta_R_interior| / max|delta_R_full|, measured at
    points EXTERNAL to the basin (kernel over basin members < tau). This is the
    Interface Principle's operative test; the global version (all points) is
    stricter because it includes the basin interior itself."""
    full = raw_delta_R(Z_eval, Z, sg, vs, members)
    interior_field = raw_delta_R(Z_eval, Z, sg, vs, interior)
    denom = np.max(np.abs(full)) + 1e-10
    Zb, sgb = Z[members], sg[members]
    d2 = (np.sum(Z_eval ** 2, 1)[:, None] + np.sum(Zb ** 2, 1)[None, :] - 2 * Z_eval @ Zb.T)
    ext = np.max(np.exp(-np.maximum(d2, 0) / (2 * sgb[None, :] ** 2)), axis=1) < tau
    return dict(global_leak=float(np.max(np.abs(interior_field)) / denom),
                external_leak=(float(np.max(np.abs(interior_field[ext])) / denom)
                               if ext.any() else float("nan")))


# ---------------------------------------------------------------- promotion
class PromotedInterface:
    def __init__(self, basin_id, boundary_centers, interior_centers, phase_origin=0):
        self.basin_id = basin_id
        self.boundary_centers = list(boundary_centers)
        self.interior_centers = list(interior_centers)
        self.phase_origin = phase_origin
        self.active = True
        self.verified = False
        self.dissolution_count = 0


def promote(agent, phase_origin=0):
    """Context-aware promotion of the agent's crystallized basins.

    interior -> FROZEN same-context reserve (kept in agent.centers; contributes
                to same-context readout via normal gating, excluded from
                cross-context readout while unverified, never written).
    boundary -> stays active, tagged with its interface id so cross-context
                pressure is absorbed as a unit.
    """
    cryst = [c for c in agent.centers if c.is_crystallized() and not c.frozen]
    if len(cryst) < MIN_BASIN:
        return []
    Z, sg, comp, basins = detect_basins(cryst)
    interfaces = []
    for members in basins:
        bnd, intr = partition_basin(Z, sg, comp, members)
        if not intr or not bnd:
            continue
        boundary = [cryst[i] for i in bnd]
        interior = [cryst[i] for i in intr]
        bid = int(comp[members[0]])
        for c in boundary:
            c.interface_group = bid
        for c in interior:
            c.frozen = True
        interfaces.append(PromotedInterface(bid, boundary, interior, phase_origin))
    return interfaces


def _contradicted(c):
    if (not c.is_crystallized()) or len(c.dissolution_log) > 0:
        return True
    if c.n_cross_updates() >= C.n_cross_min and len(c.D_history) >= C.n_cross_min:
        mu = float(np.mean(c.D_history[-C.n_cross_min:]))
        if c.v * mu < REVERSAL_THRESHOLD_INTERFACE:
            return True
    return False


def interface_crucible(interfaces):
    """Dissolve interfaces whose boundary is majority-contradicted in the current
    context. Interior is a frozen reserve and does not participate."""
    stats = dict(verified=0, dissolved=0)
    for itf in interfaces:
        if not itf.active:
            continue
        n_bad = sum(_contradicted(c) for c in itf.boundary_centers)
        itf.dissolution_count = n_bad
        if n_bad > len(itf.boundary_centers) / 2.0:
            itf.active = False; itf.verified = False; stats["dissolved"] += 1
        else:
            itf.verified = True; stats["verified"] += 1
    return stats
