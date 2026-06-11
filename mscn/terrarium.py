"""THE TERRARIUM -- the rendering substrate for the visual showcase benchmark.

One visual language for the whole apparatus: a terrain heatmap (the coherence
landscape) with living overlays -- the agent and its trail, the memory particles
(green = positive corrections, red = negative, sized by |v|, ringed when
crystallized/consolidated), the planner's committed option (flag + dotted path),
the trust halo (k_eff rendered as glow size/brightness), fog when observation
noise is high, and event banners (earthquakes, season switches, detected
contexts).

This module is PRESENTATION ONLY. The physics is the validated science stack
(`ASIWorld`, `SwitchingWorld`, `MoatWorld`, `CorridorWorld`, the KRK oracle);
the Terrarium wraps those worlds, never replaces them, and nothing rendered here
feeds back into an agent's decisions or its eval budget (terrain sampling uses
the oracle-side ``true_coherence``, which is uncounted by design). Episode
scripts and scoring live in ``terrarium_episodes.py``; rendering is OPTIONAL
there (``--render``), so every episode doubles as a headless CI check.

Outputs are FILES (no GUI in this sandbox): PNG stills and animated GIFs
(matplotlib 'Agg' frames stitched with pillow), with a hard size budget per GIF
(they get committed).

Run: ``python -m mscn.terrarium``  (self-check: renders a smoke-test frame/GIF
into a temp dir and asserts the size budget machinery works).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

ArrayF = np.ndarray

# the shared visual vocabulary (one look across every episode)
STYLE = {
    "terrain_cmap": "cividis",
    "agent": "#ffffff",
    "agent_edge": "#1a1a1a",
    "twin": "#b0b0b0",
    "trail_alpha": 0.65,
    "pos_particle": "#46d46a",
    "neg_particle": "#ff4d4d",
    "ring": "#ffd24d",
    "halo": "#aee6ff",
    "flag": "#ff9f1c",
    "banner_bg": "#101018",
    "fps": 9,
    "gif_budget_mb": 5.5,
}


# ---------------------------------------------------------------------------
#  Snapshots: what one keyframe needs to know (presentation state only)
# ---------------------------------------------------------------------------

@dataclass
class Snap:
    tick: int
    x: ArrayF
    particles: list = field(default_factory=list)   # (z, v, ringed, readable, ctx)
    k_eff: float = 1.0
    option_cell: tuple | None = None
    option_path_xy: ArrayF | None = None
    boost: bool = False
    ok: bool = True
    noise: float = 0.0
    events: list = field(default_factory=list)      # banner strings, this keyframe
    terrain: ArrayF | None = None                   # refreshed on world changes
    meters: dict = field(default_factory=dict)      # novice meters (score strip)
    extra: dict = field(default_factory=dict)


def memory_particles(agent) -> list:
    """Extract (z, v, ringed, readable, ctx) from either memory substrate:
    the classic engine when the agent carries one (Unified/ULTRA), else the
    shell's Scale centres (ringed = consolidated upward)."""
    out = []
    if hasattr(agent, "engine"):
        eng = agent.engine
        if eng.Z.shape[0]:
            g = eng._gamma()
            for i in range(eng.Z.shape[0]):
                out.append((eng.Z[i].copy(), float(eng.V[i]), bool(eng.CRY[i]),
                            bool(g[i]), int(eng.CTX[i])))
        return out
    for s in agent.scales:
        for c in s.centers:
            readable = (not agent.gate_contexts) or c.ctx == agent.ctx
            out.append((c.z.copy(), float(c.v), bool(c.transferred),
                        readable, int(c.ctx)))
    return out


def k_eff_of(agent) -> float:
    """The locally modulated responsiveness when the third ODE is wired
    (engine.delta_k), else the global k."""
    if hasattr(agent, "engine"):
        return float(np.clip(agent.k + agent.engine.delta_k(np.asarray(agent.x, float)),
                             0.0, agent.k_max))
    return float(agent.k)


def sample_terrain(world, n: int = 70) -> ArrayF:
    """Oracle-side terrain heightmap (uncounted ``true_coherence``), row 0 = lo."""
    xs = np.linspace(world.lo[0], world.hi[0], n)
    ys = np.linspace(world.lo[1], world.hi[1], n)
    return np.array([[world.true_coherence(np.array([x, y])) for x in xs]
                     for y in ys])


class Recorder:
    """Collects keyframe snapshots from a running agent/world pair. The episode
    decides WHEN to snap (every Nth tick) and which events/meters to attach;
    terrain is re-sampled only when the episode says the world changed."""

    def __init__(self, world, terrain_n: int = 70) -> None:
        self.world = world
        self.terrain_n = terrain_n
        self.snaps: list[Snap] = []
        self._terrain: ArrayF | None = None

    def refresh_terrain(self) -> None:
        self._terrain = (sample_terrain(self.world, self.terrain_n)
                         if self.world.dim >= 2 else None)

    def snap(self, agent, events: list | None = None,
             meters: dict | None = None, extra: dict | None = None) -> None:
        if self._terrain is None and self.world.dim >= 2:
            self.refresh_terrain()
        opt = getattr(agent, "option_target", None)
        path = None
        if opt is not None and self.world.dim >= 2:
            centre = self.world.lo + (np.array(opt) + 0.5) / agent.plan_res * \
                (self.world.hi - self.world.lo)
            path = np.stack([np.asarray(agent.x, float)[:2], centre[:2]])
        self.snaps.append(Snap(
            tick=agent.tick_no, x=np.asarray(agent.x, float).copy(),
            particles=memory_particles(agent), k_eff=k_eff_of(agent),
            option_cell=opt, option_path_xy=path,
            boost=getattr(agent, "boost_ticks", 0) > 0,
            ok=bool(agent.monitor["ok"]), noise=float(self.world.noise),
            events=list(events or []), terrain=self._terrain,
            meters=dict(meters or {}), extra=dict(extra or {})))


# ---------------------------------------------------------------------------
#  Frame rendering (matplotlib Agg -> PIL)
# ---------------------------------------------------------------------------

def _fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = Image.fromarray(buf[..., :3].copy())
    plt.close(fig)
    return img


def _draw_overlays(ax, snap: Snap, world, trail: list, color: str,
                   vmax_p: float) -> None:
    lo, hi = world.lo, world.hi
    if len(trail) >= 2:
        t = np.array(trail)
        ax.plot(t[:, 0], t[:, 1], color=color, lw=1.4,
                alpha=STYLE["trail_alpha"], solid_capstyle="round")
    for z, v, ringed, readable, _ctx in snap.particles:
        if abs(v) < 1e-3:
            continue
        c = STYLE["pos_particle"] if v >= 0 else STYLE["neg_particle"]
        size = 14 + 70 * min(abs(v) / vmax_p, 1.0)
        alpha = 0.85 if readable else 0.18
        if v >= 0:
            ax.scatter([z[0]], [z[1]], s=size, c=c, marker="o", alpha=alpha,
                       edgecolors=STYLE["ring"] if ringed else "none",
                       linewidths=1.4 if ringed else 0.0, zorder=4)
        else:
            ax.scatter([z[0]], [z[1]], s=size, c=c, marker="x", alpha=alpha,
                       linewidths=1.6, zorder=4)
    if snap.option_path_xy is not None:
        p = snap.option_path_xy
        ax.plot(p[:, 0], p[:, 1], ls=":", lw=1.6, color=STYLE["flag"], zorder=5)
        ax.scatter([p[1, 0]], [p[1, 1]], marker="^", s=120, c=STYLE["flag"],
                   edgecolors="k", zorder=6)
    # the trust glow: halo size/brightness follows the local k_eff
    glow = np.clip(snap.k_eff / 8.0, 0.05, 1.0)
    ax.scatter([snap.x[0]], [snap.x[1]], s=700 * glow + 80, c=STYLE["halo"],
               alpha=0.10 + 0.30 * glow, zorder=6, linewidths=0)
    ax.scatter([snap.x[0]], [snap.x[1]], s=46, c=color,
               edgecolors=STYLE["agent_edge"], linewidths=1.0, zorder=7)
    if snap.boost:
        ax.scatter([snap.x[0]], [snap.x[1]], s=210, facecolors="none",
                   edgecolors="#ff5050", linewidths=1.6, zorder=7)
    if snap.noise > 0.3:                       # fog: the world is hard to see
        rng = np.random.default_rng(snap.tick)
        fog = rng.random((24, 24))
        ax.imshow(fog, extent=[lo[0], hi[0], lo[1], hi[1]], origin="lower",
                  cmap="gray", alpha=min(0.45, 0.55 * snap.noise), zorder=3)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xticks([])
    ax.set_yticks([])


def render_frame(snap: Snap, world, *, title: str = "", trail: list | None = None,
                 vlim: tuple | None = None, size: float = 4.2) -> Image.Image:
    """One 2-D keyframe: terrain + overlays + banner + meter strip."""
    fig, ax = plt.subplots(figsize=(size, size * 1.06), dpi=100)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.075)
    if snap.terrain is not None:
        vmin, vmax = vlim if vlim else (snap.terrain.min(), snap.terrain.max())
        ax.imshow(snap.terrain, extent=[world.lo[0], world.hi[0],
                                        world.lo[1], world.hi[1]],
                  origin="lower", cmap=STYLE["terrain_cmap"], vmin=vmin, vmax=vmax)
    _draw_overlays(ax, snap, world, trail or [], STYLE["agent"], vmax_p=1.5)
    banner = title + ("   " + "  ".join(snap.events) if snap.events else "")
    fig.text(0.02, 0.965, banner, color="w", fontsize=9, family="monospace",
             va="top", bbox=dict(facecolor=STYLE["banner_bg"], pad=2.5, alpha=0.85))
    if snap.meters:
        strip = "   ".join(f"{k}: {v}" for k, v in snap.meters.items())
        fig.text(0.02, 0.012, strip, color="#dddddd", fontsize=8.5,
                 family="monospace")
    fig.patch.set_facecolor("#0c0c12")
    return _fig_to_image(fig)


def render_profile_frame(snap: Snap, world, *, title: str = "",
                         trail: list | None = None, size: float = 4.6,
                         profile_n: int = 240) -> Image.Image:
    """The 1-D side view (the canyon): terrain height = coherence profile; the
    agent walks the curve; particles sit on it; the option flag marks the
    committed destination."""
    xs = np.linspace(world.lo[0], world.hi[0], profile_n)
    prof = np.array([world.true_coherence(np.array([x])) for x in xs])
    fig, ax = plt.subplots(figsize=(size, size * 0.62), dpi=100)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06)
    ax.fill_between(xs, prof.min() - 0.3, prof, color="#3a3a55", zorder=1)
    ax.plot(xs, prof, color="#9090c0", lw=1.5, zorder=2)

    def h(p):
        return float(world.true_coherence(np.array([float(p)])))

    if trail:
        tx = [float(t[0]) for t in trail]
        ax.plot(tx, [h(p) + 0.06 for p in tx], color=STYLE["agent"], lw=1.2,
                alpha=0.5, zorder=3)
    for z, v, ringed, readable, _ctx in snap.particles:
        if abs(v) < 1e-3:
            continue
        c = STYLE["pos_particle"] if v >= 0 else STYLE["neg_particle"]
        if v >= 0:
            ax.scatter([z[0]], [h(z[0]) + 0.12], s=12 + 50 * min(abs(v), 1.5),
                       c=c, marker="o", alpha=0.85 if readable else 0.2,
                       edgecolors=STYLE["ring"] if ringed else "none",
                       linewidths=1.2 if ringed else 0.0, zorder=4)
        else:
            ax.scatter([z[0]], [h(z[0]) + 0.12], s=12 + 50 * min(abs(v), 1.5),
                       c=c, marker="x", alpha=0.85 if readable else 0.2,
                       linewidths=1.4, zorder=4)
    if snap.option_cell is not None:
        ox = world.lo[0] + (snap.option_cell[0] + 0.5) / 10.0 * \
            (world.hi[0] - world.lo[0])
        ax.scatter([ox], [h(ox) + 0.35], marker="v", s=110, c=STYLE["flag"],
                   edgecolors="k", zorder=6)
        ax.plot([snap.x[0], ox], [h(snap.x[0]) + 0.3, h(ox) + 0.3], ls=":",
                lw=1.5, color=STYLE["flag"], zorder=5)
    glow = np.clip(snap.k_eff / 8.0, 0.05, 1.0)
    ax.scatter([snap.x[0]], [h(snap.x[0]) + 0.06], s=600 * glow + 60,
               c=STYLE["halo"], alpha=0.10 + 0.30 * glow, zorder=6, linewidths=0)
    ax.scatter([snap.x[0]], [h(snap.x[0]) + 0.06], s=46, c=STYLE["agent"],
               edgecolors=STYLE["agent_edge"], linewidths=1.0, zorder=7)
    banner = title + ("   " + "  ".join(snap.events) if snap.events else "")
    fig.text(0.02, 0.955, banner, color="w", fontsize=9, family="monospace",
             va="top", bbox=dict(facecolor=STYLE["banner_bg"], pad=2.5, alpha=0.85))
    if snap.meters:
        fig.text(0.02, 0.02, "   ".join(f"{k}: {v}" for k, v in snap.meters.items()),
                 color="#dddddd", fontsize=8.5, family="monospace")
    ax.set_xlim(world.lo[0], world.hi[0])
    ax.set_ylim(prof.min() - 0.3, prof.max() + 0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.patch.set_facecolor("#0c0c12")
    ax.set_facecolor("#0c0c12")
    return _fig_to_image(fig)


def hstack_images(imgs: list[Image.Image], pad: int = 4) -> Image.Image:
    """Side-by-side composition (the twins)."""
    h = max(i.height for i in imgs)
    w = sum(i.width for i in imgs) + pad * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), "#0c0c12")
    x = 0
    for i in imgs:
        out.paste(i, (x, (h - i.height) // 2))
        x += i.width + pad
    return out


def vstack_images(imgs: list[Image.Image], pad: int = 4) -> Image.Image:
    """Stacked composition (wide profile panels)."""
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs) + pad * (len(imgs) - 1)
    out = Image.new("RGB", (w, h), "#0c0c12")
    y = 0
    for i in imgs:
        out.paste(i, ((w - i.width) // 2, y))
        y += i.height + pad
    return out


# ---------------------------------------------------------------------------
#  GIF assembly with a hard size budget (the files get committed)
# ---------------------------------------------------------------------------

def save_gif(frames: list[Image.Image], path: str, *, fps: int | None = None,
             budget_mb: float | None = None) -> dict:
    """Stitch frames to an animated GIF under the size budget: progressively
    drop every 2nd frame / downscale until it fits. Returns what was done."""
    assert frames, "no frames to save"
    fps = fps or STYLE["fps"]
    budget = (budget_mb or STYLE["gif_budget_mb"]) * 1e6
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cur, scale, dropped = list(frames), 1.0, 0
    for _attempt in range(6):
        imgs = cur if scale >= 0.999 else [
            f.resize((int(f.width * scale), int(f.height * scale)),
                     Image.LANCZOS) for f in cur]
        pal = [im.quantize(colors=128, method=Image.MEDIANCUT) for im in imgs]
        pal[0].save(path, save_all=True, append_images=pal[1:],
                    duration=int(1000 / fps), loop=0, optimize=True)
        size = os.path.getsize(path)
        if size <= budget:
            return {"path": path, "bytes": size, "frames": len(cur),
                    "scale": scale, "dropped_halvings": dropped}
        if len(cur) > 60:
            cur = cur[::2]
            dropped += 1
        else:
            scale *= 0.8
    return {"path": path, "bytes": os.path.getsize(path), "frames": len(cur),
            "scale": scale, "dropped_halvings": dropped, "over_budget": True}


def save_png(img: Image.Image, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path, optimize=True)
    return path


# ---------------------------------------------------------------------------
#  Self-check (headless smoke test of the full render path)
# ---------------------------------------------------------------------------

def main() -> None:
    import tempfile

    from .ibf_asi import ASIWorld, IBFASI

    print("terrarium self-check: world + agent + recorder + frame + gif ...")
    w = ASIWorld(seed=0, noise=0.4)
    a = IBFASI(w, seed=100, model_planner=True)
    rec = Recorder(w)
    trail: list = []
    frames = []
    for t in range(40):
        w.tick()
        a.step()
        trail.append(np.asarray(a.x, float).copy())
        if t % 2 == 0:
            rec.snap(a, events=(["TEST EVENT"] if t == 20 else []),
                     meters={"tick": t})
            frames.append(render_frame(rec.snaps[-1], w, title="smoke test",
                                       trail=trail[-40:]))
    with tempfile.TemporaryDirectory() as d:
        info = save_gif(frames, os.path.join(d, "smoke.gif"), budget_mb=2.0)
        png = save_png(frames[-1], os.path.join(d, "smoke.png"))
        assert os.path.getsize(info["path"]) <= 2.0e6, "GIF budget machinery failed"
        assert os.path.getsize(png) > 0
        print(f"  gif: {info['bytes'] / 1e6:.2f} MB, {info['frames']} frames "
              f"(scale {info['scale']:.2f})  png: ok")
    # the 1-D profile path
    from .ibf_asi_regimes import CorridorWorld
    cw = CorridorWorld(seed=0, noise=0.15, drift=0.0, phase_drift=0.02, ripple=0.05)
    ca = IBFASI(cw, seed=100, model_planner=True, n_jumps=0, two_sided_k=False)
    ca.x = np.array([-5.0])
    rec2 = Recorder(cw)
    for _ in range(30):
        cw.tick()
        ca.step()
    rec2.snap(ca)
    img = render_profile_frame(rec2.snaps[-1], cw, title="corridor smoke")
    assert img.width > 100
    print("  profile frame: ok")
    print("terrarium self-check PASSED.")


if __name__ == "__main__":
    main()
