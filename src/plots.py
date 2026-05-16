"""Figures for the WASH siting proposal.

Generates:
- `<camp>_baseline_LT_t.png` — baseline total latrine accessibility heatmap
- `<camp>_baseline_LT_f.png` — baseline female latrine accessibility
- `<camp>_chosen_sites.png`   — overlay of greedy and IP selections
- `<camp>_post_LT_t.png`      — post-placement total accessibility
- `<camp>_delta_LT_t.png`     — change map
- `<camp>_pareto.png`         — efficiency-vs-equity Pareto plot

Uses matplotlib only — no geopandas. Styling matches
`proposal/make_camp22_parseability_figure.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


PALETTE = {
    "low": "#e65446",
    "mid": "#f7ecbe",
    "high": "#408884",
    "ink": "#272f33",
    "muted": "#4a525a",
    "latrine": "#1f4e79",
    "sensitive": "#5f4490",
    "chosen_greedy": "#d9531e",
    "chosen_ip": "#2d8d9b",
    "bg": "#f8f7f2",
}


def _cmap_lmh(low: str, mid: str, high: str):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("lmh", [low, mid, high])


def heatmap(
    cell_rings: list[list[tuple[float, float]]],
    values: np.ndarray,
    camp_polygon: list[tuple[float, float]],
    *,
    title: str,
    ax: plt.Axes | None = None,
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
    overlays: list[tuple[np.ndarray, dict]] | None = None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)
    else:
        fig = ax.figure

    if cmap is None:
        cmap = _cmap_lmh(PALETTE["low"], PALETTE["mid"], PALETTE["high"])

    patches = [MplPolygon(ring) for ring in cell_rings]
    pc = PatchCollection(patches, array=np.asarray(values), cmap=cmap,
                          edgecolors="none")
    if vmin is None:
        vmin = float(np.percentile(values, 5)) if len(values) else 0.0
    if vmax is None:
        vmax = float(np.percentile(values, 95)) if len(values) else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)

    # camp boundary
    xs = [p[0] for p in camp_polygon] + [camp_polygon[0][0]]
    ys = [p[1] for p in camp_polygon] + [camp_polygon[0][1]]
    ax.plot(xs, ys, color=PALETTE["ink"], lw=1.5)

    if overlays:
        for pts, kwargs in overlays:
            if len(pts) == 0:
                continue
            ax.scatter(pts[:, 0], pts[:, 1], **kwargs)

    # bounds
    polyx = np.array([p[0] for p in camp_polygon])
    polyy = np.array([p[1] for p in camp_polygon])
    pad = 60
    ax.set_xlim(polyx.min() - pad, polyx.max() + pad)
    ax.set_ylim(polyy.min() - pad, polyy.max() + pad)
    ax.set_aspect("equal")
    ax.set_facecolor(PALETTE["bg"])
    ax.set_title(title, fontsize=11, color=PALETTE["ink"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["muted"])
        spine.set_linewidth(0.6)

    cbar = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=8, color=PALETTE["muted"])

    # scale bar 200 m
    bx0, _ = ax.get_xlim()
    by0, _ = ax.get_ylim()
    sx, sy = bx0 + 80, by0 + 40
    ax.plot([sx, sx + 200], [sy, sy], color=PALETTE["ink"], lw=2.5)
    ax.text(sx + 100, sy + 12, "200 m", ha="center", fontsize=8,
            color=PALETTE["muted"])

    return fig, ax


def pareto_plot(rows: list[dict], *, save_path: Path):
    """`rows` from results/camp22/summary.csv; one frontier per method."""
    fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=150)
    methods = {}
    for r in rows:
        methods.setdefault(r["method"], []).append(r)
    for m, group in methods.items():
        group = sorted(group, key=lambda r: r["lambda"])
        # x = equity gain, y = efficiency gain (both as Δ pop-weighted mean)
        x = [r["pop_w_mean_LT_f_after"] - r["pop_w_mean_LT_f_before"] for r in group]
        y = [r["pop_w_mean_LT_t_after"] - r["pop_w_mean_LT_t_before"] for r in group]
        ax.plot(x, y, "o-", label=m, lw=1.8, markersize=5,
                color=PALETTE["chosen_greedy"] if m == "greedy" else PALETTE["chosen_ip"])
        for r, xi, yi in zip(group, x, y):
            ax.annotate(f"λ={r['lambda']:.2f}", (xi, yi), fontsize=7,
                        xytext=(4, 4), textcoords="offset points",
                        color=PALETTE["muted"])
    ax.set_xlabel("Δ pop-weighted mean female latrine accessibility")
    ax.set_ylabel("Δ pop-weighted mean total latrine accessibility")
    ax.set_title("Pareto: efficiency vs. female-equity gain", fontsize=10)
    ax.grid(True, color=PALETTE["muted"], alpha=0.15)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
