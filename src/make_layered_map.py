"""Render a context map for a camp.

Stacks every spatial layer we have on top of the camp boundary so reviewers
can see what the optimizer is reasoning over:

- shelter footprints (CV segmentation output, light fill)
- footpath / access-road network (thin lines)
- existing latrines (small dots)
- sensitive common facilities (squares)
- camp boundary (heavy outline)

Run:
    python3 src/make_layered_map.py --camp "Camp 22" --out results/camp22/figures/layered_map.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as MplPolygon

from geo import points_in_poly
from loaders import (
    PROJECT_ROOT,
    filter_points_to_camp,
    filter_shelters_to_camp,
    load_camp_polygon,
    load_common_facilities,
    load_footpath_polylines,
    load_shelter_polygons,
)
from load_latrines import load_latrines_2022


PALETTE = {
    "bg": "#f8f7f2",
    "ink": "#272f33",
    "muted": "#4a525a",
    "boundary": "#272f33",
    "shelter_fill": "#c08560",
    "shelter_edge": "#8a5a3c",
    "footpath": "#6a7d6f",
    "latrine": "#1f4e79",
    "sensitive": "#5f4490",
    "title": "#1a2226",
}


def _filter_polylines_to_bbox(
    parts: list[list[tuple[float, float]]],
    bx0: float,
    by0: float,
    bx1: float,
    by1: float,
) -> list[np.ndarray]:
    """Keep polyline parts that have any vertex inside the bbox."""
    kept: list[np.ndarray] = []
    for part in parts:
        arr = np.asarray(part, dtype=float)
        if not len(arr):
            continue
        in_x = (arr[:, 0] >= bx0) & (arr[:, 0] <= bx1)
        in_y = (arr[:, 1] >= by0) & (arr[:, 1] <= by1)
        if (in_x & in_y).any():
            kept.append(arr)
    return kept


def render(camp_name: str, out_path: Path, *, dpi: int = 200) -> None:
    print(f"Loading {camp_name} layers...")
    camp = load_camp_polygon(camp_name)
    cf = load_common_facilities(camp_name)
    latr = load_latrines_2022()
    latr_in = filter_points_to_camp(latr.xy_merc, camp)
    sensitive = cf.sensitive_xy()
    if len(sensitive):
        sensitive = filter_points_to_camp(sensitive, camp)
    shelters_all = load_shelter_polygons(2022)
    shelters = filter_shelters_to_camp(shelters_all, camp)
    print(f"  shelters in camp: {len(shelters)}")
    footpaths = load_footpath_polylines()
    bx0 = camp.merc[:, 0].min() - 150
    by0 = camp.merc[:, 1].min() - 150
    bx1 = camp.merc[:, 0].max() + 150
    by1 = camp.merc[:, 1].max() + 150
    footpaths_in = _filter_polylines_to_bbox(footpaths, bx0, by0, bx1, by1)
    print(f"  footpath segments touching camp bbox: {len(footpaths_in)}")
    print(f"  existing latrines: {len(latr_in)}, sensitive sites: {len(sensitive)}")

    fig, ax = plt.subplots(figsize=(9.5, 8), dpi=dpi)
    ax.set_facecolor(PALETTE["bg"])

    # Shelters: a soft fill to suggest density without dominating
    if shelters:
        patches = [MplPolygon(sh["rings"][0]) for sh in shelters]
        pc = PatchCollection(
            patches,
            facecolor=PALETTE["shelter_fill"],
            edgecolor=PALETTE["shelter_edge"],
            alpha=0.55,
            linewidths=0.25,
        )
        ax.add_collection(pc)

    # Footpaths
    if footpaths_in:
        segments = []
        for arr in footpaths_in:
            for i in range(len(arr) - 1):
                segments.append([arr[i], arr[i + 1]])
        lc = LineCollection(
            segments, colors=PALETTE["footpath"], linewidths=0.9, alpha=0.85
        )
        ax.add_collection(lc)

    # Latrines
    if len(latr_in):
        ax.scatter(
            latr_in[:, 0],
            latr_in[:, 1],
            s=4,
            c=PALETTE["latrine"],
            alpha=0.7,
            linewidths=0,
            zorder=4,
        )

    # Sensitive common facilities
    if len(sensitive):
        ax.scatter(
            sensitive[:, 0],
            sensitive[:, 1],
            s=22,
            marker="s",
            facecolors=PALETTE["sensitive"],
            edgecolors=PALETTE["bg"],
            linewidths=0.4,
            alpha=0.85,
            zorder=5,
        )

    # Camp boundary
    xs = camp.merc[:, 0].tolist() + [camp.merc[0, 0]]
    ys = camp.merc[:, 1].tolist() + [camp.merc[0, 1]]
    ax.plot(xs, ys, color=PALETTE["boundary"], lw=2.2, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by0, by1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(PALETTE["muted"])
        s.set_linewidth(0.6)

    ax.set_title(
        f"{camp_name}: optimizer context layers (2022)",
        fontsize=13,
        color=PALETTE["title"],
        loc="left",
        pad=12,
    )

    # Legend
    legend_x = bx1 - 320
    legend_y = by1 - 30
    legend_items = [
        (PALETTE["shelter_fill"], "shelter footprint", "patch"),
        (PALETTE["footpath"], "footpath / access road", "line"),
        (PALETTE["latrine"], f"existing latrine ({len(latr_in)})", "dot"),
        (PALETTE["sensitive"], f"sensitive facility ({len(sensitive)})", "square"),
        (PALETTE["boundary"], "camp boundary", "line2"),
    ]
    y = legend_y
    for color, label, kind in legend_items:
        if kind == "patch":
            ax.add_patch(
                plt.Rectangle(
                    (legend_x, y - 18),
                    24,
                    18,
                    facecolor=color,
                    edgecolor=PALETTE["shelter_edge"],
                    alpha=0.55,
                    linewidth=0.4,
                )
            )
        elif kind == "line":
            ax.plot([legend_x, legend_x + 24], [y - 9, y - 9], color=color, lw=1.1)
        elif kind == "line2":
            ax.plot([legend_x, legend_x + 24], [y - 9, y - 9], color=color, lw=2.2)
        elif kind == "dot":
            ax.scatter([legend_x + 12], [y - 9], s=12, c=color, alpha=0.8, linewidths=0)
        elif kind == "square":
            ax.scatter(
                [legend_x + 12], [y - 9], s=22, marker="s", c=color, alpha=0.85, linewidths=0
            )
        ax.text(legend_x + 32, y - 12, label, fontsize=9, color=PALETTE["muted"])
        y -= 28

    # Stats box (lower-right)
    stats = [
        f"{len(shelters):,} shelter footprints",
        f"{sum((sh.get('area') or 0) for sh in shelters):.0f} m² shelter area",
        f"{len(latr_in):,} latrines",
        f"{len(footpaths_in):,} footpath segments",
        f"{len(sensitive):,} sensitive sites",
    ]
    y = by0 + 80 + (len(stats) - 1) * 22
    for s in stats:
        ax.text(bx0 + 22, y, s, fontsize=9, color=PALETTE["muted"])
        y -= 22

    # 200 m scale bar
    sx = bx0 + 40
    sy = by0 + 40
    ax.plot([sx, sx + 200], [sy, sy], color=PALETTE["ink"], lw=2.8)
    ax.plot([sx, sx], [sy - 6, sy + 6], color=PALETTE["ink"], lw=1.4)
    ax.plot([sx + 200, sx + 200], [sy - 6, sy + 6], color=PALETTE["ink"], lw=1.4)
    ax.text(sx + 100, sy + 12, "200 m", ha="center", fontsize=8, color=PALETTE["muted"])

    # North arrow
    nx, ny = bx0 + 60, by1 - 80
    ax.annotate(
        "",
        xy=(nx, ny + 40),
        xytext=(nx, ny - 10),
        arrowprops=dict(facecolor=PALETTE["ink"], width=2, headwidth=8, edgecolor="none"),
    )
    ax.text(nx, ny + 50, "N", ha="center", fontsize=9, color=PALETTE["ink"], fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camp", default="Camp 22")
    p.add_argument("--out", default=None,
                   help="output PNG path; default = results/<slug>/figures/layered_map.png")
    args = p.parse_args()
    if args.out:
        out = Path(args.out)
    else:
        slug = args.camp.lower().replace(" ", "")
        out = PROJECT_ROOT / "results" / slug / "figures" / "layered_map.png"
    render(args.camp, out)


if __name__ == "__main__":
    main()
