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
    filter_baseline_to_camp,
    filter_points_to_camp,
    filter_shelters_to_camp,
    load_accessibility,
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


def fine_pop_density(
    camp,
    shelters: list[dict],
    total_pop: float,
    cell_size_m: float = 10.0,
    smooth_sigma: float = 1.2,
):
    """Allocate `total_pop` to a fine grid by shelter-area-weighted shares.

    Each shelter's `area` (m²) is binned by its centroid into a fine cell;
    every cell's population = total_pop × (shelter_area_in_cell / total_shelter_area).
    A light Gaussian blur softens the result for cleaner visualization.

    Returns (density_2d, x_min, y_min, x_max, y_max) where density is
    people-per-cell at the chosen `cell_size_m`.
    """
    bx0 = float(camp.merc[:, 0].min())
    by0 = float(camp.merc[:, 1].min())
    bx1 = float(camp.merc[:, 0].max())
    by1 = float(camp.merc[:, 1].max())

    nx = max(1, int(np.ceil((bx1 - bx0) / cell_size_m)))
    ny = max(1, int(np.ceil((by1 - by0) / cell_size_m)))
    grid = np.zeros((ny, nx), dtype=float)

    total_area = 0.0
    for sh in shelters:
        outer = sh["rings"][0]
        rx = [p[0] for p in outer]
        ry = [p[1] for p in outer]
        cx = sum(rx) / len(rx)
        cy = sum(ry) / len(ry)
        a = float(sh.get("area") or 0.0)
        if a <= 0:
            continue
        ix = int((cx - bx0) / cell_size_m)
        iy = int((cy - by0) / cell_size_m)
        if 0 <= ix < nx and 0 <= iy < ny:
            grid[iy, ix] += a
            total_area += a

    if total_area > 0:
        grid *= (total_pop / total_area)

    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            grid = gaussian_filter(grid, sigma=smooth_sigma)
        except ImportError:
            pass

    return grid, bx0, by0, bx1, by1


def render(camp_name: str, out_path: Path, *, dpi: int = 200,
           overlay: str = "shelter", fine_cellsize: float = 10.0) -> None:
    print(f"Loading {camp_name} layers (overlay={overlay})...")
    camp = load_camp_polygon(camp_name)
    cf = load_common_facilities(camp_name)
    latr = load_latrines_2022()
    latr_in = filter_points_to_camp(latr.xy_merc, camp)
    sensitive = cf.sensitive_xy()
    if len(sensitive):
        sensitive = filter_points_to_camp(sensitive, camp)
    shelters: list[dict] = []
    baseline = None
    fine_density = None
    show_shelter_overlay = overlay in ("shelter", "fine_pop_density", "fine_pop_female")
    if show_shelter_overlay:
        shelters_all = load_shelter_polygons(2022)
        shelters = filter_shelters_to_camp(shelters_all, camp)
        print(f"  shelters in camp: {len(shelters)}")
    if overlay in ("pop_total", "pop_female", "pop_density",
                   "fine_pop_density", "fine_pop_female"):
        base_full = load_accessibility("ACC22_S2.gpkg")
        baseline = filter_baseline_to_camp(base_full, camp)
        print(f"  demand cells: {len(baseline.cell_x)}, "
              f"sum total pop: {baseline.pop_total.sum():.0f}, "
              f"sum female pop: {baseline.pop_female.sum():.0f}")
    if overlay in ("fine_pop_density", "fine_pop_female"):
        total_pop = (
            float(baseline.pop_female.sum())
            if overlay == "fine_pop_female"
            else float(baseline.pop_total.sum())
        )
        fine_density = fine_pop_density(
            camp, shelters, total_pop, cell_size_m=fine_cellsize, smooth_sigma=1.2
        )
        print(f"  fine grid: {fine_density[0].shape} at {fine_cellsize:.0f} m")
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

    overlay_pc = None
    overlay_label = ""
    overlay_image = None  # imshow handle for fine grid

    # Fine-grid population density: drawn FIRST so shelter footprints and
    # everything else sit on top.
    if fine_density is not None:
        from matplotlib.colors import LinearSegmentedColormap

        grid, gx0, gy0, gx1, gy1 = fine_density
        # mask cells outside camp polygon for cleaner edges
        ny, nx = grid.shape
        cs = (gx1 - gx0) / nx
        ix = np.arange(nx)
        iy = np.arange(ny)
        Cx = gx0 + (ix + 0.5) * cs
        Cy = gy0 + (iy + 0.5) * cs
        XX, YY = np.meshgrid(Cx, Cy)
        in_poly = points_in_poly(XX.ravel(), YY.ravel(), camp.merc_list()).reshape(grid.shape)
        masked = np.where(in_poly, grid, np.nan)

        cmap = LinearSegmentedColormap.from_list(
            "pop_fine", ["#f6efe2", "#dc9a6a", "#a33d3d"]
        )
        cmap.set_bad(color=PALETTE["bg"])
        overlay_image = ax.imshow(
            masked,
            extent=(gx0, gx1, gy0, gy1),
            origin="lower",
            cmap=cmap,
            alpha=0.95,
            interpolation="bilinear",
            zorder=1,
        )
        nz = grid[in_poly & (grid > 0)]
        if len(nz):
            overlay_image.set_clim(0, float(np.quantile(nz, 0.97)))
        overlay_label = (
            f"Female population per {fine_cellsize:.0f} m cell (2022)"
            if overlay == "fine_pop_female"
            else f"Population per {fine_cellsize:.0f} m cell (2022)"
        )

    # Shelter polygons: dim outlines when the fine heatmap is below them,
    # filled patches otherwise.
    if shelters:
        patches = [MplPolygon(sh["rings"][0]) for sh in shelters]
        if fine_density is not None:
            pc = PatchCollection(
                patches,
                facecolor="none",
                edgecolor=PALETTE["shelter_edge"],
                alpha=0.35,
                linewidths=0.4,
                zorder=2,
            )
        else:
            pc = PatchCollection(
                patches,
                facecolor=PALETTE["shelter_fill"],
                edgecolor=PALETTE["shelter_edge"],
                alpha=0.55,
                linewidths=0.25,
                zorder=2,
            )
        ax.add_collection(pc)
    if baseline is not None and overlay in ("pop_total", "pop_female", "pop_density"):
        from matplotlib.colors import LinearSegmentedColormap

        if overlay == "pop_total":
            vals = baseline.pop_total
            overlay_label = "Total population per 50 m cell (2022)"
        elif overlay == "pop_female":
            vals = baseline.pop_female
            overlay_label = "Female population per 50 m cell (2022)"
        elif overlay == "pop_density":
            # per-hectare density (50 m cell = 0.25 ha)
            vals = baseline.pop_total / 0.25
            overlay_label = "Population density (people / ha, 2022)"
        cmap = LinearSegmentedColormap.from_list(
            "pop", ["#f6efe2", "#dc9a6a", "#a33d3d"]
        )
        patches = [MplPolygon(r) for r in baseline.rings]
        overlay_pc = PatchCollection(
            patches,
            array=np.asarray(vals, dtype=float),
            cmap=cmap,
            edgecolors="none",
            alpha=0.92,
        )
        nz = vals[vals > 0]
        if len(nz):
            overlay_pc.set_clim(0, float(np.quantile(nz, 0.97)))
        ax.add_collection(overlay_pc)

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

    title_suffix = "" if overlay == "shelter" else f" — {overlay_label}"
    ax.set_title(
        f"{camp_name}: optimizer context layers (2022){title_suffix}",
        fontsize=12,
        color=PALETTE["title"],
        loc="left",
        pad=12,
    )

    if overlay_pc is not None:
        cbar = fig.colorbar(overlay_pc, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(overlay_label, fontsize=9, color=PALETTE["muted"])
        cbar.ax.tick_params(labelsize=8, color=PALETTE["muted"])
    elif overlay_image is not None:
        cbar = fig.colorbar(overlay_image, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(overlay_label, fontsize=9, color=PALETTE["muted"])
        cbar.ax.tick_params(labelsize=8, color=PALETTE["muted"])

    # Legend
    legend_x = bx1 - 320
    legend_y = by1 - 30
    legend_items = []
    if shelters:
        legend_items.append(
            (PALETTE["shelter_fill"], "shelter footprint", "patch")
        )
    legend_items += [
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
    stats: list[str] = []
    if shelters:
        stats.append(f"{len(shelters):,} shelter footprints")
        stats.append(
            f"{sum((sh.get('area') or 0) for sh in shelters):.0f} m² shelter area"
        )
    if baseline is not None:
        stats.append(f"{baseline.pop_total.sum():,.0f} total people (2022)")
        stats.append(f"{baseline.pop_female.sum():,.0f} female (2022)")
        stats.append(f"{(baseline.pop_total>0).sum()} / {len(baseline.cell_x)} cells populated")
    stats += [
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
    p.add_argument(
        "--overlay",
        default="shelter",
        choices=(
            "shelter",
            "pop_total",
            "pop_female",
            "pop_density",
            "fine_pop_density",
            "fine_pop_female",
        ),
        help="background fill for the map",
    )
    p.add_argument("--fine-cellsize", type=float, default=10.0,
                   help="cell size in meters for fine_pop_* overlays")
    p.add_argument("--out", default=None,
                   help="output PNG path; default depends on overlay")
    args = p.parse_args()
    if args.out:
        out = Path(args.out)
    else:
        slug = args.camp.lower().replace(" ", "")
        stem = "layered_map" if args.overlay == "shelter" else f"layered_map_{args.overlay}"
        out = PROJECT_ROOT / "results" / slug / "figures" / f"{stem}.png"
    render(args.camp, out, overlay=args.overlay, fine_cellsize=args.fine_cellsize)


if __name__ == "__main__":
    main()
