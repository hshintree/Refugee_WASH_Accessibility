"""Fine-grain accessibility map for a camp.

The heatmap is the **E2SFCA latrine-accessibility score** evaluated on a
finer grid (default 10 m) instead of the published 50 m grid.

Key trick. E2SFCA has two steps:

    Step 1:  R_j = n_j / Σ_i K_ij p_i              # facility competition
    Step 2:  A_i = Σ_j K_ij R_j                    # cell-side aggregation

`R_j` depends on the aggregate demand across the whole catchment — using
fine cells for Step 1 mishandles boundary effects (cells outside the camp
have no allocated population, so latrines near the edge see an
artificially low denom and `R_j` blows up). The fix:

- **Step 1 on Ahn et al.'s 50 m grid**, using the published per-cell
  populations from `ACC22_S2.gpkg`. This matches the paper exactly and
  inherits their global competition.
- **Step 2 at fine resolution**. Once `R_j` is fixed, A_i at *any* point
  is just a sum of kernel-weighted facility supply ratios — there is no
  competition coupling, so we can evaluate it at 10 m, 5 m, or a
  continuous interpolation without breaking anything.

Distance is network distance over the footpath graph by default (same
Dijkstra path the optimizer uses), with Euclidean as a `--distance`
option.

Run:
    python3 src/make_accessibility_map.py --camp "Camp 22" --cellsize 10 \\
        --metric LT_t --distance network
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

from e2sfca import D0, SIGMA, e2sfca, kernel_matrix
from geo import points_in_poly
from load_latrines import filter_latrines_to_bbox, load_latrines_2022
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
from make_layered_map import PALETTE, _filter_polylines_to_bbox
from network_distance import build_graph, network_distance_matrix


SPHERE_TARGET = 1.0 / 20.0


def _fine_grid(camp, cell_size_m: float):
    bx0 = float(camp.merc[:, 0].min())
    by0 = float(camp.merc[:, 1].min())
    bx1 = float(camp.merc[:, 0].max())
    by1 = float(camp.merc[:, 1].max())
    nx = max(1, int(np.ceil((bx1 - bx0) / cell_size_m)))
    ny = max(1, int(np.ceil((by1 - by0) / cell_size_m)))
    Cx = bx0 + (np.arange(nx) + 0.5) * cell_size_m
    Cy = by0 + (np.arange(ny) + 0.5) * cell_size_m
    XX, YY = np.meshgrid(Cx, Cy)
    return XX, YY, bx0, by0, bx1, by1, nx, ny


def _allocate_population_to_grid(
    shelters: list[dict], total_pop: float, XX: np.ndarray, YY: np.ndarray,
    cell_size_m: float, bx0: float, by0: float,
    smooth_sigma_m: float = 20.0,
):
    """Allocate population to a fine grid by centroid-binning shelter area,
    then **spreading it with a Gaussian blur** of width `smooth_sigma_m`.

    Without the blur the population sits on a few hundred isolated cells
    (centroids of the segmented shelters). E2SFCA's facility-side
    denominator `Σ K(d) p` then under-counts demand for latrines that
    happen to sit between cluster centroids, blowing up `R_j` to absurd
    values. A blur of ~20 m (≈ a typical shelter footprint's diameter)
    keeps the spatial structure visible while regularising the supply
    ratios.

    Total population is preserved by re-normalising after the blur.
    """
    ny, nx = XX.shape
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
        grid *= total_pop / total_area

    if smooth_sigma_m > 0:
        from scipy.ndimage import gaussian_filter

        sigma_cells = smooth_sigma_m / cell_size_m
        before_sum = grid.sum()
        grid = gaussian_filter(grid, sigma=sigma_cells, mode="constant", cval=0.0)
        # Re-normalise so the camp total survives the boundary mode
        after_sum = grid.sum()
        if after_sum > 0:
            grid *= before_sum / after_sum

    return grid


def render(
    camp_name: str,
    out_path: Path,
    *,
    cell_size_m: float = 10.0,
    metric: str = "LT_t",  # one of LT_t, LT_f_S1, LT_f_S2
    distance: str = "network",
    dpi: int = 200,
) -> None:
    t0 = time.time()
    print(f"Loading {camp_name} ({metric}, {distance} distance, {cell_size_m:.0f} m grid)...")
    camp = load_camp_polygon(camp_name)
    cf = load_common_facilities(camp_name)
    latr_all = load_latrines_2022()
    latr_in = filter_points_to_camp(latr_all.xy_merc, camp)
    sensitive = cf.sensitive_xy()
    if len(sensitive):
        sensitive = filter_points_to_camp(sensitive, camp)
    shelters_all = load_shelter_polygons(2022)
    shelters = filter_shelters_to_camp(shelters_all, camp)
    base_full = load_accessibility("ACC22_S2.gpkg")
    baseline = filter_baseline_to_camp(base_full, camp)

    if metric == "LT_t":
        coarse_pop = base_full.pop_total
        pop_kind = "total"
        sup_capacity = latr_all.LT
        cb_label = "Total latrine accessibility (E2SFCA)"
    elif metric == "LT_f_S1":
        coarse_pop = base_full.pop_female
        pop_kind = "female"
        sup_capacity = latr_all.LT_female_sum
        cb_label = "Female latrine accessibility (Scenario 1)"
    elif metric == "LT_f_S2":
        coarse_pop = base_full.pop_female
        pop_kind = "female"
        sup_capacity = latr_all.LT_female_S2
        cb_label = "Female latrine accessibility (Scenario 2, safety penalty)"
    else:
        raise ValueError(metric)

    print(f"  using {pop_kind} pop from ACC22_S2.gpkg (global "
          f"{coarse_pop.sum():,.0f}, in-camp {baseline.pop_total.sum():,.0f})")

    XX, YY, bx0, by0, bx1, by1, nx, ny = _fine_grid(camp, cell_size_m)
    fine_xy = np.column_stack([XX.ravel(), YY.ravel()])
    in_poly = points_in_poly(fine_xy[:, 0], fine_xy[:, 1], camp.merc_list())
    print(f"  fine grid: {nx}×{ny} = {nx*ny} cells, "
          f"{int(in_poly.sum())} inside camp")

    # Coarse demand grid for Step 1 = the published GPKG cells (global).
    coarse_xy = np.column_stack([base_full.cell_x, base_full.cell_y])

    # Supply set: all latrines within a buffer wider than the catchment
    # around the fine-grid bbox, so edge cells see cross-camp competition.
    buffer_m = float(D0) + 200.0
    latr_b, sup_mask = filter_latrines_to_bbox(
        latr_all, bx0 - buffer_m, by0 - buffer_m, bx1 + buffer_m, by1 + buffer_m,
    )
    n_caps = sup_capacity[sup_mask]
    print(f"  latrines in supply set (camp+{buffer_m:.0f} m): {len(latr_b.xy_merc)}")

    if distance == "network":
        t = time.time()
        paths = load_footpath_polylines()
        nbox = (bx0 - 2000, by0 - 2000, bx1 + 2000, by1 + 2000)
        paths_local = _filter_polylines_to_bbox(paths, *nbox)
        fg = build_graph(paths_local, snap_tol_m=0.5)
        print(f"  graph: {len(fg.nodes_xy)} nodes, "
              f"{fg.graph.nnz//2} edges, {fg.n_components} components")

        # Step 1 distance: coarse demand cells ↔ supply latrines (uses
        # global demand, so we cover all camps inside the catchment).
        D_step1 = network_distance_matrix(
            coarse_xy, latr_b.xy_merc, fg, d0=D0, buffer_m=200.0,
        )
        # Step 2 distance: fine cells ↔ supply latrines.
        D_step2 = network_distance_matrix(
            fine_xy, latr_b.xy_merc, fg, d0=D0, buffer_m=200.0,
        )
        print(f"  network distance matrices (step1 {D_step1.shape}, "
              f"step2 {D_step2.shape}): {time.time()-t:.1f}s")
    else:
        dx = coarse_xy[:, 0:1] - latr_b.xy_merc[None, :, 0]
        dy = coarse_xy[:, 1:2] - latr_b.xy_merc[None, :, 1]
        D_step1 = np.sqrt(dx * dx + dy * dy)
        dx = fine_xy[:, 0:1] - latr_b.xy_merc[None, :, 0]
        dy = fine_xy[:, 1:2] - latr_b.xy_merc[None, :, 1]
        D_step2 = np.sqrt(dx * dx + dy * dy)
        print(f"  euclidean distance matrices: step1 {D_step1.shape}, step2 {D_step2.shape}")

    # ---- Step 1: facility competition ratio R_j (coarse grid, paper's) ----
    K1 = kernel_matrix(D_step1, d0=D0, sigma=SIGMA)
    denom = (K1 * coarse_pop[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rj = np.where(denom > 0, n_caps / denom, 0.0)
    Rj = np.nan_to_num(Rj, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- Step 2: per-fine-cell accessibility = Σ_j K_step2 * R_j ----
    K2 = kernel_matrix(D_step2, d0=D0, sigma=SIGMA)
    A_flat = (K2 * Rj[None, :]).sum(axis=1)
    A = A_flat.reshape(ny, nx)
    A_masked = np.where(in_poly.reshape(ny, nx), A, np.nan)

    # Allocate population to fine grid (visualization-only — used for the
    # "share below Sphere" stat and the colormap norm).
    pop_grid = _allocate_population_to_grid(
        shelters, float(baseline.pop_total.sum()),
        XX, YY, cell_size_m, bx0, by0,
    )
    pop_flat = pop_grid.ravel()

    print(f"  E2SFCA range [{np.nanmin(A_masked):.4f}, {np.nanmax(A_masked):.4f}], "
          f"mean over populated fine cells: "
          f"{float(np.nanmean(A_masked[pop_grid > 0])):.4f}")

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 8), dpi=dpi)
    ax.set_facecolor(PALETTE["bg"])

    cmap = LinearSegmentedColormap.from_list(
        "acc", ["#a33d3d", "#dc9a6a", "#f7ecbe", "#7fb4a8", "#226e6a"]
    )
    cmap.set_bad(color=PALETTE["bg"])

    nz = A_masked[np.isfinite(A_masked) & (pop_grid > 0)]
    if len(nz):
        vmin = 0.0
        vmax = float(np.quantile(nz, 0.97))
        # Center the colormap on the Sphere target so red = below standard,
        # teal = at-or-above.
        vmax = max(vmax, SPHERE_TARGET * 1.4)
    else:
        vmin, vmax = 0.0, SPHERE_TARGET * 2

    im = ax.imshow(
        A_masked,
        extent=(bx0, bx0 + nx * cell_size_m, by0, by0 + ny * cell_size_m),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=0.95,
        interpolation="bilinear",
        zorder=1,
    )

    # Shelter outlines (subtle)
    if shelters:
        patches = [MplPolygon(sh["rings"][0]) for sh in shelters]
        ax.add_collection(
            PatchCollection(
                patches,
                facecolor="none",
                edgecolor=PALETTE["shelter_edge"],
                alpha=0.4,
                linewidths=0.35,
                zorder=2,
            )
        )

    # Footpaths
    nbox = (bx0 - 150, by0 - 150, bx1 + 150, by1 + 150)
    fp = _filter_polylines_to_bbox(load_footpath_polylines(), *nbox)
    if fp:
        segments = []
        for arr in fp:
            for i in range(len(arr) - 1):
                segments.append([arr[i], arr[i + 1]])
        ax.add_collection(
            LineCollection(
                segments, colors=PALETTE["footpath"], linewidths=0.9, alpha=0.85,
                zorder=3,
            )
        )

    # Latrines
    if len(latr_in):
        ax.scatter(latr_in[:, 0], latr_in[:, 1], s=4, c=PALETTE["latrine"],
                   alpha=0.75, linewidths=0, zorder=4)

    # Sensitive sites
    if len(sensitive):
        ax.scatter(sensitive[:, 0], sensitive[:, 1], s=22, marker="s",
                   facecolors=PALETTE["sensitive"],
                   edgecolors=PALETTE["bg"], linewidths=0.4, alpha=0.85, zorder=5)

    # Boundary
    xs = camp.merc[:, 0].tolist() + [camp.merc[0, 0]]
    ys = camp.merc[:, 1].tolist() + [camp.merc[0, 1]]
    ax.plot(xs, ys, color=PALETTE["boundary"], lw=2.2, zorder=6)

    ax.set_aspect("equal")
    ax.set_xlim(bx0 - 60, bx1 + 60)
    ax.set_ylim(by0 - 60, by1 + 60)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(PALETTE["muted"])
        s.set_linewidth(0.6)
    ax.set_title(
        f"{camp_name}: E2SFCA {metric} at {cell_size_m:.0f} m grid "
        f"({distance} distance, 2022)",
        fontsize=12, color=PALETTE["title"], loc="left", pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(cb_label, fontsize=9, color=PALETTE["muted"])
    cbar.ax.tick_params(labelsize=8, color=PALETTE["muted"])
    # mark the Sphere target on the colorbar
    cbar.ax.axhline(
        SPHERE_TARGET, color=PALETTE["ink"], lw=1.2, linestyle="--",
    )
    cbar.ax.text(
        1.6, SPHERE_TARGET, " Sphere\n target", transform=cbar.ax.get_yaxis_transform(),
        va="center", fontsize=8, color=PALETTE["ink"],
    )

    # legend
    legend_x = bx1 - 320
    legend_y = by1 - 30
    legend_items = [
        (PALETTE["footpath"], "footpath / access road", "line"),
        (PALETTE["latrine"], f"existing latrine ({len(latr_in)})", "dot"),
        (PALETTE["sensitive"], f"sensitive facility ({len(sensitive)})", "square"),
        (PALETTE["shelter_edge"], "shelter outline", "line"),
        (PALETTE["boundary"], "camp boundary", "line2"),
    ]
    y = legend_y
    for color, label, kind in legend_items:
        if kind == "line":
            ax.plot([legend_x, legend_x + 24], [y - 9, y - 9], color=color, lw=1.1)
        elif kind == "line2":
            ax.plot([legend_x, legend_x + 24], [y - 9, y - 9], color=color, lw=2.2)
        elif kind == "dot":
            ax.scatter([legend_x + 12], [y - 9], s=12, c=color, alpha=0.8, linewidths=0)
        elif kind == "square":
            ax.scatter([legend_x + 12], [y - 9], s=22, marker="s", c=color,
                       alpha=0.85, linewidths=0)
        ax.text(legend_x + 32, y - 12, label, fontsize=9, color=PALETTE["muted"])
        y -= 28

    # stats box
    stats = [
        f"{int(baseline.pop_total.sum()):,} people in camp ({pop_kind} used for R_j)",
        f"{len(latr_b.xy_merc):,} latrines in supply set",
        f"{len(shelters):,} shelter footprints",
        f"σ = {SIGMA:.0f} m, d₀ = {D0:.0f} m, Sphere = {SPHERE_TARGET:.3f}",
        f"mean A_i (populated fine): "
        f"{float(np.nanmean(A_masked[pop_grid > 0])):.4f}",
        f"share below Sphere: "
        f"{float(np.nanmean(A_masked[pop_grid > 0] < SPHERE_TARGET)):.1%}",
    ]
    y = by0 + 100 + (len(stats) - 1) * 22
    for s in stats:
        ax.text(bx0 + 22, y, s, fontsize=9, color=PALETTE["muted"])
        y -= 22

    # 200 m scale bar
    sx = bx0 + 40
    sy = by0 + 40
    ax.plot([sx, sx + 200], [sy, sy], color=PALETTE["ink"], lw=2.8)
    ax.plot([sx, sx], [sy - 6, sy + 6], color=PALETTE["ink"], lw=1.4)
    ax.plot([sx + 200, sx + 200], [sy - 6, sy + 6], color=PALETTE["ink"], lw=1.4)
    ax.text(sx + 100, sy + 12, "200 m", ha="center", fontsize=8,
            color=PALETTE["muted"])

    # North arrow
    nx_, ny_ = bx0 + 60, by1 - 80
    ax.annotate(
        "", xy=(nx_, ny_ + 40), xytext=(nx_, ny_ - 10),
        arrowprops=dict(facecolor=PALETTE["ink"], width=2, headwidth=8,
                        edgecolor="none"),
    )
    ax.text(nx_, ny_ + 50, "N", ha="center", fontsize=9, color=PALETTE["ink"],
            fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Wrote {out_path}  ({time.time()-t0:.1f}s total)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camp", default="Camp 22")
    p.add_argument("--cellsize", type=float, default=10.0)
    p.add_argument("--metric", choices=("LT_t", "LT_f_S1", "LT_f_S2"),
                   default="LT_t")
    p.add_argument("--distance", choices=("euclid", "network"), default="network")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out:
        out = Path(args.out)
    else:
        slug = args.camp.lower().replace(" ", "")
        stem = f"accessibility_fine_{args.metric}_{args.distance}_{int(args.cellsize)}m"
        out = PROJECT_ROOT / "results" / slug / "figures" / f"{stem}.png"
    render(args.camp, out, cell_size_m=args.cellsize, metric=args.metric,
           distance=args.distance)


if __name__ == "__main__":
    main()
