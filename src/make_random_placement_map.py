"""Pick K random valid latrine sites and visualize their accessibility lift.

Renders three side-by-side panels for a camp at 10 m fine grid:

1. Baseline E2SFCA accessibility.
2. Accessibility after adding `K` randomly chosen new latrines.
3. Δ-accessibility = (after − before).

Feasibility filter for random candidates ("reasonable locations"):

- Inside the camp polygon.
- NOT inside any 2022 shelter footprint (avoid building on top of housing).
- ≥ `min_latrine_setback` from existing latrines (default 25 m).
- ≥ `sensitive_buffer` from any sensitive common facility (default 50 m).

Sampling is rejection-based; we draw uniformly from the bbox and accept
the first `K` that pass all filters.

Run:
    python3 src/make_random_placement_map.py --camp "Camp 22" --K 5 --seed 7
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

from e2sfca import D0, SIGMA, kernel_matrix
from geo import point_in_poly, points_in_poly
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
GRID_M = 10.0


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


def _shelter_mask_on_grid(
    shelters: list[dict],
    bx0: float, by0: float, nx: int, ny: int, cell_size_m: float,
) -> np.ndarray:
    """Boolean mask, True where a shelter footprint covers the cell centroid.

    Uses centroid-in-polygon per cell against per-shelter bboxes to keep
    the loop cheap.
    """
    mask = np.zeros((ny, nx), dtype=bool)
    for sh in shelters:
        outer = sh["rings"][0]
        rx = [p[0] for p in outer]
        ry = [p[1] for p in outer]
        x_min, x_max = min(rx), max(rx)
        y_min, y_max = min(ry), max(ry)
        i0 = max(0, int((x_min - bx0) / cell_size_m) - 1)
        i1 = min(nx, int((x_max - bx0) / cell_size_m) + 2)
        j0 = max(0, int((y_min - by0) / cell_size_m) - 1)
        j1 = min(ny, int((y_max - by0) / cell_size_m) + 2)
        for j in range(j0, j1):
            for i in range(i0, i1):
                cx = bx0 + (i + 0.5) * cell_size_m
                cy = by0 + (j + 0.5) * cell_size_m
                if point_in_poly(cx, cy, outer):
                    mask[j, i] = True
    return mask


@dataclass
class RandomDraw:
    xy: np.ndarray  # (K, 2) accepted candidate positions
    n_attempts: int
    rejected_in_shelter: int
    rejected_near_latrine: int
    rejected_near_sensitive: int
    rejected_outside_camp: int


def sample_random_candidates(
    *,
    camp,
    bx0, by0, bx1, by1,
    shelter_mask, cell_size_m, nx, ny,
    existing_latrines_xy: np.ndarray,
    sensitive_xy: np.ndarray,
    K: int,
    min_latrine_setback: float = 25.0,
    sensitive_buffer: float = 50.0,
    seed: int = 0,
    max_attempts: int = 5000,
) -> RandomDraw:
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    counts = dict(shelter=0, latrine=0, sensitive=0, outside=0)
    attempts = 0
    poly = camp.merc_list()

    while len(accepted) < K and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(bx0, bx1)
        y = rng.uniform(by0, by1)
        if not point_in_poly(x, y, poly):
            counts["outside"] += 1
            continue
        ix = int((x - bx0) / cell_size_m)
        iy = int((y - by0) / cell_size_m)
        if 0 <= ix < nx and 0 <= iy < ny and shelter_mask[iy, ix]:
            counts["shelter"] += 1
            continue
        if len(existing_latrines_xy):
            d2 = ((existing_latrines_xy[:, 0] - x) ** 2
                  + (existing_latrines_xy[:, 1] - y) ** 2)
            if d2.min() < min_latrine_setback ** 2:
                counts["latrine"] += 1
                continue
        if len(sensitive_xy):
            d2s = ((sensitive_xy[:, 0] - x) ** 2
                   + (sensitive_xy[:, 1] - y) ** 2)
            if d2s.min() < sensitive_buffer ** 2:
                counts["sensitive"] += 1
                continue
        # also reject pile-up with previously accepted draws
        ok = True
        for prev in accepted:
            if (prev[0] - x) ** 2 + (prev[1] - y) ** 2 < min_latrine_setback ** 2:
                ok = False
                break
        if not ok:
            counts["latrine"] += 1
            continue
        accepted.append(np.array([x, y]))

    xy = np.vstack(accepted) if accepted else np.zeros((0, 2))
    return RandomDraw(
        xy=xy, n_attempts=attempts,
        rejected_in_shelter=counts["shelter"],
        rejected_near_latrine=counts["latrine"],
        rejected_near_sensitive=counts["sensitive"],
        rejected_outside_camp=counts["outside"],
    )


def _e2sfca_fine(
    fine_xy: np.ndarray, in_poly: np.ndarray, ny: int, nx: int,
    coarse_xy: np.ndarray, coarse_pop: np.ndarray,
    sup_xy: np.ndarray, sup_caps: np.ndarray,
    fg,
):
    """Run Step 1 on coarse grid, Step 2 on fine grid (network distance)."""
    D1 = network_distance_matrix(coarse_xy, sup_xy, fg, d0=D0, buffer_m=200.0)
    D2 = network_distance_matrix(fine_xy, sup_xy, fg, d0=D0, buffer_m=200.0)
    K1 = kernel_matrix(D1, d0=D0, sigma=SIGMA)
    denom = (K1 * coarse_pop[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rj = np.where(denom > 0, sup_caps / denom, 0.0)
    Rj = np.nan_to_num(Rj, nan=0.0, posinf=0.0, neginf=0.0)
    K2 = kernel_matrix(D2, d0=D0, sigma=SIGMA)
    A = (K2 * Rj[None, :]).sum(axis=1).reshape(ny, nx)
    return np.where(in_poly.reshape(ny, nx), A, np.nan)


def _draw_panel(
    ax, fig, A_masked, bx0, by0, nx, ny, cell_size_m,
    camp, shelters, footpaths_in, latrines_xy, sensitive_xy,
    extra_points: list[tuple[np.ndarray, dict]] | None,
    *, vmin, vmax, title: str, cmap, mark_sphere: bool,
):
    ax.set_facecolor(PALETTE["bg"])
    im = ax.imshow(
        A_masked,
        extent=(bx0, bx0 + nx * cell_size_m, by0, by0 + ny * cell_size_m),
        origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
        alpha=0.95, interpolation="bilinear", zorder=1,
    )
    if shelters:
        patches = [MplPolygon(sh["rings"][0]) for sh in shelters]
        ax.add_collection(PatchCollection(
            patches, facecolor="none", edgecolor=PALETTE["shelter_edge"],
            alpha=0.35, linewidths=0.3, zorder=2,
        ))
    if footpaths_in:
        segments = []
        for arr in footpaths_in:
            for i in range(len(arr) - 1):
                segments.append([arr[i], arr[i + 1]])
        ax.add_collection(LineCollection(
            segments, colors=PALETTE["footpath"], linewidths=0.7, alpha=0.8,
            zorder=3,
        ))
    if len(latrines_xy):
        ax.scatter(latrines_xy[:, 0], latrines_xy[:, 1], s=3,
                   c=PALETTE["latrine"], alpha=0.55, linewidths=0, zorder=4)
    if len(sensitive_xy):
        ax.scatter(sensitive_xy[:, 0], sensitive_xy[:, 1], s=14, marker="s",
                   facecolors=PALETTE["sensitive"], edgecolors=PALETTE["bg"],
                   linewidths=0.3, alpha=0.7, zorder=5)
    if extra_points:
        for pts, kwargs in extra_points:
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], **kwargs)
    xs = camp.merc[:, 0].tolist() + [camp.merc[0, 0]]
    ys = camp.merc[:, 1].tolist() + [camp.merc[0, 1]]
    ax.plot(xs, ys, color=PALETTE["boundary"], lw=2.0, zorder=6)
    ax.set_aspect("equal")
    polyx = camp.merc[:, 0]; polyy = camp.merc[:, 1]
    pad = 60
    ax.set_xlim(polyx.min() - pad, polyx.max() + pad)
    ax.set_ylim(polyy.min() - pad, polyy.max() + pad)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(PALETTE["muted"])
        s.set_linewidth(0.6)
    ax.set_title(title, fontsize=11, color=PALETTE["title"], pad=8)
    return im


def render(camp_name: str, out_path: Path, *, K: int = 5, seed: int = 0,
           cell_size_m: float = GRID_M, dpi: int = 200) -> None:
    t0 = time.time()
    print(f"Loading {camp_name} ...")
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

    XX, YY, bx0, by0, bx1, by1, nx, ny = _fine_grid(camp, cell_size_m)
    fine_xy = np.column_stack([XX.ravel(), YY.ravel()])
    in_poly = points_in_poly(fine_xy[:, 0], fine_xy[:, 1], camp.merc_list())

    t = time.time()
    shelter_mask = _shelter_mask_on_grid(
        shelters, bx0, by0, nx, ny, cell_size_m
    )
    print(f"  shelter mask: {shelter_mask.sum()} of {nx*ny} fine cells "
          f"({100*shelter_mask.mean():.1f}%) ({time.time()-t:.1f}s)")

    draw = sample_random_candidates(
        camp=camp, bx0=bx0, by0=by0, bx1=bx1, by1=by1,
        shelter_mask=shelter_mask, cell_size_m=cell_size_m, nx=nx, ny=ny,
        existing_latrines_xy=latr_in, sensitive_xy=sensitive,
        K=K, seed=seed,
    )
    print(f"  drew {len(draw.xy)} sites in {draw.n_attempts} attempts; "
          f"rejected: shelter={draw.rejected_in_shelter}, "
          f"latrine={draw.rejected_near_latrine}, "
          f"sensitive={draw.rejected_near_sensitive}, "
          f"outside-camp={draw.rejected_outside_camp}")
    if len(draw.xy) < K:
        raise RuntimeError(f"Only {len(draw.xy)} valid sites found in "
                            f"{draw.n_attempts} attempts; relax constraints.")

    # supply set (existing) — buffer-filter to keep matrices small
    buffer_m = float(D0) + 200.0
    latr_b, sup_mask = filter_latrines_to_bbox(
        latr_all, bx0 - buffer_m, by0 - buffer_m,
        bx1 + buffer_m, by1 + buffer_m,
    )
    sup_xy_before = latr_b.xy_merc
    sup_caps_before = latr_all.LT[sup_mask]

    sup_xy_after = np.vstack([sup_xy_before, draw.xy])
    new_caps = np.full(len(draw.xy), 1.0)  # 1 stance per new latrine
    sup_caps_after = np.concatenate([sup_caps_before, new_caps])

    # network graph
    paths = load_footpath_polylines()
    nbox = (bx0 - 2000, by0 - 2000, bx1 + 2000, by1 + 2000)
    paths_local = _filter_polylines_to_bbox(paths, *nbox)
    fg = build_graph(paths_local, snap_tol_m=0.5)
    print(f"  graph: {len(fg.nodes_xy)} nodes")

    coarse_xy = np.column_stack([base_full.cell_x, base_full.cell_y])

    print("  computing baseline E2SFCA on fine grid ...")
    A_before = _e2sfca_fine(
        fine_xy, in_poly, ny, nx, coarse_xy, base_full.pop_total,
        sup_xy_before, sup_caps_before, fg,
    )
    print("  computing after-placement E2SFCA on fine grid ...")
    A_after = _e2sfca_fine(
        fine_xy, in_poly, ny, nx, coarse_xy, base_full.pop_total,
        sup_xy_after, sup_caps_after, fg,
    )
    delta = A_after - A_before

    in_camp = in_poly.reshape(ny, nx)
    populated_fine = _shelter_mask_on_grid(
        # use shelter mask centroids in this camp to proxy "where people live"
        shelters, bx0, by0, nx, ny, cell_size_m,
    )  # boolean; used for share-below-Sphere reporting
    print(f"  baseline mean A_i (cells in shelter) = "
          f"{float(np.nanmean(A_before[populated_fine])):.4f}")
    print(f"  after    mean A_i (cells in shelter) = "
          f"{float(np.nanmean(A_after[populated_fine])):.4f}")
    print(f"  baseline share below Sphere = "
          f"{float(np.nanmean(A_before[populated_fine] < SPHERE_TARGET)):.2%}")
    print(f"  after    share below Sphere = "
          f"{float(np.nanmean(A_after[populated_fine] < SPHERE_TARGET)):.2%}")

    # --- figure ---
    fig = plt.figure(figsize=(17.5, 6.5), dpi=dpi, facecolor=PALETTE["bg"])
    ax1 = fig.add_axes([0.03, 0.07, 0.27, 0.84])
    ax2 = fig.add_axes([0.33, 0.07, 0.27, 0.84])
    ax3 = fig.add_axes([0.63, 0.07, 0.27, 0.84])
    side_ax = fig.add_axes([0.91, 0.07, 0.09, 0.84])
    side_ax.set_xlim(0, 1); side_ax.set_ylim(0, 1); side_ax.axis("off")

    cmap_acc = LinearSegmentedColormap.from_list(
        "acc", ["#a33d3d", "#dc9a6a", "#f7ecbe", "#7fb4a8", "#226e6a"]
    )
    cmap_acc.set_bad(color=PALETTE["bg"])
    cmap_div = LinearSegmentedColormap.from_list(
        "delta", ["#7e2222", "#dc9a6a", "#f8f7f2", "#7fb4a8", "#1a5f5b"]
    )
    cmap_div.set_bad(color=PALETTE["bg"])

    nz_before = A_before[np.isfinite(A_before) & populated_fine]
    nz_after = A_after[np.isfinite(A_after) & populated_fine]
    vmax = max(SPHERE_TARGET * 1.4,
               float(np.quantile(nz_before, 0.97)) if len(nz_before) else 0,
               float(np.quantile(nz_after, 0.97)) if len(nz_after) else 0)
    vmin = 0.0

    fp_box = (bx0 - 150, by0 - 150, bx1 + 150, by1 + 150)
    footpaths_in_camp = _filter_polylines_to_bbox(load_footpath_polylines(), *fp_box)

    extras_after = [
        (draw.xy, dict(s=160, marker="*",
                       facecolors=PALETTE["chosen_ip"] if False else "#d9b21a",
                       edgecolors=PALETTE["ink"], linewidths=1.0,
                       zorder=10, label="new (random) latrines")),
    ]
    extras_delta = [
        (draw.xy, dict(s=160, marker="*",
                       facecolors="#d9b21a",
                       edgecolors=PALETTE["ink"], linewidths=1.0,
                       zorder=10)),
    ]

    im1 = _draw_panel(ax1, fig, A_before, bx0, by0, nx, ny, cell_size_m,
                      camp, shelters, footpaths_in_camp, latr_in, sensitive,
                      None, vmin=vmin, vmax=vmax,
                      title="Baseline accessibility (E2SFCA, LT_t)",
                      cmap=cmap_acc, mark_sphere=True)
    im2 = _draw_panel(ax2, fig, A_after, bx0, by0, nx, ny, cell_size_m,
                      camp, shelters, footpaths_in_camp, latr_in, sensitive,
                      extras_after, vmin=vmin, vmax=vmax,
                      title=f"After + {K} random latrines",
                      cmap=cmap_acc, mark_sphere=True)
    vlim = max(1e-6, float(np.nanmax(np.abs(delta))))
    im3 = _draw_panel(ax3, fig, delta, bx0, by0, nx, ny, cell_size_m,
                      camp, shelters, footpaths_in_camp, latr_in, sensitive,
                      extras_delta, vmin=-vlim, vmax=vlim,
                      title="Δ = after − before",
                      cmap=cmap_div, mark_sphere=False)

    # colorbar for the two A panels
    cbar_ax_a = fig.add_axes([0.302, 0.20, 0.010, 0.55])
    cbar_a = fig.colorbar(im2, cax=cbar_ax_a)
    cbar_a.set_label("E2SFCA accessibility", fontsize=8.5,
                      color=PALETTE["muted"])
    cbar_a.ax.axhline(SPHERE_TARGET, color=PALETTE["ink"], lw=1.0, linestyle="--")
    cbar_a.ax.tick_params(labelsize=7, color=PALETTE["muted"])

    # colorbar for the delta panel
    cbar_ax_d = fig.add_axes([0.601, 0.20, 0.010, 0.55])
    cbar_d = fig.colorbar(im3, cax=cbar_ax_d)
    cbar_d.set_label("Δ accessibility", fontsize=8.5, color=PALETTE["muted"])
    cbar_d.ax.tick_params(labelsize=7, color=PALETTE["muted"])

    # side annotations
    side_ax.text(0.0, 0.98, "Random sites", fontsize=10,
                 color=PALETTE["title"], fontweight="bold")
    side_ax.scatter([0.05], [0.93], s=140, marker="*",
                    facecolors="#d9b21a", edgecolors=PALETTE["ink"],
                    linewidths=1.0)
    side_ax.text(0.14, 0.93, f"new latrine ({K})", fontsize=9,
                 color=PALETTE["muted"], va="center")

    side_ax.text(0.0, 0.85, "Filters", fontsize=10,
                 color=PALETTE["title"], fontweight="bold")
    notes = [
        "• not on shelter",
        "• ≥ 25 m from existing latrine",
        "• ≥ 50 m from sensitive facility",
        "• inside camp boundary",
    ]
    y = 0.81
    for n in notes:
        side_ax.text(0.0, y, n, fontsize=8.5, color=PALETTE["muted"])
        y -= 0.04

    side_ax.text(0.0, 0.58, "Numbers", fontsize=10,
                 color=PALETTE["title"], fontweight="bold")
    stat_lines = [
        f"baseline mean A_i: "
        f"{float(np.nanmean(A_before[populated_fine])):.4f}",
        f"after mean A_i: "
        f"{float(np.nanmean(A_after[populated_fine])):.4f}",
        f"baseline < Sphere: "
        f"{float(np.nanmean(A_before[populated_fine] < SPHERE_TARGET)):.1%}",
        f"after < Sphere: "
        f"{float(np.nanmean(A_after[populated_fine] < SPHERE_TARGET)):.1%}",
        f"random seed = {seed}",
        f"σ = {SIGMA:.0f} m, d₀ = {D0:.0f} m",
        f"Sphere = {SPHERE_TARGET:.3f}",
    ]
    y = 0.54
    for s in stat_lines:
        side_ax.text(0.0, y, s, fontsize=8.5, color=PALETTE["muted"])
        y -= 0.035

    side_ax.text(0.0, 0.20, "Rejected during draw", fontsize=10,
                 color=PALETTE["title"], fontweight="bold")
    rej_lines = [
        f"on shelter: {draw.rejected_in_shelter}",
        f"≤25 m of latrine: {draw.rejected_near_latrine}",
        f"≤50 m of sensitive: {draw.rejected_near_sensitive}",
        f"outside camp: {draw.rejected_outside_camp}",
        f"total attempts: {draw.n_attempts}",
    ]
    y = 0.16
    for s in rej_lines:
        side_ax.text(0.0, y, s, fontsize=8.5, color=PALETTE["muted"])
        y -= 0.030

    fig.suptitle(
        f"{camp_name}: {K} random latrines (E2SFCA on 10 m grid, network distance)",
        fontsize=13, color=PALETTE["title"], y=0.97,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"Wrote {out_path} ({time.time()-t0:.1f}s total)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--camp", default="Camp 22")
    p.add_argument("--K", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--cellsize", type=float, default=GRID_M)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out:
        out = Path(args.out)
    else:
        slug = args.camp.lower().replace(" ", "")
        out = (PROJECT_ROOT / "results" / slug / "figures"
                / f"random_K{args.K}_seed{args.seed}_10m.png")
    render(args.camp, out, K=args.K, seed=args.seed,
           cell_size_m=args.cellsize)


if __name__ == "__main__":
    main()
