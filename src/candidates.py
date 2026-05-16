"""Candidate latrine site generation on the 50 m grid.

A candidate is any 50 m grid cell that:
- lies inside the target camp polygon, AND
- is not within `MIN_LATRINE_SETBACK` of an existing latrine (avoid pure
  redundancy), AND
- is not within `SENSITIVE_BUFFER` of a sensitive common facility (food
  distribution, health, kitchens, schools, safe spaces, religious sites).

Shelter overlap is not yet enforced — the shelter polygons live in the
`Rohingya_z18_45441_year2022_2025v7.zip` file and require unzipping a
~200 MB GPKG. For the pilot we rely on the sensitive-facility buffer to
keep candidates away from built-up areas. (Adding shelter exclusion is a
TODO; see notes in run_camp22.py.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geo import points_in_poly
from loaders import (
    AccessibilityBaseline,
    CampPolygon,
    CommonFacilities,
)

GRID = 50.0
MIN_LATRINE_SETBACK = 25.0  # meters; same-cell overlap allowed at >=25 m
SENSITIVE_BUFFER = 50.0  # meters; matches the Sphere "safe latrine" envelope


@dataclass
class Candidates:
    xy: np.ndarray  # (N, 2) cell centroid in Web Mercator
    cell_index: np.ndarray  # index into the demand grid (-1 if not a demand cell)
    g_index: np.ndarray  # GPKG-style g_index when matched, else -1


def _min_distance_from(xy: np.ndarray, points: np.ndarray) -> np.ndarray:
    """For each xy[i], minimum Euclidean distance to any row of `points`.

    Returns +inf when `points` is empty.
    """
    if len(points) == 0:
        return np.full(xy.shape[0], np.inf)
    dx = xy[:, 0:1] - points[None, :, 0]
    dy = xy[:, 1:2] - points[None, :, 1]
    return np.sqrt(dx * dx + dy * dy).min(axis=1)


def generate(
    camp: CampPolygon,
    baseline: AccessibilityBaseline,
    existing_latrines_xy: np.ndarray,
    common: CommonFacilities,
    *,
    min_latrine_setback: float = MIN_LATRINE_SETBACK,
    sensitive_buffer: float = SENSITIVE_BUFFER,
    use_demand_grid_only: bool = True,
) -> Candidates:
    """Build feasible candidate set.

    Args:
        camp: camp polygon (we keep candidates strictly inside).
        baseline: published accessibility table (provides the demand grid).
        existing_latrines_xy: (N, 2) Web Mercator coords of existing 2022
            latrines.
        common: common-facility table for the camp; sensitive subset is
            used as exclusion centers.
        use_demand_grid_only: if True, candidates are restricted to the
            already-populated demand cells (recommended — guarantees each
            candidate has a g_index we can map back). If False, we also
            tile the camp polygon with extra empty 50 m cells.
    """
    # Camp-22-only baseline cells (demand grid)
    mask_camp = points_in_poly(baseline.cell_x, baseline.cell_y, camp.merc_list())
    cell_xy = np.column_stack([baseline.cell_x[mask_camp], baseline.cell_y[mask_camp]])
    cell_local_idx = np.where(mask_camp)[0]
    cell_g = baseline.g_index[mask_camp]

    cand_xy_list = [cell_xy]
    cand_cell_idx_list = [cell_local_idx]
    cand_g_list = [cell_g]

    if not use_demand_grid_only:
        # Tile camp bbox at 50 m; keep cells inside polygon and not already
        # present in the demand grid.
        bx0 = camp.merc[:, 0].min()
        by0 = camp.merc[:, 1].min()
        bx1 = camp.merc[:, 0].max()
        by1 = camp.merc[:, 1].max()
        # Snap to the demand grid (it's axis-aligned at 50 m).
        if len(cell_xy):
            ax = cell_xy[0, 0] - GRID * round((cell_xy[0, 0] - bx0) / GRID)
            ay = cell_xy[0, 1] - GRID * round((cell_xy[0, 1] - by0) / GRID)
        else:
            ax, ay = bx0, by0
        xs = np.arange(ax, bx1, GRID)
        ys = np.arange(ay, by1, GRID)
        XX, YY = np.meshgrid(xs, ys)
        ex_xy = np.column_stack([XX.ravel(), YY.ravel()])
        in_poly = points_in_poly(ex_xy[:, 0], ex_xy[:, 1], camp.merc_list())
        ex_xy = ex_xy[in_poly]
        # Drop those already in cell_xy
        if len(ex_xy):
            d_to_demand = _min_distance_from(ex_xy, cell_xy)
            new_only = d_to_demand > GRID * 0.51
            ex_xy = ex_xy[new_only]
        cand_xy_list.append(ex_xy)
        cand_cell_idx_list.append(np.full(len(ex_xy), -1, dtype=int))
        cand_g_list.append(np.full(len(ex_xy), -1, dtype=int))

    xy = np.concatenate(cand_xy_list, axis=0)
    cell_idx = np.concatenate(cand_cell_idx_list, axis=0)
    g_index = np.concatenate(cand_g_list, axis=0)

    # Apply exclusion filters.
    d_latr = _min_distance_from(xy, existing_latrines_xy)
    d_sens = _min_distance_from(xy, common.sensitive_xy())
    keep = (d_latr >= min_latrine_setback) & (d_sens >= sensitive_buffer)

    return Candidates(
        xy=xy[keep],
        cell_index=cell_idx[keep],
        g_index=g_index[keep],
    )
