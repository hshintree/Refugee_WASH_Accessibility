"""Spatial helpers: grid, areal weighting, CRS — mirrors Accessibility/utils/spatial.R."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


def read_vector(path: str | Path, dst_crs) -> gpd.GeoDataFrame:
    path = str(path)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"No CRS in {path}")
    return gdf.to_crs(dst_crs)


def make_grid(
    boundary: gpd.GeoDataFrame,
    cellsize: tuple[float, float] | float = (50.0, 50.0),
    *,
    clip: bool = True,
) -> gpd.GeoDataFrame:
    """Square grid over bbox of boundary; when clip=True, keep only cells intersecting boundary."""
    if isinstance(cellsize, (int, float)):
        cellsize = (float(cellsize), float(cellsize))
    b_union = boundary.union_all()
    minx, miny, maxx, maxy = b_union.bounds
    xs = np.arange(minx, maxx, cellsize[0])
    ys = np.arange(miny, maxy, cellsize[1])
    polys = [box(x0, y0, x0 + cellsize[0], y0 + cellsize[1]) for x0 in xs for y0 in ys]
    grid = gpd.GeoDataFrame(geometry=polys, crs=boundary.crs)
    grid["g_index"] = np.arange(1, len(grid) + 1, dtype=np.int64)
    if clip:
        grid = grid[grid.intersects(b_union)].copy()
    return grid[["g_index", "geometry"]]


def weighted_sum(
    source: gpd.GeoDataFrame,
    target: gpd.GeoDataFrame,
    *,
    aux: gpd.GeoDataFrame | None = None,
    aux_attr: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Areal interpolation matching R `weighted_sum`:
    - Without aux: sum of intersection areas of `source` with each grid cell.
    - With aux: apportion source counts via camp overlap area, then aggregate to grid.
    """
    if aux is None:
        inter = gpd.overlay(
            source,
            target,
            how="intersection",
            keep_geom_type=False,
        )
        if inter.empty:
            return gpd.GeoDataFrame(columns=["g_index", "area"])
        inter["area"] = inter.geometry.area
        by_cell = inter.groupby("g_index", as_index=False)["area"].sum()
        out = target.merge(by_cell, on="g_index", how="left")
        return out[out["area"].notna()].copy()

    if not aux_attr:
        raise ValueError("aux_attr required when aux is set")

    first = gpd.overlay(source, aux, how="intersection", keep_geom_type=False)
    if first.empty:
        raise ValueError("No overlap between source and auxiliary camp polygons.")
    first["src_aux_area"] = first.geometry.area
    sum_by_camp = first.groupby("CampLabel", as_index=False)["src_aux_area"].sum().rename(
        columns={"src_aux_area": "sum_area"}
    )
    first = first.merge(sum_by_camp, on="CampLabel", how="left")
    first["prop"] = first["src_aux_area"] / first["sum_area"].replace(0, np.nan)
    for c in aux_attr:
        first[c] = first[c] * first["prop"]

    second = gpd.overlay(target, first, how="intersection", keep_geom_type=False)
    second["area_split"] = second.geometry.area
    second["w"] = second["area_split"] / second["src_aux_area"].replace(0, np.nan)
    for c in aux_attr:
        second[c] = second[c] * second["w"]

    agg = second.groupby("g_index", as_index=False)[list(aux_attr)].sum(min_count=1)
    out = target.merge(agg, on="g_index", how="left")
    return out.dropna(subset=aux_attr).copy()
