"""E2SFCA and distance matrices — mirrors Accessibility/utils/ACC.R."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

SIGMA = 396.0


def e2sfca(p: np.ndarray, n: np.ndarray, D: np.ndarray, d0: float) -> np.ndarray:
    """Gaussian E2SFCA (same algebra as R e2sfca_cpp_parallel)."""
    p = np.asarray(p, dtype=np.float64).ravel()
    n = np.asarray(n, dtype=np.float64).ravel()
    D = np.asarray(D, dtype=np.float64)
    if D.shape != (p.size, n.size):
        raise ValueError(f"Expected D shape {(p.size, n.size)}, got {D.shape}")
    sigma_sq = SIGMA**2
    mask = D <= d0
    K = np.where(mask, np.exp(-(D**2) / sigma_sq), 0.0)
    denom = (K * p[:, np.newaxis]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rj = np.where(denom > 0, n / denom, 0.0)
    return (K * rj).sum(axis=1)


def acc(
    demand: gpd.GeoDataFrame,
    demand_attr: str,
    supply: gpd.GeoDataFrame,
    supply_attr: str,
    distmat: np.ndarray,
    *,
    d0: float = 1609.0,
    acolname: str = "tsfca",
) -> gpd.GeoDataFrame:
    """Append accessibility column to demand layer."""
    p = demand[demand_attr].to_numpy(dtype=np.float64)
    p = np.where(np.isnan(p), 0.0, p)
    n = supply[supply_attr].to_numpy(dtype=np.float64)
    n = np.where(np.isnan(n), 0.0, n)
    if p.size != distmat.shape[0] or n.size != distmat.shape[1]:
        raise ValueError(
            f"dim mismatch demand {p.size}, supply {n.size}, D {distmat.shape}"
        )
    out = demand.copy()
    out[acolname] = e2sfca(p, n, distmat, d0)
    return out


def _explode_lines(geoms) -> list[LineString]:
    out: list[LineString] = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        if g.geom_type == "LineString":
            out.append(g)
        elif g.geom_type == "MultiLineString":
            out.extend([x for x in g.geoms if not x.is_empty])
        elif g.geom_type == "GeometryCollection":
            for sub in g.geoms:
                if sub.geom_type == "LineString":
                    out.append(sub)
                elif sub.geom_type == "MultiLineString":
                    out.extend([x for x in sub.geoms if not x.is_empty])
    return out


def _densify_line(ls: LineString, step: float) -> LineString:
    if ls.length <= 0 or step <= 0:
        return ls
    dists = np.arange(0.0, ls.length, step)
    coords = [ls.interpolate(d) for d in dists]
    coords.append(Point(ls.coords[-1]))
    return LineString([(p.x, p.y) for p in coords])


def roads_to_graph_edges(
    roads: gpd.GeoDataFrame, *, densify_m: float = 10.0
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]], np.ndarray]:
    """
    Build edge list and coordinate lookup from road LineStrings.
    Returns (edge_from_idx, edge_to_idx, node_coords, edge_weights).
    """
    lines_gs = _explode_lines(roads.geometry)
    if not lines_gs:
        raise ValueError("No line geometry in roads layer.")
    merged = unary_union(lines_gs)
    if merged.geom_type == "LineString":
        lines = [merged]
    elif merged.geom_type == "MultiLineString":
        lines = list(merged.geoms)
    else:
        lines = _explode_lines([merged])

    xy_key: dict[tuple[float, float], int] = {}
    coords_list: list[tuple[float, float]] = []

    def _get_idx(x: float, y: float) -> int:
        key = (round(float(x), 5), round(float(y), 5))
        if key not in xy_key:
            xy_key[key] = len(coords_list)
            coords_list.append(key)
        return xy_key[key]

    from_idx: list[int] = []
    to_idx: list[int] = []
    weights: list[float] = []

    for ls in lines:
        dls = _densify_line(ls, densify_m) if densify_m > 0 else ls
        pts = list(dls.coords)
        for i in range(len(pts) - 1):
            x0, y0 = pts[i][:2]
            x1, y1 = pts[i + 1][:2]
            a = _get_idx(x0, y0)
            b = _get_idx(x1, y1)
            if a == b:
                continue
            w = float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
            u, v = (a, b) if a < b else (b, a)
            from_idx.append(u)
            to_idx.append(v)
            weights.append(w)

    # Collapse parallel edges — keep the shortest segment (pedestrian can use best link).
    best: dict[tuple[int, int], float] = {}
    for u, v, w in zip(from_idx, to_idx, weights):
        e = (u, v)
        if e not in best or w < best[e]:
            best[e] = w
    from_idx = np.fromiter((t[0] for t in best), dtype=np.int32)
    to_idx = np.fromiter((t[1] for t in best), dtype=np.int32)
    weights = np.fromiter(best.values(), dtype=np.float64)

    return (from_idx, to_idx, coords_list, weights)


def point_to_road_dr(points: gpd.GeoSeries, roads_union) -> np.ndarray:
    """Per-point shortest distance to road linework (like R st_distance pt, st_union(roads))."""
    out = np.empty(len(points), dtype=np.float64)
    for i, p in enumerate(points):
        out[i] = float(p.distance(roads_union))
    return out


def accdist(
    demand: gpd.GeoDataFrame,
    supply: gpd.GeoDataFrame,
    *,
    roads: gpd.GeoDataFrame | None = None,
    densify_m: float = 10.0,
) -> np.ndarray:
    """
    Demand (m) × supply (n) distance matrix.
    Euclidean if roads is None; else network distances with R-style hybrid shortcut.
    """
    d_pts = demand.geometry.centroid
    s_pts = supply.geometry.centroid
    coord_dem = np.column_stack([d_pts.x.to_numpy(), d_pts.y.to_numpy()])
    coord_sup = np.column_stack([s_pts.x.to_numpy(), s_pts.y.to_numpy()])
    eD = cdist(coord_dem, coord_sup, metric="euclidean")

    if roads is None:
        return eD

    roads_union = unary_union(_explode_lines(roads.geometry))
    d_ro = point_to_road_dr(d_pts, roads_union)
    d_rd = point_to_road_dr(s_pts, roads_union)
    A = np.add.outer(d_ro, d_rd)

    from_i, to_i, nodes, w = roads_to_graph_edges(roads, densify_m=densify_m)
    n_v = len(nodes)
    coords = np.asarray(nodes, dtype=np.float64)
    tree = cKDTree(coords)

    _, snap_dem = tree.query(coord_dem)
    _, snap_sup = tree.query(coord_sup)

    import igraph as ig

    edge_list = list(zip(from_i.tolist(), to_i.tolist()))
    g = ig.Graph(n=n_v, edges=edge_list, directed=False)
    g.es["weight"] = w.tolist()

    # Shortest paths between snapped nodes
    block = 256
    parts = []
    for i0 in range(0, snap_dem.shape[0], block):
        sd = snap_dem[i0 : i0 + block]
        dm = g.distances(
            source=sd.tolist(),
            target=snap_sup.tolist(),
            weights="weight",
            mode="all",
        )
        parts.append(np.asarray(dm, dtype=np.float64))
    D = np.vstack(parts)

    D = D + d_ro[:, np.newaxis]
    D = D + d_rd[np.newaxis, :]
    D = D + ((eD - D) * (eD < A))
    D[~np.isfinite(D)] = np.inf
    return D


def accdist_euclidean(demand: gpd.GeoDataFrame, supply: gpd.GeoDataFrame) -> np.ndarray:
    return accdist(demand, supply, roads=None)
