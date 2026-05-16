"""Network distance over the camp footpath graph (no geopandas / sfnetworks).

Mirrors what Ahn et al.'s R pipeline does with `cppRouting` and `sfnetworks`,
in pure Python:

1. Build an undirected graph from the footpath polylines. Each vertex of
   each polyline becomes a node; consecutive vertices in the same polyline
   become an edge with cost equal to the Euclidean segment length. Vertices
   from different polylines that are within `snap_tol` meters of each
   other are merged into a single node (junctions).
2. For each origin/destination point, find the nearest graph node and the
   `d_snap` Euclidean offset to it.
3. Run `scipy.sparse.csgraph.dijkstra` with `limit = d0_plus_buffer` to get
   shortest-path distances from origins to all destinations.
4. Output `D[i, j] = d_snap_orig[i] + d_path[i, j] + d_snap_dest[j]`, and
   then `min(D[i,j], euclidean[i,j])` (matches the R code's safety net).

This produces a distance matrix compatible with `e2sfca.e2sfca` and
`marginal.compute`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass
class FootpathGraph:
    nodes_xy: np.ndarray  # (N, 2) node positions in EPSG:3857 meters
    graph: csr_matrix  # (N, N) symmetric, weighted in meters
    n_components: int  # connectivity sanity check


def build_graph(
    polylines: list[list[tuple[float, float]]],
    *,
    snap_tol_m: float = 0.5,
) -> FootpathGraph:
    """Build a sparse undirected graph from polyline parts.

    `snap_tol_m` is the grid size used to deduplicate near-identical
    vertices into shared junctions; 0.5 m is plenty for footpaths whose
    raw vertex coordinates already come from line-feature digitisation.
    """
    if not polylines:
        return FootpathGraph(nodes_xy=np.zeros((0, 2)), graph=csr_matrix((0, 0)), n_components=0)

    # 1) Build a vertex index by snapping to a coarse grid.
    key_to_id: dict[tuple[int, int], int] = {}
    nodes: list[tuple[float, float]] = []

    def node_id(x: float, y: float) -> int:
        k = (int(round(x / snap_tol_m)), int(round(y / snap_tol_m)))
        idx = key_to_id.get(k)
        if idx is None:
            idx = len(nodes)
            key_to_id[k] = idx
            nodes.append((x, y))
        return idx

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for part in polylines:
        if len(part) < 2:
            continue
        prev = node_id(part[0][0], part[0][1])
        for x, y in part[1:]:
            curr = node_id(x, y)
            if curr == prev:
                continue
            dx = nodes[curr][0] - nodes[prev][0]
            dy = nodes[curr][1] - nodes[prev][1]
            d = float(np.hypot(dx, dy))
            rows.append(prev)
            cols.append(curr)
            data.append(d)
            rows.append(curr)
            cols.append(prev)
            data.append(d)
            prev = curr

    N = len(nodes)
    nodes_xy = np.asarray(nodes, dtype=float)
    if not data:
        return FootpathGraph(nodes_xy=nodes_xy, graph=csr_matrix((N, N)), n_components=0)
    graph = csr_matrix((data, (rows, cols)), shape=(N, N))

    # connectivity (informational)
    from scipy.sparse.csgraph import connected_components

    n_comp, _ = connected_components(graph, directed=False)
    return FootpathGraph(nodes_xy=nodes_xy, graph=graph, n_components=n_comp)


def snap_to_graph(
    xy: np.ndarray, fg: FootpathGraph
) -> tuple[np.ndarray, np.ndarray]:
    """Return (node_idx, snap_distance) of length len(xy).

    Uses a 2D KD-tree on the graph's nodes; O(log N) per query.
    """
    if len(fg.nodes_xy) == 0 or len(xy) == 0:
        return np.zeros(len(xy), dtype=int), np.full(len(xy), np.inf)
    tree = cKDTree(fg.nodes_xy)
    d, idx = tree.query(xy, k=1)
    return idx.astype(int), d.astype(float)


def network_distance_matrix(
    dem_xy: np.ndarray,
    sup_xy: np.ndarray,
    fg: FootpathGraph,
    *,
    d0: float = 1609.0,
    buffer_m: float = 200.0,
    short_cut_to_euclid: bool = True,
) -> np.ndarray:
    """Return an (M, K) distance matrix in meters.

    For each origin/destination pair the value is
    `d_snap_orig + d_path + d_snap_dest` (origin walks to nearest network
    node, traverses footpaths, then walks to destination). When the
    short-cut option is on, pairs for which the snap-distance pair-sum
    `d_snap_orig + d_snap_dest` already exceeds the Euclidean distance get
    replaced by the Euclidean value — mirrors `accdist`'s safety branch
    in `Accessibility/utils/ACC.R`. Without that branch network distance
    is always ≥ Euclidean, so using `min(D_net, D_euclid)` blindly is wrong
    — that would make the network metric a no-op.

    Pairs whose Dijkstra distance exceeds `d0 + buffer_m` are returned as
    `+inf`; anything past the catchment is irrelevant to E2SFCA and early
    termination saves a lot of work.
    """
    M = len(dem_xy)
    K = len(sup_xy)
    if M == 0 or K == 0 or len(fg.nodes_xy) == 0:
        return np.full((M, K), np.inf)

    o_idx, o_snap = snap_to_graph(dem_xy, fg)
    d_idx, d_snap = snap_to_graph(sup_xy, fg)

    limit = float(d0 + buffer_m)
    path = dijkstra(fg.graph, indices=o_idx, limit=limit, directed=False)
    d_path = path[:, d_idx]
    D = d_path + o_snap[:, None] + d_snap[None, :]

    if short_cut_to_euclid:
        dx = dem_xy[:, 0:1] - sup_xy[None, :, 0]
        dy = dem_xy[:, 1:2] - sup_xy[None, :, 1]
        eD = np.sqrt(dx * dx + dy * dy)
        snap_sum = o_snap[:, None] + d_snap[None, :]
        # If the snap-cost alone already exceeds Euclidean, going via the
        # network would be wasteful — pretend the user just walks direct.
        # This is the only case where Euclidean can beat the snap+path sum.
        D = np.where(snap_sum > eD, eD, D)

    return D


def cache_path(camp_slug: str, kind: str = "demand_to_supply") -> Path:
    from loaders import PROJECT_ROOT

    return PROJECT_ROOT / "src" / "cache" / f"netdist_{camp_slug}_{kind}.npz"


def cached_or_compute(
    cache: Path,
    dem_xy: np.ndarray,
    sup_xy: np.ndarray,
    fg: FootpathGraph,
    *,
    d0: float = 1609.0,
) -> np.ndarray:
    if cache.exists():
        D = np.load(cache)["D"]
        if D.shape == (len(dem_xy), len(sup_xy)):
            return D
    D = network_distance_matrix(dem_xy, sup_xy, fg, d0=d0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, D=D)
    return D
