"""Lon/lat ↔ local meters and polygon utilities.

We use the same local equirectangular projection as
`proposal/make_camp22_parseability_figure.py` so coordinates line up across
the project. Camp 22 spans ~1.5 km; the approximation is accurate to better
than 0.1 m at that scale.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

EARTH_R = 6_371_000.0


def lonlat_to_local_m(
    lon: float, lat: float, lon0: float, lat0: float
) -> tuple[float, float]:
    x = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_R
    return x, y


def lonlat_array_to_local_m(
    lons: np.ndarray, lats: np.ndarray, lon0: float, lat0: float
) -> tuple[np.ndarray, np.ndarray]:
    x = np.radians(lons - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(lats - lat0) * EARTH_R
    return x, y


def local_m_to_lonlat(
    x: float, y: float, lon0: float, lat0: float
) -> tuple[float, float]:
    lat = lat0 + math.degrees(y / EARTH_R)
    lon = lon0 + math.degrees(x / (EARTH_R * math.cos(math.radians(lat0))))
    return lon, lat


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def points_in_poly(
    xs: np.ndarray, ys: np.ndarray, poly: list[tuple[float, float]]
) -> np.ndarray:
    """Vectorized ray-casting; returns boolean array."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    inside = np.zeros(xs.shape, dtype=bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = (yi > ys) != (yj > ys)
        denom = (yj - yi) or 1e-12
        x_int = (xj - xi) * (ys - yi) / denom + xi
        crosses = cond & (xs < x_int)
        inside ^= crosses
        j = i
    return inside


def polygon_centroid(poly: list[tuple[float, float]]) -> tuple[float, float]:
    """Area-weighted centroid for a simple polygon (lon/lat or m, same formula)."""
    a = 0.0
    cx = 0.0
    cy = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        a += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    a *= 0.5
    if abs(a) < 1e-12:
        # degenerate; fall back to mean
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return sum(xs) / n, sum(ys) / n
    return cx / (6 * a), cy / (6 * a)


def polygon_bounds(poly: Iterable[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)
