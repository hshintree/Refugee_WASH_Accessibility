"""Web Mercator (EPSG:3857) forward/inverse.

The Ahn et al. GeoPackage outputs (`ACC22.gpkg`, `ACC22_S2.gpkg`, etc.)
store the 50 m grid in EPSG:3857. We project lon/lat inputs into the same
system so coordinates line up across all layers and distances are
consistent with the published accessibility values.

Note: at Camp 22's latitude (~21.2 N) the Web Mercator scale factor is
1/cos(lat) ≈ 1.071, so a "50 m" grid cell is ~46.7 m on the ground and a
1609 m catchment is ~1502 m on the ground. We preserve the convention to
keep our outputs directly comparable with Ahn et al.
"""

from __future__ import annotations

import math

import numpy as np

R_MERC = 6_378_137.0


def lonlat_to_merc(lon: float | np.ndarray, lat: float | np.ndarray):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = np.radians(lon) * R_MERC
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R_MERC
    return x, y


def merc_to_lonlat(x: float | np.ndarray, y: float | np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lon = np.degrees(x / R_MERC)
    lat = np.degrees(2 * np.arctan(np.exp(y / R_MERC)) - np.pi / 2)
    return lon, lat


def lonlat_pair_to_merc(lon: float, lat: float) -> tuple[float, float]:
    """Scalar variant returning a tuple of floats."""
    x = math.radians(lon) * R_MERC
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R_MERC
    return x, y
