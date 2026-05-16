"""Minimal GeoPackage reader.

GPKG is just SQLite with conventions. Geometry is stored as WKB prefixed by
a small "GP" header (envelope info). We need polygon centroids and ring
vertices for grid cells, so we implement just enough WKB parsing to handle
POLYGON and MULTIPOLYGON in either endian.

We avoid geopandas/fiona on purpose — see memory:feedback-no-heavy-geo-deps.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


WKB_POINT = 1
WKB_LINESTRING = 2
WKB_POLYGON = 3
WKB_MULTIPOINT = 4
WKB_MULTILINESTRING = 5
WKB_MULTIPOLYGON = 6


@dataclass
class GpkgFeature:
    attrs: dict
    rings: list[list[tuple[float, float]]]  # outer ring of each polygon
    srs_id: int


def _parse_gpkg_header(blob: bytes) -> tuple[int, int]:
    """Return (header_length, srs_id). Magic 'GP', version, flags, srs, envelope."""
    if blob[0:2] != b"GP":
        raise ValueError("Not a GPKG geometry blob")
    version = blob[2]
    flags = blob[3]
    # bit 0 = empty geometry, bit 1-3 envelope type, bit 0 of flags-low = endian for header
    envelope_type = (flags >> 1) & 0x07
    header_endian = "<" if (flags & 0x01) else ">"
    srs_id = struct.unpack(header_endian + "i", blob[4:8])[0]
    envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}.get(envelope_type, 0)
    return 8 + envelope_bytes, srs_id


def _parse_ring(buf: bytes, off: int, endian: str) -> tuple[list[tuple[float, float]], int]:
    (npoints,) = struct.unpack_from(endian + "I", buf, off)
    off += 4
    pts = []
    for _ in range(npoints):
        x, y = struct.unpack_from(endian + "2d", buf, off)
        pts.append((x, y))
        off += 16
    return pts, off


def _parse_polygon(buf: bytes, off: int, endian: str) -> tuple[list[tuple[float, float]], int]:
    """Return (outer_ring, new_offset). We discard holes for our use case."""
    (nrings,) = struct.unpack_from(endian + "I", buf, off)
    off += 4
    outer: list[tuple[float, float]] = []
    for r in range(nrings):
        ring, off = _parse_ring(buf, off, endian)
        if r == 0:
            outer = ring
    return outer, off


def _parse_wkb_geometry(buf: bytes, off: int = 0) -> list[list[tuple[float, float]]]:
    endian_byte = buf[off]
    endian = "<" if endian_byte == 1 else ">"
    off += 1
    (wkb_type,) = struct.unpack_from(endian + "I", buf, off)
    off += 4
    if wkb_type == WKB_POLYGON:
        outer, _ = _parse_polygon(buf, off, endian)
        return [outer]
    if wkb_type == WKB_MULTIPOLYGON:
        (npoly,) = struct.unpack_from(endian + "I", buf, off)
        off += 4
        polys: list[list[tuple[float, float]]] = []
        for _ in range(npoly):
            inner_endian_byte = buf[off]
            inner_endian = "<" if inner_endian_byte == 1 else ">"
            off += 1
            (inner_type,) = struct.unpack_from(inner_endian + "I", buf, off)
            off += 4
            if inner_type != WKB_POLYGON:
                raise ValueError(f"MultiPolygon contains non-polygon {inner_type}")
            outer, off = _parse_polygon(buf, off, inner_endian)
            polys.append(outer)
        return polys
    raise ValueError(f"Unsupported WKB type {wkb_type}")


def read_features(path: Path, table: str | None = None) -> Iterator[GpkgFeature]:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    try:
        if table is None:
            row = con.execute(
                "SELECT table_name FROM gpkg_contents WHERE data_type='features' LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError(f"No feature table in {path}")
            table = row[0]
        geom_col = con.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
            (table,),
        ).fetchone()[0]
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        attr_cols = [c for c in cols if c != geom_col]
        sql = (
            f"SELECT {','.join('`'+c+'`' for c in attr_cols)}, `{geom_col}` FROM {table}"
        )
        for row in con.execute(sql):
            attrs = dict(zip(attr_cols, row[:-1]))
            blob = row[-1]
            if blob is None:
                yield GpkgFeature(attrs=attrs, rings=[], srs_id=0)
                continue
            hdr_len, srs_id = _parse_gpkg_header(blob)
            rings = _parse_wkb_geometry(blob, hdr_len)
            yield GpkgFeature(attrs=attrs, rings=rings, srs_id=srs_id)
    finally:
        con.close()
