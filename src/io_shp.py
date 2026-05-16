"""Minimal shapefile reader.

Supports point (type 1) and polyline (type 3) records, read directly from
.shp bytes (no fiona/geopandas). Coordinates are returned as raw doubles
in the file's CRS — we project later via `projection.py`.

Reference: ESRI Shapefile Technical Description (1998).
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path
from typing import Iterator


def _iter_records(data: bytes) -> Iterator[tuple[int, bytes]]:
    """Yield (shape_type, content_bytes) for each record."""
    # File header is 100 bytes.
    offset = 100
    while offset + 8 <= len(data):
        _, content_len_words = struct.unpack(">2i", data[offset : offset + 8])
        content_len = content_len_words * 2
        content = data[offset + 8 : offset + 8 + content_len]
        offset += 8 + content_len
        if len(content) < 4:
            continue
        (shape_type,) = struct.unpack("<i", content[:4])
        yield shape_type, content


def read_points(data: bytes) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for shape_type, content in _iter_records(data):
        if shape_type == 0:  # null
            continue
        if shape_type != 1:
            raise ValueError(f"Expected point shapefile; got shape type {shape_type}")
        x, y = struct.unpack("<2d", content[4:20])
        pts.append((x, y))
    return pts


def read_polylines(data: bytes) -> list[list[tuple[float, float]]]:
    """Return list of parts; each part is a list of (x,y) vertices.

    Shape type 3 is PolyLine. A record can contain multiple parts (multi-
    linestring). We flatten parts so each becomes its own polyline.
    """
    parts_out: list[list[tuple[float, float]]] = []
    for shape_type, content in _iter_records(data):
        if shape_type == 0:
            continue
        if shape_type != 3:
            raise ValueError(f"Expected polyline shapefile; got shape type {shape_type}")
        # bbox (4 doubles), numParts, numPoints
        num_parts, num_points = struct.unpack("<2i", content[36:44])
        part_offsets = list(struct.unpack(f"<{num_parts}i", content[44 : 44 + 4 * num_parts]))
        pts_offset = 44 + 4 * num_parts
        all_pts = list(
            struct.iter_unpack("<2d", content[pts_offset : pts_offset + 16 * num_points])
        )
        part_offsets.append(num_points)
        for i in range(num_parts):
            seg = all_pts[part_offsets[i] : part_offsets[i + 1]]
            parts_out.append([tuple(p) for p in seg])
    return parts_out


def read_points_from_zip(zip_path: Path, member: str) -> list[tuple[float, float]]:
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(member)
    return read_points(data)


def read_polylines_from_path(shp_path: Path) -> list[list[tuple[float, float]]]:
    return read_polylines(shp_path.read_bytes())
