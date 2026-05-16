"""KML camp outlines: OGR often drops ExtendedData, so we merge SimpleData from XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import geopandas as gpd
import pandas as pd

_KML_NS = {"k": "http://www.opengis.net/kml/2.2"}


def read_kml_camp_polygons(path: Path | str) -> gpd.GeoDataFrame:
    """
    Read camp outlines with SimpleData attributes preserved.
    Geometries are read by GeoPandas; fields are parsed from Placemarks in document order
    (matched to OGR feature order — same as R's terra::vect on this file).
    """
    path = Path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    rows: list[dict[str, str]] = []
    for pm in root.findall(".//k:Placemark", _KML_NS):
        row: dict[str, str] = {}
        for sd in pm.findall(".//k:SimpleData", _KML_NS):
            name = sd.attrib.get("name")
            if name:
                row[name] = (sd.text or "").strip()
        if row:
            rows.append(row)

    gdf = gpd.read_file(path)
    if len(rows) != len(gdf):
        raise ValueError(
            f"KML SimpleData count ({len(rows)}) != geometry count ({len(gdf)}); check driver/KML."
        )
    meta = pd.DataFrame(rows)
    return gpd.GeoDataFrame(
        pd.concat([meta.reset_index(drop=True), gdf.reset_index(drop=True)["geometry"]], axis=1),
        geometry="geometry",
        crs=gdf.crs,
    )
