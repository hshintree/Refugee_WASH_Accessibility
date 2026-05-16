"""KML parsing for camp boundary polygons."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"kml": "http://www.opengis.net/kml/2.2"}


def read_camp_polygon(kml_path: Path, smsd_camp: str) -> list[tuple[float, float]]:
    """Return the outer ring of the camp's polygon as (lon, lat) tuples."""
    tree = ET.parse(kml_path)
    for placemark in tree.findall(".//kml:Placemark", NS):
        data = {
            sd.attrib.get("name"): (sd.text or "")
            for sd in placemark.findall(".//kml:SimpleData", NS)
        }
        if data.get("SMSDCamp") != smsd_camp:
            continue
        coords_text = placemark.findtext(".//kml:coordinates", namespaces=NS)
        if not coords_text:
            raise ValueError(f"No coordinates found for {smsd_camp}")
        coords: list[tuple[float, float]] = []
        for token in coords_text.split():
            lon, lat, *_ = token.split(",")
            coords.append((float(lon), float(lat)))
        return coords
    raise ValueError(f"Could not find {smsd_camp} in {kml_path}")


def list_camps(kml_path: Path) -> list[str]:
    tree = ET.parse(kml_path)
    names = []
    for placemark in tree.findall(".//kml:Placemark", NS):
        for sd in placemark.findall(".//kml:SimpleData", NS):
            if sd.attrib.get("name") == "SMSDCamp":
                names.append(sd.text or "")
                break
    return names
