from __future__ import annotations

import math
import struct
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ACCESS = ROOT / "Refugee_WASH_Accessibility-main" / "Accessibility"
OUT = ROOT / "proposal" / "figures"


def read_kml_camp_polygon(kml_path: Path, smsd_camp: str) -> list[tuple[float, float]]:
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(kml_path)
    for placemark in tree.findall(".//kml:Placemark", ns):
        data = {
            sd.attrib.get("name"): (sd.text or "")
            for sd in placemark.findall(".//kml:SimpleData", ns)
        }
        if data.get("SMSDCamp") != smsd_camp:
            continue
        coords_text = placemark.findtext(".//kml:coordinates", namespaces=ns)
        if not coords_text:
            raise ValueError(f"No coordinates found for {smsd_camp}")
        coords = []
        for token in coords_text.split():
            lon, lat, *_ = token.split(",")
            coords.append((float(lon), float(lat)))
        return coords
    raise ValueError(f"Could not find {smsd_camp} in {kml_path}")


def read_point_shapefile_from_zip(zip_path: Path, shp_member: str) -> list[tuple[float, float]]:
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(shp_member)

    points: list[tuple[float, float]] = []
    offset = 100
    while offset + 8 <= len(data):
        _, content_len_words = struct.unpack(">2i", data[offset : offset + 8])
        content_len = content_len_words * 2
        content = data[offset + 8 : offset + 8 + content_len]
        offset += 8 + content_len
        if len(content) < 4:
            continue
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type == 0:
            continue
        if shape_type != 1:
            raise ValueError(f"Expected point shapefile; got shape type {shape_type}")
        x, y = struct.unpack("<2d", content[4:20])
        points.append((x, y))
    return points


def lonlat_to_local_m(
    lon: float, lat: float, lon0: float, lat0: float
) -> tuple[float, float]:
    r = 6_371_000.0
    x = math.radians(lon - lon0) * r * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * r
    return x, y


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill=(39, 47, 51),
) -> None:
    draw.text(xy, text, font=font, fill=fill)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    camp_name = "Camp 22"
    polygon_ll = read_kml_camp_polygon(
        ACCESS / "data" / "camp_outline" / "20230412_a1_camp_outlines.kml",
        camp_name,
    )
    lon0 = sum(lon for lon, _ in polygon_ll) / len(polygon_ll)
    lat0 = sum(lat for _, lat in polygon_ll) / len(polygon_ll)
    polygon_m = [lonlat_to_local_m(lon, lat, lon0, lat0) for lon, lat in polygon_ll]

    latrine_points = read_point_shapefile_from_zip(
        ACCESS / "data" / "facility" / "Rohingya_refugee_response.zip",
        "Rohingya_refugee_response/WASH_Latrine_20220531.shp",
    )
    latrine_m = [
        lonlat_to_local_m(lon, lat, lon0, lat0)
        for lon, lat in latrine_points
        if point_in_poly(*lonlat_to_local_m(lon, lat, lon0, lat0), polygon_m)
    ]

    common = pd.read_excel(
        ROOT / "co_facilities_mapping_dataset_v26.1_hdx.xlsx",
        sheet_name="FM_Database",
        usecols=[
            "Camp_Name",
            "Sector",
            "Facility Type (sector prefer)",
            "GPS_Latitude",
            "GPS_Longitude",
        ],
    )
    common = common[common["Camp_Name"].astype(str).eq(camp_name)].copy()
    common["GPS_Latitude"] = pd.to_numeric(common["GPS_Latitude"], errors="coerce")
    common["GPS_Longitude"] = pd.to_numeric(common["GPS_Longitude"], errors="coerce")
    common = common.dropna(subset=["GPS_Latitude", "GPS_Longitude"])
    common["x"], common["y"] = zip(
        *[
            lonlat_to_local_m(lon, lat, lon0, lat0)
            for lon, lat in zip(common["GPS_Longitude"], common["GPS_Latitude"])
        ]
    )
    common = common[[point_in_poly(x, y, polygon_m) for x, y in zip(common.x, common.y)]]

    sensitive_terms = (
        "Health",
        "Food",
        "Protection",
        "Education",
        "Religious",
        "Safe",
        "Kitchen",
        "GFD",
        "Learning",
    )
    common["is_sensitive"] = common.apply(
        lambda r: any(
            term.lower()
            in f"{r['Sector']} {r['Facility Type (sector prefer)']}".lower()
            for term in sensitive_terms
        ),
        axis=1,
    )
    sensitive = common[common["is_sensitive"]]

    xs = [p[0] for p in polygon_m]
    ys = [p[1] for p in polygon_m]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    pad = 90
    minx, maxx = minx - pad, maxx + pad
    miny, maxy = miny - pad, maxy + pad

    # A deliberately simple proxy heatmap: nearby existing latrines increase score.
    # This is not the full E2SFCA; it is a parseability sketch from the available layers.
    step = 35.0
    sigma = 140.0
    grid = []
    for y in np.arange(miny, maxy, step):
        for x in np.arange(minx, maxx, step):
            if not point_in_poly(x, y, polygon_m):
                continue
            if latrine_m:
                d2 = np.array([(x - lx) ** 2 + (y - ly) ** 2 for lx, ly in latrine_m])
                score = float(np.exp(-d2 / (sigma**2)).sum())
                nearest_latrine = float(np.sqrt(d2.min()))
            else:
                score = 0.0
                nearest_latrine = float("inf")
            nearest_sensitive = float("inf")
            if len(sensitive):
                ds = np.array([(x - sx) ** 2 + (y - sy) ** 2 for sx, sy in zip(sensitive.x, sensitive.y)])
                nearest_sensitive = float(np.sqrt(ds.min()))
            grid.append((x, y, score, nearest_latrine, nearest_sensitive))

    scores = np.array([g[2] for g in grid])
    lo, hi = np.percentile(scores, [5, 95]) if len(scores) else (0, 1)
    if hi <= lo:
        hi = lo + 1

    candidates = [g for g in grid if g[3] > 75 and g[4] > 35]
    if len(candidates) < 5:
        candidates = [g for g in grid if g[3] > 70 and g[4] > 20]
    candidates.sort(key=lambda g: g[2])
    chosen = []
    for cand in candidates:
        x, y = cand[0], cand[1]
        if all(math.hypot(x - cx, y - cy) > 120 for cx, cy, *_ in chosen):
            chosen.append(cand)
        if len(chosen) == 5:
            break

    w, h = 1500, 1060
    margin_l, margin_r, margin_t, margin_b = 90, 420, 96, 94
    plot_w = w - margin_l - margin_r
    plot_h = h - margin_t - margin_b

    def screen(pt: tuple[float, float]) -> tuple[int, int]:
        x, y = pt
        sx = margin_l + (x - minx) / (maxx - minx) * plot_w
        sy = margin_t + (maxy - y) / (maxy - miny) * plot_h
        return int(round(sx)), int(round(sy))

    img = Image.new("RGB", (w, h), "#f8f7f2")
    draw = ImageDraw.Draw(img, "RGBA")
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, 17)
        small_font = ImageFont.truetype(font_path, 14)
        title_font = ImageFont.truetype(bold_path, 25)
        section_font = ImageFont.truetype(bold_path, 17)
    except OSError:
        font = small_font = title_font = section_font = ImageFont.load_default()

    # Heatmap cells
    cell_px_x = max(3, int(plot_w * step / (maxx - minx)) + 1)
    cell_px_y = max(3, int(plot_h * step / (maxy - miny)) + 1)
    low = np.array([230, 84, 70])
    mid = np.array([247, 236, 190])
    high = np.array([64, 136, 132])
    for x, y, score, *_ in grid:
        t = max(0.0, min(1.0, (score - lo) / (hi - lo)))
        if t < 0.5:
            c = low * (1 - 2 * t) + mid * (2 * t)
        else:
            c = mid * (2 - 2 * t) + high * (2 * t - 1)
        sx, sy = screen((x, y))
        draw.rectangle(
            [sx - cell_px_x // 2, sy - cell_px_y // 2, sx + cell_px_x // 2, sy + cell_px_y // 2],
            fill=tuple(c.astype(int).tolist() + [185]),
        )

    # Boundary
    boundary = [screen(p) for p in polygon_m]
    draw.polygon(boundary, outline=(38, 48, 52, 255), width=3)

    # Facilities
    for x, y in latrine_m:
        sx, sy = screen((x, y))
        draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(31, 78, 121, 180))
    for x, y in zip(sensitive.x, sensitive.y):
        sx, sy = screen((x, y))
        draw.rectangle([sx - 3, sy - 3, sx + 3, sy + 3], fill=(95, 68, 144, 200))
    for x, y, *_ in chosen:
        sx, sy = screen((x, y))
        draw.ellipse([sx - 9, sy - 9, sx + 9, sy + 9], fill=(255, 255, 255, 235), outline=(20, 20, 20, 255), width=2)
        draw.line([sx - 6, sy, sx + 6, sy], fill=(20, 20, 20, 255), width=2)
        draw.line([sx, sy - 6, sx, sy + 6], fill=(20, 20, 20, 255), width=2)

    # Title and notes
    draw_label(draw, (46, 30), "Camp 22 parseability sketch: latrine access proxy + candidate sites", title_font)
    draw_label(draw, (46, 64), "Not full E2SFCA yet; this checks that the camp outline, WASH shapefile, and XLSX context layer line up.", font, fill=(74, 82, 86))

    lx = w - margin_r + 40
    ly = margin_t + 30
    draw_label(draw, (lx, ly), "Layers parsed", section_font)
    ly += 34
    legend_items = [
        ((31, 78, 121), "Existing 2022 latrine points"),
        ((95, 68, 144), "Sensitive/common facilities from XLSX"),
        ((20, 20, 20), "Illustrative low-proximity sites"),
    ]
    for color, label in legend_items:
        draw.rectangle([lx, ly + 2, lx + 20, ly + 22], fill=color + (220,))
        draw_label(draw, (lx + 26, ly), label, font, fill=(58, 66, 70))
        ly += 34

    ly += 14
    draw_label(draw, (lx, ly), "Counts", section_font)
    ly += 32
    for line in [
        f"Latrines inside camp: {len(latrine_m):,}",
        f"Common facilities inside camp: {len(common):,}",
        f"Sensitive/context points: {len(sensitive):,}",
        f"Illustrative low-proximity sites: {len(chosen)}",
    ]:
        draw_label(draw, (lx, ly), line, font, fill=(58, 66, 70))
        ly += 28

    ly += 14
    draw_label(draw, (lx, ly), "Heatmap", section_font)
    ly += 32
    draw_label(draw, (lx, ly), "red = lower crude latrine proximity", font, fill=(58, 66, 70))
    ly += 26
    draw_label(draw, (lx, ly), "teal = higher crude latrine proximity", font, fill=(58, 66, 70))
    ly += 34
    bar_w = 170
    for i in range(bar_w):
        t = i / (bar_w - 1)
        if t < 0.5:
            c = low * (1 - 2 * t) + mid * (2 * t)
        else:
            c = mid * (2 - 2 * t) + high * (2 * t - 1)
        draw.rectangle([lx + i, ly, lx + i + 1, ly + 14], fill=tuple(c.astype(int).tolist() + [255]))
    draw_label(draw, (lx, ly + 22), "low", small_font, fill=(58, 66, 70))
    draw_label(draw, (lx + bar_w - 34, ly + 22), "high", small_font, fill=(58, 66, 70))

    # Simple north arrow and scale bar.
    ax, ay = margin_l + 30, margin_t + 34
    draw.line([ax, ay + 40, ax, ay], fill=(38, 48, 52, 255), width=3)
    draw.polygon([(ax, ay - 8), (ax - 6, ay + 8), (ax + 6, ay + 8)], fill=(38, 48, 52, 255))
    draw_label(draw, (ax - 5, ay + 48), "N", small_font)
    scale_m = 200
    scale_px = int(plot_w * scale_m / (maxx - minx))
    sx, sy = margin_l + 25, h - margin_b + 30
    draw.line([sx, sy, sx + scale_px, sy], fill=(38, 48, 52, 255), width=4)
    draw.line([sx, sy - 7, sx, sy + 7], fill=(38, 48, 52, 255), width=2)
    draw.line([sx + scale_px, sy - 7, sx + scale_px, sy + 7], fill=(38, 48, 52, 255), width=2)
    draw_label(draw, (sx, sy + 14), "200 m", small_font, fill=(58, 66, 70))

    output = OUT / "camp22_parseability_sketch.png"
    img.save(output)
    print(output)


if __name__ == "__main__":
    main()
