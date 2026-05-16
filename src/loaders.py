"""High-level loaders for the WASH project layers.

Returns numpy arrays in EPSG:3857 (Web Mercator), matching the projection
of the Ahn et al. ACC*.gpkg outputs. See `projection.py` for the rationale.

This module is the single source of truth for which file is which. New
camps or layers should be added here, not in callers.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from io_gpkg import read_features
from io_kml import read_camp_polygon, list_camps
from io_shp import read_points_from_zip, read_polylines_from_path
from projection import lonlat_to_merc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACCESS = PROJECT_ROOT / "Refugee_WASH_Accessibility-main" / "Accessibility"
CAMP_OUTLINE_KML = ACCESS / "data" / "camp_outline" / "20230412_a1_camp_outlines.kml"
POPULATION_CSV = ACCESS / "data" / "camp_outline" / "Population.csv"
FACILITY_ZIP = ACCESS / "data" / "facility" / "Rohingya_refugee_response.zip"
FOOTPATH_SHP = ACCESS / "data" / "road" / "20250910_Access_Road_Footpath_all_camps.shp"
COMMON_FACILITIES_XLSX = PROJECT_ROOT / "co_facilities_mapping_dataset_v26.1_hdx.xlsx"
ACC_OUT_DIR = ACCESS / "out"


# --- Camp polygon -----------------------------------------------------------


@dataclass
class CampPolygon:
    name: str
    lonlat: np.ndarray  # (N,2)
    merc: np.ndarray  # (N,2)

    def merc_list(self) -> list[tuple[float, float]]:
        return [(float(x), float(y)) for x, y in self.merc]


def load_camp_polygon(name: str) -> CampPolygon:
    poly_ll = read_camp_polygon(CAMP_OUTLINE_KML, name)
    lonlat = np.array(poly_ll, dtype=float)
    mx, my = lonlat_to_merc(lonlat[:, 0], lonlat[:, 1])
    merc = np.column_stack([mx, my])
    return CampPolygon(name=name, lonlat=lonlat, merc=merc)


def all_camp_names() -> list[str]:
    return list_camps(CAMP_OUTLINE_KML)


# --- Existing WASH facilities -----------------------------------------------


def _load_wash_points(member: str) -> np.ndarray:
    pts = read_points_from_zip(FACILITY_ZIP, member)
    arr = np.array(pts, dtype=float) if pts else np.zeros((0, 2))
    if len(arr) == 0:
        return arr
    mx, my = lonlat_to_merc(arr[:, 0], arr[:, 1])
    return np.column_stack([mx, my])


def load_existing_latrines_2022() -> np.ndarray:
    return _load_wash_points("Rohingya_refugee_response/WASH_Latrine_20220531.shp")


def load_existing_latrines_2024() -> np.ndarray:
    return _load_wash_points("Rohingya_refugee_response/WASH_Latrine_20240815.shp")


def load_handpumps_2022() -> np.ndarray:
    return _load_wash_points("Rohingya_refugee_response/WASH_handpump_20220531.shp")


def load_bathing_2022() -> np.ndarray:
    return _load_wash_points("Rohingya_refugee_response/WASH_Bath_20220531.shp")


# --- Common (ISCG) facilities ----------------------------------------------


SENSITIVE_KEYWORDS = (
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


@dataclass
class CommonFacilities:
    df: pd.DataFrame  # columns: Sector, FacilityType, x_merc, y_merc, is_sensitive

    def sensitive_xy(self) -> np.ndarray:
        sub = self.df[self.df["is_sensitive"]]
        if len(sub) == 0:
            return np.zeros((0, 2))
        return sub[["x_merc", "y_merc"]].to_numpy()


def load_common_facilities(camp_name: str) -> CommonFacilities:
    df = pd.read_excel(
        COMMON_FACILITIES_XLSX,
        sheet_name="FM_Database",
        usecols=[
            "Camp_Name",
            "Sector",
            "Facility Type (sector prefer)",
            "GPS_Latitude",
            "GPS_Longitude",
        ],
    )
    df = df[df["Camp_Name"].astype(str).eq(camp_name)].copy()
    df["GPS_Latitude"] = pd.to_numeric(df["GPS_Latitude"], errors="coerce")
    df["GPS_Longitude"] = pd.to_numeric(df["GPS_Longitude"], errors="coerce")
    df = df.dropna(subset=["GPS_Latitude", "GPS_Longitude"]).copy()
    mx, my = lonlat_to_merc(df["GPS_Longitude"].to_numpy(), df["GPS_Latitude"].to_numpy())
    df["x_merc"] = mx
    df["y_merc"] = my
    df = df.rename(columns={"Facility Type (sector prefer)": "FacilityType"})
    sector = df["Sector"].fillna("").astype(str)
    ftype = df["FacilityType"].fillna("").astype(str)
    haystack = (sector + " " + ftype).str.lower()
    df["is_sensitive"] = haystack.apply(
        lambda s: any(k.lower() in s for k in SENSITIVE_KEYWORDS)
    )
    return CommonFacilities(df=df.reset_index(drop=True))


# --- Footpath network -------------------------------------------------------


def load_footpath_polylines() -> list[list[tuple[float, float]]]:
    """Return polylines reprojected to Web Mercator."""
    parts_ll = read_polylines_from_path(FOOTPATH_SHP)
    out: list[list[tuple[float, float]]] = []
    for part in parts_ll:
        arr = np.array(part, dtype=float)
        mx, my = lonlat_to_merc(arr[:, 0], arr[:, 1])
        out.append(list(zip(mx.tolist(), my.tolist())))
    return out


# --- Population ------------------------------------------------------------


def load_camp_population_table() -> pd.DataFrame:
    return pd.read_csv(POPULATION_CSV)


def camp_population_totals(camp_name: str) -> dict[str, float] | None:
    """Match a camp's name (e.g. 'Camp 22') to a Population.csv row using the
    `Camp` column (e.g. 'Camp22'). Returns Total/Female/Male for 2022.
    """
    df = load_camp_population_table()
    key = camp_name.replace(" ", "").replace("Camp", "Camp")  # 'Camp 22' -> 'Camp22'
    row = df[df["Camp"].astype(str).str.replace(" ", "") == key]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {
        "total_2022": float(r["Total22Feb"]),
        "female_2022": float(r["Female22Feb"]),
        "male_2022": float(r["Male22Feb"]),
        "total_2025": float(r["Total25Jan"]),
        "female_2025": float(r["Female25Jan"]),
        "male_2025": float(r["Male25Jan"]),
    }


# --- Accessibility baseline (Ahn et al. output) -----------------------------


@dataclass
class AccessibilityBaseline:
    cell_x: np.ndarray  # centroid x (merc), shape (N,)
    cell_y: np.ndarray  # centroid y (merc), shape (N,)
    rings: list[list[tuple[float, float]]]  # per-cell outer ring for plotting
    g_index: np.ndarray  # shape (N,)
    pop_total: np.ndarray
    pop_female: np.ndarray
    pop_male: np.ndarray
    lt_total: np.ndarray  # latrine accessibility, total
    lt_male: np.ndarray
    lt_female: np.ndarray
    s_total: np.ndarray  # bathing accessibility
    s_male: np.ndarray
    s_female: np.ndarray


def load_accessibility(gpkg_name: str = "ACC22_S2.gpkg") -> AccessibilityBaseline:
    path = ACC_OUT_DIR / gpkg_name
    xs = []
    ys = []
    rings_all = []
    g_index = []
    pop_t = []
    pop_f = []
    pop_m = []
    lt_t = []
    lt_m = []
    lt_f = []
    s_t = []
    s_m = []
    s_f = []
    for feat in read_features(path):
        ring = feat.rings[0] if feat.rings else []
        if not ring:
            continue
        # centroid of axis-aligned 50m square is just the midpoint
        rx = [p[0] for p in ring]
        ry = [p[1] for p in ring]
        cx = (min(rx) + max(rx)) / 2
        cy = (min(ry) + max(ry)) / 2
        xs.append(cx)
        ys.append(cy)
        rings_all.append(ring)
        a = feat.attrs
        g_index.append(a.get("g_index", 0))
        pop_t.append(a.get("Total22Feb", 0.0) or 0.0)
        pop_f.append(a.get("Female22Feb", 0.0) or 0.0)
        pop_m.append(a.get("Male22Feb", 0.0) or 0.0)
        lt_t.append(a.get("LT_t", 0.0) or 0.0)
        lt_m.append(a.get("LT_m", 0.0) or 0.0)
        lt_f.append(a.get("LT_f", 0.0) or 0.0)
        s_t.append(a.get("S_t", 0.0) or 0.0)
        s_m.append(a.get("S_m", 0.0) or 0.0)
        s_f.append(a.get("S_f", 0.0) or 0.0)
    return AccessibilityBaseline(
        cell_x=np.array(xs),
        cell_y=np.array(ys),
        rings=rings_all,
        g_index=np.array(g_index, dtype=int),
        pop_total=np.array(pop_t),
        pop_female=np.array(pop_f),
        pop_male=np.array(pop_m),
        lt_total=np.array(lt_t),
        lt_male=np.array(lt_m),
        lt_female=np.array(lt_f),
        s_total=np.array(s_t),
        s_male=np.array(s_m),
        s_female=np.array(s_f),
    )


def filter_baseline_to_camp(
    baseline: AccessibilityBaseline, camp: CampPolygon
) -> AccessibilityBaseline:
    from geo import points_in_poly  # local import to avoid cycle issues

    mask = points_in_poly(baseline.cell_x, baseline.cell_y, camp.merc_list())
    rings = [r for keep, r in zip(mask, baseline.rings) if keep]
    return AccessibilityBaseline(
        cell_x=baseline.cell_x[mask],
        cell_y=baseline.cell_y[mask],
        rings=rings,
        g_index=baseline.g_index[mask],
        pop_total=baseline.pop_total[mask],
        pop_female=baseline.pop_female[mask],
        pop_male=baseline.pop_male[mask],
        lt_total=baseline.lt_total[mask],
        lt_male=baseline.lt_male[mask],
        lt_female=baseline.lt_female[mask],
        s_total=baseline.s_total[mask],
        s_male=baseline.s_male[mask],
        s_female=baseline.s_female[mask],
    )


def filter_points_to_camp(points_xy: np.ndarray, camp: CampPolygon) -> np.ndarray:
    if len(points_xy) == 0:
        return points_xy
    from geo import points_in_poly

    mask = points_in_poly(points_xy[:, 0], points_xy[:, 1], camp.merc_list())
    return points_xy[mask]
