"""Load existing latrine positions and per-latrine capacity columns.

Joins the 2022 latrine point shapefile with the matching DBF so each
latrine carries its Web-Mercator (x, y) plus the stance counts used by the
Ahn et al. supply attributes (`LT`, `LT_Male_sum`, `LT_Female_sum`, and the
Scenario 2 female-safety variant).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from io_dbf import read_dbf_from_zip
from io_shp import read_points_from_zip
from loaders import FACILITY_ZIP
from projection import lonlat_to_merc


@dataclass
class Latrines:
    xy_merc: np.ndarray  # (N, 2)
    LT: np.ndarray  # total stances (used for LT_t)
    LT_male_sum: np.ndarray  # LT_all_gen + LT_Male (used for LT_m)
    LT_female_sum: np.ndarray  # LT_all_gen + LT_Female (used for LT_f, Scenario 1)
    LT_female_S2: np.ndarray  # LT_all_gen * 0.75 + LT_Female (Scenario 2)
    df: pd.DataFrame  # full table for reference


def load_latrines_2022() -> Latrines:
    pts = read_points_from_zip(
        FACILITY_ZIP, "Rohingya_refugee_response/WASH_Latrine_20220531.shp"
    )
    df = read_dbf_from_zip(
        FACILITY_ZIP, "Rohingya_refugee_response/WASH_Latrine_20220531.dbf"
    )
    if len(pts) != len(df):
        raise ValueError(f"shp/dbf length mismatch: {len(pts)} vs {len(df)}")
    arr = np.array(pts, dtype=float)
    mx, my = lonlat_to_merc(arr[:, 0], arr[:, 1])
    xy = np.column_stack([mx, my])

    LT = df["LT"].fillna(0).to_numpy(dtype=float)
    LT_all_gen = df["LT_all_gen"].fillna(0).to_numpy(dtype=float)
    LT_Male = df["LT_Male"].fillna(0).to_numpy(dtype=float)
    LT_Female = df["LT_Female"].fillna(0).to_numpy(dtype=float)

    LT_male_sum = LT_all_gen + LT_Male
    LT_female_sum = LT_all_gen + LT_Female
    LT_female_S2 = LT_all_gen * 0.75 + LT_Female

    return Latrines(
        xy_merc=xy,
        LT=LT,
        LT_male_sum=LT_male_sum,
        LT_female_sum=LT_female_sum,
        LT_female_S2=LT_female_S2,
        df=df.reset_index(drop=True),
    )


def filter_latrines_to_bbox(
    latr: Latrines, x_min: float, y_min: float, x_max: float, y_max: float
) -> tuple[Latrines, np.ndarray]:
    """Keep latrines within a bbox. Returns (filtered, mask)."""
    x = latr.xy_merc[:, 0]
    y = latr.xy_merc[:, 1]
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return (
        Latrines(
            xy_merc=latr.xy_merc[mask],
            LT=latr.LT[mask],
            LT_male_sum=latr.LT_male_sum[mask],
            LT_female_sum=latr.LT_female_sum[mask],
            LT_female_S2=latr.LT_female_S2[mask],
            df=latr.df[mask].reset_index(drop=True),
        ),
        mask,
    )
