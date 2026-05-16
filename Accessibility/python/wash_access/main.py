"""
Reproduce Accessibility/main.R: grid-based E2SFCA for WASH facilities.

Run from repo root or set WASH_ACC_DATA_ROOT to the Accessibility/ directory.

  PYTHONPATH=Accessibility/python python -m wash_access.main
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from shapely import make_valid as shapely_make_valid
from .acc import acc, accdist, accdist_euclidean
from .config import get_data_root, get_out_dir
from .kml_io import read_kml_camp_polygons
from .spatial import make_grid, read_vector, weighted_sum


def _vsizip(zip_path: Path, *inner_parts: str) -> str:
    inner = "/".join(inner_parts)
    return f"/vsizip/{zip_path.resolve()}/{inner}"


def clip_to_boundary(gdf: gpd.GeoDataFrame, bnd: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    u = bnd.union_all()
    return gdf[gdf.intersects(u)].copy()


def load_camps_with_pop(data: Path, aoi) -> gpd.GeoDataFrame:
    kml = data / "data/camp_outline/20230412_a1_camp_outlines.kml"
    pop = pd.read_csv(data / "data/camp_outline/Population.csv")
    camps = read_kml_camp_polygons(kml).merge(pop, on="CampLabel", how="left")
    return camps.to_crs(aoi)


def get_acc_d(acc25: gpd.GeoDataFrame, acc22: gpd.GeoDataFrame, aoigrid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    selected = ["LT_t", "P_t", "S_t", "AVG_t"]
    pop22 = ["Total22Feb", "Male22Feb", "Female22Feb"]
    pop25 = ["Total25Jan", "Male25Jan", "Female25Jan"]
    a22 = acc22[["g_index", *pop22, *selected]].copy()
    a22 = a22.rename(
        columns=dict(zip(pop22 + selected, pop22 + [f"{c}_22" for c in selected]))
    )
    a25 = acc25[["g_index", *pop25, *selected]].copy()
    a25 = a25.rename(
        columns=dict(zip(pop25 + selected, pop25 + [f"{c}_25" for c in selected]))
    )
    out = aoigrid.merge(a22, on="g_index", how="inner").merge(a25, on="g_index", how="inner")
    out["d_pop_t"] = out["Total25Jan"] - out["Total22Feb"]
    out["d_pop_m"] = out["Male25Jan"] - out["Male22Feb"]
    out["d_pop_f"] = out["Female25Jan"] - out["Female22Feb"]
    out["d_LT_t"] = out["LT_t_25"] - out["LT_t_22"]
    out["d_P_t"] = out["P_t_25"] - out["P_t_22"]
    out["d_S_t"] = out["S_t_25"] - out["S_t_22"]
    out["d_AVG_t"] = out["AVG_t_25"] - out["AVG_t_22"]
    keep = [
        "geometry",
        "g_index",
        "d_pop_t",
        "d_pop_m",
        "d_pop_f",
        "d_LT_t",
        "d_P_t",
        "d_S_t",
        "d_AVG_t",
    ]
    return out[[c for c in keep if c in out.columns]]


def validation_plot(
    acc25: gpd.GeoDataFrame, camps: gpd.GeoDataFrame, data: Path, out_dir: Path
) -> None:
    wash_path = data / "data/facility/Overview-and-Monitoring-of-WASH-Per-Camp_Round_5_October-31_2024.csv"
    if not wash_path.exists():
        return
    rep = pd.read_csv(wash_path)
    merged = gpd.sjoin(acc25, camps[["SMSDCamp", "geometry"]], how="inner", predicate="intersects")
    by_camp = merged.groupby("SMSDCamp", as_index=False)["LT_t"].mean().rename(
        columns={"LT_t": "acc25_A1mean"}
    )
    joined = by_camp.merge(
        rep[["SMSDCamp", "people_per_functional_latrine"]],
        on="SMSDCamp",
        how="inner",
    )
    rho, pval = stats.spearmanr(joined["acc25_A1mean"], joined["people_per_functional_latrine"])
    (out_dir / "FigureS5_cortest.txt").write_text(
        f"Spearman correlation test\n rho = {rho}\n p-value = {pval}\n n = {len(joined)}\n",
        encoding="utf-8",
    )
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.91, 4.92))
        ax.scatter(joined["acc25_A1mean"], joined["people_per_functional_latrine"], s=22, alpha=0.8)
        m, b = np.polyfit(joined["acc25_A1mean"], joined["people_per_functional_latrine"], 1)
        x0 = joined["acc25_A1mean"]
        ax.plot(x0, m * x0 + b, color="#e31a1c", linewidth=1)
        ax.set_xlabel("Our accessibility scores to latrines (2025)")
        ax.set_ylabel("People per functional latrine (2024)")
        ax.text(
            0.98,
            0.98,
            f"Spearman ρ = {rho:.2f}\np = {pval:.1e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        fig.tight_layout()
        fig.savefig(out_dir / "FigureS5_validation.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass


def run(
    *,
    data_root: Path | None = None,
    out_dir: Path | None = None,
    use_road_network: bool = True,
    run_validation: bool = True,
    densify_m: float = 10.0,
) -> None:
    data = Path(data_root or get_data_root())
    out = Path(out_dir or get_out_dir())
    out.mkdir(parents=True, exist_ok=True)

    crs = rasterio.open(data / "data/img_sample/115258_198184.png").crs
    cellsize = 50

    print("Reading layers …")
    bnd = read_vector(data / "data/result/Camp_100m_buffer.shp", crs)
    camps = load_camps_with_pop(data, crs)

    b_fp = data / "data/result/Rohingya_z18_45441_year2022_2025v7.zip"
    R2022 = read_vector(
        _vsizip(b_fp, "Rohingya_z18_45441_year2022_v7.gpkg"),
        crs,
    )
    R2025 = read_vector(
        _vsizip(b_fp, "Rohingya_z18_00000_year2025_v7.gpkg"),
        crs,
    )
    R2022["i_index"] = np.arange(1, len(R2022) + 1)
    R2025["i_index"] = np.arange(1, len(R2025) + 1)

    fac = data / "data/facility/Rohingya_refugee_response.zip"
    f_latr22 = read_vector(_vsizip(fac, "Rohingya_refugee_response", "WASH_Latrine_20220531.shp"), crs)
    f_latr24 = read_vector(_vsizip(fac, "Rohingya_refugee_response", "WASH_Latrine_20240815.shp"), crs)
    f_shower22 = read_vector(_vsizip(fac, "Rohingya_refugee_response", "WASH_Bath_20220531.shp"), crs)
    f_shower24 = read_vector(_vsizip(fac, "Rohingya_refugee_response", "WASH_Bath_20240815.shp"), crs)
    f_pumps22 = read_vector(_vsizip(fac, "Rohingya_refugee_response", "WASH_handpump_20220531.shp"), crs)
    f_pumps24 = read_vector(_vsizip(fac, "Rohingya_refugee_response", "WASH_handpump_20240815.shp"), crs)

    f_latr22 = clip_to_boundary(f_latr22, bnd)
    f_latr22["LT_Male"] = f_latr22["LT_Male"].replace(np.nan, 0)
    f_latr22["LT_all_gen"] = f_latr22["LT_all_gen"].replace(np.nan, 0)
    f_latr22["LT_Male_sum"] = f_latr22["LT_all_gen"] + f_latr22["LT_Male"]
    f_latr22["LT_Female"] = f_latr22["LT_Female"].replace(np.nan, 0)
    f_latr22["LT_Female_sum"] = f_latr22["LT_all_gen"] + f_latr22["LT_Female"]

    f_latr24 = clip_to_boundary(f_latr24, bnd)
    f_latr24["nb_Latrine"] = pd.to_numeric(f_latr24["nb_Latrine"], errors="coerce").fillna(0)

    f_shower22 = clip_to_boundary(f_shower22, bnd)
    f_shower22["Bathing_M"] = f_shower22["Bathing_M"] + f_shower22["Bath_gen_u"]
    f_shower22["Bathing_F"] = f_shower22["Bathing_F"] + f_shower22["Bath_gen_u"]

    f_shower24 = clip_to_boundary(f_shower24, bnd)
    f_pumps22 = clip_to_boundary(f_pumps22, bnd)
    f_pumps22["TW"] = pd.to_numeric(f_pumps22["TW"], errors="coerce").fillna(0)
    f_pumps24 = clip_to_boundary(f_pumps24, bnd)
    f_pumps24["nb_TW"] = pd.to_numeric(f_pumps24["nb_TW"], errors="coerce").fillna(0)

    roads = read_vector(data / "data/road/20250910_Access_Road_Footpath_all_camps.shp", crs)
    roads["geometry"] = roads["geometry"].apply(shapely_make_valid)

    print("Building grid & areal interpolation …")
    aoigrid = make_grid(bnd, cellsize=(cellsize, cellsize), clip=True)
    aux_attr = ["Total22Feb", "Female22Feb", "Male22Feb", "Total25Jan", "Female25Jan", "Male25Jan"]
    R2022_grid = weighted_sum(R2022, aoigrid, aux=camps, aux_attr=aux_attr[:3])
    R2025_grid = weighted_sum(R2025, aoigrid, aux=camps, aux_attr=aux_attr[3:6])
    gc.collect()

    if use_road_network:
        print("Scenario I — road-network distances (long step) …")
        D_latr24 = accdist(R2025_grid, f_latr24, roads=roads, densify_m=densify_m)
        D_p24 = accdist(R2025_grid, f_pumps24, roads=roads, densify_m=densify_m)
        D_s24 = accdist(R2025_grid, f_shower24, roads=roads, densify_m=densify_m)

        ACC25 = acc(R2025_grid, "Total25Jan", f_latr24, "nb_Latrine", D_latr24, acolname="LT_t")
        ACC25 = acc(ACC25, "Total25Jan", f_pumps24, "nb_TW", D_p24, acolname="P_t")
        ACC25 = acc(ACC25, "Total25Jan", f_shower24, "nb_WR", D_s24, acolname="S_t")
        ACC25["AVG_t"] = (ACC25["LT_t"] + ACC25["P_t"] + ACC25["S_t"]) / 3.0
        del D_latr24, D_p24, D_s24
        gc.collect()

        print("Road OD matrices for 2022 …")
        D_latr22 = accdist(R2022_grid, f_latr22, roads=roads, densify_m=densify_m)
        D_p22 = accdist(R2022_grid, f_pumps22, roads=roads, densify_m=densify_m)
        D_s22 = accdist(R2022_grid, f_shower22, roads=roads, densify_m=densify_m)

        ACC22 = acc(R2022_grid, "Total22Feb", f_latr22, "LT", D_latr22, acolname="LT_t")
        ACC22 = acc(ACC22, "Male22Feb", f_latr22, "LT_Male_sum", D_latr22, acolname="LT_m")
        ACC22 = acc(ACC22, "Female22Feb", f_latr22, "LT_Female_sum", D_latr22, acolname="LT_f")
        ACC22 = acc(ACC22, "Total22Feb", f_pumps22, "TW", D_p22, acolname="P_t")
        ACC22 = acc(ACC22, "Total22Feb", f_shower22, "Bathing", D_s22, acolname="S_t")
        ACC22 = acc(ACC22, "Male22Feb", f_shower22, "Bathing_M", D_s22, acolname="S_m")
        ACC22 = acc(ACC22, "Female22Feb", f_shower22, "Bathing_F", D_s22, acolname="S_f")
        ACC22["AVG_t"] = (ACC22["LT_t"] + ACC22["P_t"] + ACC22["S_t"]) / 3.0

        ACC22.to_file(out / "ACC22.gpkg", driver="GPKG")
        ACC25.to_file(out / "ACC25.gpkg", driver="GPKG")
        get_acc_d(ACC25, ACC22, aoigrid).to_file(out / "ACC_D.gpkg", driver="GPKG")

        if run_validation:
            validation_plot(ACC25, camps, data, out)

        print("Scenario II …")
        f_latr22_s2 = f_latr22.copy()
        f_latr22_s2["LT_Female_sum"] = f_latr22_s2["LT_all_gen"] * 0.75 + f_latr22_s2["LT_Female"]
        f_shower22_s2 = f_shower22.copy()
        f_shower22_s2["Bathing_F"] = f_shower22_s2["Bathing_F"] * 0.75 + f_shower22_s2["Bath_gen_u"]

        acc_s2 = acc(R2022_grid, "Total22Feb", f_latr22_s2, "LT", D_latr22, acolname="LT_t")
        acc_s2 = acc(acc_s2, "Male22Feb", f_latr22_s2, "LT_Male_sum", D_latr22, acolname="LT_m")
        acc_s2 = acc(acc_s2, "Female22Feb", f_latr22_s2, "LT_Female_sum", D_latr22, acolname="LT_f")
        acc_s2 = acc(acc_s2, "Total22Feb", f_shower22_s2, "Bathing", D_s22, acolname="S_t")
        acc_s2 = acc(acc_s2, "Male22Feb", f_shower22_s2, "Bathing_M", D_s22, acolname="S_m")
        acc_s2 = acc(acc_s2, "Female22Feb", f_shower22_s2, "Bathing_F", D_s22, acolname="S_f")
        acc_s2.to_file(out / "ACC22_S2.gpkg", driver="GPKG")
        del D_latr22
        gc.collect()
    else:
        print("Skipping road-network scenario (--no-road-network).")

    print("Euclidean-distance E2SFCA …")
    D_latr22 = accdist_euclidean(R2022_grid, f_latr22)
    D_p22 = accdist_euclidean(R2022_grid, f_pumps22)
    D_s22 = accdist_euclidean(R2022_grid, f_shower22)

    ACC22e = acc(R2022_grid, "Total22Feb", f_latr22, "LT", D_latr22, acolname="LT_t")
    ACC22e = acc(ACC22e, "Male22Feb", f_latr22, "LT_Male_sum", D_latr22, acolname="LT_m")
    ACC22e = acc(ACC22e, "Female22Feb", f_latr22, "LT_Female_sum", D_latr22, acolname="LT_f")
    ACC22e = acc(ACC22e, "Total22Feb", f_pumps22, "TW", D_p22, acolname="P_t")
    ACC22e = acc(ACC22e, "Total22Feb", f_shower22, "Bathing", D_s22, acolname="S_t")
    ACC22e = acc(ACC22e, "Male22Feb", f_shower22, "Bathing_M", D_s22, acolname="S_m")
    ACC22e = acc(ACC22e, "Female22Feb", f_shower22, "Bathing_F", D_s22, acolname="S_f")
    ACC22e["AVG_t"] = (ACC22e["LT_t"] + ACC22e["P_t"] + ACC22e["S_t"]) / 3.0
    del D_latr22, D_p22, D_s22

    D_latr24 = accdist_euclidean(R2025_grid, f_latr24)
    D_p24 = accdist_euclidean(R2025_grid, f_pumps24)
    D_s24 = accdist_euclidean(R2025_grid, f_shower24)
    ACC25e = acc(R2025_grid, "Total25Jan", f_latr24, "nb_Latrine", D_latr24, acolname="LT_t")
    ACC25e = acc(ACC25e, "Total25Jan", f_pumps24, "nb_TW", D_p24, acolname="P_t")
    ACC25e = acc(ACC25e, "Total25Jan", f_shower24, "nb_WR", D_s24, acolname="S_t")
    ACC25e["AVG_t"] = (ACC25e["LT_t"] + ACC25e["P_t"] + ACC25e["S_t"]) / 3.0

    ACC22e.to_file(out / "ACC22_euclidean.gpkg", driver="GPKG")
    ACC25e.to_file(out / "ACC25_euclidean.gpkg", driver="GPKG")
    get_acc_d(ACC25e, ACC22e, aoigrid).to_file(out / "ACC_D_euclidean.gpkg", driver="GPKG")

    print(f"Done. Outputs written to {out}/")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="WASH accessibility E2SFCA (Python port)")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to Accessibility/ folder (default: ../ relative to this package)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: DATA_ROOT/out)",
    )
    p.add_argument(
        "--no-road-network",
        action="store_true",
        help="Skip slow road-network step; still writes Euclidean GPKGs",
    )
    p.add_argument(
        "--no-validation",
        action="store_true",
        help="Do not write FigureS5 validation outputs",
    )
    p.add_argument(
        "--densify-m",
        type=float,
        default=10.0,
        help="Road line densification spacing (meters) for network edges",
    )
    args = p.parse_args(argv)
    run(
        data_root=args.data_root,
        out_dir=args.out_dir,
        use_road_network=not args.no_road_network,
        run_validation=not args.no_validation,
        densify_m=args.densify_m,
    )


if __name__ == "__main__":
    main()
